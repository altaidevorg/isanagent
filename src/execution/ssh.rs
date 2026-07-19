//! SSH remote execution (Phase 4): one **TCP+SSH session per provider session** (connect on
//! `create_session`); Python uses the same framed REPL worker as local (variables persist across
//! `execution_run` calls). Shell mode runs `bash -s` with stdin per run. `cwd_mode` /
//! `cwd_relative` refer to **remote** paths only (never the agent workspace sandbox).

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as B64_ENGINE;
use base64::Engine as _;
use dashmap::DashMap;
use russh::client;
use russh::keys::{self, PrivateKeyWithHashAlg};
use russh::ChannelMsg;
use russh::ChannelStream;
use russh::Disconnect;
use tokio::io::{split, ReadHalf, WriteHalf};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::capabilities::{
    NetworkPolicy, ProviderCapabilities, ProviderCapabilitiesSnapshot, SessionCapabilities,
};
use super::error::ExecutionError;
use super::ids::SessionId;
use super::provider::ExecutionProvider;
use super::repl_framing;
use super::run::{CwdPolicy, RunResult, RunSpec, SessionCreateRequest, SessionHandle};

/// SSH RFC 4254: stderr extended data stream.
const SSH_STDERR: u32 = 1;

/// Build-time config for [`SshExecutionProvider`] (from `AppConfig` in the harness).
#[derive(Debug, Clone)]
pub struct SshExecutionProviderConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub remote_workdir: String,
    pub remote_python: String,
    pub identity_path: Option<String>,
    /// INSECURE escape hatch: when true, ANY server key is accepted (no verification). Default
    /// false — the provider verifies host keys via a trust-on-first-use `known_hosts` store.
    pub accept_unknown_host_keys: bool,
    /// File where verified host-key fingerprints are recorded (`host:port SHA256:...` per line).
    pub known_hosts_path: PathBuf,
    pub max_run_timeout_secs: u64,
    pub max_output_bytes: usize,
    pub max_sessions: usize,
}

struct SshClientHandler {
    accept_unknown_host_keys: bool,
    host_label: String,
    known_hosts_path: PathBuf,
}

/// Outcome of comparing a presented server key against the `known_hosts` store.
#[derive(Debug, PartialEq, Eq)]
enum HostKeyVerdict {
    /// Fingerprint matches the recorded one.
    Match,
    /// First time we have seen this host; fingerprint recorded.
    TrustedOnFirstUse,
    /// A different fingerprint is already recorded — possible MITM; connection must be refused.
    Mismatch { recorded: String },
}

/// Read the recorded fingerprint for `host_label` from the known_hosts file, if any.
fn read_recorded_host_key(path: &Path, host_label: &str) -> std::io::Result<Option<String>> {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((host, fingerprint)) = line.split_once(' ') {
            if host == host_label {
                return Ok(Some(fingerprint.trim().to_string()));
            }
        }
    }
    Ok(None)
}

fn append_known_host(path: &Path, host_label: &str, fingerprint: &str) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{host_label} {fingerprint}")
}

/// Trust-on-first-use host-key verification: record an unknown host, confirm a matching one,
/// and flag a changed key (possible MITM) so the caller can refuse the connection.
fn verify_or_record_known_host(
    path: &Path,
    host_label: &str,
    fingerprint: &str,
) -> std::io::Result<HostKeyVerdict> {
    match read_recorded_host_key(path, host_label)? {
        Some(recorded) if recorded == fingerprint => Ok(HostKeyVerdict::Match),
        Some(recorded) => Ok(HostKeyVerdict::Mismatch { recorded }),
        None => {
            append_known_host(path, host_label, fingerprint)?;
            Ok(HostKeyVerdict::TrustedOnFirstUse)
        }
    }
}

impl russh::client::Handler for SshClientHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        server_public_key: &keys::PublicKey,
    ) -> Result<bool, Self::Error> {
        // Explicit insecure override: accept any key without verification. Off by default.
        if self.accept_unknown_host_keys {
            log::warn!(
                "ssh: accept_unknown_host_keys=true — skipping host-key verification for {} (INSECURE; vulnerable to MITM)",
                self.host_label
            );
            return Ok(true);
        }
        let fingerprint = server_public_key
            .fingerprint(keys::HashAlg::Sha256)
            .to_string();
        match verify_or_record_known_host(&self.known_hosts_path, &self.host_label, &fingerprint) {
            Ok(HostKeyVerdict::Match) => Ok(true),
            Ok(HostKeyVerdict::TrustedOnFirstUse) => {
                log::info!(
                    "ssh: trusting new host key for {} on first use ({})",
                    self.host_label,
                    fingerprint
                );
                Ok(true)
            }
            Ok(HostKeyVerdict::Mismatch { recorded }) => {
                log::error!(
                    "ssh: HOST KEY MISMATCH for {} — refusing connection (possible MITM). \
                     recorded={recorded} presented={fingerprint}. If the change is legitimate, \
                     remove the stale line from {}.",
                    self.host_label,
                    self.known_hosts_path.display()
                );
                Ok(false)
            }
            Err(e) => {
                log::error!(
                    "ssh: known_hosts verification failed for {} ({e}) — refusing connection (fail closed)",
                    self.host_label
                );
                Ok(false)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum SshExecMode {
    Python,
    Shell,
}

struct SshPythonRepl {
    cwd: String,
    read: ReadHalf<ChannelStream<client::Msg>>,
    write: WriteHalf<ChannelStream<client::Msg>>,
}

struct SshConnected {
    handle: client::Handle<SshClientHandler>,
    python_repl: Option<SshPythonRepl>,
}

struct SshSession {
    mode: SshExecMode,
    run_cancel: Mutex<Option<CancellationToken>>,
    connected: Mutex<Option<SshConnected>>,
}

/// SSH-backed [`ExecutionProvider`] (Linux-oriented remote: `bash` + stdin-fed `python` / `bash -s`).
pub struct SshExecutionProvider {
    config: SshExecutionProviderConfig,
    private_key: Option<Arc<keys::PrivateKey>>,
    caps: ProviderCapabilities,
    sessions: DashMap<SessionId, Arc<SshSession>>,
}

impl SshExecutionProvider {
    pub fn new(config: SshExecutionProviderConfig) -> Result<Self, ExecutionError> {
        validate_remote_host(&config.host)?;
        validate_remote_user(&config.user)?;
        let wd = validate_remote_workdir(&config.remote_workdir)?;
        validate_remote_python(&config.remote_python)?;

        let private_key = if let Some(ref p) = config.identity_path {
            let key = keys::load_secret_key(p, None).map_err(|e| {
                ExecutionError::InvalidArgument(format!("ssh: could not load identity_file: {e}"))
            })?;
            Some(Arc::new(key))
        } else {
            None
        };

        if private_key.is_none() {
            let pw = std::env::var("SSH_PASSWORD")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            if pw.is_none() {
                return Err(ExecutionError::InvalidArgument(
                    "ssh: set [harness.execution.ssh].identity_file to a private key, or set \
                     SSH_PASSWORD in the environment (password is not read from config.toml)"
                        .into(),
                ));
            }
        }

        let mut caps = ProviderCapabilities::minimal("ssh");
        caps.languages = vec!["python".into(), "shell".into()];
        caps.supports_persistent_sessions = true;
        caps.supports_interrupt = false;
        caps.supports_package_install = false;
        caps.supports_remote_shell = false;
        caps.jupyter_kernel = false;
        caps.network_policy = NetworkPolicy::Full;
        caps.max_output_bytes_default = Some(config.max_output_bytes as u64);

        Ok(Self {
            config: SshExecutionProviderConfig {
                remote_workdir: wd,
                ..config
            },
            private_key,
            caps,
            sessions: DashMap::new(),
        })
    }

    fn pick_mode(&self, req: &SessionCreateRequest) -> Result<SshExecMode, ExecutionError> {
        let lang = req
            .language
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        match lang {
            None | Some("python") | Some("py") => Ok(SshExecMode::Python),
            Some("shell") | Some("sh") | Some("bash") => Ok(SshExecMode::Shell),
            Some(other) => Err(ExecutionError::InvalidArgument(format!(
                "unsupported language for ssh provider: {other} (supported: python, shell)"
            ))),
        }
    }

    fn session_caps(&self, id: &SessionId, mode: &SshExecMode) -> SessionCapabilities {
        let active_language = match mode {
            SshExecMode::Python => Some("python".into()),
            SshExecMode::Shell => Some("shell".into()),
        };
        let wd = &self.config.remote_workdir;
        let mut ext = std::collections::BTreeMap::new();
        if matches!(mode, SshExecMode::Python) {
            ext.insert("ssh_python_repl".into(), serde_json::Value::Bool(true));
        }
        SessionCapabilities {
            session_id: id.clone(),
            schema_version: 1,
            provider_id: "ssh".into(),
            active_language,
            gpu_visible: None,
            working_directory_display: Some(format!(
                "ssh {}@{}:{} (session_default cwd {}; each run mkdir -p that path on the remote before cd; sandbox_relative = remote path under that dir or absolute on remote)",
                self.config.user, self.config.host, self.config.port, wd
            )),
            provider_snapshot: ProviderCapabilitiesSnapshot {
                supports_interrupt: self.caps.supports_interrupt,
                supports_package_install: self.caps.supports_package_install,
                supports_remote_shell: self.caps.supports_remote_shell,
                jupyter_kernel: self.caps.jupyter_kernel,
                network_policy: self.caps.network_policy,
            },
            extensions: ext,
        }
    }
}

/// Resolve remote working directory for one run. `sandbox_relative` is **not** the agent sandbox:
/// absolute paths are used as-is on the remote; relative paths join `[harness.execution.ssh].remote_workdir`.
pub fn resolve_ssh_run_cwd(remote_root: &str, cwd: &CwdPolicy) -> Result<String, ExecutionError> {
    match cwd {
        CwdPolicy::SessionDefault => Ok(remote_root.to_string()),
        CwdPolicy::SandboxRelative(rel) => {
            let t = rel.trim();
            if t.is_empty() {
                return Err(ExecutionError::InvalidArgument(
                    "ssh: cwd_relative must be non-empty when cwd_mode is sandbox_relative".into(),
                ));
            }
            if t.starts_with('/') {
                validate_remote_workdir(t)
            } else {
                for seg in t.split('/') {
                    if seg == ".." {
                        return Err(ExecutionError::InvalidArgument(
                            "ssh: cwd_relative must not contain '..' path segments".into(),
                        ));
                    }
                }
                let root = remote_root.trim_end_matches('/');
                let rel = t.trim_start_matches('/');
                let joined = format!("{root}/{rel}");
                validate_remote_workdir(&joined)
            }
        }
    }
}

/// `mkdir -p` then `cd` into `wd` (must already pass [`validate_remote_workdir`] so it contains no `'`.
fn remote_mkdir_cd_prefix(wd: &str) -> Result<String, ExecutionError> {
    if wd.contains('\'') {
        return Err(ExecutionError::InvalidArgument(
            "ssh: remote path must not contain single quotes".into(),
        ));
    }
    Ok(format!("mkdir -p '{wd}' && cd '{wd}' && "))
}

/// One-shot shell: ensure remote cwd exists, then `exec bash -s`; user source is written to session stdin.
fn ssh_remote_shell_line(wd: &str) -> Result<String, ExecutionError> {
    let prefix = remote_mkdir_cd_prefix(wd)?;
    let line = format!("{prefix}exec bash -s");
    validate_safe_remote_exec_line(&line)?;
    Ok(line)
}

fn ssh_python_repl_exec_line(wd: &str, py: &str) -> Result<String, ExecutionError> {
    let enc = B64_ENGINE.encode(repl_framing::PYTHON_REPL_BOOTSTRAP.as_bytes());
    if enc.len() > 400_000 {
        return Err(ExecutionError::InvalidArgument(
            "ssh: repl bootstrap encoding unexpectedly large".into(),
        ));
    }
    let prefix = remote_mkdir_cd_prefix(wd)?;
    let line = format!(
        "{prefix}exec '{py}' -u -c 'import base64;exec(compile(__import__(\"base64\").standard_b64decode(\"{enc}\"),\"<isanagent>\",\"exec\"))'"
    );
    validate_safe_remote_exec_line(&line)?;
    Ok(line)
}

fn validate_safe_remote_exec_line(line: &str) -> Result<(), ExecutionError> {
    if line.contains('`') {
        return Err(ExecutionError::InvalidArgument(
            "ssh: unexpected shell metacharacters in remote exec line".into(),
        ));
    }
    Ok(())
}

/// Absolute POSIX path on the remote: `/` + safe characters only, no `..`.
pub fn validate_remote_workdir(raw: &str) -> Result<String, ExecutionError> {
    let t = raw.trim();
    if t.is_empty() {
        return Err(ExecutionError::InvalidArgument(
            "ssh remote_workdir must be non-empty".into(),
        ));
    }
    if !t.starts_with('/') {
        return Err(ExecutionError::InvalidArgument(format!(
            "ssh remote_workdir must be an absolute path (got {t:?})"
        )));
    }
    if t.contains("..") {
        return Err(ExecutionError::InvalidArgument(
            "ssh remote_workdir must not contain '..'".into(),
        ));
    }
    if t.len() > 512 {
        return Err(ExecutionError::InvalidArgument(
            "ssh remote_workdir exceeds 512 characters".into(),
        ));
    }
    for ch in t.chars() {
        if ch.is_ascii_alphanumeric() || ch == '/' || ch == '_' || ch == '-' || ch == '.' {
            continue;
        }
        return Err(ExecutionError::InvalidArgument(format!(
            "ssh remote_workdir contains disallowed character {ch:?} (allowed: A–Z, a–z, 0–9, /, _, -, .)"
        )));
    }
    Ok(t.to_string())
}

fn validate_remote_host(host: &str) -> Result<(), ExecutionError> {
    let t = host.trim();
    if t.is_empty() {
        return Err(ExecutionError::InvalidArgument(
            "ssh host must be non-empty".into(),
        ));
    }
    if t.chars().any(|c| c.is_whitespace()) {
        return Err(ExecutionError::InvalidArgument(
            "ssh host must not contain whitespace".into(),
        ));
    }
    if t.len() > 253 {
        return Err(ExecutionError::InvalidArgument(
            "ssh host exceeds 253 characters".into(),
        ));
    }
    Ok(())
}

fn validate_remote_user(user: &str) -> Result<(), ExecutionError> {
    let t = user.trim();
    if t.is_empty() {
        return Err(ExecutionError::InvalidArgument(
            "ssh user must be non-empty".into(),
        ));
    }
    if t.len() > 128 {
        return Err(ExecutionError::InvalidArgument(
            "ssh user exceeds 128 characters".into(),
        ));
    }
    for ch in t.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
            continue;
        }
        return Err(ExecutionError::InvalidArgument(format!(
            "ssh user contains disallowed character {ch:?}"
        )));
    }
    Ok(())
}

fn validate_remote_python(py: &str) -> Result<(), ExecutionError> {
    let t = py.trim();
    if t.is_empty() {
        return Err(ExecutionError::InvalidArgument(
            "ssh remote_python must be non-empty".into(),
        ));
    }
    if t.len() > 256 {
        return Err(ExecutionError::InvalidArgument(
            "ssh remote_python exceeds 256 characters".into(),
        ));
    }
    for ch in t.chars() {
        if ch.is_ascii_alphanumeric() || ch == '/' || ch == '_' || ch == '-' || ch == '.' {
            continue;
        }
        return Err(ExecutionError::InvalidArgument(format!(
            "ssh remote_python contains disallowed character {ch:?}"
        )));
    }
    Ok(())
}

async fn authenticate_session(
    private_key: &Option<Arc<keys::PrivateKey>>,
    handle: &mut client::Handle<SshClientHandler>,
    user: &str,
) -> Result<(), ExecutionError> {
    let user = user.to_string();
    if let Some(key) = private_key {
        let rsa_hash = handle
            .best_supported_rsa_hash()
            .await
            .map_err(|e| ExecutionError::Provider(format!("ssh: rsa hash probe: {e}")))?;
        let auth = handle
            .authenticate_publickey(
                user.clone(),
                PrivateKeyWithHashAlg::new(key.clone(), rsa_hash.flatten()),
            )
            .await
            .map_err(|e| ExecutionError::Provider(format!("ssh: publickey auth: {e}")))?;
        if auth.success() {
            return Ok(());
        }
    }
    let pw = std::env::var("SSH_PASSWORD")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(pw) = pw {
        let auth = handle
            .authenticate_password(user.clone(), pw)
            .await
            .map_err(|e| ExecutionError::Provider(format!("ssh: password auth: {e}")))?;
        if auth.success() {
            return Ok(());
        }
    }
    Err(ExecutionError::Provider(
        "ssh: all configured authentication methods failed".into(),
    ))
}

async fn open_ssh_handle(
    config: &SshExecutionProviderConfig,
    private_key: &Option<Arc<keys::PrivateKey>>,
) -> Result<client::Handle<SshClientHandler>, ExecutionError> {
    let client_cfg = Arc::new(client::Config::default());
    let handler = SshClientHandler {
        accept_unknown_host_keys: config.accept_unknown_host_keys,
        host_label: format!("{}:{}", config.host, config.port),
        known_hosts_path: config.known_hosts_path.clone(),
    };
    let addrs = (config.host.as_str(), config.port);
    let mut handle = client::connect(client_cfg, addrs, handler)
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: connect: {e}")))?;

    authenticate_session(private_key, &mut handle, &config.user).await?;
    Ok(handle)
}

const SSH_REPL_PROBE_MAX_EACH: usize = 8192;

async fn open_ssh_python_repl_once(
    conn: &mut SshConnected,
    cwd: &str,
    py: &str,
) -> Result<SshPythonRepl, ExecutionError> {
    let ch = conn
        .handle
        .channel_open_session()
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: open session channel: {e}")))?;
    let line = ssh_python_repl_exec_line(cwd, py)?;
    ch.exec(true, line)
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: exec: {e}")))?;
    let stream = ch.into_stream();
    let (mut read, mut write) = split(stream);
    match repl_framing::repl_round_trip(&mut write, &mut read, "pass", None, None, SSH_REPL_PROBE_MAX_EACH).await
    {
        Ok((_stdout, _stderr, 0)) => Ok(SshPythonRepl {
            cwd: cwd.to_string(),
            read,
            write,
        }),
        Ok((stdout, stderr, code)) => Err(ExecutionError::Provider(format!(
            "ssh: Python REPL failed self-test on remote (exit {code}); stdout={stdout:?} stderr={stderr:?}. \
             Check [harness.execution.ssh].remote_python and disk permissions for the cwd."
        ))),
        Err(e) => Err(ExecutionError::Provider(format!(
            "ssh: Python REPL dropped during startup ({e}). \
             Typical causes: remote_python missing on PATH, shell exec line rejected by sshd, or mkdir/cd failing for the cwd. \
             If the host was never seen before, set accept_unknown_host_keys or add the host key."
        ))),
    }
}

async fn ensure_ssh_python_repl(
    conn: &mut SshConnected,
    cwd: &str,
    py: &str,
) -> Result<(), ExecutionError> {
    let same = conn.python_repl.as_ref().is_some_and(|r| r.cwd == cwd);
    if same {
        return Ok(());
    }
    conn.python_repl.take();
    let mut last = None::<ExecutionError>;
    for _ in 0..2 {
        match open_ssh_python_repl_once(conn, cwd, py).await {
            Ok(repl) => {
                conn.python_repl = Some(repl);
                return Ok(());
            }
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| {
        ExecutionError::Provider("ssh: Python REPL could not be started".into())
    }))
}

async fn run_ssh_channel_oneway(
    handle: &mut client::Handle<SshClientHandler>,
    remote_exec_line: &str,
    stdin_body: Vec<u8>,
    max_output_bytes: usize,
) -> Result<RunResult, ExecutionError> {
    let max_each = (max_output_bytes / 2).max(1024);
    let mut channel = handle
        .channel_open_session()
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: open session channel: {e}")))?;

    channel
        .exec(true, remote_exec_line)
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: exec: {e}")))?;

    channel
        .data(Cursor::new(stdin_body))
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: stdin: {e}")))?;

    channel
        .eof()
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: eof: {e}")))?;

    let mut stdout = Vec::<u8>::new();
    let mut stderr = Vec::<u8>::new();
    let mut code: Option<u32> = None;

    loop {
        let Some(msg) = channel.wait().await else {
            break;
        };
        match msg {
            ChannelMsg::Data { data } if stdout.len() < max_each => {
                let take = (max_each - stdout.len()).min(data.len());
                stdout.extend_from_slice(&data[..take]);
            }
            ChannelMsg::ExtendedData { data, ext }
                if ext == SSH_STDERR && stderr.len() < max_each =>
            {
                let take = (max_each - stderr.len()).min(data.len());
                stderr.extend_from_slice(&data[..take]);
            }
            ChannelMsg::ExitStatus { exit_status } => {
                code = Some(exit_status);
            }
            ChannelMsg::Eof | ChannelMsg::Close => {}
            _ => {}
        }
    }

    Ok(RunResult::new(
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr),
        code.map(|c| c as i32),
    ))
}

fn truncate_run_result(mut r: RunResult, max_total: usize) -> RunResult {
    let max_each = (max_total / 2).max(1024);
    r.stdout = repl_framing::truncate_utf8_str_cap(&r.stdout, max_each);
    r.stderr = repl_framing::truncate_utf8_str_cap(&r.stderr, max_each);
    r
}

#[async_trait]
impl ExecutionProvider for SshExecutionProvider {
    fn provider_id(&self) -> &str {
        "ssh"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.caps.clone()
    }

    async fn create_session(
        &self,
        req: SessionCreateRequest,
    ) -> Result<SessionHandle, ExecutionError> {
        if self.config.max_sessions > 0 && self.sessions.len() >= self.config.max_sessions {
            return Err(ExecutionError::limit_exceeded(
                "sessions",
                format!("max_sessions={} reached", self.config.max_sessions),
            ));
        }
        let mode = self.pick_mode(&req)?;
        let handle = open_ssh_handle(&self.config, &self.private_key).await?;
        let id = SessionId::new(uuid::Uuid::new_v4().to_string());
        let caps = self.session_caps(&id, &mode);
        let session = Arc::new(SshSession {
            mode: mode.clone(),
            run_cancel: Mutex::new(None),
            connected: Mutex::new(Some(SshConnected {
                handle,
                python_repl: None,
            })),
        });
        self.sessions.insert(id.clone(), session);
        Ok(SessionHandle {
            id,
            capabilities: caps,
        })
    }

    async fn close_session(&self, session_id: &SessionId) -> Result<(), ExecutionError> {
        if let Some((_, sess)) = self.sessions.remove(session_id) {
            if let Some(t) = sess.run_cancel.lock().await.take() {
                t.cancel();
            }
            let mut cg = sess.connected.lock().await;
            if let Some(mut c) = cg.take() {
                c.python_repl.take();
                let _ = c.handle.disconnect(Disconnect::ByApplication, "", "").await;
            }
            return Ok(());
        }
        Err(ExecutionError::InvalidSession(session_id.to_string()))
    }

    async fn run(
        &self,
        session_id: &SessionId,
        spec: RunSpec,
    ) -> Result<RunResult, ExecutionError> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| ExecutionError::InvalidSession(session_id.to_string()))?;
        let session = session.value().clone();
        let sess = session.clone();
        let mode = session.mode.clone();

        let cancel = CancellationToken::new();
        {
            let mut slot = session.run_cancel.lock().await;
            if slot.is_some() {
                return Err(ExecutionError::unsupported(
                    "run",
                    "session already has an active run",
                ));
            }
            *slot = Some(cancel.clone());
        }

        let timeout_secs = spec
            .timeout_secs
            .min(self.config.max_run_timeout_secs)
            .max(1);

        let cwd = resolve_ssh_run_cwd(&self.config.remote_workdir, &spec.cwd)?;
        let code = spec.code;
        let remote_py = self.config.remote_python.clone();
        let max_out = self.config.max_output_bytes;
        let sid = session_id.to_string();

        let work = async move {
            let mut cg = sess.connected.lock().await;
            let conn = cg.as_mut().ok_or_else(|| {
                ExecutionError::InvalidSession(format!("{sid} (ssh session is not connected)"))
            })?;
            match mode {
                SshExecMode::Python => {
                    ensure_ssh_python_repl(conn, &cwd, &remote_py).await?;
                    let repl = conn.python_repl.as_mut().ok_or_else(|| {
                        ExecutionError::Provider("ssh: repl failed to start".into())
                    })?;
                    let max_each = (max_out / 2).max(1024);
                    let (stdout, stderr, st) = repl_framing::repl_round_trip(
                        &mut repl.write,
                        &mut repl.read,
                        &code,
                        None,
                        None,
                        max_each,
                    )
                    .await?;
                    Ok(RunResult::new(stdout, stderr, Some(st)))
                }
                SshExecMode::Shell => {
                    let line = ssh_remote_shell_line(&cwd)?;
                    run_ssh_channel_oneway(&mut conn.handle, &line, code.into_bytes(), max_out)
                        .await
                }
            }
        };

        let mut jh = tokio::spawn(work);
        let sleep = tokio::time::sleep(Duration::from_secs(timeout_secs));
        tokio::pin!(sleep);
        let result: Result<RunResult, ExecutionError> = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                jh.abort();
                Err(ExecutionError::Cancelled)
            }
            _ = &mut sleep => {
                jh.abort();
                Err(ExecutionError::Timeout { timeout_secs })
            }
            joined = &mut jh => match joined {
                Ok(Ok(r)) => Ok(truncate_run_result(r, max_out)),
                Ok(Err(e)) => Err(e),
                Err(e) if e.is_cancelled() => Err(ExecutionError::Cancelled),
                Err(e) => Err(ExecutionError::Provider(format!("ssh run join: {e}"))),
            },
        };

        if result.is_err() {
            let mut cg = session.connected.lock().await;
            if let Some(c) = cg.as_mut() {
                c.python_repl.take();
            }
        }

        *session.run_cancel.lock().await = None;
        result
    }

    async fn cancel(&self, session_id: &SessionId) -> Result<(), ExecutionError> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| ExecutionError::InvalidSession(session_id.to_string()))?;
        let session = session.value().clone();
        if let Some(t) = session.run_cancel.lock().await.clone() {
            t.cancel();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_workdir_accepts_tmp() {
        assert_eq!(
            validate_remote_workdir("/tmp/isanagent-exec").unwrap(),
            "/tmp/isanagent-exec"
        );
    }

    // 0.4: trust-on-first-use host-key verification.
    fn tofu_temp_path() -> PathBuf {
        std::env::temp_dir()
            .join(format!("isanagent-knownhosts-{}", uuid::Uuid::new_v4()))
            .join("known_hosts")
    }

    #[test]
    fn host_key_tofu_records_then_matches() {
        let path = tofu_temp_path();
        // First sight of the host -> trusted on first use and recorded.
        assert_eq!(
            verify_or_record_known_host(&path, "10.0.0.5:22", "SHA256:abc").unwrap(),
            HostKeyVerdict::TrustedOnFirstUse
        );
        // Same fingerprint next time -> match.
        assert_eq!(
            verify_or_record_known_host(&path, "10.0.0.5:22", "SHA256:abc").unwrap(),
            HostKeyVerdict::Match
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn host_key_mismatch_is_detected() {
        let path = tofu_temp_path();
        verify_or_record_known_host(&path, "host:22", "SHA256:original").unwrap();
        // A different key for the same host -> mismatch (possible MITM); caller must refuse.
        assert_eq!(
            verify_or_record_known_host(&path, "host:22", "SHA256:changed").unwrap(),
            HostKeyVerdict::Mismatch {
                recorded: "SHA256:original".to_string()
            }
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn host_key_store_is_per_host() {
        let path = tofu_temp_path();
        verify_or_record_known_host(&path, "a:22", "SHA256:aaa").unwrap();
        // A different host is independent -> first use, not a mismatch.
        assert_eq!(
            verify_or_record_known_host(&path, "b:22", "SHA256:bbb").unwrap(),
            HostKeyVerdict::TrustedOnFirstUse
        );
        assert_eq!(
            verify_or_record_known_host(&path, "a:22", "SHA256:aaa").unwrap(),
            HostKeyVerdict::Match
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn validate_workdir_rejects_relative() {
        assert!(validate_remote_workdir("tmp/x").is_err());
    }

    #[test]
    fn validate_workdir_rejects_dotdot() {
        assert!(validate_remote_workdir("/tmp/../etc").is_err());
    }

    #[test]
    fn resolve_sandbox_relative_joins_root() {
        assert_eq!(
            resolve_ssh_run_cwd(
                "/home/ubuntu",
                &CwdPolicy::SandboxRelative("proj/gpu".into())
            )
            .unwrap(),
            "/home/ubuntu/proj/gpu"
        );
    }

    #[test]
    fn resolve_absolute_in_sandbox_relative() {
        assert_eq!(
            resolve_ssh_run_cwd(
                "/home/ubuntu",
                &CwdPolicy::SandboxRelative("/var/log".into())
            )
            .unwrap(),
            "/var/log"
        );
    }

    #[test]
    fn remote_shell_line() {
        let s = ssh_remote_shell_line("/tmp/w").unwrap();
        assert!(s.contains("mkdir -p '/tmp/w'"));
        assert!(s.contains("cd '/tmp/w'"));
        assert!(s.contains("bash -s"));
    }

    #[test]
    fn python_repl_line_contains_b64() {
        let s = ssh_python_repl_exec_line("/tmp/w", "python3").unwrap();
        assert!(s.contains("mkdir -p '/tmp/w'"));
        assert!(s.contains("cd '/tmp/w'"));
        assert!(s.contains("python3"));
        assert!(s.contains("standard_b64decode"));
    }
}
