//! SSH remote execution (Phase 4 MVP): one connection + exec per `run`; sessions track language only.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use dashmap::DashMap;
use russh::client;
use russh::keys::{self, PrivateKeyWithHashAlg};
use russh::ChannelMsg;
use russh::Disconnect;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::capabilities::{
    NetworkPolicy, ProviderCapabilities, ProviderCapabilitiesSnapshot, SessionCapabilities,
};
use super::error::ExecutionError;
use super::ids::SessionId;
use super::provider::ExecutionProvider;
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
    pub accept_unknown_host_keys: bool,
    pub max_run_timeout_secs: u64,
    pub max_output_bytes: usize,
    pub max_sessions: usize,
}

struct SshClientHandler {
    accept_unknown_host_keys: bool,
}

impl russh::client::Handler for SshClientHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &keys::PublicKey,
    ) -> Result<bool, Self::Error> {
        Ok(self.accept_unknown_host_keys)
    }
}

#[derive(Debug, Clone)]
enum SshExecMode {
    Python,
    Shell,
}

struct SshSession {
    mode: SshExecMode,
    run_cancel: Mutex<Option<CancellationToken>>,
}

/// SSH-backed [`ExecutionProvider`] (Linux-oriented remote: `bash` + `base64` + `python` / `bash -s`).
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
        SessionCapabilities {
            session_id: id.clone(),
            schema_version: 1,
            provider_id: "ssh".into(),
            active_language,
            gpu_visible: None,
            working_directory_display: Some(format!(
                "ssh {}@{}:{} (cwd {})",
                self.config.user, self.config.host, self.config.port, wd
            )),
            provider_snapshot: ProviderCapabilitiesSnapshot {
                supports_interrupt: self.caps.supports_interrupt,
                supports_package_install: self.caps.supports_package_install,
                supports_remote_shell: self.caps.supports_remote_shell,
                jupyter_kernel: self.caps.jupyter_kernel,
                network_policy: self.caps.network_policy,
            },
            extensions: Default::default(),
        }
    }

    fn build_remote_command(
        &self,
        mode: &SshExecMode,
        code: &str,
    ) -> Result<String, ExecutionError> {
        let b64 = B64.encode(code.as_bytes());
        if !b64
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
        {
            return Err(ExecutionError::InvalidArgument(
                "ssh: internal base64 encoding produced unexpected characters".into(),
            ));
        }
        let wd = &self.config.remote_workdir;
        let py = &self.config.remote_python;
        // Inner script uses single-quoted segments (paths are validation-restricted). Avoid
        // `bash -lc '...'` nesting by double-encoding and using `bash -c "printf ..."`.
        let inner_script = match mode {
            SshExecMode::Python => {
                format!("cd '{wd}' && printf '%s' '{b64}' | base64 -d | exec '{py}' -u -")
            }
            SshExecMode::Shell => {
                format!("cd '{wd}' && printf '%s' '{b64}' | base64 -d | exec bash -s")
            }
        };
        if inner_script.contains('"') || inner_script.contains('$') || inner_script.contains('`') {
            return Err(ExecutionError::InvalidArgument(
                "ssh: refused to build remote command with unexpected shell metacharacters".into(),
            ));
        }
        let inner_b64 = B64.encode(inner_script.as_bytes());
        if inner_b64.contains('\'') {
            return Err(ExecutionError::InvalidArgument(
                "ssh: inner script base64 unexpectedly contained a single quote".into(),
            ));
        }
        Ok(format!(
            "bash -c \"printf %s '{inner_b64}' | base64 -d | bash\""
        ))
    }
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

async fn run_ssh_exec(
    config: SshExecutionProviderConfig,
    private_key: Option<Arc<keys::PrivateKey>>,
    remote_command: String,
    max_output_bytes: usize,
) -> Result<RunResult, ExecutionError> {
    let max_each = (max_output_bytes / 2).max(1024);
    let client_cfg = Arc::new(client::Config::default());
    let handler = SshClientHandler {
        accept_unknown_host_keys: config.accept_unknown_host_keys,
    };
    let addrs = (config.host.as_str(), config.port);
    let mut handle = client::connect(client_cfg, addrs, handler)
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: connect: {e}")))?;

    authenticate_session(&private_key, &mut handle, &config.user).await?;

    let mut channel = handle
        .channel_open_session()
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: open session channel: {e}")))?;

    channel
        .exec(true, remote_command)
        .await
        .map_err(|e| ExecutionError::Provider(format!("ssh: exec: {e}")))?;

    let mut stdout = Vec::<u8>::new();
    let mut stderr = Vec::<u8>::new();
    let mut code: Option<u32> = None;

    loop {
        let Some(msg) = channel.wait().await else {
            break;
        };
        match msg {
            ChannelMsg::Data { data } => {
                if stdout.len() < max_each {
                    let take = (max_each - stdout.len()).min(data.len());
                    stdout.extend_from_slice(&data[..take]);
                }
            }
            ChannelMsg::ExtendedData { data, ext } => {
                if ext == SSH_STDERR && stderr.len() < max_each {
                    let take = (max_each - stderr.len()).min(data.len());
                    stderr.extend_from_slice(&data[..take]);
                }
            }
            ChannelMsg::ExitStatus { exit_status } => {
                code = Some(exit_status);
            }
            ChannelMsg::Eof | ChannelMsg::Close => {}
            _ => {}
        }
    }

    let _ = handle.disconnect(Disconnect::ByApplication, "", "").await;

    Ok(RunResult::new(
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr),
        code.map(|c| c as i32),
    ))
}

fn truncate_run_result(mut r: RunResult, max_total: usize) -> RunResult {
    let max_each = (max_total / 2).max(1024);
    r.stdout = truncate_utf8_string(&r.stdout, max_each);
    r.stderr = truncate_utf8_string(&r.stderr, max_each);
    r
}

fn truncate_utf8_string(s: &str, max: usize) -> String {
    let mut o = s.to_string();
    if o.len() <= max {
        return o;
    }
    let mut end = max;
    while end > 0 && !o.is_char_boundary(end) {
        end -= 1;
    }
    o.truncate(end);
    o.push_str("\n... (truncated)");
    o
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
        let id = SessionId::new(uuid::Uuid::new_v4().to_string());
        let caps = self.session_caps(&id, &mode);
        let session = Arc::new(SshSession {
            mode,
            run_cancel: Mutex::new(None),
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
            return Ok(());
        }
        Err(ExecutionError::InvalidSession(session_id.to_string()))
    }

    async fn run(
        &self,
        session_id: &SessionId,
        spec: RunSpec,
    ) -> Result<RunResult, ExecutionError> {
        if !matches!(spec.cwd, CwdPolicy::SessionDefault) {
            return Err(ExecutionError::unsupported(
                "run",
                "ssh provider only supports cwd_mode session_default (remote cwd is fixed in config)",
            ));
        }

        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| ExecutionError::InvalidSession(session_id.to_string()))?;
        let session = session.value().clone();
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

        let remote_command = self.build_remote_command(&mode, &spec.code)?;
        let cfg = self.config.clone();
        let pk = self.private_key.clone();
        let max_out = self.config.max_output_bytes;

        let work = async move { run_ssh_exec(cfg, pk, remote_command, max_out).await };

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

    #[test]
    fn validate_workdir_rejects_relative() {
        assert!(validate_remote_workdir("tmp/x").is_err());
    }

    #[test]
    fn validate_workdir_rejects_dotdot() {
        assert!(validate_remote_workdir("/tmp/../etc").is_err());
    }
}
