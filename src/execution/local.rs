//! Local subprocess execution (Phase 1): sandbox cwd, timeouts, output caps, best-effort cancel.
//!
//! **Python** can run in **REPL mode** (default): one long-lived `python` per session with a shared
//! namespace across `run` calls, or **subprocess mode**: a fresh interpreter per `run` (legacy).

use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::capabilities::{
    NetworkPolicy, ProviderCapabilities, ProviderCapabilitiesSnapshot, SessionCapabilities,
};
use super::error::ExecutionError;
use super::ids::SessionId;
use super::provider::ExecutionProvider;
use super::run::{CwdPolicy, RunResult, RunSpec, SessionCreateRequest, SessionHandle};
use crate::tools::builtin::resolve_path;

/// How user code is executed for a session.
#[derive(Debug, Clone)]
pub enum LocalExecMode {
    /// Python: either a persistent REPL child or one fresh subprocess per `run`.
    Python {
        executable: String,
        /// When true, one interpreter per session (variables persist). When false, each `run` is isolated.
        repl: bool,
    },
    /// POSIX `sh -c` or Windows `cmd /C` (shell — use only with trusted prompts).
    Shell,
}

/// Configuration for [`LocalExecutionProvider`].
#[derive(Debug, Clone)]
pub struct LocalExecutionConfig {
    /// Agent sandbox root (same boundary as `resolve_path(..., restrict)`).
    pub sandbox_dir: PathBuf,
    pub restrict_to_workspace: bool,
    /// Upper bound on `RunSpec.timeout_secs` (minimum 1 after clamp).
    pub max_run_timeout_secs: u64,
    /// Max combined bytes read from stdout + stderr (each pipe gets half, minimum 1024 each).
    pub max_output_bytes: usize,
    /// Max concurrent sessions (0 = unlimited).
    pub max_sessions: usize,
    /// Default `python` on PATH unless overridden.
    pub python_executable: String,
    /// When true (default), local Python sessions use a persistent REPL; when false, each run is a new process.
    pub python_repl: bool,
}

impl LocalExecutionConfig {
    pub fn new(sandbox_dir: PathBuf, restrict_to_workspace: bool) -> Self {
        Self {
            sandbox_dir,
            restrict_to_workspace,
            max_run_timeout_secs: 300,
            max_output_bytes: 256 * 1024,
            max_sessions: 32,
            python_executable: "python".to_string(),
            python_repl: true,
        }
    }
}

/// Local process-backed [`ExecutionProvider`].
pub struct LocalExecutionProvider {
    config: LocalExecutionConfig,
    caps: ProviderCapabilities,
    sessions: DashMap<SessionId, Arc<LocalSession>>,
}

struct LocalSession {
    root_cwd: PathBuf,
    mode: LocalExecMode,
    /// PID of the running child (if any), for cancel when the run task holds the handle.
    active_pid: Mutex<Option<u32>>,
    /// Present while a `run` is in flight (also used to reject overlapping runs).
    run_cancel: Mutex<Option<CancellationToken>>,
    /// Persistent Python worker (REPL mode only). Dropped on close, cancel, timeout, or cwd change.
    python_repl: Mutex<Option<ReplPythonChild>>,
}

/// One long-lived `python -u -c <bootstrap>` with framed stdin/stdout.
struct ReplPythonChild {
    cwd: PathBuf,
    stdin: tokio::process::ChildStdin,
    stdout: tokio::process::ChildStdout,
    child: Child,
}

/// Max UTF-8 source bytes per REPL round-trip (defense in depth; matches Python worker bound).
const MAX_REPL_SOURCE_BYTES: usize = 16 * 1024 * 1024;

/// Embedded worker: read `>I` length + UTF-8 source, `exec` in shared namespace, reply `>III` + payloads.
const PYTHON_REPL_BOOTSTRAP: &str = r#"import struct,sys,traceback,io,contextlib
_g={}
while 1:
 h=sys.stdin.buffer.read(4)
 if len(h)<4:
  sys.exit(0)
 n=int.from_bytes(h,'big')
 if n>16777216 or n<0:
  sys.exit(2)
 s=sys.stdin.buffer.read(n).decode('utf-8','replace')
 o,e=io.StringIO(),io.StringIO()
 c=0
 try:
  with contextlib.redirect_stdout(o),contextlib.redirect_stderr(e):
   exec(compile(s,'<isanagent>','exec'),_g,_g)
 except Exception:
  traceback.print_exc(file=e)
  c=1
 ob,eb=o.getvalue().encode('utf-8','backslashreplace'),e.getvalue().encode('utf-8','backslashreplace')
 sys.stdout.buffer.write(struct.pack('>III',c,len(ob),len(eb))+ob+eb)
 sys.stdout.buffer.flush()
"#;

impl LocalExecutionProvider {
    pub fn new(config: LocalExecutionConfig) -> Result<Self, ExecutionError> {
        let sandbox = &config.sandbox_dir;
        if !sandbox.is_dir() {
            return Err(ExecutionError::InvalidArgument(format!(
                "sandbox_dir is not a directory: {}",
                sandbox.display()
            )));
        }

        let mut caps = ProviderCapabilities::minimal("local");
        caps.languages = vec!["python".into(), "shell".into()];
        caps.supports_persistent_sessions = true;
        caps.supports_interrupt = true;
        caps.supports_package_install = false;
        caps.supports_remote_shell = false;
        caps.jupyter_kernel = false;
        caps.network_policy = NetworkPolicy::Off;
        caps.max_output_bytes_default = Some(config.max_output_bytes as u64);

        Ok(Self {
            config,
            caps,
            sessions: DashMap::new(),
        })
    }

    fn pick_mode(&self, req: &SessionCreateRequest) -> Result<LocalExecMode, ExecutionError> {
        let lang = req
            .language
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        match lang {
            None | Some("python") | Some("py") => Ok(LocalExecMode::Python {
                executable: self.config.python_executable.clone(),
                repl: self.config.python_repl,
            }),
            Some("shell") | Some("sh") | Some("bash") => Ok(LocalExecMode::Shell),
            Some(other) => Err(ExecutionError::InvalidArgument(format!(
                "unsupported language for local provider: {other} (supported: python, shell)"
            ))),
        }
    }

    fn session_capabilities(
        &self,
        id: &SessionId,
        mode: &LocalExecMode,
        cwd_display: &str,
    ) -> SessionCapabilities {
        let active_language = match mode {
            LocalExecMode::Python { .. } => Some("python".into()),
            LocalExecMode::Shell => Some("shell".into()),
        };
        let mut ext = std::collections::BTreeMap::new();
        if let LocalExecMode::Python { repl, .. } = mode {
            ext.insert("local_python_repl".into(), serde_json::Value::Bool(*repl));
        }
        SessionCapabilities {
            session_id: id.clone(),
            schema_version: 1,
            provider_id: "local".into(),
            active_language,
            gpu_visible: None,
            working_directory_display: Some(cwd_display.to_string()),
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

    fn resolve_run_cwd(
        &self,
        session: &LocalSession,
        spec: &RunSpec,
    ) -> Result<PathBuf, ExecutionError> {
        match &spec.cwd {
            CwdPolicy::SessionDefault => Ok(session.root_cwd.clone()),
            CwdPolicy::SandboxRelative(rel) => resolve_path(
                rel.as_str(),
                &self.config.sandbox_dir,
                self.config.restrict_to_workspace,
            )
            .map_err(ExecutionError::InvalidArgument),
        }
    }

    /// Spawns or reuses the REPL child when `cwd` matches; otherwise kills the old child and spawns in `cwd`.
    async fn ensure_python_repl(
        &self,
        session: &LocalSession,
        executable: &str,
        cwd: &Path,
    ) -> Result<(), ExecutionError> {
        let mut slot = session.python_repl.lock().await;
        let cwd = cwd.to_path_buf();
        let respawn = match slot.as_ref() {
            None => true,
            Some(ch) => ch.cwd != cwd,
        };
        if respawn {
            if let Some(mut old) = slot.take() {
                let _ = old.child.kill().await;
            }
            let mut cmd = Command::new(executable);
            cmd.arg("-u").arg("-c").arg(PYTHON_REPL_BOOTSTRAP);
            cmd.current_dir(&cwd);
            cmd.stdin(Stdio::piped());
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::null());
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                unsafe {
                    cmd.as_std_mut().pre_exec(|| {
                        if libc::setpgid(0, 0) != 0 {
                            return Err(std::io::Error::last_os_error());
                        }
                        Ok(())
                    });
                }
            }
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                cmd.as_std_mut().creation_flags(CREATE_NO_WINDOW);
            }
            let mut child = cmd
                .spawn()
                .map_err(|e| ExecutionError::Provider(format!("local repl spawn: {e}")))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| ExecutionError::Provider("local repl missing stdin".into()))?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| ExecutionError::Provider("local repl missing stdout".into()))?;
            *slot = Some(ReplPythonChild {
                cwd,
                stdin,
                stdout,
                child,
            });
        }
        Ok(())
    }
}

async fn shutdown_repl_locked(m: &Mutex<Option<ReplPythonChild>>) {
    let mut slot = m.lock().await;
    if let Some(mut ch) = slot.take() {
        let _ = ch.child.kill().await;
    }
}

async fn read_exact<R: tokio::io::AsyncRead + Unpin>(
    r: &mut R,
    buf: &mut [u8],
) -> Result<(), ExecutionError> {
    let mut off = 0;
    while off < buf.len() {
        let n = r
            .read(&mut buf[off..])
            .await
            .map_err(|e| ExecutionError::Provider(format!("repl read: {e}")))?;
        if n == 0 {
            return Err(ExecutionError::Provider(
                "repl: unexpected EOF from child".into(),
            ));
        }
        off += n;
    }
    Ok(())
}

/// Read `total` bytes from `r`, keeping at most `cap` bytes in the returned buffer (rest discarded).
async fn read_exact_capped<R: tokio::io::AsyncRead + Unpin>(
    r: &mut R,
    total: usize,
    cap: usize,
) -> Result<Vec<u8>, ExecutionError> {
    let mut out = Vec::new();
    let mut remaining = total;
    let mut buf = [0u8; 16384];
    while remaining > 0 {
        let chunk = remaining.min(buf.len());
        read_exact(r, &mut buf[..chunk]).await?;
        if out.len() < cap {
            let room = cap.saturating_sub(out.len());
            let take = chunk.min(room);
            out.extend_from_slice(&buf[..take]);
        }
        remaining -= chunk;
    }
    Ok(out)
}

#[async_trait]
impl ExecutionProvider for LocalExecutionProvider {
    fn provider_id(&self) -> &str {
        "local"
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
        let root = resolve_path(
            ".",
            &self.config.sandbox_dir,
            self.config.restrict_to_workspace,
        )
        .map_err(ExecutionError::InvalidArgument)?;

        let id = SessionId::new(uuid::Uuid::new_v4().to_string());
        let cwd_display = root.display().to_string();
        let caps = self.session_capabilities(&id, &mode, &cwd_display);
        let session = Arc::new(LocalSession {
            root_cwd: root,
            mode,
            active_pid: Mutex::new(None),
            run_cancel: Mutex::new(None),
            python_repl: Mutex::new(None),
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
            shutdown_repl_locked(&sess.python_repl).await;
            if let Some(pid) = sess.active_pid.lock().await.take() {
                kill_process_best_effort(pid);
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

        let cwd = self.resolve_run_cwd(&session, &spec)?;
        let timeout_secs = spec
            .timeout_secs
            .min(self.config.max_run_timeout_secs)
            .max(1);
        let timeout = Duration::from_secs(timeout_secs);
        let max_each = (self.config.max_output_bytes / 2).max(1024);

        let result: Result<RunResult, ExecutionError> = match &session.mode {
            LocalExecMode::Python {
                executable,
                repl: true,
            } => {
                let exec_owned = executable.clone();
                let code = spec.code.clone();
                let work = async {
                    self.ensure_python_repl(&session, exec_owned.as_str(), &cwd)
                        .await?;
                    if let Some(pid) = session
                        .python_repl
                        .lock()
                        .await
                        .as_ref()
                        .and_then(|ch| ch.child.id())
                    {
                        *session.active_pid.lock().await = Some(pid);
                    }
                    let mut g = session.python_repl.lock().await;
                    let repl = g
                        .as_mut()
                        .ok_or_else(|| ExecutionError::Provider("local repl unavailable".into()))?;
                    let cbytes = code.as_bytes();
                    if cbytes.len() > MAX_REPL_SOURCE_BYTES {
                        return Err(ExecutionError::InvalidArgument(format!(
                            "code exceeds max repl bytes ({MAX_REPL_SOURCE_BYTES})"
                        )));
                    }
                    let len_u = cbytes.len() as u32;
                    repl.stdin
                        .write_all(&len_u.to_be_bytes())
                        .await
                        .map_err(|e| ExecutionError::Provider(format!("repl stdin: {e}")))?;
                    repl.stdin
                        .write_all(cbytes)
                        .await
                        .map_err(|e| ExecutionError::Provider(format!("repl stdin: {e}")))?;
                    repl.stdin
                        .flush()
                        .await
                        .map_err(|e| ExecutionError::Provider(format!("repl stdin: {e}")))?;
                    let mut hdr = [0u8; 12];
                    read_exact(&mut repl.stdout, &mut hdr).await?;
                    let st = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
                    let olen = u32::from_be_bytes(hdr[4..8].try_into().unwrap()) as usize;
                    let elen = u32::from_be_bytes(hdr[8..12].try_into().unwrap()) as usize;
                    const MAX_REPLY: usize = 64 * 1024 * 1024;
                    if olen > MAX_REPLY || elen > MAX_REPLY {
                        return Err(ExecutionError::Provider(
                            "repl: invalid output length from worker".into(),
                        ));
                    }
                    let out_raw = read_exact_capped(&mut repl.stdout, olen, max_each).await?;
                    let err_raw = read_exact_capped(&mut repl.stdout, elen, max_each).await?;
                    let stdout = string_from_utf8_lossy_trim_cap(out_raw, max_each);
                    let stderr = string_from_utf8_lossy_trim_cap(err_raw, max_each);
                    Ok(RunResult::new(stdout, stderr, Some(st as i32)))
                };

                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        if let Some(p) = session.active_pid.lock().await.take() {
                            kill_process_best_effort(p);
                        }
                        shutdown_repl_locked(&session.python_repl).await;
                        Err(ExecutionError::Cancelled)
                    }
                    out = tokio::time::timeout(timeout, work) => match out {
                        Err(_) => {
                            if let Some(p) = session.active_pid.lock().await.take() {
                                kill_process_best_effort(p);
                            }
                            shutdown_repl_locked(&session.python_repl).await;
                            Err(ExecutionError::Timeout { timeout_secs })
                        }
                        Ok(Err(e)) => {
                            if let Some(p) = session.active_pid.lock().await.take() {
                                kill_process_best_effort(p);
                            }
                            shutdown_repl_locked(&session.python_repl).await;
                            Err(e)
                        }
                        Ok(Ok(r)) => Ok(r),
                    },
                }
            }
            _ => {
                let (mut cmd, stdin_body) = build_command(&session.mode, &spec.code)?;
                cmd.current_dir(&cwd);
                cmd.stdin(if stdin_body.is_some() {
                    Stdio::piped()
                } else {
                    Stdio::null()
                });
                cmd.stdout(Stdio::piped());
                cmd.stderr(Stdio::piped());
                #[cfg(unix)]
                {
                    use std::os::unix::process::CommandExt;
                    unsafe {
                        cmd.as_std_mut().pre_exec(|| {
                            if libc::setpgid(0, 0) != 0 {
                                return Err(std::io::Error::last_os_error());
                            }
                            Ok(())
                        });
                    }
                }
                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                    cmd.as_std_mut().creation_flags(CREATE_NO_WINDOW);
                }

                let child = cmd
                    .spawn()
                    .map_err(|e| ExecutionError::Provider(e.to_string()))?;
                let pid = child.id();
                if let Some(pid) = pid {
                    *session.active_pid.lock().await = Some(pid);
                }

                let work = async move {
                    match tokio::time::timeout(
                        timeout,
                        drain_child_pipes(child, max_each, stdin_body),
                    )
                    .await
                    {
                        Err(_) => {
                            if let Some(p) = pid {
                                kill_process_best_effort(p);
                            }
                            Err(ExecutionError::Timeout { timeout_secs })
                        }
                        Ok(Err(e)) => Err(e),
                        Ok(Ok(r)) => Ok(r),
                    }
                };

                let mut jh = tokio::spawn(work);
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        if let Some(p) = pid {
                            kill_process_best_effort(p);
                        }
                        jh.abort();
                        Err(ExecutionError::Cancelled)
                    }
                    joined = &mut jh => match joined {
                        Ok(inner) => inner,
                        Err(e) if e.is_cancelled() => Err(ExecutionError::Cancelled),
                        Err(e) => Err(ExecutionError::Provider(format!("run task join: {e}"))),
                    },
                }
            }
        };

        *session.run_cancel.lock().await = None;
        *session.active_pid.lock().await = None;

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
        if let Some(pid) = session.active_pid.lock().await.take() {
            kill_process_best_effort(pid);
        }
        shutdown_repl_locked(&session.python_repl).await;
        Ok(())
    }
}

/// Read from `reader` until EOF, retaining at most `max_capture` bytes in the returned buffer and
/// discarding the rest so the peer pipe never fills indefinitely.
async fn read_pipe_limited(
    mut reader: impl tokio::io::AsyncRead + Unpin,
    max_capture: usize,
) -> std::io::Result<Vec<u8>> {
    let mut captured = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        if captured.len() < max_capture {
            let room = max_capture.saturating_sub(captured.len());
            let take = n.min(room);
            captured.extend_from_slice(&buf[..take]);
        }
    }
    Ok(captured)
}

async fn drain_child_pipes(
    mut child: Child,
    max_each: usize,
    stdin_body: Option<Vec<u8>>,
) -> Result<RunResult, ExecutionError> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ExecutionError::Provider("local child missing stdout pipe".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ExecutionError::Provider("local child missing stderr pipe".into()))?;
    let stdin = child.stdin.take();

    let stdin_fut = async move {
        match (stdin, stdin_body) {
            (Some(mut s), Some(body)) => {
                s.write_all(&body).await?;
                s.shutdown().await?;
            }
            (None, Some(_)) => {
                return Err(std::io::Error::other("local child missing stdin pipe"));
            }
            _ => {}
        }
        Ok::<(), std::io::Error>(())
    };

    let (out, err, _, status) = tokio::try_join!(
        read_pipe_limited(stdout, max_each),
        read_pipe_limited(stderr, max_each),
        stdin_fut,
        child.wait(),
    )
    .map_err(|e| ExecutionError::Provider(e.to_string()))?;

    let stdout = string_from_utf8_lossy_trim_cap(out, max_each);
    let stderr = string_from_utf8_lossy_trim_cap(err, max_each);
    Ok(RunResult::new(stdout, stderr, status.code()))
}

fn string_from_utf8_lossy_trim_cap(bytes: Vec<u8>, max_chars: usize) -> String {
    let mut s = String::from_utf8_lossy(&bytes).into_owned();
    if s.len() <= max_chars {
        return s;
    }
    let mut end = max_chars;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s.push_str("\n... (truncated)");
    s
}

/// Returns the command and optional stdin payload (UTF-8 source) when the interpreter reads code
/// from stdin instead of argv.
fn build_command(
    mode: &LocalExecMode,
    code: &str,
) -> Result<(Command, Option<Vec<u8>>), ExecutionError> {
    let (c, stdin) = match mode {
        LocalExecMode::Python { executable, .. } => {
            let mut c = Command::new(executable);
            // Unbuffered; code is sent on stdin to avoid command-line length limits.
            c.arg("-u").arg("-");
            (c, Some(code.as_bytes().to_vec()))
        }
        LocalExecMode::Shell => {
            if cfg!(target_os = "windows") {
                let mut c = Command::new("cmd");
                c.arg("/C").arg(code);
                (c, None)
            } else {
                let mut c = Command::new("sh");
                c.arg("-c").arg(code);
                (c, None)
            }
        }
    };
    Ok((c, stdin))
}

#[cfg(windows)]
fn kill_process_best_effort(pid: u32) {
    if pid <= 1 {
        return;
    }
    let _ = StdCommand::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(unix)]
fn kill_process_best_effort(pid: u32) {
    // `kill(-1, SIGKILL)` would target every signalable process for this user — never do that.
    if pid <= 1 {
        return;
    }
    // Child runs in its own process group (`setpgid` in `pre_exec`); negative PID targets the group.
    let pgid = -(pid as i32);
    // SAFETY: `kill` from the parent after fork/exec is allowed here.
    unsafe {
        let _ = libc::kill(pgid, libc::SIGKILL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_sandbox() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("isanagent-exec-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn rejects_non_dir_sandbox() {
        let cfg = LocalExecutionConfig::new(
            PathBuf::from("/nonexistent/path/that/should/not/exist"),
            true,
        );
        assert!(LocalExecutionProvider::new(cfg).is_err());
    }

    fn echo_hello_case() -> (&'static str, String) {
        if cfg!(windows) {
            // `cmd /C echo` is reliable for stdio; Windows Store `python` shims can yield empty stdout in CI.
            ("shell", "echo hello-exec".into())
        } else {
            ("python", r#"print("hello-exec")"#.into())
        }
    }

    fn long_running_case() -> (&'static str, String) {
        if cfg!(windows) {
            ("shell", "ping -n 120 127.0.0.1 >nul".into())
        } else {
            ("python", "import time; time.sleep(60)".into())
        }
    }

    fn timeout_probe_case() -> (&'static str, String) {
        if cfg!(windows) {
            ("shell", "ping -n 60 127.0.0.1 >nul".into())
        } else {
            ("python", "import time; time.sleep(30)".into())
        }
    }

    #[tokio::test]
    async fn local_echo_stdout() {
        let (lang, code) = echo_hello_case();
        let dir = temp_sandbox();
        let cfg = LocalExecutionConfig::new(dir.clone(), true);
        let prov = LocalExecutionProvider::new(cfg).unwrap();
        let h = prov
            .create_session(SessionCreateRequest {
                language: Some(lang.into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let r = prov
            .run(&h.id, RunSpec::new(code, 30))
            .await
            .unwrap_or_else(|e| panic!("run failed: {e:?}"));
        assert!(
            r.stdout.contains("hello-exec") || r.stderr.contains("hello-exec"),
            "stdout={:?} stderr={:?} code={:?}",
            r.stdout,
            r.stderr,
            r.exit_code
        );
        assert_eq!(r.exit_code, Some(0));
        prov.close_session(&h.id).await.unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn cwd_sandbox_relative() {
        let dir = temp_sandbox();
        let sub = dir.join("pkg");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("marker.txt"), "x").unwrap();
        let cfg = LocalExecutionConfig::new(dir.clone(), true);
        let prov = LocalExecutionProvider::new(cfg).unwrap();
        let h = prov
            .create_session(SessionCreateRequest {
                language: Some((if cfg!(windows) { "shell" } else { "python" }).to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        let code = if cfg!(windows) {
            "type marker.txt".to_string()
        } else {
            r#"import pathlib; print(pathlib.Path("marker.txt").read_text())"#.to_string()
        };
        let mut spec = RunSpec::new(code, 30);
        spec.cwd = CwdPolicy::SandboxRelative("pkg".into());
        let r = prov.run(&h.id, spec).await.unwrap();
        assert!(r.stdout.trim().contains('x'), "{r:?}");
        prov.close_session(&h.id).await.unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn timeout_returns_timeout_error() {
        let (lang, code) = timeout_probe_case();
        let dir = temp_sandbox();
        let cfg = LocalExecutionConfig::new(dir.clone(), true);
        let prov = LocalExecutionProvider::new(cfg).unwrap();
        let h = prov
            .create_session(SessionCreateRequest {
                language: Some(lang.into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let r = prov.run(&h.id, RunSpec::new(code, 1)).await;
        assert!(
            matches!(r, Err(ExecutionError::Timeout { timeout_secs: 1 })),
            "unexpected result: {r:?}"
        );
        prov.close_session(&h.id).await.unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn cancel_mid_run() {
        let (lang, code) = long_running_case();
        let dir = temp_sandbox();
        let cfg = LocalExecutionConfig::new(dir.clone(), true);
        let prov = Arc::new(LocalExecutionProvider::new(cfg).unwrap());
        let h = prov
            .create_session(SessionCreateRequest {
                language: Some(lang.into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let sid = h.id.clone();
        let p2 = prov.clone();
        let run = tokio::spawn(async move { p2.run(&sid, RunSpec::new(code, 120)).await });
        tokio::time::sleep(Duration::from_millis(200)).await;
        prov.cancel(&h.id).await.unwrap();
        let r = run.await.unwrap();
        assert!(
            matches!(r, Err(ExecutionError::Cancelled)),
            "expected Cancelled, got {:?}",
            r
        );
        prov.close_session(&h.id).await.ok();
        let _ = fs::remove_dir_all(&dir);
    }

    /// REPL mode needs a real `python` on PATH; Windows Store shims are flaky for CI, so Unix-only.
    #[tokio::test]
    #[cfg(unix)]
    async fn local_python_repl_persists_variables() {
        let dir = temp_sandbox();
        let cfg = LocalExecutionConfig::new(dir.clone(), true);
        assert!(cfg.python_repl);
        let prov = LocalExecutionProvider::new(cfg).unwrap();
        let h = prov
            .create_session(SessionCreateRequest {
                language: Some("python".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let r1 = prov
            .run(&h.id, RunSpec::new("x = 42", 30))
            .await
            .unwrap_or_else(|e| panic!("first run: {e:?}"));
        assert_eq!(r1.exit_code, Some(0), "{r1:?}");
        let r2 = prov
            .run(&h.id, RunSpec::new("print(x)", 30))
            .await
            .unwrap_or_else(|e| panic!("second run: {e:?}"));
        assert_eq!(r2.exit_code, Some(0), "{r2:?}");
        assert!(
            r2.stdout.contains("42"),
            "expected repl namespace; stdout={:?}",
            r2.stdout
        );
        prov.close_session(&h.id).await.unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn local_python_subprocess_mode_no_shared_namespace() {
        let dir = temp_sandbox();
        let mut cfg = LocalExecutionConfig::new(dir.clone(), true);
        cfg.python_repl = false;
        let prov = LocalExecutionProvider::new(cfg).unwrap();
        let h = prov
            .create_session(SessionCreateRequest {
                language: Some("python".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let r1 = prov.run(&h.id, RunSpec::new("x = 42", 30)).await.unwrap();
        assert_eq!(r1.exit_code, Some(0));
        let r2 = prov.run(&h.id, RunSpec::new("print(x)", 30)).await.unwrap();
        assert_ne!(r2.exit_code, Some(0), "{r2:?}");
        assert!(
            r2.stderr.contains("NameError") || r2.stdout.contains("NameError"),
            "{r2:?}"
        );
        prov.close_session(&h.id).await.unwrap();
        let _ = fs::remove_dir_all(&dir);
    }
}
