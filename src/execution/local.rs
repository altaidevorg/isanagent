//! Local subprocess execution (Phase 1): sandbox cwd, timeouts, output caps, best-effort cancel.

use std::path::PathBuf;
use std::process::{Command as StdCommand, Stdio};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use tokio::process::Command;
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
    /// `python_executable -c <code>` (argv, no shell).
    Python { executable: String },
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
}

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
            extensions: Default::default(),
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

        let mut cmd = build_command(&session.mode, &spec.code)?;
        cmd.current_dir(&cwd);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
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

        // `wait_with_output` is more reliable on Windows than manual `try_join` on pipes.
        let work = async move {
            match tokio::time::timeout(timeout, child.wait_with_output()).await {
                Err(_) => {
                    if let Some(p) = pid {
                        kill_process_best_effort(p);
                    }
                    Err(ExecutionError::Timeout { timeout_secs })
                }
                Ok(Err(e)) => Err(ExecutionError::Provider(e.to_string())),
                Ok(Ok(out)) => {
                    let stdout = truncate_utf8_bytes(out.stdout, max_each);
                    let stderr = truncate_utf8_bytes(out.stderr, max_each);
                    Ok(RunResult::new(stdout, stderr, out.status.code()))
                }
            }
        };

        let mut jh = tokio::spawn(work);
        let result: Result<RunResult, ExecutionError> = async move {
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
        .await;

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
        Ok(())
    }
}

fn build_command(mode: &LocalExecMode, code: &str) -> Result<Command, ExecutionError> {
    let c = match mode {
        LocalExecMode::Python { executable } => {
            let mut c = Command::new(executable);
            // Unbuffered so short-lived `print` reaches the pipe before exit (notably on Windows).
            c.arg("-u").arg("-c").arg(code);
            c
        }
        LocalExecMode::Shell => {
            if cfg!(target_os = "windows") {
                let mut c = Command::new("cmd");
                c.arg("/C").arg(code);
                c
            } else {
                let mut c = Command::new("sh");
                c.arg("-c").arg(code);
                c
            }
        }
    };
    Ok(c)
}

fn truncate_utf8_bytes(bytes: Vec<u8>, max: usize) -> String {
    let mut s = String::from_utf8_lossy(&bytes).into_owned();
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s.push_str("\n... (truncated)");
    s
}

fn kill_process_best_effort(pid: u32) {
    if pid == 0 {
        return;
    }
    if cfg!(target_os = "windows") {
        let _ = StdCommand::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    } else {
        let _ = StdCommand::new("kill")
            .args(["-9", "--", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
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
}
