//! Gated harness tools for the execution plane (`[harness.execution] enabled = true`).

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use log::info;
use serde_json::{json, Value};

use crate::execution::{
    CwdPolicy, ExecutionError, ExecutionHarness, RunSpec, SessionCreateRequest, SessionId,
};
use crate::traits::Tool;

fn exec_err(e: ExecutionError) -> String {
    e.to_string()
}

fn require_session_id(args: &Value) -> Result<SessionId, String> {
    let id = args
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Missing 'session_id'".to_string())?;
    Ok(SessionId::new(id.to_string()))
}

fn parse_cwd(args: &Value) -> Result<CwdPolicy, String> {
    let mode = args
        .get("cwd_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("session_default");
    match mode {
        "session_default" => Ok(CwdPolicy::SessionDefault),
        "sandbox_relative" => {
            let rel = args
                .get("cwd_relative")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "cwd_mode \"sandbox_relative\" requires cwd_relative".to_string())?;
            if rel.trim().is_empty() {
                return Err("cwd_relative must be non-empty".to_string());
            }
            Ok(CwdPolicy::SandboxRelative(rel.to_string()))
        }
        other => Err(format!(
            "Invalid cwd_mode {other:?}; use session_default or sandbox_relative"
        )),
    }
}

/// Open an execution session (local subprocess or Jupyter kernel per config).
pub struct ExecutionSessionCreateTool {
    pub harness: Arc<ExecutionHarness>,
}

#[async_trait]
impl Tool for ExecutionSessionCreateTool {
    fn name(&self) -> &str {
        "execution_session_create"
    }

    fn description(&self) -> &str {
        "Create an execution session (requires [harness.execution] enabled). Provider is [harness.execution] default_provider: local runs under the workspace sandbox; jupyter uses a configured Jupyter Server kernel. Returns session_id, session capabilities, and a short provider capability summary. Use execution_run to execute code; execution_session_close when done."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "label": { "type": "string", "description": "Optional label for logs" },
                "language": {
                    "type": "string",
                    "description": "Optional language hint. Local: python, py, shell, sh, bash. Jupyter: python, py, r, R (ir kernel)."
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let req = SessionCreateRequest {
            label: args
                .get("label")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            language: args
                .get("language")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        };
        let handle = self
            .harness
            .provider()
            .create_session(req)
            .await
            .map_err(exec_err)?;
        let summary = self.harness.capabilities_summary();
        let v = json!({
            "session_id": handle.id,
            "session_capabilities": handle.capabilities,
            "provider_capabilities": summary,
        });
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
    }
}

/// Run code in an existing session (subject to provider timeout and output caps).
pub struct ExecutionRunTool {
    pub harness: Arc<ExecutionHarness>,
}

#[async_trait]
impl Tool for ExecutionRunTool {
    fn name(&self) -> &str {
        "execution_run"
    }

    fn description(&self) -> &str {
        "Run code in an execution_session_create session. Args: session_id, code, timeout_secs (optional), cwd_mode (optional session_default|sandbox_relative), cwd_relative when sandbox_relative. Output may be truncated per [harness.execution] limits."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string" },
                "code": { "type": "string" },
                "timeout_secs": { "type": "integer", "description": "Wall-clock limit for this run (clamped to harness max_wall_secs)" },
                "cwd_mode": { "type": "string", "description": "session_default (default) or sandbox_relative" },
                "cwd_relative": { "type": "string", "description": "Path relative to sandbox when cwd_mode is sandbox_relative" }
            },
            "required": ["session_id", "code"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let sid = require_session_id(&args)?;
        let code = args
            .get("code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'code'".to_string())?
            .to_string();
        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(60);
        let cwd = parse_cwd(&args)?;
        let mut spec = RunSpec::new(code, timeout_secs);
        spec.cwd = cwd;
        let started = Instant::now();
        let result = self
            .harness
            .provider()
            .run(&sid, spec)
            .await
            .map_err(exec_err)?;
        let prov = self.harness.provider().provider_id();
        info!(
            "execution_run provider={} session={} exit={:?} stdout_len={} stderr_len={} duration_ms={}",
            prov,
            sid,
            result.exit_code,
            result.stdout.len(),
            result.stderr.len(),
            started.elapsed().as_millis()
        );
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }
}

/// Best-effort interrupt of the current run (preflight: provider must support_interrupt).
pub struct ExecutionCancelTool {
    pub harness: Arc<ExecutionHarness>,
}

#[async_trait]
impl Tool for ExecutionCancelTool {
    fn name(&self) -> &str {
        "execution_cancel"
    }

    fn description(&self) -> &str {
        "Cancel the in-flight run for a session (local: SIGKILL / taskkill best-effort; jupyter: server interrupt + cooperative cancel). Preflight: only available when provider capabilities include supports_interrupt."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string" }
            },
            "required": ["session_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        if !self.harness.provider().capabilities().supports_interrupt {
            return Err(
                "execution_cancel unsupported: provider capabilities.supports_interrupt is false"
                    .to_string(),
            );
        }
        let sid = require_session_id(&args)?;
        self.harness
            .provider()
            .cancel(&sid)
            .await
            .map_err(exec_err)?;
        Ok("cancel requested for session".to_string())
    }
}

/// Tear down a session and release provider resources.
pub struct ExecutionSessionCloseTool {
    pub harness: Arc<ExecutionHarness>,
}

#[async_trait]
impl Tool for ExecutionSessionCloseTool {
    fn name(&self) -> &str {
        "execution_session_close"
    }

    fn description(&self) -> &str {
        "Close an execution session created with execution_session_create. Cancels any in-flight run for that session."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string" }
            },
            "required": ["session_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let sid = require_session_id(&args)?;
        self.harness
            .provider()
            .close_session(&sid)
            .await
            .map_err(exec_err)?;
        Ok("session closed".to_string())
    }
}

/// Host-oriented snapshot (capabilities + optional python -V).
pub struct ExecutionEnvInfoTool {
    pub harness: Arc<ExecutionHarness>,
}

#[async_trait]
impl Tool for ExecutionEnvInfoTool {
    fn name(&self) -> &str {
        "execution_env_info"
    }

    fn description(&self) -> &str {
        "Return provider capability summary. For local execution, also runs python_executable -V on the agent host (best effort). For jupyter, that probe is still the host interpreter (sanity check only); the kernel Python environment is whatever the Jupyter server started for that kernelspec."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "python_executable": {
                    "type": "string",
                    "description": "Override for -V probe (defaults to harness python_executable)"
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let exe = args
            .get("python_executable")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| self.harness.python_executable());
        let mut v = json!({
            "provider_capabilities": self.harness.capabilities_summary(),
        });
        let probe = tokio::task::spawn_blocking({
            let exe = exe.to_string();
            move || std::process::Command::new(&exe).arg("-V").output()
        })
        .await
        .map_err(|e| e.to_string())?;
        if let Ok(out) = probe {
            let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !line.is_empty() {
                v["python_version_line"] = json!(line);
            } else {
                let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                if !err.is_empty() {
                    v["python_version_error"] = json!(err);
                }
            }
        }
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecutionProvider;
    use std::fs;

    fn temp_sandbox() -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("isanagent-exec-tool-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn create_run_close_roundtrip() {
        let dir = temp_sandbox();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), true);
        let prov: Arc<dyn ExecutionProvider> =
            Arc::new(crate::execution::LocalExecutionProvider::new(cfg).expect("local provider"));
        let harness = Arc::new(ExecutionHarness::new(prov, "python"));
        let create = ExecutionSessionCreateTool {
            harness: harness.clone(),
        };
        let lang = if cfg!(windows) { "shell" } else { "python" };
        let out = create
            .execute(json!({ "language": lang }))
            .await
            .expect("create");
        let v: Value = serde_json::from_str(&out).expect("json");
        let sid = v["session_id"].as_str().expect("session id");
        let run = ExecutionRunTool {
            harness: harness.clone(),
        };
        let code = if cfg!(windows) {
            "echo ok-tool"
        } else {
            "print('ok-tool')"
        };
        let r = run
            .execute(json!({
                "session_id": sid,
                "code": code,
                "timeout_secs": 30,
            }))
            .await
            .expect("run");
        assert!(r.contains("ok-tool"), "run out={r}");
        let close = ExecutionSessionCloseTool { harness };
        close.execute(json!({ "session_id": sid })).await.unwrap();
        let _ = fs::remove_dir_all(&dir);
    }
}
