//! Gated harness tools for the execution plane (`[harness.execution] enabled = true`).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

use crate::bus::BusMessage;
use crate::bus::TelemetryEvent;
use crate::execution::{
    sanitize_session_segment, CwdPolicy, ExecutionError, ExecutionHarness, RunSpec,
    SessionCreateRequest, SessionId,
};
use crate::tool_runtime::current_tool_exec_ctx;
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

fn best_effort_git_head(workspace_dir: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["-C", workspace_dir.to_str()?, "rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s.chars().take(40).collect())
    }
}

#[derive(Serialize)]
struct ExecutionManifestLine<'a> {
    ts: &'a str,
    chat_id: &'a str,
    channel: &'a str,
    provider_id: &'a str,
    session_id: &'a str,
    exit_code: Option<i32>,
    duration_ms: u64,
    stdout_len: usize,
    stderr_len: usize,
    artifact_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    git_head: Option<&'a str>,
}

async fn append_execution_manifest(
    workspace_dir: &Path,
    line: ExecutionManifestLine<'_>,
) -> Result<(), String> {
    let dir = workspace_dir.join(".system_generated");
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("manifest mkdir: {e}"))?;
    let path = dir.join("execution_runs.jsonl");
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|e| format!("manifest open: {e}"))?;
    let json = serde_json::to_string(&line).map_err(|e| e.to_string())?;
    f.write_all(json.as_bytes())
        .await
        .map_err(|e| format!("manifest write: {e}"))?;
    f.write_all(b"\n")
        .await
        .map_err(|e| format!("manifest nl: {e}"))?;
    Ok(())
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
        "Create an execution session (requires [harness.execution] enabled). Provider is [harness.execution] default_provider: local runs under the workspace sandbox; jupyter uses a configured Jupyter Server kernel; ssh opens one SSH session to a configured host (reused until execution_session_close). Returns session_id, session capabilities, and a short provider capability summary. Use execution_run to execute code; execution_session_close when done. Jupyter may write binary display_data (PNG, JPEG, large CSV/JSON) under the sandbox `.execution_artifacts/{session_id}/{run_id}/` and list them in RunResult.attachments."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "label": { "type": "string", "description": "Optional label for logs" },
                "language": {
                    "type": "string",
                    "description": "Optional language hint. Local: python, py, shell, sh, bash. Jupyter: python, py, r, R (ir kernel). SSH: python, py, shell, sh, bash (remote exec with code on stdin)."
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
            "artifact_root_relative": format!(".execution_artifacts/{}/", sanitize_session_segment(&handle.id)),
        });
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
    }
}

/// Run code in an existing session (subject to provider timeout and output caps).
pub struct ExecutionRunTool {
    pub harness: Arc<ExecutionHarness>,
    pub outbound_tx: mpsc::Sender<BusMessage>,
}

#[async_trait]
impl Tool for ExecutionRunTool {
    fn name(&self) -> &str {
        "execution_run"
    }

    fn description(&self) -> &str {
        "Run code in an execution_session_create session. Args: session_id, code, timeout_secs (optional), cwd_mode (optional session_default|sandbox_relative), cwd_relative when sandbox_relative. stdout/stderr may be truncated per [harness.execution] limits. Jupyter: binary plots and large CSV/JSON are saved under `.execution_artifacts/{session_id}/<run>/` and returned in `attachments` with sandbox-relative paths; use execution_artifact_list to browse. A line is appended to workspace `.system_generated/execution_runs.jsonl` per run (metadata only, no code)."
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
        let duration_ms = started.elapsed().as_millis() as u64;
        let artifact_count = result.attachments.len();
        info!(
            "execution_run provider={} session={} exit={:?} stdout_len={} stderr_len={} duration_ms={} artifacts={}",
            prov,
            sid,
            result.exit_code,
            result.stdout.len(),
            result.stderr.len(),
            duration_ms,
            artifact_count
        );

        let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let (chat_id, channel) = current_tool_exec_ctx()
            .map(|c| (c.chat_id, c.channel))
            .unwrap_or_else(|| (String::new(), String::new()));
        let git_head = best_effort_git_head(self.harness.workspace_dir());

        let manifest = ExecutionManifestLine {
            ts: &ts,
            chat_id: &chat_id,
            channel: &channel,
            provider_id: prov,
            session_id: sid.as_str(),
            exit_code: result.exit_code,
            duration_ms,
            stdout_len: result.stdout.len(),
            stderr_len: result.stderr.len(),
            artifact_count,
            git_head: git_head.as_deref(),
        };
        if let Err(e) = append_execution_manifest(self.harness.workspace_dir(), manifest).await {
            log::warn!("execution manifest append failed: {e}");
        }

        let _ = self
            .outbound_tx
            .send(BusMessage::Telemetry(
                TelemetryEvent::ExecutionRunFinished {
                    chat_id: chat_id.clone(),
                    channel: channel.clone(),
                    provider_id: prov.to_string(),
                    session_id: sid.to_string(),
                    exit_code: result.exit_code,
                    duration_ms,
                    stdout_len: result.stdout.len(),
                    stderr_len: result.stderr.len(),
                    artifact_count,
                    git_head: git_head.clone(),
                },
            ))
            .await;

        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }
}

/// List files under `.execution_artifacts/{session_id}/` for a session (sandbox-relative paths).
pub struct ExecutionArtifactListTool {
    pub harness: Arc<ExecutionHarness>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ArtifactListEntry {
    path: String,
    size_bytes: u64,
}

#[async_trait]
impl Tool for ExecutionArtifactListTool {
    fn name(&self) -> &str {
        "execution_artifact_list"
    }

    fn description(&self) -> &str {
        "List materialized execution artifacts for a session under the sandbox `.execution_artifacts/{session_id}/` (all runs). Returns sandbox-relative paths and sizes. Use read_file on a listed path to fetch text; binary files are best opened outside the agent."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string", "description": "Execution session id from execution_session_create" },
                "limit": { "type": "integer", "description": "Max file entries (default 100, max 500)" }
            },
            "required": ["session_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let sid = require_session_id(&args)?;
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(100)
            .clamp(1, 500) as usize;
        let seg = sanitize_session_segment(&sid);
        let base = self
            .harness
            .sandbox_dir()
            .join(crate::execution::ARTIFACT_ROOT_DIR)
            .join(&seg);
        if !base.starts_with(self.harness.sandbox_dir()) {
            return Err("invalid artifact path".to_string());
        }
        let mut entries: Vec<ArtifactListEntry> = Vec::new();
        if base.is_dir() {
            let mut rd = tokio::fs::read_dir(&base)
                .await
                .map_err(|e| format!("read_dir: {e}"))?;
            while let Some(ent) = rd.next_entry().await.map_err(|e| e.to_string())? {
                if entries.len() >= limit {
                    break;
                }
                let p = ent.path();
                let meta = ent.metadata().await.map_err(|e| e.to_string())?;
                if meta.is_file() {
                    let rel = p
                        .strip_prefix(self.harness.sandbox_dir())
                        .map_err(|_| "artifact path escaped sandbox".to_string())?;
                    entries.push(ArtifactListEntry {
                        path: rel.to_string_lossy().replace('\\', "/"),
                        size_bytes: meta.len(),
                    });
                } else if meta.is_dir() {
                    let mut sub = tokio::fs::read_dir(&p)
                        .await
                        .map_err(|e| format!("read_dir sub: {e}"))?;
                    while let Some(f) = sub.next_entry().await.map_err(|e| e.to_string())? {
                        if entries.len() >= limit {
                            break;
                        }
                        let fp = f.path();
                        let m = f.metadata().await.map_err(|e| e.to_string())?;
                        if m.is_file() {
                            let rel = fp
                                .strip_prefix(self.harness.sandbox_dir())
                                .map_err(|_| "artifact path escaped sandbox".to_string())?;
                            entries.push(ArtifactListEntry {
                                path: rel.to_string_lossy().replace('\\', "/"),
                                size_bytes: m.len(),
                            });
                        }
                    }
                }
            }
        }
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        let v = json!({
            "session_id": sid,
            "artifact_session_dir_relative": format!("{}/{}", crate::execution::ARTIFACT_ROOT_DIR, seg),
            "entries": entries,
            "truncated": entries.len() >= limit,
        });
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
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
        "Return provider capability summary. For local execution, also runs python_executable -V on the agent host (best effort). For jupyter, that probe is still the host interpreter (sanity check only); the kernel Python environment is whatever the Jupyter server started for that kernelspec. For ssh, the probe is still the agent host interpreter (not the remote remote_python)."
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
            "artifact_limits": self.harness.artifact_limits(),
            "artifact_root_relative": crate::execution::ARTIFACT_ROOT_DIR,
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
    use crate::execution::ArtifactLimits;

    fn temp_dirs() -> (std::path::PathBuf, std::path::PathBuf) {
        let root =
            std::env::temp_dir().join(format!("isanagent-exec-test-{}", uuid::Uuid::new_v4()));
        let sandbox = root.join("sandbox");
        std::fs::create_dir_all(&sandbox).unwrap();
        (root, sandbox)
    }

    #[tokio::test]
    async fn create_run_close_roundtrip() {
        let (ws, dir) = temp_dirs();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), true);
        let prov: Arc<dyn crate::execution::ExecutionProvider> =
            Arc::new(crate::execution::LocalExecutionProvider::new(cfg).expect("local provider"));
        let harness = Arc::new(ExecutionHarness::new(
            prov,
            "python",
            ws.clone(),
            dir.clone(),
            ArtifactLimits::default(),
        ));
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
        let (otx, mut orx) = mpsc::channel::<BusMessage>(8);
        let run = ExecutionRunTool {
            harness: harness.clone(),
            outbound_tx: otx,
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
        let _ = orx.try_recv();
        let close = ExecutionSessionCloseTool { harness };
        close.execute(json!({ "session_id": sid })).await.unwrap();
        let _ = std::fs::remove_dir_all(&ws);
    }
}
