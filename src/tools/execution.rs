//! Gated harness tools for the execution plane (`[harness.execution] enabled = true`).

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::bus::BusMessage;
use crate::channels::terminal::build_execution_stream_notice;
use crate::execution::{
    persist_successful_execution_run, sanitize_session_segment, CwdPolicy, ExecutionError,
    ExecutionHarness, ExecutionJobManager, PersistSuccessfulExecutionRunParams, RunEvent, RunSpec,
    SessionCreateRequest, SessionId, SpawnBackgroundRunRequest,
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

const MAX_EXEC_DESCRIPTION_CHARS: usize = 200;

/// Optional short human-facing line for terminal UI and audits (truncated).
fn parse_optional_execution_description(args: &Value) -> Option<String> {
    args.get("description")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            let t = s.to_string();
            let n = t.chars().count();
            if n <= MAX_EXEC_DESCRIPTION_CHARS {
                return t;
            }
            format!(
                "{}…",
                t.chars()
                    .take(MAX_EXEC_DESCRIPTION_CHARS.saturating_sub(1))
                    .collect::<String>()
            )
        })
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
        "Create an execution session (requires [harness.execution] enabled). Provider is [harness.execution] default_provider: local runs under the workspace sandbox; jupyter uses a Jupyter Server kernel (new or `resume_jupyter_kernel_id`); ssh opens one SSH session to a configured host (reused until execution_session_close); colab_mcp launches a local Colab MCP bridge process and targets a notebook execution tool exposed by the browser session. Returns session_id, session capabilities, and a short provider capability summary. Use execution_run or execution_run_background to execute code; execution_session_close when done. Jupyter may write binary display_data under the sandbox `.execution_artifacts/{session_id}/{run_id}/` and list them in RunResult.attachments."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "label": { "type": "string", "description": "Optional label for logs" },
                "language": {
                    "type": "string",
                    "description": "Optional language hint. Local: python, py, shell, sh, bash. Jupyter: python, py, r, R (ir kernel). SSH: python, py, shell, sh, bash (remote exec with code on stdin). Colab MCP MVP currently expects python."
                },
                "resume_jupyter_kernel_id": {
                    "type": "string",
                    "description": "Jupyter only: reuse an existing kernel id from `session_capabilities.extensions.jupyter_kernel_id` (or a prior server listing). Fails if the kernel no longer exists."
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
            resume_jupyter_kernel_id: args
                .get("resume_jupyter_kernel_id")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
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
        "Run code in an execution_session_create session. Args: session_id, code, timeout_secs (optional — omit only for quick runs; for generation/training set explicitly up to max_wall_secs; use smaller values for probes), description (optional, strongly recommended for Ratatui and logs: short human summary of intent), cwd_mode, cwd_relative. stdout/stderr may be truncated per [harness.execution] limits. Jupyter: live stream events are emitted on the bus for the terminal UI; binary plots and large CSV/JSON are saved under `.execution_artifacts/{session_id}/<run>/` and returned in `attachments`. Each run writes a journal under workspace `.system_generated/execution_history/{provider}/{session_id}/{run_id}/` (`run.json` + `source.txt`). A metadata line is appended to `.system_generated/execution_runs.jsonl` (no code)."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string" },
                "code": { "type": "string" },
                "timeout_secs": { "type": "integer", "description": "Wall-clock limit (clamped to max_wall_secs). Prefer explicit values for long work; use small values for quick checks." },
                "description": { "type": "string", "description": "Short human-facing summary for terminal UI and audits (max ~200 chars); omit only for trivial runs" },
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
            .unwrap_or(self.harness.default_run_timeout_secs);
        let cwd = parse_cwd(&args)?;
        let run_description = parse_optional_execution_description(&args);
        let run_id = uuid::Uuid::new_v4().to_string();
        let started = Instant::now();
        let started_ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

        let prov = self.harness.provider().provider_id();
        let (event_tx, event_rx) = if prov == "jupyter" {
            let (tx, rx) = mpsc::channel::<RunEvent>(128);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let (chat_id, channel) = current_tool_exec_ctx()
            .map(|c| (c.chat_id.clone(), c.channel.clone()))
            .unwrap_or_else(|| (String::new(), String::new()));

        if let Some(mut rx) = event_rx {
            let ob = self.outbound_tx.clone();
            let sid_s = sid.to_string();
            let rid = run_id.clone();
            let cid = chat_id.clone();
            let ch = channel.clone();
            let stream_desc = run_description.clone();
            tokio::spawn(async move {
                while let Some(ev) = rx.recv().await {
                    let payload = serde_json::to_string(&ev).unwrap_or_default();
                    let msg = build_execution_stream_notice(
                        &cid,
                        &ch,
                        &sid_s,
                        &rid,
                        &payload,
                        stream_desc.as_deref(),
                    );
                    let _ = ob.send(BusMessage::Outbound(msg)).await;
                }
            });
        }

        let mut spec = RunSpec::new(code.clone(), timeout_secs);
        spec.cwd = cwd;
        spec.run_id = Some(run_id.clone());
        spec.run_event_tx = event_tx;
        spec.description = run_description.clone();

        let result = self
            .harness
            .provider()
            .run(&sid, spec)
            .await
            .map_err(exec_err)?;
        let duration_ms = started.elapsed().as_millis() as u64;
        let artifact_count = result.attachments.len();
        let exit_s = match result.exit_code {
            None => "none".to_string(),
            Some(0) => "0".to_string(),
            Some(n) => format!("exit {n}"),
        };
        info!(
            "execution_run provider={} session={} exit={} stdout_len={} stderr_len={} duration_ms={} artifacts={}",
            prov,
            sid,
            exit_s,
            result.stdout.len(),
            result.stderr.len(),
            duration_ms,
            artifact_count
        );

        persist_successful_execution_run(PersistSuccessfulExecutionRunParams {
            harness: self.harness.as_ref(),
            outbound_tx: &self.outbound_tx,
            provider_id: prov,
            sid: &sid,
            run_id: &run_id,
            code: &code,
            result: &result,
            started_ts: &started_ts,
            chat_id: &chat_id,
            channel: &channel,
            duration_ms,
            job_id: None,
            run_description: run_description.as_deref(),
        })
        .await;

        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }
}

/// Start a run on a background task; returns `job_id` immediately (same session concurrency rules as `execution_run`).
pub struct ExecutionRunBackgroundTool {
    pub harness: Arc<ExecutionHarness>,
    pub jobs: Arc<ExecutionJobManager>,
}

#[async_trait]
impl Tool for ExecutionRunBackgroundTool {
    fn name(&self) -> &str {
        "execution_run_background"
    }

    fn description(&self) -> &str {
        "Same as execution_run (session_id, code, optional timeout_secs, description, cwd_*) but returns immediately with a job_id. Poll execution_job_status / execution_job_result; use execution_job_cancel or execution_cancel on the session to interrupt when supported. One active run or background job per session. Jobs are process-local and lost on agent exit. Always pass description for runs expected to take more than ~30s so the terminal strip stays readable."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string" },
                "code": { "type": "string" },
                "timeout_secs": { "type": "integer", "description": "Wall-clock limit (defaults to default_execution_timeout_secs when omitted, clamped to max_wall_secs). Set explicitly for long jobs; use small values for quick checks." },
                "description": { "type": "string", "description": "Short human-facing summary for terminal UI and audits (max ~200 chars); strongly recommended for background work" },
                "cwd_mode": { "type": "string", "description": "session_default (default) or sandbox_relative" },
                "cwd_relative": { "type": "string", "description": "Path relative to sandbox when cwd_mode is sandbox_relative" },
                "label": { "type": "string", "description": "Optional label for operator logs" }
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
            .unwrap_or(self.harness.default_run_timeout_secs);
        let cwd = parse_cwd(&args)?;
        let label = args
            .get("label")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let run_description = parse_optional_execution_description(&args);
        let (chat_id, channel) = current_tool_exec_ctx()
            .map(|c| (c.chat_id.clone(), c.channel.clone()))
            .unwrap_or_else(|| (String::new(), String::new()));
        let job_id = self.jobs.spawn_run(SpawnBackgroundRunRequest {
            sid: sid.clone(),
            code,
            timeout_secs,
            cwd,
            label,
            run_description,
            chat_id,
            channel,
        })?;
        let v = json!({
            "job_id": job_id,
            "session_id": sid.to_string(),
            "provider_id": self.harness.provider().provider_id(),
        });
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
    }
}

pub struct ExecutionJobStatusTool {
    pub jobs: Arc<ExecutionJobManager>,
}

#[async_trait]
impl Tool for ExecutionJobStatusTool {
    fn name(&self) -> &str {
        "execution_job_status"
    }

    fn description(&self) -> &str {
        "Return status, timestamps, and error (if any) for a background execution job started with execution_run_background."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string" }
            },
            "required": ["job_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let job_id = args
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'job_id'".to_string())?;
        let v = self.jobs.job_status_json(job_id).await?;
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
    }
}

pub struct ExecutionJobResultTool {
    pub jobs: Arc<ExecutionJobManager>,
    pub max_tool_output_chars: usize,
}

#[async_trait]
impl Tool for ExecutionJobResultTool {
    fn name(&self) -> &str {
        "execution_job_result"
    }

    fn description(&self) -> &str {
        "When the job is terminal, return the RunResult-shaped JSON (stdout/stderr/attachments). While running, returns a short JSON message. Output may be truncated to the session max_tool_output_chars cap."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string" }
            },
            "required": ["job_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let job_id = args
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'job_id'".to_string())?;
        self.jobs
            .job_result_pretty(job_id, self.max_tool_output_chars)
            .await
    }
}

pub struct ExecutionJobListTool {
    pub jobs: Arc<ExecutionJobManager>,
}

#[async_trait]
impl Tool for ExecutionJobListTool {
    fn name(&self) -> &str {
        "execution_job_list"
    }

    fn description(&self) -> &str {
        "List recent in-memory background execution jobs (optional session_id filter, limit default 50, max 500). Jobs are evicted after completion when the registry is full."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string", "description": "When set, only jobs for this execution session" },
                "limit": { "type": "integer", "description": "Max rows (default 50, max 500)" }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let session = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(SessionId::new);
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let v = self.jobs.list_jobs_json(session, limit).await;
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
    }
}

pub struct ExecutionJobCancelTool {
    pub jobs: Arc<ExecutionJobManager>,
}

#[async_trait]
impl Tool for ExecutionJobCancelTool {
    fn name(&self) -> &str {
        "execution_job_cancel"
    }

    fn description(&self) -> &str {
        "Best-effort interrupt for a background job by job_id (maps to execution_cancel for that job's session). Same capability gate as execution_cancel."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string" }
            },
            "required": ["job_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let job_id = args
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'job_id'".to_string())?;
        self.jobs.cancel_job(job_id).await?;
        Ok(format!("cancel requested for job {job_id}"))
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
        let max_wall = self.harness.max_wall_secs;
        let def_timeout = self.harness.default_run_timeout_secs;
        let timeout_policy = format!(
            "Per-run wall clock is capped at max_wall_secs ({max_wall}). If you omit timeout_secs on execution_run / execution_run_background, default_run_timeout_secs ({def_timeout}) applies. For long generation, training, or heavy I/O, set timeout_secs explicitly (up to the cap). For quick probes and tight polling loops, use a small timeout_secs. Prefer execution_run_background when work may block the reasoning loop for many minutes, then poll execution_job_status. Call execution_env_info anytime to re-read these caps. Use the description field on execution_run / execution_run_background so the terminal UI and audits show intent instead of raw ids."
        );
        let mut v = json!({
            "provider_capabilities": self.harness.capabilities_summary(),
            "artifact_limits": self.harness.artifact_limits(),
            "artifact_root_relative": crate::execution::ARTIFACT_ROOT_DIR,
            "max_wall_secs": max_wall,
            "default_run_timeout_secs": def_timeout,
            "timeout_policy": timeout_policy,
        });
        let probe = tokio::task::spawn_blocking({
            let exe = exe.to_string();
            move || {
                crate::execution::build_python_host_command(&exe)
                    .arg("-V")
                    .output()
            }
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
    use crate::execution::ExecutionJobManager;
    use std::time::Duration;

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
            60,
            3600,
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

    #[tokio::test]
    async fn background_job_poll_and_result() {
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
            60,
            3600,
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
        let (otx, _orx) = mpsc::channel::<BusMessage>(8);
        let jobs = Arc::new(ExecutionJobManager::new(harness.clone(), otx));
        let bg = ExecutionRunBackgroundTool {
            harness: harness.clone(),
            jobs: jobs.clone(),
        };
        let code = if cfg!(windows) {
            "echo bg-job"
        } else {
            "print('bg-job')"
        };
        let started = bg
            .execute(json!({
                "session_id": sid,
                "code": code,
                "timeout_secs": 30,
            }))
            .await
            .expect("bg");
        let jv: Value = serde_json::from_str(&started).expect("json");
        let jid = jv["job_id"].as_str().expect("job_id");
        let status_tool = ExecutionJobStatusTool { jobs: jobs.clone() };
        let mut terminal = false;
        for _ in 0..80 {
            let s = status_tool
                .execute(json!({ "job_id": jid }))
                .await
                .expect("status");
            if s.contains("\"terminal\": true") {
                terminal = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(terminal, "job did not reach terminal state");
        let result_tool = ExecutionJobResultTool {
            jobs: jobs.clone(),
            max_tool_output_chars: 20_000,
        };
        let r = result_tool
            .execute(json!({ "job_id": jid }))
            .await
            .expect("result");
        assert!(r.contains("bg-job"), "result={r}");
        let list = ExecutionJobListTool { jobs: jobs.clone() };
        let listed = list.execute(json!({})).await.expect("list");
        assert!(listed.contains(jid));
        let close = ExecutionSessionCloseTool { harness };
        close.execute(json!({ "session_id": sid })).await.unwrap();
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    async fn execution_env_info_includes_timeout_caps() {
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
            90,
            1200,
        ));
        let env = ExecutionEnvInfoTool { harness };
        let out = env.execute(json!({})).await.expect("env");
        assert!(out.contains("\"max_wall_secs\": 1200"), "out={out}");
        assert!(
            out.contains("\"default_run_timeout_secs\": 90"),
            "out={out}"
        );
        assert!(out.contains("timeout_policy"));
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    async fn background_job_carries_description_in_status_and_list() {
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
            60,
            3600,
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
        let (otx, _orx) = mpsc::channel::<BusMessage>(8);
        let jobs = Arc::new(ExecutionJobManager::new(harness.clone(), otx));
        let bg = ExecutionRunBackgroundTool {
            harness: harness.clone(),
            jobs: jobs.clone(),
        };
        let code = if cfg!(windows) {
            "echo desc-test"
        } else {
            "print('desc-test')"
        };
        let started = bg
            .execute(json!({
                "session_id": sid,
                "code": code,
                "timeout_secs": 30,
                "description": "Unit test background label",
            }))
            .await
            .expect("bg");
        let jv: Value = serde_json::from_str(&started).expect("json");
        let jid = jv["job_id"].as_str().expect("job_id");
        for _ in 0..80 {
            let st = ExecutionJobStatusTool { jobs: jobs.clone() };
            let s = st.execute(json!({ "job_id": jid })).await.expect("st");
            if s.contains("\"terminal\": true") {
                assert!(s.contains("Unit test background label"));
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let list = ExecutionJobListTool { jobs: jobs.clone() };
        let listed = list.execute(json!({})).await.expect("list");
        assert!(listed.contains("Unit test background label"));
        let close = ExecutionSessionCloseTool { harness };
        close.execute(json!({ "session_id": sid })).await.unwrap();
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    async fn background_job_cancel_requests_interrupt() {
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
            60,
            3600,
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
        let (otx, _orx) = mpsc::channel::<BusMessage>(8);
        let jobs = Arc::new(ExecutionJobManager::new(harness.clone(), otx));
        let bg = ExecutionRunBackgroundTool {
            harness: harness.clone(),
            jobs: jobs.clone(),
        };
        let code = if cfg!(windows) {
            "ping 127.0.0.1 -n 30 >nul"
        } else {
            "import time\ntime.sleep(30)"
        };
        let started = bg
            .execute(json!({
                "session_id": sid,
                "code": code,
                "timeout_secs": 60,
            }))
            .await
            .expect("bg");
        let jv: Value = serde_json::from_str(&started).expect("json");
        let jid = jv["job_id"].as_str().expect("job_id");
        tokio::time::sleep(Duration::from_millis(80)).await;
        let cancel = ExecutionJobCancelTool { jobs: jobs.clone() };
        cancel
            .execute(json!({ "job_id": jid }))
            .await
            .expect("cancel");
        let status_tool = ExecutionJobStatusTool { jobs };
        let mut saw = false;
        for _ in 0..120 {
            let s = status_tool
                .execute(json!({ "job_id": jid }))
                .await
                .expect("status");
            if s.contains("cancelled") || s.contains("timeout") || s.contains("failed") {
                saw = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(40)).await;
        }
        assert!(saw, "expected terminal non-success after cancel");
        let close = ExecutionSessionCloseTool { harness };
        close.execute(json!({ "session_id": sid })).await.unwrap();
        let _ = std::fs::remove_dir_all(&ws);
    }
}
