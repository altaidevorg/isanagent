//! Harness tools for the execution plane (omitted unless `AppConfig::execution_harness_enabled()` is false).

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::bus::BusMessage;
use crate::channels::terminal::build_execution_stream_notice;
use crate::execution::{
    persist_successful_execution_run, run_with_auto_promote, sanitize_session_segment,
    AdoptInflightRequest, AutoPromoteOutcome, CwdPolicy, ExecutionError, ExecutionHarness,
    ExecutionJobManager, InflightSyncRegistry, PersistSuccessfulExecutionRunParams, RunEvent,
    RunResult, RunSpec, SessionCreateRequest, SessionId, SpawnBackgroundRunRequest,
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
        "Create an execution session (requires [harness.execution] enabled). \
         Defaults to [harness.execution] default_provider; pass `provider` to pick another \
         currently-available provider when multiple are configured. Provider semantics: \
         local runs under the workspace sandbox; jupyter uses a Jupyter Server kernel \
         (new or `resume_jupyter_kernel_id`); ssh opens one SSH connection to a configured host \
         (reused until execution_session_close; Python keeps a persistent remote REPL; shell is one-off per run; \
         execution_run cwd fields refer to remote paths only for ssh). Misconfigured providers are pruned at startup \
         (warning logged) and won't appear in `provider`'s allowed values for this run, so \
         passing a known-bad provider is unnecessary. Returns session_id, session capabilities, \
         and a short provider capability summary. Use execution_run or execution_run_background \
         to execute code; execution_session_close when done. Jupyter may write binary \
         display_data under the sandbox `.execution_artifacts/{session_id}/{run_id}/` and list \
         them in RunResult.attachments."
    }

    fn parameters(&self) -> Value {
        let avail = self.harness.available_providers();
        let avail_strs: Vec<String> = avail.iter().map(|s| s.to_string()).collect();
        let default_id = self.harness.default_provider_id().to_string();
        serde_json::json!({
            "type": "object",
            "properties": {
                "label": { "type": "string", "description": "Optional label for logs" },
                "language": {
                    "type": "string",
                    "description": "Optional language hint. Local: shell, sh, bash, cmd, powershell (all map to the host shell — use python_run or exec+uv for Python). Jupyter: python, py, r, R (ir kernel). SSH: python, py, shell, sh, bash (remote exec with code on stdin)."
                },
                "resume_jupyter_kernel_id": {
                    "type": "string",
                    "description": "Jupyter only: reuse an existing kernel id from `session_capabilities.extensions.jupyter_kernel_id` (or a prior server listing). Fails if the kernel no longer exists."
                },
                "provider": {
                    "type": "string",
                    "enum": avail_strs,
                    "description": format!(
                        "Optional provider id; omit to use the default ({default_id}). Currently available providers (those that built successfully at startup): [{}].",
                        avail.join(", ")
                    )
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
        let provider_arg = args
            .get("provider")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let provider = self.harness.provider_for(provider_arg)?;
        let provider_id = provider.provider_id().to_string();
        let handle = provider.create_session(req).await.map_err(exec_err)?;
        // Bind this session to the provider that created it so subsequent execution_run /
        // execution_session_close calls route to the right provider in mixed-provider setups.
        self.harness
            .register_session(handle.id.as_str(), &provider_id);
        let summary = self.harness.capabilities_summary();
        let v = json!({
            "session_id": handle.id,
            "session_capabilities": handle.capabilities,
            "provider_capabilities": summary,
            "provider": provider_id,
            "artifact_root_relative": format!(".execution_artifacts/{}/", sanitize_session_segment(&handle.id)),
        });
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
    }
}

/// Run code in an existing session (subject to provider timeout and output caps).
pub struct ExecutionRunTool {
    pub harness: Arc<ExecutionHarness>,
    pub outbound_tx: mpsc::Sender<BusMessage>,
    /// Job manager used to auto-promote long synchronous runs into background jobs.
    /// `None` disables auto-promote (e.g. unit tests or harnesses without a job registry).
    pub jobs: Option<Arc<ExecutionJobManager>>,
    /// Registry that maps `chat_id -> oneshot::Sender<()>`; the `/background` slash command
    /// pushes onto it to let the user manually promote a sync run before the timer fires.
    pub inflight: Option<Arc<InflightSyncRegistry>>,
}

#[async_trait]
impl Tool for ExecutionRunTool {
    fn name(&self) -> &str {
        "execution_run"
    }

    fn description(&self) -> &str {
        "Run code in an execution_session_create session. Args: session_id, code, timeout_secs (optional — omit only for quick runs; for generation/training set explicitly up to max_wall_secs; use smaller values for probes), description (optional, strongly recommended for Ratatui and logs: short human summary of intent), cwd_mode, cwd_relative. stdout/stderr may be truncated per [harness.execution] limits. Jupyter: live stream events are emitted on the bus for the terminal UI; binary plots and large CSV/JSON are saved under `.execution_artifacts/{session_id}/<run>/` and returned in `attachments`. SSH: cwd_mode applies on the remote host (sandbox_relative joins under [harness.execution.ssh].remote_workdir or uses an absolute remote path); Python variables persist across runs in the same session. Each run writes a journal under workspace `.system_generated/execution_history/{provider}/{session_id}/{run_id}/` (`run.json` + `source.txt`). A metadata line is appended to `.system_generated/execution_runs.jsonl` (no code)."
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

        // Resolve the provider that owns this session (or fall back to default for legacy
        // single-provider call sites that bypass register_session).
        let session_provider = self.harness.provider_for_session(sid.as_str());
        let prov = session_provider.provider_id();
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

        let auto_promote = self.harness.auto_promote_after_secs;
        let promote_enabled = self.jobs.is_some()
            && auto_promote > 0
            && auto_promote < timeout_secs
            && !chat_id.is_empty();

        if !promote_enabled {
            let result = session_provider.run(&sid, spec).await.map_err(exec_err)?;
            return self
                .finalize_sync_run(FinalizeSyncRunParams {
                    sid: &sid,
                    code: &code,
                    result: &result,
                    started,
                    started_ts: &started_ts,
                    run_id: &run_id,
                    chat_id: &chat_id,
                    channel: &channel,
                    run_description: run_description.as_deref(),
                    provider_id: prov,
                })
                .await;
        }

        let jobs = self.jobs.clone().expect("jobs checked above");
        let inflight = self.inflight.clone();
        let promote_rx = inflight.as_ref().map(|reg| reg.register(&chat_id));
        let (promote_signal_rx, _inflight_guard) = match promote_rx {
            Some((rx, guard)) => (Some(rx), Some(guard)),
            None => (None, None),
        };

        let provider = session_provider.clone();
        let sid_for_work = sid.clone();
        let work = async move { provider.run(&sid_for_work, spec).await };

        let chat_id_for_promote = chat_id.clone();
        let channel_for_promote = channel.clone();
        let sid_for_promote = sid.clone();
        let description_for_promote = run_description.clone();

        let outcome = run_with_auto_promote::<Result<RunResult, ExecutionError>, _, _>(
            work,
            Duration::from_secs(auto_promote),
            promote_signal_rx,
            move |handle, _reason| {
                let req = AdoptInflightRequest {
                    sid: sid_for_promote,
                    tool_name: "execution_run".to_string(),
                    label: None,
                    description: description_for_promote,
                    chat_id: chat_id_for_promote,
                    channel: channel_for_promote,
                    join: handle,
                };
                jobs.adopt_inflight(req).unwrap_or_else(|e| {
                    log::warn!("execution_run: adopt_inflight failed: {e}");
                    String::new()
                })
            },
        )
        .await;

        match outcome {
            AutoPromoteOutcome::Completed(Ok(result)) => {
                self.finalize_sync_run(FinalizeSyncRunParams {
                    sid: &sid,
                    code: &code,
                    result: &result,
                    started,
                    started_ts: &started_ts,
                    run_id: &run_id,
                    chat_id: &chat_id,
                    channel: &channel,
                    run_description: run_description.as_deref(),
                    provider_id: prov,
                })
                .await
            }
            AutoPromoteOutcome::Completed(Err(e)) => Err(exec_err(e)),
            AutoPromoteOutcome::Promoted { job_id, reason } => {
                let envelope = json!({
                    "auto_promoted": true,
                    "reason": reason.as_str(),
                    "job_id": job_id,
                    "session_id": sid.to_string(),
                    "tool_name": "execution_run",
                    "follow_up": "Use execution_job_status / execution_job_result to retrieve the run result when it finishes. Use execution_job_cancel for best-effort interrupt.",
                });
                serde_json::to_string_pretty(&envelope).map_err(|e| e.to_string())
            }
        }
    }
}

struct FinalizeSyncRunParams<'a> {
    sid: &'a SessionId,
    code: &'a str,
    result: &'a RunResult,
    started: Instant,
    started_ts: &'a str,
    run_id: &'a str,
    chat_id: &'a str,
    channel: &'a str,
    run_description: Option<&'a str>,
    provider_id: &'a str,
}

impl ExecutionRunTool {
    async fn finalize_sync_run(&self, p: FinalizeSyncRunParams<'_>) -> Result<String, String> {
        let duration_ms = p.started.elapsed().as_millis() as u64;
        let artifact_count = p.result.attachments.len();
        let exit_s = match p.result.exit_code {
            None => "none".to_string(),
            Some(0) => "0".to_string(),
            Some(n) => format!("exit {n}"),
        };
        info!(
            "execution_run provider={} session={} exit={} stdout_len={} stderr_len={} duration_ms={} artifacts={}",
            p.provider_id,
            p.sid,
            exit_s,
            p.result.stdout.len(),
            p.result.stderr.len(),
            duration_ms,
            artifact_count
        );

        persist_successful_execution_run(PersistSuccessfulExecutionRunParams {
            harness: self.harness.as_ref(),
            outbound_tx: &self.outbound_tx,
            provider_id: p.provider_id,
            sid: p.sid,
            run_id: p.run_id,
            code: p.code,
            result: p.result,
            started_ts: p.started_ts,
            chat_id: p.chat_id,
            channel: p.channel,
            duration_ms,
            job_id: None,
            run_description: p.run_description,
        })
        .await;

        serde_json::to_string_pretty(p.result).map_err(|e| e.to_string())
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
        "Same as execution_run (session_id, code, optional timeout_secs, description, cwd_*) but returns immediately with a job_id. When the job reaches a terminal state, the harness may enqueue a synthetic follow-up message (unless wake_on_job_terminal is false in config) so you can call execution_job_status / execution_job_result without waiting for the user; otherwise poll manually. Use execution_job_cancel or execution_cancel on the session to interrupt when supported. One active run or background job per session. Jobs are process-local and lost on agent exit. Always pass description for runs expected to take more than ~30s so the terminal strip stays readable."
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
            "provider_id": self.harness.provider_for_session(sid.as_str()).provider_id(),
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
        "When the job is terminal, return the RunResult-shaped JSON (stdout/stderr/attachments). While running, returns a short JSON message. Output may be truncated to the session max_tool_output_chars cap. After a background job finishes, a follow-up user turn may be injected automatically—call this tool to fetch the full result."
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

pub struct ExecutionReadLogTool {
    pub jobs: Arc<ExecutionJobManager>,
    pub harness: Arc<ExecutionHarness>,
}

#[async_trait]
impl Tool for ExecutionReadLogTool {
    fn name(&self) -> &str {
        "execution_read_log"
    }

    fn description(&self) -> &str {
        "Read raw stdout or stderr logs for a given background execution job or run_id without fetching the entire output. Useful for inspecting massive logs from tools or background processes that are otherwise truncated in the result. You can read specific lines by specifying start_line and end_line (1-indexed, inclusive) capped at a maximum of 100 lines per call."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string", "description": "The job ID to read logs from. Provide either this or run_id." },
                "run_id": { "type": "string", "description": "The run ID to read logs from. Provide either this or job_id." },
                "stream": { "type": "string", "enum": ["stdout", "stderr"], "description": "Which stream to read." },
                "start_line": { "type": "integer", "description": "Starting line number (1-indexed, inclusive)" },
                "end_line": { "type": "integer", "description": "Ending line number (1-indexed, inclusive)" }
            },
            "required": ["stream", "start_line", "end_line"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let stream = args
            .get("stream")
            .and_then(|v| v.as_str())
            .unwrap_or("stdout");
        let start_line = args
            .get("start_line")
            .and_then(|v| v.as_u64())
            .ok_or("Missing 'start_line' argument")?;
        let end_line = args
            .get("end_line")
            .and_then(|v| v.as_u64())
            .ok_or("Missing 'end_line' argument")?;

        let run_id;
        let mut sid = None;
        let mut prov = None;

        if let Some(jid) = args.get("job_id").and_then(|v| v.as_str()) {
            if let Some(job) = self.jobs.get(jid) {
                run_id = job.run_id.clone();
                sid = Some(job.session_id.to_string());
                prov = Some(
                    self.harness
                        .provider_for_session(job.session_id.as_str())
                        .provider_id()
                        .to_string(),
                );
            } else {
                return Err(format!("Job not found: {jid}"));
            }
        } else if let Some(run) = args.get("run_id").and_then(|v| v.as_str()) {
            run_id = run.to_string();
        } else {
            return Err("Must provide either job_id or run_id".to_string());
        }

        let target_file = format!("{stream}.txt");

        let mut log_path = None;
        if let (Some(s), Some(p)) = (sid, prov) {
            use crate::execution::run_history_dir;
            let sid_obj = crate::execution::SessionId::new(s);
            log_path = Some(
                run_history_dir(self.harness.workspace_dir(), &p, &sid_obj, &run_id)
                    .join(&target_file),
            );
        } else {
            let hist_dir = self
                .harness
                .workspace_dir()
                .join(".system_generated")
                .join("execution_history");
            if let Ok(mut entries1) = tokio::fs::read_dir(&hist_dir).await {
                while let Ok(Some(e1)) = entries1.next_entry().await {
                    if let Ok(mut entries2) = tokio::fs::read_dir(e1.path()).await {
                        while let Ok(Some(e2)) = entries2.next_entry().await {
                            let candidate = e2.path().join(&run_id).join(&target_file);
                            if tokio::fs::metadata(&candidate).await.is_ok() {
                                log_path = Some(candidate);
                                break;
                            }
                        }
                    }
                    if log_path.is_some() {
                        break;
                    }
                }
            }
        }

        let Some(path) = log_path else {
            return Err(format!("Log file not found for run_id {run_id}."));
        };

        if tokio::fs::metadata(&path).await.is_err() {
            return Err(format!("Log file {} does not exist.", path.display()));
        }

        use tokio::io::{AsyncBufReadExt, BufReader};
        let file = tokio::fs::File::open(&path)
            .await
            .map_err(|e| format!("Failed to open log file: {e}"))?;

        let mut reader = BufReader::new(file);

        let start = start_line.max(1) as usize;
        let end = end_line as usize;

        if end < start {
            return Err("end_line must be greater than or equal to start_line".to_string());
        }

        let actual_end = start + 99.min(end - start);

        let mut lines = Vec::new();
        let mut current_line = 1;
        let mut buf = String::new();

        loop {
            buf.clear();
            match reader.read_line(&mut buf).await {
                Ok(0) => break,
                Ok(_) => {
                    if current_line >= start && current_line <= actual_end {
                        lines.push(buf.trim_end_matches(&['\r', '\n'][..]).to_string());
                    }
                    current_line += 1;
                    if current_line > actual_end {
                        break;
                    }
                }
                Err(e) => return Err(format!("Read error: {e}")),
            }
        }

        if lines.is_empty() {
            return Ok(format!("No lines found in the specified range. The file might have fewer than {start} lines."));
        }

        let content = lines.join("\n");
        let res = format!(
            "--- Log preview: {stream} (Lines {start} to {}) ---\n{}",
            current_line - 1,
            content
        );
        Ok(res)
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
        "Best-effort interrupt for a background job by job_id. Prefers the cooperative provider cancel (same capability gate as execution_cancel); when the provider does not support cooperative interrupts, falls back to aborting the local wait."
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
        let outcome = self.jobs.cancel_job(job_id).await?;
        let mut payload = serde_json::json!({
            "job_id": job_id,
            "cancel_kind": outcome.cancel_kind,
            "message": format!("cancel requested for job {job_id}"),
        });
        if let Some(note) = outcome.note {
            if let Value::Object(ref mut m) = payload {
                m.insert("note".to_string(), Value::String(note));
            }
        }
        serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())
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
        let sid = require_session_id(&args)?;
        let provider = self.harness.provider_for_session(sid.as_str());
        if !provider.capabilities().supports_interrupt {
            return Err(
                "execution_cancel unsupported: provider capabilities.supports_interrupt is false"
                    .to_string(),
            );
        }
        provider.cancel(&sid).await.map_err(exec_err)?;
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
        let provider = self.harness.provider_for_session(sid.as_str());
        provider.close_session(&sid).await.map_err(exec_err)?;
        // Drop the session→provider mapping; future ops on this id fall back to default and
        // typically error from the provider (session not found), which is the expected UX.
        self.harness.unregister_session(sid.as_str());
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
    #[ignore = "Local provider rejects language=python; port to python_run or language=shell (Phase 0.0c follow-up)"]
    async fn execution_session_create_accepts_explicit_provider() {
        // With a single-provider harness, an explicit `provider: "local"` should resolve cleanly
        // and the response should echo the provider id back so the model can audit its choice.
        let (ws, dir) = temp_dirs();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
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
            0,
        ));
        let create = ExecutionSessionCreateTool {
            harness: harness.clone(),
        };
        let lang = if cfg!(windows) { "shell" } else { "python" };
        let out = create
            .execute(json!({ "language": lang, "provider": "local" }))
            .await
            .expect("create");
        let v: Value = serde_json::from_str(&out).expect("json");
        assert_eq!(v["provider"].as_str(), Some("local"));
        let sid = v["session_id"].as_str().expect("session id").to_string();
        // Mapping should be live so subsequent ops route to the same provider.
        assert_eq!(harness.provider_for_session(&sid).provider_id(), "local");

        let close = ExecutionSessionCloseTool { harness };
        close.execute(json!({ "session_id": sid })).await.unwrap();
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    async fn execution_session_create_rejects_unknown_provider() {
        let (ws, dir) = temp_dirs();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
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
            0,
        ));
        let create = ExecutionSessionCreateTool { harness };
        let err = create
            .execute(json!({ "provider": "nope" }))
            .await
            .expect_err("expected provider rejection");
        assert!(err.contains("available providers"), "err={err}");
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    #[ignore = "Local provider rejects language=python; port to python_run or language=shell (Phase 0.0c follow-up)"]
    async fn create_run_close_roundtrip() {
        let (ws, dir) = temp_dirs();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
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
            0,
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
            jobs: None,
            inflight: None,
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
    #[ignore = "Local provider rejects language=python; port to python_run or language=shell (Phase 0.0c follow-up)"]
    async fn background_job_poll_and_result() {
        let (ws, dir) = temp_dirs();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
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
            0,
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
        let jobs = Arc::new(ExecutionJobManager::new(harness.clone(), otx, None, false));
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
    #[ignore = "Local provider rejects language=python; port to python_run or language=shell (Phase 0.0c follow-up)"]
    async fn execution_run_auto_promotes_when_bound_smaller_than_runtime() {
        let (ws, dir) = temp_dirs();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
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
            // Tight 1s auto-promote bound so any 5s+ run is forced into the background.
            1,
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
        let (otx, _orx) = mpsc::channel::<BusMessage>(64);
        let jobs = Arc::new(ExecutionJobManager::new(
            harness.clone(),
            otx.clone(),
            None,
            false,
        ));
        let inflight = Arc::new(InflightSyncRegistry::new());
        let run = ExecutionRunTool {
            harness: harness.clone(),
            outbound_tx: otx,
            jobs: Some(jobs.clone()),
            inflight: Some(inflight.clone()),
        };
        let code = if cfg!(windows) {
            "ping 127.0.0.1 -n 6 >nul"
        } else {
            "import time\ntime.sleep(5)\nprint('done')"
        };
        let res = crate::tool_runtime::with_tool_exec_scope(
            crate::tool_runtime::ToolExecCtx::new("terminal", "auto-promote-chat", None),
            async {
                run.execute(json!({
                    "session_id": sid,
                    "code": code,
                    "timeout_secs": 60,
                    "description": "auto-promote-test",
                }))
                .await
            },
        )
        .await
        .expect("run");
        let envelope: Value = serde_json::from_str(&res).expect("envelope is JSON");
        assert_eq!(envelope["auto_promoted"].as_bool(), Some(true), "out={res}");
        assert_eq!(
            envelope["reason"].as_str(),
            Some("auto_promote_after_secs"),
            "out={res}"
        );
        let jid = envelope["job_id"]
            .as_str()
            .expect("job_id missing in envelope");
        // Drain the spawned job so the test does not leave background work running.
        for _ in 0..200 {
            let st = ExecutionJobStatusTool { jobs: jobs.clone() };
            let s = st.execute(json!({ "job_id": jid })).await.expect("st");
            if s.contains("\"terminal\": true") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let close = ExecutionSessionCloseTool { harness };
        close.execute(json!({ "session_id": sid })).await.unwrap();
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    async fn execution_env_info_includes_timeout_caps() {
        let (ws, dir) = temp_dirs();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
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
            0,
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
    #[ignore = "Local provider rejects language=python; port to python_run or language=shell (Phase 0.0c follow-up)"]
    async fn background_job_carries_description_in_status_and_list() {
        let (ws, dir) = temp_dirs();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
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
            0,
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
        let jobs = Arc::new(ExecutionJobManager::new(harness.clone(), otx, None, false));
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
    #[ignore = "Local provider rejects language=python; port to python_run or language=shell (Phase 0.0c follow-up)"]
    async fn background_job_cancel_requests_interrupt() {
        let (ws, dir) = temp_dirs();
        let cfg = crate::execution::LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
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
            0,
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
        let jobs = Arc::new(ExecutionJobManager::new(harness.clone(), otx, None, false));
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
