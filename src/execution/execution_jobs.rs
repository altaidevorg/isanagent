//! In-process background execution jobs (`execution_run_background`). Jobs end when the agent process exits.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use log::{info, warn};
use serde::Serialize;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, RwLock};
use tokio::task::{AbortHandle, JoinHandle};

use crate::bus::{BusMessage, InboundMessage, TelemetryEvent, METADATA_SYNTHETIC_JOB_FOLLOWUP};
use crate::channels::terminal::{
    build_execution_job_notice, build_execution_job_started_notice, build_execution_stream_notice,
};
use crate::execution::error::ExecutionError;
use crate::execution::{persist_successful_execution_run, PersistSuccessfulExecutionRunParams};
use crate::execution::{CwdPolicy, ExecutionHarness, RunEvent, RunResult, RunSpec, SessionId};

const JOB_RUNNING: u8 = 1;
const JOB_COMPLETED: u8 = 2;
const JOB_FAILED: u8 = 3;
const JOB_CANCELLED: u8 = 4;
const JOB_TIMEOUT: u8 = 5;

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn job_status_str(s: u8) -> &'static str {
    match s {
        JOB_RUNNING => "running",
        JOB_COMPLETED => "completed",
        JOB_FAILED => "failed",
        JOB_CANCELLED => "cancelled",
        JOB_TIMEOUT => "timeout",
        _ => "unknown",
    }
}

/// Human-facing exit code text (never `Debug` on `Option`).
fn format_exit_for_user(code: Option<i32>) -> String {
    match code {
        None => "none".to_string(),
        Some(0) => "0".to_string(),
        Some(n) => format!("exit {n}"),
    }
}

struct SessionBusyDrop {
    map: Arc<DashMap<String, ()>>,
    key: String,
}

impl Drop for SessionBusyDrop {
    fn drop(&mut self) {
        self.map.remove(&self.key);
    }
}

#[derive(Serialize)]
struct JobAuditLine<'a> {
    ts: &'a str,
    job_id: &'a str,
    session_id: &'a str,
    provider_id: &'a str,
    status: &'a str,
    duration_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
}

async fn append_job_audit_line(
    workspace_dir: &Path,
    line: &JobAuditLine<'_>,
) -> Result<(), String> {
    let dir = workspace_dir.join(".system_generated");
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("execution_jobs audit mkdir: {e}"))?;
    let path = dir.join("execution_jobs.jsonl");
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|e| format!("execution_jobs audit open: {e}"))?;
    let json = serde_json::to_string(line).map_err(|e| e.to_string())?;
    f.write_all(json.as_bytes())
        .await
        .map_err(|e| format!("execution_jobs audit write: {e}"))?;
    f.write_all(b"\n")
        .await
        .map_err(|e| format!("execution_jobs audit nl: {e}"))?;
    Ok(())
}

struct JobFinishedTelemetry<'a> {
    chat_id: &'a str,
    channel: &'a str,
    job_id: &'a str,
    session_id: &'a str,
    provider_id: &'a str,
    status: &'a str,
    duration_ms: u64,
    exit_code: Option<i32>,
    stdout_len: usize,
    stderr_len: usize,
    artifact_count: usize,
    description: Option<String>,
}

async fn send_job_finished(tx: &mpsc::Sender<BusMessage>, msg: JobFinishedTelemetry<'_>) {
    let _ = tx
        .send(BusMessage::Telemetry(
            TelemetryEvent::ExecutionJobFinished {
                chat_id: msg.chat_id.to_string(),
                channel: msg.channel.to_string(),
                job_id: msg.job_id.to_string(),
                session_id: msg.session_id.to_string(),
                provider_id: msg.provider_id.to_string(),
                status: msg.status.to_string(),
                duration_ms: msg.duration_ms,
                exit_code: msg.exit_code,
                stdout_len: msg.stdout_len,
                stderr_len: msg.stderr_len,
                artifact_count: msg.artifact_count,
                description: msg.description,
            },
        ))
        .await;
}

pub struct ExecutionJobRecord {
    pub job_id: String,
    pub session_id: SessionId,
    pub run_id: String,
    pub label: Option<String>,
    /// Human-facing summary for UI and audits (optional).
    pub description: Option<String>,
    /// Originating tool name (`execution_run`, `execution_run_background`, …).
    pub tool_name: String,
    pub status: AtomicU8,
    /// Wall-clock finish time for eviction ordering (`0` while not terminal).
    pub finished_unix_ms: AtomicU64,
    pub started_rfc3339: String,
    pub finished_rfc3339: RwLock<Option<String>>,
    pub error: RwLock<Option<String>>,
    pub result: RwLock<Option<RunResult>>,
    pub chat_id: String,
    pub channel: String,
    /// Abort handle for the spawned task; used by `cancel_job_force` for best-effort cancel
    /// (e.g. when the underlying provider does not support cooperative cancel).
    pub abort: Mutex<Option<AbortHandle>>,
    /// `Some("provider")` when cancelled via cooperative provider cancel; `Some("abort")` when
    /// best-effort `JoinHandle::abort` was used; `None` while running or for other terminal states.
    pub cancel_kind: RwLock<Option<&'static str>>,
}

impl ExecutionJobRecord {
    pub fn status_name(&self) -> &'static str {
        job_status_str(self.status.load(Ordering::Acquire))
    }

    pub fn is_terminal(&self) -> bool {
        self.status.load(Ordering::Acquire) >= JOB_COMPLETED
    }
}

struct ExecutionJobManagerInner {
    harness: Arc<ExecutionHarness>,
    outbound_tx: mpsc::Sender<BusMessage>,
    /// When set and `wake_on_job_terminal` is true, terminal jobs enqueue a synthetic [`BusMessage::Inbound`].
    inbound_bus_tx: Option<mpsc::Sender<BusMessage>>,
    wake_on_job_terminal: bool,
    jobs: DashMap<String, Arc<ExecutionJobRecord>>,
    session_busy: Arc<DashMap<String, ()>>,
    max_jobs: usize,
}

/// Enqueue a synthetic user message so the agent can call `execution_job_result` without waiting for the user.
async fn send_job_terminal_followup_inbound_if_configured(
    inner: &ExecutionJobManagerInner,
    chat_id: &str,
    channel: &str,
    job_id: &str,
    status_label: &str,
    session_id: &str,
    tool_name: &str,
) {
    if !inner.wake_on_job_terminal {
        return;
    }
    let Some(ref tx) = inner.inbound_bus_tx else {
        return;
    };
    let content = format!(
        "System: Background execution job `{job_id}` finished with status `{status_label}` (session `{session_id}`, tool `{tool_name}`). \
Call `execution_job_status` if you are unsure of the state, then `execution_job_result` (and `execution_artifact_list` when relevant). \
Summarize outcomes or errors for the user and update `todo_write` if you use it for this work."
    );
    let mut metadata = HashMap::new();
    metadata.insert(
        METADATA_SYNTHETIC_JOB_FOLLOWUP.to_string(),
        serde_json::Value::Bool(true),
    );
    metadata.insert(
        "execution_job_id".to_string(),
        serde_json::Value::String(job_id.to_string()),
    );
    metadata.insert(
        crate::bus::METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS.to_string(),
        serde_json::Value::Bool(true),
    );
    let inbound = InboundMessage {
        channel: channel.to_string(),
        sender_id: "execution_job".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content,
        attachments: vec![],
        metadata,
    };
    if let Err(e) = tx.send(BusMessage::Inbound(inbound)).await {
        warn!("execution job follow-up inbound: bus send failed: {e}");
    }
}

/// Arguments for [`ExecutionJobManager::spawn_run`].
#[derive(Debug)]
pub struct SpawnBackgroundRunRequest {
    pub sid: SessionId,
    pub code: String,
    pub timeout_secs: u64,
    pub cwd: CwdPolicy,
    pub label: Option<String>,
    pub run_description: Option<String>,
    pub chat_id: String,
    pub channel: String,
}

/// Process-local registry for background runs.
#[derive(Clone)]
pub struct ExecutionJobManager {
    inner: Arc<ExecutionJobManagerInner>,
}

impl ExecutionJobManager {
    pub fn new(
        harness: Arc<ExecutionHarness>,
        outbound_tx: mpsc::Sender<BusMessage>,
        inbound_bus_tx: Option<mpsc::Sender<BusMessage>>,
        wake_on_job_terminal: bool,
    ) -> Self {
        Self {
            inner: Arc::new(ExecutionJobManagerInner {
                harness,
                outbound_tx,
                inbound_bus_tx,
                wake_on_job_terminal,
                jobs: DashMap::new(),
                session_busy: Arc::new(DashMap::new()),
                max_jobs: 512,
            }),
        }
    }

    pub fn get(&self, job_id: &str) -> Option<Arc<ExecutionJobRecord>> {
        self.inner.jobs.get(job_id).map(|e| e.value().clone())
    }

    pub fn list_job_ids_for_session(&self, sid: &SessionId) -> Vec<String> {
        self.inner
            .jobs
            .iter()
            .filter(|e| e.value().session_id == *sid)
            .map(|e| e.key().clone())
            .collect()
    }

    /// Drop oldest **terminal** jobs until under `max_jobs` so new spawns can proceed.
    fn evict_terminal_jobs_if_at_cap(&self) {
        while self.inner.jobs.len() >= self.inner.max_jobs {
            let mut oldest: Option<(u64, String)> = None;
            for e in self.inner.jobs.iter() {
                let rec = e.value();
                if !rec.is_terminal() {
                    continue;
                }
                let ms = rec.finished_unix_ms.load(Ordering::Acquire);
                if ms == 0 {
                    continue;
                }
                oldest = match oldest {
                    None => Some((ms, e.key().clone())),
                    Some((cur, _)) if ms < cur => Some((ms, e.key().clone())),
                    Some(x) => Some(x),
                };
            }
            match oldest {
                Some((_, id)) => {
                    self.inner.jobs.remove(&id);
                }
                None => break,
            }
        }
    }

    /// Best-effort interrupt for the session backing this job (same as `execution_cancel` for that session).
    ///
    /// Prefers the cooperative provider-level cancel (`provider.cancel`) when supported.
    /// Falls back to `cancel_job_force` (best-effort `JoinHandle::abort`) when the provider
    /// does not support cooperative interrupts. On the abort path the local
    /// wait is dropped immediately, but remote work may continue running.
    pub async fn cancel_job(&self, job_id: &str) -> Result<CancelOutcome, String> {
        let rec = self
            .get(job_id)
            .ok_or_else(|| "Unknown job_id".to_string())?;
        if rec.is_terminal() {
            return Err("Job already finished".to_string());
        }
        // Cancel through the provider that owns this session.
        let session_provider = self
            .inner
            .harness
            .provider_for_session(rec.session_id.as_str());
        if session_provider.capabilities().supports_interrupt {
            session_provider
                .cancel(&rec.session_id)
                .await
                .map_err(|e| e.to_string())?;
            *rec.cancel_kind.write().await = Some("provider");
            return Ok(CancelOutcome {
                cancel_kind: "provider",
                note: None,
            });
        }
        // Provider does not support cooperative cancel; try best-effort task abort.
        self.cancel_job_force(job_id).await?;
        Ok(CancelOutcome {
            cancel_kind: "abort",
            note: Some(
                "Best-effort abort: the local wait was dropped, but the remote work may keep running until it finishes naturally."
                    .to_string(),
            ),
        })
    }

    pub async fn job_status_json(&self, job_id: &str) -> Result<serde_json::Value, String> {
        let r = self
            .get(job_id)
            .ok_or_else(|| "Unknown job_id".to_string())?;
        let finished = r.finished_rfc3339.read().await.clone();
        let err = r.error.read().await.clone();
        let cancel_kind = *r.cancel_kind.read().await;
        let mut payload = json!({
            "job_id": r.job_id,
            "session_id": r.session_id.to_string(),
            "run_id": r.run_id,
            "label": r.label,
            "description": r.description,
            "tool_name": r.tool_name,
            "status": r.status_name(),
            "started_rfc3339": r.started_rfc3339,
            "finished_rfc3339": finished,
            "error": err,
            "terminal": r.is_terminal(),
        });
        if let Some(kind) = cancel_kind {
            if let serde_json::Value::Object(ref mut m) = payload {
                m.insert("cancel_kind".to_string(), json!(kind));
                if kind == "abort" {
                    m.insert(
                        "cancel_note".to_string(),
                        json!("Best-effort abort: local wait was dropped; remote work may keep running until it finishes naturally."),
                    );
                }
            }
        }
        Ok(payload)
    }

    /// Pretty JSON for a finished job’s [`RunResult`], or a structured error payload; running jobs get a short message.
    pub async fn job_result_pretty(
        &self,
        job_id: &str,
        max_chars: usize,
    ) -> Result<String, String> {
        let r = self
            .get(job_id)
            .ok_or_else(|| "Unknown job_id".to_string())?;
        if !r.is_terminal() {
            return serde_json::to_string_pretty(
                &json!({
                    "status": r.status_name(),
                    "message": "Job still running; use execution_job_status or wait, then call execution_job_result again."
                }),
            )
            .map_err(|e| e.to_string());
        }
        let res = r.result.read().await.clone();
        match res {
            Some(result) => {
                let mut s = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
                crate::utils::truncate_utf8_safe(
                    &mut s,
                    max_chars,
                    "\n\n… (truncated to configured max_tool_output_chars)\n",
                );
                Ok(s)
            }
            None => {
                let err = r.error.read().await.clone().unwrap_or_default();
                serde_json::to_string_pretty(&json!({
                    "status": r.status_name(),
                    "run_result": serde_json::Value::Null,
                    "error": err,
                }))
                .map_err(|e| e.to_string())
            }
        }
    }

    pub async fn list_jobs_json(
        &self,
        session: Option<SessionId>,
        limit: usize,
    ) -> serde_json::Value {
        let limit = limit.clamp(1, 500);
        let mut rows: Vec<serde_json::Value> = Vec::new();
        for e in self.inner.jobs.iter() {
            if rows.len() >= limit {
                break;
            }
            let rec = e.value();
            if let Some(ref sid) = session {
                if rec.session_id != *sid {
                    continue;
                }
            }
            let finished = rec.finished_rfc3339.read().await.clone();
            let err = rec.error.read().await.clone();
            rows.push(json!({
                "job_id": rec.job_id,
                "session_id": rec.session_id.to_string(),
                "run_id": rec.run_id,
                "status": rec.status_name(),
                "label": rec.label,
                "description": rec.description,
                "started_rfc3339": rec.started_rfc3339,
                "finished_rfc3339": finished,
                "error": err,
            }));
        }
        json!({ "jobs": rows })
    }

    /// Returns `job_id` immediately; the run executes on a Tokio task. One active job (or synchronous run) per session.
    pub fn spawn_run(&self, req: SpawnBackgroundRunRequest) -> Result<String, String> {
        let SpawnBackgroundRunRequest {
            sid,
            code,
            timeout_secs,
            cwd,
            label,
            run_description,
            chat_id,
            channel,
        } = req;

        self.evict_terminal_jobs_if_at_cap();
        if self.inner.jobs.len() >= self.inner.max_jobs {
            return Err(format!(
                "Too many execution jobs in memory (max {}). Wait for jobs to finish or restart the agent.",
                self.inner.max_jobs
            ));
        }
        let sid_str = sid.to_string();
        match self.inner.session_busy.entry(sid_str.clone()) {
            Entry::Occupied(_) => Err(
                "This session already has an active execution run or background job; wait, poll execution_job_status, or use execution_cancel / execution_job_cancel."
                    .to_string(),
            ),
            Entry::Vacant(e) => {
                e.insert(());
                Ok(())
            }
        }?;

        let job_id = uuid::Uuid::new_v4().to_string();
        let run_id = uuid::Uuid::new_v4().to_string();
        let started_ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

        let rec = Arc::new(ExecutionJobRecord {
            job_id: job_id.clone(),
            session_id: sid.clone(),
            run_id: run_id.clone(),
            label: label.clone(),
            description: run_description.clone(),
            tool_name: "execution_run_background".to_string(),
            status: AtomicU8::new(JOB_RUNNING),
            started_rfc3339: started_ts.clone(),
            finished_unix_ms: AtomicU64::new(0),
            finished_rfc3339: RwLock::new(None),
            error: RwLock::new(None),
            result: RwLock::new(None),
            chat_id: chat_id.clone(),
            channel: channel.clone(),
            abort: Mutex::new(None),
            cancel_kind: RwLock::new(None),
        });
        self.inner.jobs.insert(job_id.clone(), rec.clone());

        // Notify the UI (multi-job strip) that a new job has started.
        send_job_started(
            &self.inner.outbound_tx,
            &chat_id,
            &channel,
            &job_id,
            &sid.to_string(),
            &rec.tool_name,
            run_description.as_deref(),
        );

        let inner = self.inner.clone();
        let sid_spawn = sid.clone();
        let job_id_for_task = job_id.clone();
        let stream_desc = run_description.clone();
        let rec_for_task = rec.clone();

        let join = tokio::spawn(async move {
            let rec = rec_for_task;
            let _busy = SessionBusyDrop {
                map: inner.session_busy.clone(),
                key: sid_str,
            };
            let session_provider = inner.harness.provider_for_session(sid_spawn.as_str());
            let prov = session_provider.provider_id().to_string();
            let started = Instant::now();
            let is_jupyter = prov == "jupyter";

            let (event_tx, event_rx) = if is_jupyter {
                let (tx, rx) = mpsc::channel::<RunEvent>(128);
                (Some(tx), Some(rx))
            } else {
                (None, None)
            };

            if let Some(mut rx) = event_rx {
                let ob = inner.outbound_tx.clone();
                let sid_s = sid_spawn.to_string();
                let rid = run_id.clone();
                let cid = chat_id.clone();
                let ch = channel.clone();
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
            spec.description = rec.description.clone();

            let run_out = session_provider.run(&sid_spawn, spec).await;
            let duration_ms = started.elapsed().as_millis() as u64;
            let ts_finish = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
            *rec.finished_rfc3339.write().await = Some(ts_finish.clone());
            let ws = inner.harness.workspace_dir().clone();

            match run_out {
                Ok(result) => {
                    rec.finished_unix_ms.store(now_unix_ms(), Ordering::Release);
                    rec.status.store(JOB_COMPLETED, Ordering::Release);
                    *rec.result.write().await = Some(result.clone());
                    persist_successful_execution_run(PersistSuccessfulExecutionRunParams {
                        harness: inner.harness.as_ref(),
                        outbound_tx: &inner.outbound_tx,
                        provider_id: &prov,
                        sid: &sid_spawn,
                        run_id: &run_id,
                        code: &code,
                        result: &result,
                        started_ts: &started_ts,
                        chat_id: &chat_id,
                        channel: &channel,
                        duration_ms,
                        job_id: Some(job_id_for_task.as_str()),
                        run_description: rec.description.as_deref(),
                    })
                    .await;
                    send_job_finished(
                        &inner.outbound_tx,
                        JobFinishedTelemetry {
                            chat_id: &chat_id,
                            channel: &channel,
                            job_id: &job_id_for_task,
                            session_id: &sid_spawn.to_string(),
                            provider_id: &prov,
                            status: "completed",
                            duration_ms,
                            exit_code: result.exit_code,
                            stdout_len: result.stdout.len(),
                            stderr_len: result.stderr.len(),
                            artifact_count: result.attachments.len(),
                            description: rec.description.clone(),
                        },
                    )
                    .await;
                    let exit_s = format_exit_for_user(result.exit_code);
                    let summary = match rec
                        .description
                        .as_deref()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                    {
                        Some(d) => {
                            format!("{} — completed (exit {}, {} ms)", d, exit_s, duration_ms)
                        }
                        None => format!(
                            "Execution job {} completed (exit {}, {} ms)",
                            job_id_for_task, exit_s, duration_ms
                        ),
                    };
                    let notice = build_execution_job_notice(
                        &chat_id,
                        &channel,
                        &job_id_for_task,
                        &sid_spawn.to_string(),
                        "completed",
                        &summary,
                        rec.description.as_deref(),
                        Some(rec.tool_name.as_str()),
                    );
                    let _ = inner.outbound_tx.send(BusMessage::Outbound(notice)).await;
                    send_job_terminal_followup_inbound_if_configured(
                        &inner,
                        &chat_id,
                        &channel,
                        &job_id_for_task,
                        "completed",
                        &sid_spawn.to_string(),
                        &rec.tool_name,
                    )
                    .await;
                    if let Err(e) = append_job_audit_line(
                        &ws,
                        &JobAuditLine {
                            ts: &ts_finish,
                            job_id: &job_id_for_task,
                            session_id: sid_spawn.as_str(),
                            provider_id: &prov,
                            status: "completed",
                            duration_ms,
                            error: None,
                            description: rec.description.as_deref(),
                        },
                    )
                    .await
                    {
                        warn!("execution_jobs audit: {e}");
                    }
                    info!(
                        "execution_job_done job={} session={} provider={} status=completed",
                        job_id_for_task, sid_spawn, prov
                    );
                }
                Err(e) => {
                    let (st_u8, status_label) = match &e {
                        ExecutionError::Timeout { .. } => (JOB_TIMEOUT, "timeout"),
                        ExecutionError::Cancelled => (JOB_CANCELLED, "cancelled"),
                        _ => (JOB_FAILED, "failed"),
                    };
                    rec.finished_unix_ms.store(now_unix_ms(), Ordering::Release);
                    rec.status.store(st_u8, Ordering::Release);
                    let es = e.to_string();
                    *rec.error.write().await = Some(es.clone());
                    send_job_finished(
                        &inner.outbound_tx,
                        JobFinishedTelemetry {
                            chat_id: &chat_id,
                            channel: &channel,
                            job_id: &job_id_for_task,
                            session_id: &sid_spawn.to_string(),
                            provider_id: &prov,
                            status: status_label,
                            duration_ms,
                            exit_code: None,
                            stdout_len: 0,
                            stderr_len: 0,
                            artifact_count: 0,
                            description: rec.description.clone(),
                        },
                    )
                    .await;
                    let summary = match rec
                        .description
                        .as_deref()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                    {
                        Some(d) => format!("{} — {status_label} ({es})", d),
                        None => format!("Execution job {job_id_for_task}: {status_label} ({es})"),
                    };
                    let notice = build_execution_job_notice(
                        &chat_id,
                        &channel,
                        &job_id_for_task,
                        &sid_spawn.to_string(),
                        status_label,
                        &summary,
                        rec.description.as_deref(),
                        Some(rec.tool_name.as_str()),
                    );
                    let _ = inner.outbound_tx.send(BusMessage::Outbound(notice)).await;
                    send_job_terminal_followup_inbound_if_configured(
                        &inner,
                        &chat_id,
                        &channel,
                        &job_id_for_task,
                        status_label,
                        &sid_spawn.to_string(),
                        &rec.tool_name,
                    )
                    .await;
                    if let Err(log_e) = append_job_audit_line(
                        &ws,
                        &JobAuditLine {
                            ts: &ts_finish,
                            job_id: &job_id_for_task,
                            session_id: sid_spawn.as_str(),
                            provider_id: &prov,
                            status: status_label,
                            duration_ms,
                            error: Some(es.as_str()),
                            description: rec.description.as_deref(),
                        },
                    )
                    .await
                    {
                        warn!("execution_jobs audit: {log_e}");
                    }
                    info!(
                        "execution_job_done job={} session={} provider={} status={}",
                        job_id_for_task, sid_spawn, prov, status_label
                    );
                }
            }
        });

        if let Ok(mut g) = rec.abort.lock() {
            *g = Some(join.abort_handle());
        }

        Ok(job_id)
    }

    /// Spawn an arbitrary future as a tracked background job.
    ///
    /// The future's `RunResult` (or `ExecutionError`) is recorded just like `spawn_run`'s.
    pub fn spawn_arbitrary(&self, req: SpawnArbitraryRequest) -> Result<String, String> {
        let SpawnArbitraryRequest {
            sid,
            tool_name,
            label,
            description,
            chat_id,
            channel,
            work,
        } = req;

        self.evict_terminal_jobs_if_at_cap();
        if self.inner.jobs.len() >= self.inner.max_jobs {
            return Err(format!(
                "Too many execution jobs in memory (max {}). Wait for jobs to finish or restart the agent.",
                self.inner.max_jobs
            ));
        }

        let job_id = uuid::Uuid::new_v4().to_string();
        let run_id = uuid::Uuid::new_v4().to_string();
        let started_ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

        let rec = Arc::new(ExecutionJobRecord {
            job_id: job_id.clone(),
            session_id: sid.clone(),
            run_id: run_id.clone(),
            label,
            description: description.clone(),
            tool_name: tool_name.clone(),
            status: AtomicU8::new(JOB_RUNNING),
            started_rfc3339: started_ts.clone(),
            finished_unix_ms: AtomicU64::new(0),
            finished_rfc3339: RwLock::new(None),
            error: RwLock::new(None),
            result: RwLock::new(None),
            chat_id: chat_id.clone(),
            channel: channel.clone(),
            abort: Mutex::new(None),
            cancel_kind: RwLock::new(None),
        });
        self.inner.jobs.insert(job_id.clone(), rec.clone());

        send_job_started(
            &self.inner.outbound_tx,
            &chat_id,
            &channel,
            &job_id,
            &sid.to_string(),
            &tool_name,
            description.as_deref(),
        );

        let inner = self.inner.clone();
        let sid_spawn = sid.clone();
        let job_id_for_task = job_id.clone();
        let rec_for_task = rec.clone();

        let join = tokio::spawn(async move {
            let rec = rec_for_task;
            let prov = inner
                .harness
                .provider_for_session(sid_spawn.as_str())
                .provider_id()
                .to_string();
            let started = Instant::now();
            let run_out = work.await;
            finalize_arbitrary_job(FinalizeArbitraryParams {
                inner: inner.clone(),
                rec: rec.clone(),
                sid: sid_spawn,
                job_id: job_id_for_task,
                chat_id,
                channel,
                provider_id: prov,
                started,
                tool_name,
                run_out,
            })
            .await;
        });

        if let Ok(mut g) = rec.abort.lock() {
            *g = Some(join.abort_handle());
        }

        Ok(job_id)
    }

    /// Adopt an already-running [`JoinHandle`] (auto-promote path) into the job manager.
    ///
    /// Used by sync tools (e.g. `execution_run`) that started their work
    /// inline, then crossed the auto-promote bound and need to hand the in-flight task to the
    /// job manager. The manager takes over completion bookkeeping (status, telemetry, journal).
    pub fn adopt_inflight(&self, req: AdoptInflightRequest) -> Result<String, String> {
        let AdoptInflightRequest {
            sid,
            tool_name,
            label,
            description,
            chat_id,
            channel,
            join,
        } = req;

        self.evict_terminal_jobs_if_at_cap();
        if self.inner.jobs.len() >= self.inner.max_jobs {
            return Err(format!(
                "Too many execution jobs in memory (max {}). Wait for jobs to finish or restart the agent.",
                self.inner.max_jobs
            ));
        }

        let job_id = uuid::Uuid::new_v4().to_string();
        let run_id = uuid::Uuid::new_v4().to_string();
        let started_ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

        let rec = Arc::new(ExecutionJobRecord {
            job_id: job_id.clone(),
            session_id: sid.clone(),
            run_id: run_id.clone(),
            label,
            description: description.clone(),
            tool_name: tool_name.clone(),
            status: AtomicU8::new(JOB_RUNNING),
            started_rfc3339: started_ts.clone(),
            finished_unix_ms: AtomicU64::new(0),
            finished_rfc3339: RwLock::new(None),
            error: RwLock::new(None),
            result: RwLock::new(None),
            chat_id: chat_id.clone(),
            channel: channel.clone(),
            abort: Mutex::new(Some(join.abort_handle())),
            cancel_kind: RwLock::new(None),
        });
        self.inner.jobs.insert(job_id.clone(), rec.clone());

        send_job_started(
            &self.inner.outbound_tx,
            &chat_id,
            &channel,
            &job_id,
            &sid.to_string(),
            &tool_name,
            description.as_deref(),
        );

        let inner = self.inner.clone();
        let sid_spawn = sid.clone();
        let job_id_for_task = job_id.clone();

        tokio::spawn(async move {
            let prov = inner
                .harness
                .provider_for_session(sid_spawn.as_str())
                .provider_id()
                .to_string();
            let started = Instant::now();
            let run_out = match join.await {
                Ok(v) => v,
                Err(e) if e.is_cancelled() => Err(ExecutionError::Cancelled),
                Err(e) => Err(ExecutionError::Provider(format!("join error: {e}"))),
            };
            finalize_arbitrary_job(FinalizeArbitraryParams {
                inner: inner.clone(),
                rec: rec.clone(),
                sid: sid_spawn,
                job_id: job_id_for_task,
                chat_id,
                channel,
                provider_id: prov,
                started,
                tool_name,
                run_out,
            })
            .await;
        });

        Ok(job_id)
    }

    /// Best-effort cancel by aborting the spawned tokio task. Used when the provider does not
    /// support cooperative cancel. The remote work may keep running on the other side until it finishes naturally.
    pub async fn cancel_job_force(&self, job_id: &str) -> Result<(), String> {
        let rec = self
            .get(job_id)
            .ok_or_else(|| "Unknown job_id".to_string())?;
        if rec.is_terminal() {
            return Err("Job already finished".to_string());
        }
        let handle = match rec.abort.lock() {
            Ok(mut g) => g.take(),
            Err(_) => None,
        };
        match handle {
            Some(h) => {
                h.abort();
                rec.status.store(JOB_CANCELLED, Ordering::Release);
                rec.finished_unix_ms.store(now_unix_ms(), Ordering::Release);
                let ts_finish =
                    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
                *rec.finished_rfc3339.write().await = Some(ts_finish);
                *rec.error.write().await = Some(
                    "cancelled via cancel_job_force (best-effort abort; remote work may continue)"
                        .to_string(),
                );
                *rec.cancel_kind.write().await = Some("abort");
                Ok(())
            }
            None => Err(
                "execution_job_cancel unsupported: no abort handle available for this job"
                    .to_string(),
            ),
        }
    }
}

/// Outcome of [`ExecutionJobManager::cancel_job`]; tells callers whether the cancel went through
/// the cooperative provider path or the best-effort `JoinHandle::abort` path.
#[derive(Debug, Clone)]
pub struct CancelOutcome {
    pub cancel_kind: &'static str,
    pub note: Option<String>,
}

fn send_job_started(
    tx: &mpsc::Sender<BusMessage>,
    chat_id: &str,
    channel: &str,
    job_id: &str,
    session_id: &str,
    tool_name: &str,
    description: Option<&str>,
) {
    let notice = build_execution_job_started_notice(
        chat_id,
        channel,
        job_id,
        session_id,
        tool_name,
        description,
    );
    let _ = tx.try_send(BusMessage::Outbound(notice));
}

/// Boxed work future for [`ExecutionJobManager::spawn_arbitrary`].
pub type ArbitraryWork =
    Pin<Box<dyn Future<Output = Result<RunResult, ExecutionError>> + Send + 'static>>;

/// Arguments for [`ExecutionJobManager::spawn_arbitrary`].
pub struct SpawnArbitraryRequest {
    pub sid: SessionId,
    pub tool_name: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub chat_id: String,
    pub channel: String,
    pub work: ArbitraryWork,
}

/// Arguments for [`ExecutionJobManager::adopt_inflight`].
pub struct AdoptInflightRequest {
    pub sid: SessionId,
    pub tool_name: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub chat_id: String,
    pub channel: String,
    pub join: JoinHandle<Result<RunResult, ExecutionError>>,
}

struct FinalizeArbitraryParams {
    inner: Arc<ExecutionJobManagerInner>,
    rec: Arc<ExecutionJobRecord>,
    sid: SessionId,
    job_id: String,
    chat_id: String,
    channel: String,
    provider_id: String,
    started: Instant,
    tool_name: String,
    run_out: Result<RunResult, ExecutionError>,
}

async fn finalize_arbitrary_job(p: FinalizeArbitraryParams) {
    let FinalizeArbitraryParams {
        inner,
        rec,
        sid,
        job_id,
        chat_id,
        channel,
        provider_id,
        started,
        tool_name,
        run_out,
    } = p;
    let duration_ms = started.elapsed().as_millis() as u64;
    let ts_finish = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    *rec.finished_rfc3339.write().await = Some(ts_finish.clone());
    let ws = inner.harness.workspace_dir().clone();

    match run_out {
        Ok(result) => {
            rec.finished_unix_ms.store(now_unix_ms(), Ordering::Release);
            rec.status.store(JOB_COMPLETED, Ordering::Release);
            *rec.result.write().await = Some(result.clone());
            send_job_finished(
                &inner.outbound_tx,
                JobFinishedTelemetry {
                    chat_id: &chat_id,
                    channel: &channel,
                    job_id: &job_id,
                    session_id: &sid.to_string(),
                    provider_id: &provider_id,
                    status: "completed",
                    duration_ms,
                    exit_code: result.exit_code,
                    stdout_len: result.stdout.len(),
                    stderr_len: result.stderr.len(),
                    artifact_count: result.attachments.len(),
                    description: rec.description.clone(),
                },
            )
            .await;
            let exit_s = format_exit_for_user(result.exit_code);
            let summary = match rec
                .description
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                Some(d) => format!("{} - completed (exit {}, {} ms)", d, exit_s, duration_ms),
                None => format!(
                    "{} job {} completed (exit {}, {} ms)",
                    tool_name, job_id, exit_s, duration_ms
                ),
            };
            let notice = build_execution_job_notice(
                &chat_id,
                &channel,
                &job_id,
                &sid.to_string(),
                "completed",
                &summary,
                rec.description.as_deref(),
                Some(rec.tool_name.as_str()),
            );
            let _ = inner.outbound_tx.send(BusMessage::Outbound(notice)).await;
            send_job_terminal_followup_inbound_if_configured(
                &inner,
                &chat_id,
                &channel,
                &job_id,
                "completed",
                &sid.to_string(),
                &rec.tool_name,
            )
            .await;
            if let Err(e) = append_job_audit_line(
                &ws,
                &JobAuditLine {
                    ts: &ts_finish,
                    job_id: &job_id,
                    session_id: sid.as_str(),
                    provider_id: &provider_id,
                    status: "completed",
                    duration_ms,
                    error: None,
                    description: rec.description.as_deref(),
                },
            )
            .await
            {
                warn!("execution_jobs audit: {e}");
            }
            info!(
                "execution_job_done job={} session={} provider={} status=completed tool={}",
                job_id, sid, provider_id, tool_name
            );
        }
        Err(e) => {
            let (st_u8, status_label) = match &e {
                ExecutionError::Timeout { .. } => (JOB_TIMEOUT, "timeout"),
                ExecutionError::Cancelled => (JOB_CANCELLED, "cancelled"),
                _ => (JOB_FAILED, "failed"),
            };
            rec.finished_unix_ms.store(now_unix_ms(), Ordering::Release);
            rec.status.store(st_u8, Ordering::Release);
            let es = e.to_string();
            *rec.error.write().await = Some(es.clone());
            send_job_finished(
                &inner.outbound_tx,
                JobFinishedTelemetry {
                    chat_id: &chat_id,
                    channel: &channel,
                    job_id: &job_id,
                    session_id: &sid.to_string(),
                    provider_id: &provider_id,
                    status: status_label,
                    duration_ms,
                    exit_code: None,
                    stdout_len: 0,
                    stderr_len: 0,
                    artifact_count: 0,
                    description: rec.description.clone(),
                },
            )
            .await;
            let summary = match rec
                .description
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                Some(d) => format!("{} - {status_label} ({es})", d),
                None => format!("{tool_name} job {job_id}: {status_label} ({es})"),
            };
            let notice = build_execution_job_notice(
                &chat_id,
                &channel,
                &job_id,
                &sid.to_string(),
                status_label,
                &summary,
                rec.description.as_deref(),
                Some(rec.tool_name.as_str()),
            );
            let _ = inner.outbound_tx.send(BusMessage::Outbound(notice)).await;
            send_job_terminal_followup_inbound_if_configured(
                &inner,
                &chat_id,
                &channel,
                &job_id,
                status_label,
                &sid.to_string(),
                &rec.tool_name,
            )
            .await;
            if let Err(log_e) = append_job_audit_line(
                &ws,
                &JobAuditLine {
                    ts: &ts_finish,
                    job_id: &job_id,
                    session_id: sid.as_str(),
                    provider_id: &provider_id,
                    status: status_label,
                    duration_ms,
                    error: Some(es.as_str()),
                    description: rec.description.as_deref(),
                },
            )
            .await
            {
                warn!("execution_jobs audit: {log_e}");
            }
            info!(
                "execution_job_done job={} session={} provider={} status={} tool={}",
                job_id, sid, provider_id, status_label, tool_name
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{ArtifactLimits, LocalExecutionConfig, LocalExecutionProvider};
    use std::time::Duration;

    fn temp_workspace() -> (std::path::PathBuf, std::path::PathBuf) {
        let root =
            std::env::temp_dir().join(format!("isanagent-jobs-test-{}", uuid::Uuid::new_v4()));
        let sandbox = root.join("sandbox");
        std::fs::create_dir_all(&sandbox).unwrap();
        (root, sandbox)
    }

    fn build_jobs() -> (
        Arc<ExecutionJobManager>,
        mpsc::Receiver<BusMessage>,
        std::path::PathBuf,
    ) {
        let (ws, dir) = temp_workspace();
        let cfg = LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
        let prov: Arc<dyn crate::execution::ExecutionProvider> =
            Arc::new(LocalExecutionProvider::new(cfg).expect("local provider"));
        let harness = Arc::new(ExecutionHarness::new(
            prov,
            "python",
            ws.clone(),
            dir,
            ArtifactLimits::default(),
            60,
            3600,
            0,
        ));
        let (otx, orx) = mpsc::channel::<BusMessage>(64);
        (
            Arc::new(ExecutionJobManager::new(harness, otx, None, false)),
            orx,
            ws,
        )
    }

    fn build_jobs_with_followup_wake(
        wake_on_job_terminal: bool,
    ) -> (
        Arc<ExecutionJobManager>,
        mpsc::Receiver<BusMessage>,
        mpsc::Receiver<BusMessage>,
    ) {
        let (ws, dir) = temp_workspace();
        let cfg = LocalExecutionConfig::new(dir.clone(), dir.clone(), true);
        let prov: Arc<dyn crate::execution::ExecutionProvider> =
            Arc::new(LocalExecutionProvider::new(cfg).expect("local provider"));
        let harness = Arc::new(ExecutionHarness::new(
            prov,
            "python",
            ws,
            dir,
            ArtifactLimits::default(),
            60,
            3600,
            0,
        ));
        let (outbound_tx, outbound_rx) = mpsc::channel::<BusMessage>(64);
        let (inbound_tx, inbound_rx) = mpsc::channel::<BusMessage>(64);
        let jobs = Arc::new(ExecutionJobManager::new(
            harness,
            outbound_tx,
            Some(inbound_tx),
            wake_on_job_terminal,
        ));
        (jobs, outbound_rx, inbound_rx)
    }

    fn fake_session() -> SessionId {
        SessionId::new("test-session")
    }

    #[tokio::test]
    async fn spawn_arbitrary_records_completion() {
        let (jobs, _orx, _ws) = build_jobs();
        let work: ArbitraryWork = Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(RunResult::new("hello", "", Some(0)))
        });
        let jid = jobs
            .spawn_arbitrary(SpawnArbitraryRequest {
                sid: fake_session(),
                tool_name: "execution_run".to_string(),
                label: None,
                description: Some("unit-arbitrary".to_string()),
                chat_id: "chat-test".to_string(),
                channel: "terminal".to_string(),
                work,
            })
            .expect("spawn_arbitrary ok");
        // Drain to terminal.
        for _ in 0..100 {
            let v = jobs.job_status_json(&jid).await.expect("status");
            if v["terminal"].as_bool() == Some(true) {
                assert_eq!(v["status"].as_str(), Some("completed"));
                assert_eq!(v["tool_name"].as_str(), Some("execution_run"));
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("spawn_arbitrary job never reached terminal state");
    }

    #[tokio::test]
    async fn cancel_job_force_aborts_long_running_task() {
        let (jobs, _orx, _ws) = build_jobs();
        let work: ArbitraryWork = Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(RunResult::new("", "", Some(0)))
        });
        let jid = jobs
            .spawn_arbitrary(SpawnArbitraryRequest {
                sid: fake_session(),
                tool_name: "execution_run".to_string(),
                label: None,
                description: None,
                chat_id: "chat-test".to_string(),
                channel: "terminal".to_string(),
                work,
            })
            .expect("spawn ok");
        // Give the task a tick to start.
        tokio::time::sleep(Duration::from_millis(40)).await;
        jobs.cancel_job_force(&jid).await.expect("cancel ok");
        // Give the abort + finalize a tick to settle.
        for _ in 0..100 {
            let v = jobs.job_status_json(&jid).await.expect("status");
            if v["terminal"].as_bool() == Some(true) {
                assert_eq!(v["status"].as_str(), Some("cancelled"));
                assert_eq!(v["cancel_kind"].as_str(), Some("abort"));
                assert!(
                    v.get("cancel_note").is_some(),
                    "expected cancel_note for abort path: {v}"
                );
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("cancel_job_force did not transition job to cancelled state");
    }

    #[tokio::test]
    async fn cancel_job_force_rejects_unknown_or_terminal() {
        let (jobs, _orx, _ws) = build_jobs();
        let err = jobs
            .cancel_job_force("no-such-job")
            .await
            .expect_err("must error");
        assert!(err.contains("Unknown"), "unexpected error: {err}");

        let work: ArbitraryWork = Box::pin(async move { Ok(RunResult::new("", "", Some(0))) });
        let jid = jobs
            .spawn_arbitrary(SpawnArbitraryRequest {
                sid: fake_session(),
                tool_name: "execution_run".to_string(),
                label: None,
                description: None,
                chat_id: "chat-test".to_string(),
                channel: "terminal".to_string(),
                work,
            })
            .expect("spawn ok");
        for _ in 0..100 {
            let v = jobs.job_status_json(&jid).await.expect("status");
            if v["terminal"].as_bool() == Some(true) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let err = jobs
            .cancel_job_force(&jid)
            .await
            .expect_err("terminal cancel must error");
        assert!(err.contains("already finished"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn terminal_job_emits_synthetic_followup_inbound_with_metadata() {
        let (jobs, _outbound_rx, mut inbound_rx) = build_jobs_with_followup_wake(true);
        let work: ArbitraryWork = Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(RunResult::new("done", "", Some(0)))
        });
        let job_id = jobs
            .spawn_arbitrary(SpawnArbitraryRequest {
                sid: fake_session(),
                tool_name: "execution_run".to_string(),
                label: None,
                description: Some("followup-test".to_string()),
                chat_id: "chat-followup".to_string(),
                channel: "terminal".to_string(),
                work,
            })
            .expect("spawn_arbitrary ok");

        let msg = tokio::time::timeout(Duration::from_secs(3), inbound_rx.recv())
            .await
            .expect("expected synthetic followup inbound")
            .expect("channel closed before followup");

        let inbound = match msg {
            BusMessage::Inbound(inbound) => inbound,
            other => panic!("expected inbound followup, got: {other:?}"),
        };
        assert_eq!(inbound.channel, "terminal");
        assert_eq!(inbound.chat_id, "chat-followup");
        assert_eq!(inbound.sender_id, "execution_job");
        assert_eq!(
            inbound
                .metadata
                .get(crate::bus::METADATA_SYNTHETIC_JOB_FOLLOWUP),
            Some(&serde_json::Value::Bool(true))
        );
        assert_eq!(
            inbound
                .metadata
                .get(crate::bus::METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS),
            Some(&serde_json::Value::Bool(true))
        );
        assert_eq!(
            inbound.metadata.get("execution_job_id"),
            Some(&serde_json::Value::String(job_id))
        );
        assert!(
            inbound.content.contains("execution_job_result"),
            "followup content should direct result retrieval: {}",
            inbound.content
        );
    }

    #[tokio::test]
    async fn wake_disabled_does_not_emit_synthetic_followup_inbound() {
        let (jobs, _outbound_rx, mut inbound_rx) = build_jobs_with_followup_wake(false);
        let work: ArbitraryWork = Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(RunResult::new("done", "", Some(0)))
        });
        let _job_id = jobs
            .spawn_arbitrary(SpawnArbitraryRequest {
                sid: fake_session(),
                tool_name: "execution_run".to_string(),
                label: None,
                description: Some("followup-disabled".to_string()),
                chat_id: "chat-no-followup".to_string(),
                channel: "terminal".to_string(),
                work,
            })
            .expect("spawn_arbitrary ok");

        let no_msg = tokio::time::timeout(Duration::from_millis(250), inbound_rx.recv()).await;
        assert!(
            no_msg.is_err(),
            "unexpected synthetic followup inbound when wake_on_job_terminal=false"
        );
    }
}
