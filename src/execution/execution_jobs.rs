//! In-process background execution jobs (`execution_run_background`). Jobs end when the agent process exits.

use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use log::{info, warn};
use serde::Serialize;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, RwLock};

use crate::bus::{BusMessage, TelemetryEvent};
use crate::channels::terminal::{build_execution_job_notice, build_execution_stream_notice};
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
    pub status: AtomicU8,
    /// Wall-clock finish time for eviction ordering (`0` while not terminal).
    pub finished_unix_ms: AtomicU64,
    pub started_rfc3339: String,
    pub finished_rfc3339: RwLock<Option<String>>,
    pub error: RwLock<Option<String>>,
    pub result: RwLock<Option<RunResult>>,
    pub chat_id: String,
    pub channel: String,
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
    jobs: DashMap<String, Arc<ExecutionJobRecord>>,
    session_busy: Arc<DashMap<String, ()>>,
    max_jobs: usize,
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
pub struct ExecutionJobManager {
    inner: Arc<ExecutionJobManagerInner>,
}

impl ExecutionJobManager {
    pub fn new(harness: Arc<ExecutionHarness>, outbound_tx: mpsc::Sender<BusMessage>) -> Self {
        Self {
            inner: Arc::new(ExecutionJobManagerInner {
                harness,
                outbound_tx,
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
    pub async fn cancel_job(&self, job_id: &str) -> Result<(), String> {
        let rec = self
            .get(job_id)
            .ok_or_else(|| "Unknown job_id".to_string())?;
        if rec.is_terminal() {
            return Err("Job already finished".to_string());
        }
        if !self
            .inner
            .harness
            .provider()
            .capabilities()
            .supports_interrupt
        {
            return Err(
                "execution_job_cancel unsupported: provider capabilities.supports_interrupt is false"
                    .to_string(),
            );
        }
        self.inner
            .harness
            .provider()
            .cancel(&rec.session_id)
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn job_status_json(&self, job_id: &str) -> Result<serde_json::Value, String> {
        let r = self
            .get(job_id)
            .ok_or_else(|| "Unknown job_id".to_string())?;
        let finished = r.finished_rfc3339.read().await.clone();
        let err = r.error.read().await.clone();
        Ok(json!({
            "job_id": r.job_id,
            "session_id": r.session_id.to_string(),
            "run_id": r.run_id,
            "label": r.label,
            "description": r.description,
            "status": r.status_name(),
            "started_rfc3339": r.started_rfc3339,
            "finished_rfc3339": finished,
            "error": err,
            "terminal": r.is_terminal(),
        }))
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
                if s.len() > max_chars {
                    s.truncate(max_chars);
                    s.push_str("\n\n… (truncated to configured max_tool_output_chars)\n");
                }
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
            status: AtomicU8::new(JOB_RUNNING),
            started_rfc3339: started_ts.clone(),
            finished_unix_ms: AtomicU64::new(0),
            finished_rfc3339: RwLock::new(None),
            error: RwLock::new(None),
            result: RwLock::new(None),
            chat_id: chat_id.clone(),
            channel: channel.clone(),
        });
        self.inner.jobs.insert(job_id.clone(), rec.clone());

        let inner = self.inner.clone();
        let sid_spawn = sid.clone();
        let job_id_for_task = job_id.clone();
        let stream_desc = run_description.clone();

        tokio::spawn(async move {
            let _busy = SessionBusyDrop {
                map: inner.session_busy.clone(),
                key: sid_str,
            };
            let prov = inner.harness.provider().provider_id().to_string();
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

            let run_out = inner.harness.provider().run(&sid_spawn, spec).await;
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
                    );
                    let _ = inner.outbound_tx.send(BusMessage::Outbound(notice)).await;
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
                    );
                    let _ = inner.outbound_tx.send(BusMessage::Outbound(notice)).await;
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

        Ok(job_id)
    }
}
