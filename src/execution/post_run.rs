//! Shared persistence after a successful `ExecutionProvider::run` (journal, manifest, telemetry).

use std::path::Path;

use log::warn;
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

use crate::bus::{BusMessage, TelemetryEvent};

use super::harness::ExecutionHarness;
use super::ids::SessionId;
use super::run::RunResult;
use super::run_history::{write_run_journal, RunJournalParams};

#[derive(Serialize)]
struct ExecutionManifestLine<'a> {
    ts: &'a str,
    chat_id: &'a str,
    channel: &'a str,
    provider_id: &'a str,
    session_id: &'a str,
    run_id: &'a str,
    exit_code: Option<i32>,
    duration_ms: u64,
    stdout_len: usize,
    stderr_len: usize,
    artifact_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    git_head: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    job_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
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

/// Arguments for [`persist_successful_execution_run`].
pub struct PersistSuccessfulExecutionRunParams<'a> {
    pub harness: &'a ExecutionHarness,
    pub outbound_tx: &'a mpsc::Sender<BusMessage>,
    pub provider_id: &'a str,
    pub sid: &'a SessionId,
    pub run_id: &'a str,
    pub code: &'a str,
    pub result: &'a RunResult,
    pub started_ts: &'a str,
    pub chat_id: &'a str,
    pub channel: &'a str,
    pub duration_ms: u64,
    pub job_id: Option<&'a str>,
    pub run_description: Option<&'a str>,
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

/// Run journal, `execution_runs.jsonl` line, and `ExecutionRunFinished` telemetry.
pub async fn persist_successful_execution_run(p: PersistSuccessfulExecutionRunParams<'_>) {
    let PersistSuccessfulExecutionRunParams {
        harness,
        outbound_tx,
        provider_id,
        sid,
        run_id,
        code,
        result,
        started_ts,
        chat_id,
        channel,
        duration_ms,
        job_id,
        run_description,
    } = p;

    let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let git_head = best_effort_git_head(harness.workspace_dir());

    let journal_ext = harness
        .provider_for_session(sid.as_str())
        .session_journal_extensions(sid);
    let jupyter_kernel_id = journal_ext
        .as_ref()
        .and_then(|m| m.get("jupyter_kernel_id"))
        .and_then(|v| v.as_str());
    let jupyter_notebook_path = journal_ext
        .as_ref()
        .and_then(|m| m.get("jupyter_notebook_sync_path"))
        .and_then(|v| v.as_str());

    if let Err(e) = write_run_journal(RunJournalParams {
        workspace_dir: harness.workspace_dir(),
        provider_id,
        session_id: sid,
        run_id,
        code,
        result,
        jupyter_kernel_id,
        jupyter_notebook_path,
        started_rfc3339: started_ts,
        finished_rfc3339: &ts,
        duration_ms,
    })
    .await
    {
        warn!("execution run journal: {e}");
    }

    let manifest = ExecutionManifestLine {
        ts: &ts,
        chat_id,
        channel,
        provider_id,
        session_id: sid.as_str(),
        run_id,
        exit_code: result.exit_code,
        duration_ms,
        stdout_len: result.stdout.len(),
        stderr_len: result.stderr.len(),
        artifact_count: result.attachments.len(),
        git_head: git_head.as_deref(),
        job_id,
        description: run_description,
    };
    if let Err(e) = append_execution_manifest(harness.workspace_dir(), manifest).await {
        warn!("execution manifest append failed: {e}");
    }

    let _ = outbound_tx
        .send(BusMessage::Telemetry(
            TelemetryEvent::ExecutionRunFinished {
                chat_id: chat_id.to_string(),
                channel: channel.to_string(),
                provider_id: provider_id.to_string(),
                session_id: sid.to_string(),
                exit_code: result.exit_code,
                duration_ms,
                stdout_len: result.stdout.len(),
                stderr_len: result.stderr.len(),
                artifact_count: result.attachments.len(),
                git_head: git_head.clone(),
                description: run_description.map(|s| s.to_string()),
            },
        ))
        .await;
}
