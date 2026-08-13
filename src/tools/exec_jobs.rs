//! In-process background job registry for shell `exec` commands.
//!
//! Provides auto-promotion of long-running `exec` calls to background tasks,
//! incremental stdout/stderr buffering, interactive stdin injection (`exec_send`),
//! status inspection (`exec_status`), and synthetic [`BusMessage::Inbound`] agent wakeups upon completion.

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;

use crate::bus::{BusMessage, InboundMessage, METADATA_SYNTHETIC_JOB_FOLLOWUP};

pub const EXEC_JOB_RUNNING: u8 = 1;
pub const EXEC_JOB_COMPLETED: u8 = 2;
pub const EXEC_JOB_FAILED: u8 = 3;
pub const EXEC_JOB_CANCELLED: u8 = 4;

pub fn exec_status_str(s: u8) -> &'static str {
    match s {
        EXEC_JOB_RUNNING => "running",
        EXEC_JOB_COMPLETED => "completed",
        EXEC_JOB_FAILED => "failed",
        EXEC_JOB_CANCELLED => "cancelled",
        _ => "unknown",
    }
}

pub struct ExecJobRecord {
    pub command_id: String,
    pub command: String,
    pub cwd: String,
    pub status: AtomicU8,
    pub exit_code: RwLock<Option<i32>>,
    pub stdout_buf: RwLock<String>,
    pub stderr_buf: RwLock<String>,
    pub stdin_tx: Mutex<Option<tokio::process::ChildStdin>>,
    pub started_at: Instant,
    pub started_rfc3339: String,
    pub finished_rfc3339: RwLock<Option<String>>,
    pub chat_id: String,
    pub channel: String,
    pub join_handle: Mutex<Option<JoinHandle<()>>>,
}

impl ExecJobRecord {
    pub fn is_terminal(&self) -> bool {
        self.status.load(Ordering::Acquire) >= EXEC_JOB_COMPLETED
    }

    pub fn status_name(&self) -> &'static str {
        exec_status_str(self.status.load(Ordering::Acquire))
    }
}

#[derive(Clone)]
pub struct ExecJobRegistry {
    jobs: Arc<DashMap<String, Arc<ExecJobRecord>>>,
    bus_tx: Option<mpsc::Sender<BusMessage>>,
}

impl ExecJobRegistry {
    pub fn new(bus_tx: Option<mpsc::Sender<BusMessage>>) -> Self {
        Self {
            jobs: Arc::new(DashMap::new()),
            bus_tx,
        }
    }

    pub fn register_job(&self, record: Arc<ExecJobRecord>) {
        self.jobs.insert(record.command_id.clone(), record);
    }

    pub fn get_job(&self, command_id: &str) -> Option<Arc<ExecJobRecord>> {
        self.jobs.get(command_id).map(|r| r.value().clone())
    }

    pub fn list_jobs(&self) -> Vec<Arc<ExecJobRecord>> {
        self.jobs.iter().map(|r| r.value().clone()).collect()
    }

    pub async fn send_synthetic_completion_inbound(
        &self,
        chat_id: &str,
        channel: &str,
        command_id: &str,
        command: &str,
        status_name: &str,
        exit_code: Option<i32>,
    ) {
        let Some(ref tx) = self.bus_tx else {
            return;
        };
        let code_str = exit_code
            .map(|c| c.to_string())
            .unwrap_or_else(|| "none".to_string());
        let content = format!(
            "System: Background shell exec command `{command_id}` (`{command}`) finished with status `{status_name}` (exit code: {code_str}). Use `exec_status` to view final output lines if needed."
        );
        let mut metadata = HashMap::new();
        metadata.insert(
            METADATA_SYNTHETIC_JOB_FOLLOWUP.to_string(),
            serde_json::Value::Bool(true),
        );
        metadata.insert(
            "exec_command_id".to_string(),
            serde_json::Value::String(command_id.to_string()),
        );
        metadata.insert(
            crate::bus::METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS.to_string(),
            serde_json::Value::Bool(true),
        );
        let inbound = InboundMessage {
            channel: channel.to_string(),
            sender_id: "exec_job".to_string(),
            chat_id: chat_id.to_string(),
            thread_id: None,
            content,
            attachments: vec![],
            metadata,
        };
        let _ = tx.send(BusMessage::Inbound(inbound)).await;
    }
}
