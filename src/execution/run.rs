//! Run requests and results (`RunSpec`, `RunResult`, session open/close payloads).

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::capabilities::SessionCapabilities;
use super::ids::SessionId;
use super::run_events::RunEvent;

/// Request to open a session (Phase 0 contract; executor validates in later phases).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SessionCreateRequest {
    /// Optional human/agent label for logs.
    pub label: Option<String>,
    /// Hint such as `python`, `rust`, `bash` (provider interprets).
    pub language: Option<String>,
    /// **Jupyter only:** attach to an existing kernel id (from a prior session or `session_capabilities`).
    /// When set, no new `POST /api/kernels` is performed; the kernel must still exist on the server.
    #[serde(default)]
    pub resume_jupyter_kernel_id: Option<String>,
}

/// Handle returned after a session is created.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionHandle {
    pub id: SessionId,
    pub capabilities: SessionCapabilities,
}

/// How the provider resolves the working directory for a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CwdPolicy {
    /// Provider default for this session (e.g. sandbox root or kernel cwd).
    #[default]
    SessionDefault,
    /// Path relative to the agent sandbox; resolved by the executor in later phases.
    SandboxRelative(String),
}

/// Single execution unit (code cell, script chunk, etc.).
#[derive(Debug, Clone)]
pub struct RunSpec {
    /// Source text to evaluate or execute.
    pub code: String,
    /// Wall-clock limit for this run (seconds). Executor may clamp to global max.
    pub timeout_secs: u64,
    pub cwd: CwdPolicy,
    /// Stable id for this run (artifacts, run journal). When `None`, Jupyter generates one internally
    /// (other providers may ignore).
    pub run_id: Option<String>,
    /// Live events (Jupyter streams). Not serialized.
    pub run_event_tx: Option<mpsc::Sender<RunEvent>>,
    /// Short human-facing summary for terminal UI and audits (not sent to the kernel).
    pub description: Option<String>,
}

impl RunSpec {
    pub fn new(code: impl Into<String>, timeout_secs: u64) -> Self {
        Self {
            code: code.into(),
            timeout_secs,
            cwd: CwdPolicy::default(),
            run_id: None,
            run_event_tx: None,
            description: None,
        }
    }
}

/// Reference to a side artifact (plot file, table export) produced by a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunAttachmentRef {
    pub id: String,
    /// Workspace- or session-relative path when materialized on disk.
    pub path: Option<String>,
    /// Optional MIME hint (`image/png`, `text/csv`, …).
    pub mime_hint: Option<String>,
}

/// Outcome of `ExecutionProvider::run` (stdout/stderr may be truncated by the executor).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunResult {
    pub stdout: String,
    pub stderr: String,
    /// Process exit code when applicable; `None` if not applicable or unknown.
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub attachments: Vec<RunAttachmentRef>,
}

impl RunResult {
    pub fn new(
        stdout: impl Into<String>,
        stderr: impl Into<String>,
        exit_code: Option<i32>,
    ) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: stderr.into(),
            exit_code,
            attachments: Vec::new(),
        }
    }
}
