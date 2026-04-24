//! Optional live events during `execution_run` (Jupyter iopub stream → bus / UI).

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Coarse-grained execution events for observability (terminal UI, telemetry).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunEvent {
    StdoutChunk { text: String },
    StderrChunk { text: String },
    KernelBusy,
    KernelIdle,
    DisplayDataSummary { mime: String },
    RunFinished,
}

/// Throttle high-frequency stream events (per run instance).
#[derive(Debug)]
pub struct RunEventThrottle {
    min_interval: Duration,
    last_emit: Mutex<Instant>,
}

impl RunEventThrottle {
    pub fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_emit: Mutex::new(Instant::now() - min_interval * 2),
        }
    }

    /// Returns true if caller should emit now (and updates last emit time).
    pub fn should_emit(&self) -> bool {
        let mut last = self.last_emit.lock().unwrap();
        let now = Instant::now();
        if now.duration_since(*last) >= self.min_interval {
            *last = now;
            true
        } else {
            false
        }
    }
}
