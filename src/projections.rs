//! Authoritative session projection engine.
//!
//! Provides finished, structured state snapshots (run status, todos, subagents, jobs)
//! directly to clients (Altai App, ACP, Desktop) so frontends never need to scrape
//! raw tool arguments or maintain divergent mirror stores.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An authoritative projection snapshot representing active session state.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SessionProjection {
    pub chat_id: String,
    pub seq: u64,
    pub timestamp_rfc3339: String,
    pub run_status: String,
    #[serde(default)]
    pub todos: Vec<Value>,
    #[serde(default)]
    pub subagents: Vec<Value>,
    #[serde(default)]
    pub jobs: Vec<Value>,
}

impl SessionProjection {
    pub fn new(chat_id: impl Into<String>, seq: u64, run_status: impl Into<String>) -> Self {
        Self {
            chat_id: chat_id.into(),
            seq,
            timestamp_rfc3339: chrono::Utc::now().to_rfc3339(),
            run_status: run_status.into(),
            todos: Vec::new(),
            subagents: Vec::new(),
            jobs: Vec::new(),
        }
    }

    pub fn with_todos(mut self, todos: Vec<Value>) -> Self {
        self.todos = todos;
        self
    }

    pub fn with_subagents(mut self, subagents: Vec<Value>) -> Self {
        self.subagents = subagents;
        self
    }

    pub fn with_jobs(mut self, jobs: Vec<Value>) -> Self {
        self.jobs = jobs;
        self
    }
}

/// Thread-safe sequence generator for emitting monotonically increasing session projections.
#[derive(Debug, Default)]
pub struct ProjectionSequencer {
    counter: std::sync::atomic::AtomicU64,
}

impl ProjectionSequencer {
    /// Creates a new sequencer starting after `initial`.
    pub fn new(initial: u64) -> Self {
        Self {
            counter: std::sync::atomic::AtomicU64::new(initial),
        }
    }

    /// Increments and returns the next monotonic sequence number.
    pub fn next_seq(&self) -> u64 {
        self.counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_and_deserializes_projection() {
        let proj = SessionProjection::new("chat-123", 42, "running")
            .with_todos(vec![serde_json::json!({
                "id": "todo-1",
                "content": "Implement projections",
                "status": "in_progress"
            })])
            .with_jobs(vec![serde_json::json!({
                "command_id": "exec-abc",
                "status": "running"
            })]);

        let json = serde_json::to_string(&proj).expect("serialize");
        let parsed: SessionProjection = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.chat_id, "chat-123");
        assert_eq!(parsed.seq, 42);
        assert_eq!(parsed.run_status, "running");
        assert_eq!(parsed.todos.len(), 1);
        assert_eq!(parsed.jobs.len(), 1);
    }
}
