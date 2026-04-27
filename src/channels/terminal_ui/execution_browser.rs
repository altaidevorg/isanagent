//! Load execution run index from `execution_runs.jsonl` and per-run journals for the terminal UI.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::execution::{run_history_dir, RunJournal, SessionId};

/// One line from `workspace/.system_generated/execution_runs.jsonl` (newer builds include `run_id`).
#[derive(Debug, Clone, Deserialize)]
pub struct ExecutionRunListItem {
    pub ts: String,
    pub chat_id: String,
    #[serde(default)]
    pub channel: String,
    pub provider_id: String,
    pub session_id: String,
    #[serde(default)]
    pub run_id: Option<String>,
    pub exit_code: Option<i32>,
    pub duration_ms: u64,
    #[serde(default)]
    pub stdout_len: usize,
    #[serde(default)]
    pub stderr_len: usize,
    #[serde(default)]
    pub description: Option<String>,
}

/// Loaded `source.txt` + `run.json` for the executions detail pane.
#[derive(Debug, Clone)]
pub struct ExecutionRunDetail {
    pub source: String,
    pub journal: RunJournal,
}

const MAX_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;

/// Read and parse `execution_runs.jsonl`, keep lines whose `chat_id` matches, newest first.
pub fn load_runs_for_chat(
    workspace_dir: &Path,
    chat_id: &str,
) -> Result<Vec<ExecutionRunListItem>, String> {
    let path = workspace_dir
        .join(".system_generated")
        .join("execution_runs.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let meta = std::fs::metadata(&path).map_err(|e| format!("execution_runs stat: {e}"))?;
    if meta.len() > MAX_MANIFEST_BYTES {
        return Err(format!(
            "execution_runs.jsonl too large ({} MiB); max {} MiB",
            meta.len() / (1024 * 1024),
            MAX_MANIFEST_BYTES / (1024 * 1024)
        ));
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("execution_runs read: {e}"))?;
    let mut out: Vec<ExecutionRunListItem> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let item: ExecutionRunListItem = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if item.chat_id == chat_id {
            out.push(item);
        }
    }
    out.reverse();
    Ok(out)
}

/// Load `source.txt` and `run.json` under the run journal directory.
pub fn load_run_detail(
    workspace_dir: &Path,
    item: &ExecutionRunListItem,
) -> Result<ExecutionRunDetail, String> {
    let run_id = item
        .run_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "This run has no run_id in the manifest (older isanagent); journal path unknown."
                .to_string()
        })?;
    let sid = SessionId::new(item.session_id.clone());
    let dir: PathBuf = run_history_dir(workspace_dir, &item.provider_id, &sid, run_id);
    let source_path = dir.join("source.txt");
    let json_path = dir.join("run.json");
    if !source_path.is_file() || !json_path.is_file() {
        return Err(format!("Run journal missing under {}", dir.display()));
    }
    let source = std::fs::read_to_string(&source_path).map_err(|e| format!("source.txt: {e}"))?;
    let json = std::fs::read_to_string(&json_path).map_err(|e| format!("run.json: {e}"))?;
    let journal: RunJournal =
        serde_json::from_str(&json).map_err(|e| format!("run.json parse: {e}"))?;
    Ok(ExecutionRunDetail { source, journal })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_without_run_id_deserializes() {
        let j = r#"{"ts":"2020-01-01T00:00:00Z","chat_id":"c1","channel":"terminal","provider_id":"local","session_id":"s1","exit_code":0,"duration_ms":1,"stdout_len":0,"stderr_len":0}"#;
        let row: ExecutionRunListItem = serde_json::from_str(j).unwrap();
        assert!(row.run_id.is_none());
    }

    #[test]
    fn manifest_with_run_id_deserializes() {
        let j = r#"{"ts":"t","chat_id":"c","channel":"terminal","provider_id":"jupyter","session_id":"sess","run_id":"r1","exit_code":0,"duration_ms":2,"stdout_len":1,"stderr_len":0}"#;
        let row: ExecutionRunListItem = serde_json::from_str(j).unwrap();
        assert_eq!(row.run_id.as_deref(), Some("r1"));
    }
}
