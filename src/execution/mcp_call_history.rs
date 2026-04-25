//! Durable per-call journal under workspace `.system_generated/execution_history/colab_mcp_tool_call/`.
//!
//! Mirrors [`run_history::write_run_journal`] for raw Colab MCP `tools/call` invocations so they
//! land in the same `.system_generated/execution_history/...` tree the rest of the harness uses.
//! Both the synchronous and the auto-promoted background completion paths funnel through
//! [`write_mcp_call_journal`]; the synchronous path additionally appends a one-line summary to a
//! sibling `.system_generated/colab_mcp_calls.jsonl` manifest via [`append_mcp_call_manifest`].

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use super::ids::SessionId;

const MAX_INLINE_TEXT: usize = 64 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub struct McpCallJournal {
    pub schema_version: u32,
    pub provider_id: String,
    pub session_id: String,
    pub call_id: String,
    pub tool_name: String,
    pub started_rfc3339: String,
    pub finished_rfc3339: String,
    pub duration_ms: u64,
    /// `completed`, `failed`, `cancelled`, `timeout`.
    pub status: String,
    pub auto_promoted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub result_truncated: bool,
    pub result: String,
}

fn truncate_field(s: String, max: usize) -> (String, bool) {
    if s.len() <= max {
        (s, false)
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        let mut t = s[..end].to_string();
        t.push_str("\n... (truncated in result.json; full output in tool result)");
        (t, true)
    }
}

/// `workspace_dir/.system_generated/execution_history/colab_mcp_tool_call/{session_seg}/{call_id}/`
pub fn mcp_call_history_dir(
    workspace_dir: &Path,
    session_id: &SessionId,
    call_id: &str,
) -> PathBuf {
    let session_seg = super::artifacts::sanitize_session_segment(session_id);
    workspace_dir
        .join(".system_generated")
        .join("execution_history")
        .join("colab_mcp_tool_call")
        .join(&session_seg)
        .join(call_id)
}

/// Inputs for [`write_mcp_call_journal`].
pub struct McpCallJournalParams<'a> {
    pub workspace_dir: &'a Path,
    pub provider_id: &'a str,
    pub session_id: &'a SessionId,
    pub call_id: &'a str,
    pub tool_name: &'a str,
    pub arguments: &'a serde_json::Value,
    pub started_rfc3339: &'a str,
    pub finished_rfc3339: &'a str,
    pub duration_ms: u64,
    pub status: &'a str,
    pub auto_promoted: bool,
    pub job_id: Option<&'a str>,
    pub description: Option<&'a str>,
    /// Either the raw MCP `tools/call` result (JSON-stringified) or an error string.
    pub result: &'a str,
}

/// Writes `request.json`, `result.txt`, and `call.json`; returns the journal directory.
pub async fn write_mcp_call_journal(p: McpCallJournalParams<'_>) -> Result<PathBuf, String> {
    let dir = mcp_call_history_dir(p.workspace_dir, p.session_id, p.call_id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("mcp call journal mkdir {}: {e}", dir.display()))?;

    let request = serde_json::json!({
        "tool_name": p.tool_name,
        "arguments": redact_large_blobs(p.arguments),
        "description": p.description,
    });
    let request_pretty =
        serde_json::to_string_pretty(&request).map_err(|e| format!("request serialize: {e}"))?;
    tokio::fs::write(dir.join("request.json"), request_pretty.as_bytes())
        .await
        .map_err(|e| format!("mcp call request write: {e}"))?;

    tokio::fs::write(dir.join("result.txt"), p.result.as_bytes())
        .await
        .map_err(|e| format!("mcp call result write: {e}"))?;

    let (result_text, result_truncated) = truncate_field(p.result.to_string(), MAX_INLINE_TEXT);
    let journal = McpCallJournal {
        schema_version: 1,
        provider_id: p.provider_id.to_string(),
        session_id: p.session_id.to_string(),
        call_id: p.call_id.to_string(),
        tool_name: p.tool_name.to_string(),
        started_rfc3339: p.started_rfc3339.to_string(),
        finished_rfc3339: p.finished_rfc3339.to_string(),
        duration_ms: p.duration_ms,
        status: p.status.to_string(),
        auto_promoted: p.auto_promoted,
        job_id: p.job_id.map(str::to_string),
        description: p.description.map(str::to_string),
        result_truncated,
        result: result_text,
    };
    let json =
        serde_json::to_string_pretty(&journal).map_err(|e| format!("call.json serialize: {e}"))?;
    tokio::fs::write(dir.join("call.json"), json.as_bytes())
        .await
        .map_err(|e| format!("mcp call call.json write: {e}"))?;

    Ok(dir)
}

/// One-line summary for `.system_generated/colab_mcp_calls.jsonl`.
#[derive(Serialize)]
pub struct McpCallManifestLine<'a> {
    pub ts: &'a str,
    pub chat_id: &'a str,
    pub channel: &'a str,
    pub provider_id: &'a str,
    pub session_id: &'a str,
    pub call_id: &'a str,
    pub tool_name: &'a str,
    pub status: &'a str,
    pub duration_ms: u64,
    pub auto_promoted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<&'a str>,
    pub result_len: usize,
}

pub async fn append_mcp_call_manifest(
    workspace_dir: &Path,
    line: McpCallManifestLine<'_>,
) -> Result<(), String> {
    let dir = workspace_dir.join(".system_generated");
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("mcp call manifest mkdir: {e}"))?;
    let path = dir.join("colab_mcp_calls.jsonl");
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|e| format!("mcp call manifest open: {e}"))?;
    let json = serde_json::to_string(&line).map_err(|e| e.to_string())?;
    f.write_all(json.as_bytes())
        .await
        .map_err(|e| format!("mcp call manifest write: {e}"))?;
    f.write_all(b"\n")
        .await
        .map_err(|e| format!("mcp call manifest nl: {e}"))?;
    Ok(())
}

/// Replace large string fields inside a JSON value with `<redacted N bytes>` placeholders so the
/// journaled `request.json` does not balloon when callers pass huge code payloads or images.
fn redact_large_blobs(v: &serde_json::Value) -> serde_json::Value {
    const MAX_STRING: usize = 8 * 1024;
    match v {
        serde_json::Value::String(s) if s.len() > MAX_STRING => serde_json::Value::String(format!(
            "<redacted {} bytes; truncated for journal>\n{}",
            s.len(),
            &s[..MAX_STRING.min(s.len())]
        )),
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(redact_large_blobs).collect())
        }
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), redact_large_blobs(v)))
                .collect(),
        ),
        _ => v.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_large_blobs_truncates_long_strings() {
        let big = "x".repeat(20 * 1024);
        let v = serde_json::json!({ "code": big.clone(), "small": "ok" });
        let red = redact_large_blobs(&v);
        let red_code = red.get("code").and_then(|v| v.as_str()).unwrap();
        assert!(red_code.starts_with("<redacted 20480 bytes"));
        assert_eq!(red.get("small").and_then(|v| v.as_str()), Some("ok"));
    }
}
