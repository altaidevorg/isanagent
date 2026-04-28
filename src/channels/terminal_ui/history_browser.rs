use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryKind {
    User,
    Assistant,
    ToolCall,
    ToolResult,
    Error,
    Other,
}

impl HistoryKind {
    pub fn label(self) -> &'static str {
        match self {
            HistoryKind::User => "User",
            HistoryKind::Assistant => "Assistant",
            HistoryKind::ToolCall => "ToolCall",
            HistoryKind::ToolResult => "ToolResult",
            HistoryKind::Error => "Error",
            HistoryKind::Other => "",
        }
    }
}

#[derive(Debug, Clone)]
pub struct HistoryListItem {
    pub ts: String,
    pub kind: HistoryKind,
    pub preview: String,
    pub raw: Value,
}

#[derive(Debug, Clone)]
pub struct HistoryDetail {
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ChatSessionItem {
    pub chat_id: String,
    pub last_ts: String,
    pub record_count: usize,
}

const MAX_LOG_BYTES: u64 = 24 * 1024 * 1024;
const PREVIEW_MAX: usize = 120;

pub fn load_recent_chat_sessions(
    workspace_dir: &Path,
    limit: usize,
) -> Result<(Vec<ChatSessionItem>, Option<String>), String> {
    let text = read_conversation_jsonl(workspace_dir)?;
    let mut bad_lines = 0usize;
    let mut map: HashMap<String, (String, usize, bool, usize)> = HashMap::new();
    for (line_idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                bad_lines += 1;
                continue;
            }
        };
        let Some(cid) = value_chat_id(&v) else {
            continue;
        };
        let ts = value_ts(&v).unwrap_or_else(|| "unknown-ts".to_string());
        let new_has_real_ts = ts != "unknown-ts";
        let e = map
            .entry(cid)
            .or_insert_with(|| (ts.clone(), 0usize, new_has_real_ts, line_idx));
        if new_has_real_ts {
            if !e.2 || ts > e.0 {
                e.0 = ts.clone();
            }
            e.2 = true;
            e.3 = line_idx;
        } else if !e.2 && line_idx >= e.3 {
            e.0 = ts.clone();
            e.3 = line_idx;
        }
        e.1 += 1;
    }
    let mut sortable = map
        .into_iter()
        .map(
            |(chat_id, (last_ts, record_count, has_real_ts, last_idx))| {
                (
                    ChatSessionItem {
                        chat_id,
                        last_ts,
                        record_count,
                    },
                    has_real_ts,
                    last_idx,
                )
            },
        )
        .collect::<Vec<_>>();
    sortable.sort_by(|a, b| match (a.1, b.1) {
        (true, true) => b.0.last_ts.cmp(&a.0.last_ts),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => b.2.cmp(&a.2),
    });
    let mut out = sortable
        .into_iter()
        .map(|(item, _, _)| item)
        .collect::<Vec<_>>();
    if limit > 0 && out.len() > limit {
        out.truncate(limit);
    }
    let warning = if bad_lines > 0 {
        Some(format!("Skipped {bad_lines} malformed log line(s)."))
    } else {
        None
    };
    Ok((out, warning))
}

pub fn load_history_for_chat(
    workspace_dir: &Path,
    chat_id: &str,
) -> Result<(Vec<HistoryListItem>, Option<String>), String> {
    let text = read_conversation_jsonl(workspace_dir)?;
    let mut out = Vec::new();
    let mut bad_lines = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                bad_lines += 1;
                continue;
            }
        };
        if value_chat_id(&v).as_deref() != Some(chat_id) {
            continue;
        }
        let ts = value_ts(&v).unwrap_or_else(|| "unknown-ts".to_string());
        let kind = infer_kind(&v);
        let preview = infer_preview(&v, kind);
        out.push(HistoryListItem {
            ts,
            kind,
            preview,
            raw: v,
        });
    }
    out.reverse();
    let warning = if bad_lines > 0 {
        Some(format!("Skipped {bad_lines} malformed log line(s)."))
    } else {
        None
    };
    Ok((out, warning))
}

fn read_conversation_jsonl(workspace_dir: &Path) -> Result<String, String> {
    let path = workspace_dir
        .join(".system_generated")
        .join("logs")
        .join("conversation.jsonl");
    if !path.exists() {
        return Ok(String::new());
    }
    let meta = std::fs::metadata(&path).map_err(|e| format!("conversation.jsonl stat: {e}"))?;
    if meta.len() > MAX_LOG_BYTES {
        return Err(format!(
            "conversation.jsonl too large ({} MiB); max {} MiB",
            meta.len() / (1024 * 1024),
            MAX_LOG_BYTES / (1024 * 1024)
        ));
    }
    std::fs::read_to_string(&path).map_err(|e| format!("conversation.jsonl read: {e}"))
}

pub fn load_history_detail(item: &HistoryListItem) -> HistoryDetail {
    let content = serde_json::to_string_pretty(&item.raw)
        .unwrap_or_else(|_| "<failed to pretty-print record>".to_string());
    HistoryDetail { content }
}

fn value_chat_id(v: &Value) -> Option<String> {
    v.get("chat_id")
        .and_then(|x| x.as_str())
        .map(str::to_string)
        .or_else(|| {
            v.as_object().and_then(|obj| {
                if obj.len() == 1 {
                    let inner = obj.values().next()?;
                    inner
                        .get("chat_id")
                        .and_then(|x| x.as_str())
                        .map(str::to_string)
                } else {
                    None
                }
            })
        })
        .or_else(|| {
            v.get("inbound")
                .and_then(|x| x.get("chat_id"))
                .and_then(|x| x.as_str())
                .map(str::to_string)
        })
        .or_else(|| {
            v.get("outbound")
                .and_then(|x| x.get("chat_id"))
                .and_then(|x| x.as_str())
                .map(str::to_string)
        })
}

fn value_ts(v: &Value) -> Option<String> {
    v.get("ts")
        .and_then(|x| x.as_str())
        .map(str::to_string)
        .or_else(|| {
            v.as_object().and_then(|obj| {
                if obj.len() == 1 {
                    let inner = obj.values().next()?;
                    inner.get("ts").and_then(|x| x.as_str()).map(str::to_string)
                } else {
                    None
                }
            })
        })
        .or_else(|| {
            v.get("timestamp")
                .and_then(|x| x.as_str())
                .map(str::to_string)
        })
        .or_else(|| v.get("time").and_then(|x| x.as_str()).map(str::to_string))
}

fn infer_kind(v: &Value) -> HistoryKind {
    if let Some(obj) = v.as_object() {
        if obj.len() == 1 {
            if let Some((k, _inner)) = obj.iter().next() {
                return match k.as_str() {
                    "ToolCall" | "ToolCallStarted" => HistoryKind::ToolCall,
                    "ToolResult" | "ToolCallFinished" => HistoryKind::ToolResult,
                    "AgentThought"
                    | "AgentUsage"
                    | "ExecutionRunFinished"
                    | "ExecutionJobFinished" => HistoryKind::Other,
                    _ => HistoryKind::Other,
                };
            }
        }
    }

    if v.get("sender_id").and_then(|x| x.as_str()).is_some() {
        return HistoryKind::User;
    }
    if v.get("channel").is_some() && v.get("chat_id").is_some() && v.get("content").is_some() {
        return HistoryKind::Assistant;
    }

    let t = v
        .get("type")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if t.contains("error") || t.contains("fail") {
        return HistoryKind::Error;
    }
    if t.contains("tool_call") || t.contains("toolcall") || t == "call" {
        return HistoryKind::ToolCall;
    }
    if t.contains("tool_result") || t.contains("toolresult") || t == "result" {
        return HistoryKind::ToolResult;
    }
    if t.contains("inbound") || t.contains("user") {
        return HistoryKind::User;
    }
    if t.contains("outbound") || t.contains("assistant") {
        return HistoryKind::Assistant;
    }
    if let Some(k) = v
        .get("role")
        .and_then(|x| x.as_str())
        .map(|s| s.to_ascii_lowercase())
    {
        if k == "user" {
            return HistoryKind::User;
        }
        if k == "assistant" {
            return HistoryKind::Assistant;
        }
        if k == "tool" {
            return HistoryKind::ToolResult;
        }
    }
    if v.get("inbound").is_some() {
        return HistoryKind::User;
    }
    if v.get("outbound").is_some() {
        return HistoryKind::Assistant;
    }
    HistoryKind::Other
}

fn infer_preview(v: &Value, kind: HistoryKind) -> String {
    let txt = v
        .get("content")
        .and_then(|x| x.as_str())
        .or_else(|| {
            v.get("inbound")
                .and_then(|x| x.get("content"))
                .and_then(|x| x.as_str())
        })
        .or_else(|| {
            v.get("outbound")
                .and_then(|x| x.get("content"))
                .and_then(|x| x.as_str())
        })
        .or_else(|| {
            v.get("message")
                .and_then(|x| x.as_str())
                .or_else(|| v.get("text").and_then(|x| x.as_str()))
        })
        .unwrap_or("");
    if txt.is_empty() {
        return kind.label().to_string();
    }
    let normalized = txt.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_display(&normalized, PREVIEW_MAX)
}

fn truncate_display(s: &str, max: usize) -> String {
    if max == 0 || s.is_empty() {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out = String::new();
    for ch in s.chars().take(max.saturating_sub(1)) {
        out.push(ch);
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_filters_and_skips_bad_lines() {
        let td =
            std::env::temp_dir().join(format!("isanagent-history-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&td);
        let logs_dir = td.join(".system_generated").join("logs");
        std::fs::create_dir_all(&logs_dir).unwrap();
        let p = logs_dir.join("conversation.jsonl");
        let data = r#"{"ts":"2026-01-01T00:00:00Z","chat_id":"c1","type":"inbound","content":"hello"}
not-json
        {"ts":"2026-01-01T00:00:01Z","chat_id":"c2","type":"outbound","content":"x"}
{"ts":"2026-01-01T00:00:02Z","chat_id":"c1","type":"tool_result","content":"done"}"#;
        std::fs::write(&p, data).unwrap();
        let (items, warn) = load_history_for_chat(&td, "c1").unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].kind, HistoryKind::ToolResult);
        assert_eq!(items[1].kind, HistoryKind::User);
        assert!(warn.unwrap_or_default().contains("Skipped 1"));
        let _ = std::fs::remove_dir_all(&td);
    }

    #[test]
    fn aggregates_recent_chat_sessions() {
        let td =
            std::env::temp_dir().join(format!("isanagent-sessions-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&td);
        let logs_dir = td.join(".system_generated").join("logs");
        std::fs::create_dir_all(&logs_dir).unwrap();
        let p = logs_dir.join("conversation.jsonl");
        let data = r#"{"ts":"2026-01-01T00:00:00Z","chat_id":"a","type":"inbound","content":"hello"}
{"ts":"2026-01-01T00:00:02Z","chat_id":"b","type":"outbound","content":"x"}
{"ts":"2026-01-01T00:00:03Z","chat_id":"a","type":"tool_result","content":"done"}"#;
        std::fs::write(&p, data).unwrap();
        let (items, warn) = load_recent_chat_sessions(&td, 10).unwrap();
        assert!(warn.is_none());
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].chat_id, "a");
        assert_eq!(items[0].record_count, 2);
        assert_eq!(items[1].chat_id, "b");
        let _ = std::fs::remove_dir_all(&td);
    }
}
