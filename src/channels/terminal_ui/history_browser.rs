use std::collections::HashMap;
use std::path::Path;

use rusqlite::{params, Connection};

#[derive(Debug, Clone)]
pub struct HistorySessionListItem {
    pub chat_id: String,
    pub last_ts: Option<String>,
    pub message_count: usize,
    last_id: i64,
}

#[derive(Debug, Clone)]
pub struct HistoryMessage {
    pub role: String,
    pub content: String,
    pub ts: Option<String>,
}

fn db_path(workspace_dir: &Path) -> std::path::PathBuf {
    workspace_dir
        .join(".system_generated")
        .join("agent_memory.db")
}

fn chat_id_from_thread_id(thread_id: &str) -> Option<String> {
    let mut parts = thread_id.splitn(3, ':');
    let channel = parts.next()?;
    if channel != "terminal" {
        return None;
    }
    let chat_id = parts.next()?.trim();
    if chat_id.is_empty() {
        None
    } else {
        Some(chat_id.to_string())
    }
}

fn summarize_content(raw: Option<String>) -> String {
    let Some(raw) = raw else {
        return String::new();
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.starts_with('[') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(arr) = v.as_array() {
                let mut out = Vec::new();
                for p in arr {
                    if let Some(t) = p.get("type").and_then(|x| x.as_str()) {
                        match t {
                            "text" => {
                                if let Some(s) = p.get("text").and_then(|x| x.as_str()) {
                                    let s = s.trim();
                                    if !s.is_empty() {
                                        out.push(s.to_string());
                                    }
                                }
                            }
                            "image_url" => out.push("[image]".to_string()),
                            _ => {}
                        }
                    }
                }
                if !out.is_empty() {
                    return out.join("\n");
                }
            }
        }
    }
    let mut out = trimmed.to_string();
    if let Some(start) = out.find("[RUNTIME CONTEXT]") {
        if let Some(end_rel) = out[start..].find("---ISANAGENT_RUNTIME_CONTEXT_END---") {
            let end = start + end_rel + "---ISANAGENT_RUNTIME_CONTEXT_END---".len();
            out.replace_range(start..end, "");
        }
    }
    let mut lines: Vec<&str> = out.lines().map(str::trim_end).collect();
    while matches!(lines.first(), Some(s) if s.trim().is_empty()) {
        lines.remove(0);
    }
    while matches!(lines.last(), Some(s) if s.trim().is_empty()) {
        lines.pop();
    }
    let mut normalized: Vec<String> = Vec::new();
    let mut prev_blank = false;
    for ln in lines {
        let blank = ln.trim().is_empty();
        if blank {
            if !prev_blank {
                normalized.push(String::new());
            }
            prev_blank = true;
        } else {
            normalized.push(ln.to_string());
            prev_blank = false;
        }
    }
    normalized.join("\n").trim().to_string()
}

pub fn load_terminal_sessions(workspace_dir: &Path) -> Result<Vec<HistorySessionListItem>, String> {
    let conn = Connection::open(db_path(workspace_dir))
        .map_err(|e| format!("open agent_memory.db: {e}"))?;
    let mut stmt = conn
        .prepare(
            "SELECT thread_id, id, created_at
             FROM messages
             WHERE thread_id LIKE 'terminal:%'
             ORDER BY id DESC",
        )
        .map_err(|e| format!("prepare terminal sessions query: {e}"))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("query terminal sessions: {e}"))?;

    let mut agg: HashMap<String, HistorySessionListItem> = HashMap::new();
    while let Some(row) = rows.next().map_err(|e| format!("read row: {e}"))? {
        let thread_id: String = row.get(0).map_err(|e| format!("read thread_id: {e}"))?;
        let Some(chat_id) = chat_id_from_thread_id(&thread_id) else {
            continue;
        };
        let id: i64 = row.get(1).map_err(|e| format!("read id: {e}"))?;
        let ts: Option<String> = row.get(2).ok();
        let entry = agg
            .entry(chat_id.clone())
            .or_insert_with(|| HistorySessionListItem {
                chat_id,
                last_ts: ts.clone(),
                message_count: 0,
                last_id: id,
            });
        entry.message_count = entry.message_count.saturating_add(1);
        if id > entry.last_id {
            entry.last_id = id;
            entry.last_ts = ts;
        } else if entry.last_ts.is_none() {
            entry.last_ts = ts;
        }
    }

    let mut out: Vec<HistorySessionListItem> = agg.into_values().collect();
    out.sort_by_key(|b| std::cmp::Reverse(b.last_id));
    Ok(out)
}

pub fn load_chat_messages(
    workspace_dir: &Path,
    chat_id: &str,
) -> Result<Vec<HistoryMessage>, String> {
    let conn = Connection::open(db_path(workspace_dir))
        .map_err(|e| format!("open agent_memory.db: {e}"))?;
    let thread_prefix = format!("terminal:{chat_id}:%");
    let mut stmt = conn
        .prepare(
            "SELECT role, content, created_at
             FROM messages
             WHERE thread_id LIKE ?1
             ORDER BY id ASC",
        )
        .map_err(|e| format!("prepare chat messages query: {e}"))?;
    let mut rows = stmt
        .query(params![thread_prefix])
        .map_err(|e| format!("query chat messages: {e}"))?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(|e| format!("read row: {e}"))? {
        let role: String = row.get(0).map_err(|e| format!("read role: {e}"))?;
        let content_raw: Option<String> = row.get(1).ok();
        let ts: Option<String> = row.get(2).ok();
        let content = summarize_content(content_raw);
        out.push(HistoryMessage { role, content, ts });
    }
    Ok(out)
}
