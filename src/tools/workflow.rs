//! Session-scoped workflow tools (todos, tool discovery, user clarification).

use async_trait::async_trait;
use chrono::Utc;
use log::info;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

use crate::bus::{BusMessage, OutboundMessage};
use crate::clarification::{ClarificationHub, METADATA_CLARIFICATION};
use crate::memory::{configure_agent_sqlite_connection, ensure_harness_todos_schema};
use crate::tool_runtime::current_tool_exec_ctx;
use crate::traits::Tool;

use super::search_tool_index;

/// One row in a session todo list.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TodoRow {
    pub content: String,
    pub status: String,
}

/// Legacy JSON file shape (`<workspace>/todos/<sha>.json`) for one-time migration into SQLite.
#[derive(Debug, Deserialize, Serialize)]
struct TodoFile {
    chat_id: String,
    items: Vec<TodoRow>,
}

fn todo_replace_sqlite(db_path: &Path, chat_id: &str, items: &[TodoRow]) -> Result<(), String> {
    let conn =
        Connection::open(db_path).map_err(|e| format!("open SQLite {:?}: {}", db_path, e))?;
    configure_agent_sqlite_connection(&conn).map_err(|e| format!("SQLite busy_timeout: {}", e))?;
    ensure_harness_todos_schema(&conn).map_err(|e| format!("harness_todos schema: {}", e))?;
    let json = serde_json::to_string(items).map_err(|e| format!("serialize todo items: {}", e))?;
    let now = Utc::now().timestamp_millis();
    conn.execute(
        "INSERT INTO harness_todos (chat_id, items_json, updated_at_ms) VALUES (?1, ?2, ?3)
         ON CONFLICT(chat_id) DO UPDATE SET
           items_json = excluded.items_json,
           updated_at_ms = excluded.updated_at_ms",
        params![chat_id, json, now],
    )
    .map_err(|e| format!("upsert harness_todos: {}", e))?;
    Ok(())
}

fn todo_load_sqlite(db_path: &Path, chat_id: &str) -> Result<Option<Vec<TodoRow>>, String> {
    let conn =
        Connection::open(db_path).map_err(|e| format!("open SQLite {:?}: {}", db_path, e))?;
    configure_agent_sqlite_connection(&conn).map_err(|e| format!("SQLite busy_timeout: {}", e))?;
    ensure_harness_todos_schema(&conn).map_err(|e| format!("harness_todos schema: {}", e))?;
    let out: Option<String> = conn
        .query_row(
            "SELECT items_json FROM harness_todos WHERE chat_id = ?1",
            params![chat_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("select harness_todos: {}", e))?;
    match out {
        None => Ok(None),
        Some(s) => {
            let items: Vec<TodoRow> =
                serde_json::from_str(&s).map_err(|e| format!("parse todo items: {}", e))?;
            Ok(Some(items))
        }
    }
}

/// Import `*.json` from a legacy directory (hashed filenames) into `harness_todos`, then remove each file.
fn migrate_legacy_json_todos(conn: &Connection, legacy_dir: &Path) -> Result<u32, String> {
    if !legacy_dir.is_dir() {
        return Ok(0);
    }
    let mut migrated = 0u32;
    let entries = fs::read_dir(legacy_dir)
        .map_err(|e| format!("read legacy todos dir {:?}: {}", legacy_dir, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("legacy todos entry: {}", e))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let raw = fs::read_to_string(&path).map_err(|e| format!("read {:?}: {}", path, e))?;
        let file: TodoFile = match serde_json::from_str(&raw) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let items_json =
            serde_json::to_string(&file.items).map_err(|e| format!("serialize items: {}", e))?;
        let now = Utc::now().timestamp_millis();
        conn.execute(
            "INSERT INTO harness_todos (chat_id, items_json, updated_at_ms) VALUES (?1, ?2, ?3)
             ON CONFLICT(chat_id) DO UPDATE SET
               items_json = excluded.items_json,
               updated_at_ms = excluded.updated_at_ms",
            params![file.chat_id, items_json, now],
        )
        .map_err(|e| format!("migrate upsert: {}", e))?;
        fs::remove_file(&path).map_err(|e| format!("remove migrated {:?}: {}", path, e))?;
        migrated += 1;
    }
    Ok(migrated)
}

/// Persists todo lists per `chat_id` in the agent SQLite DB (`harness_todos` table).
#[derive(Clone)]
pub struct TodoStore {
    db_path: PathBuf,
}

impl TodoStore {
    /// Opens the DB at `db_path` (same file as [`crate::memory::SqliteMemoryActor`]), ensures schema, and optionally migrates legacy `<workspace>/todos/*.json` files.
    pub fn try_new(db_path: PathBuf, legacy_json_dir: Option<PathBuf>) -> Result<Self, String> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create DB parent {:?}: {}", parent, e))?;
        }
        let conn =
            Connection::open(&db_path).map_err(|e| format!("open SQLite {:?}: {}", db_path, e))?;
        configure_agent_sqlite_connection(&conn)
            .map_err(|e| format!("SQLite busy_timeout: {}", e))?;
        ensure_harness_todos_schema(&conn).map_err(|e| format!("harness_todos schema: {}", e))?;
        if let Some(ref dir) = legacy_json_dir {
            let n = migrate_legacy_json_todos(&conn, dir)?;
            if n > 0 {
                info!(
                    "Migrated {} todo file(s) from {:?} into {}",
                    n,
                    dir,
                    db_path.display()
                );
            }
        }
        drop(conn);
        Ok(Self { db_path })
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn replace(&self, chat_id: &str, items: Vec<TodoRow>) -> Result<(), String> {
        todo_replace_sqlite(&self.db_path, chat_id, &items)
    }

    pub fn load(&self, chat_id: &str) -> Result<Option<Vec<TodoRow>>, String> {
        todo_load_sqlite(&self.db_path, chat_id)
    }
}

const MAX_TODO_ITEMS: usize = 200;

/// Replace the structured todo list for this chat session.
pub struct TodoWriteTool {
    pub store: TodoStore,
}

fn normalize_todo_status(s: &str) -> Result<(), String> {
    match s {
        "pending" | "in_progress" | "completed" => Ok(()),
        _ => Err(format!(
            "Invalid status {:?}; use pending, in_progress, or completed.",
            s
        )),
    }
}

fn format_todo_list(chat_id: &str, rows: &[TodoRow]) -> String {
    let mut out = String::from("# Todo list\n\n");
    if rows.is_empty() {
        out.push_str("(empty)\n");
        return out;
    }
    let icon = |s: &str| match s {
        "completed" => "[x]",
        "in_progress" => "[~]",
        _ => "[ ]",
    };
    for (i, row) in rows.iter().enumerate() {
        out.push_str(&format!(
            "{}. {} {}\n",
            i + 1,
            icon(&row.status),
            row.content
        ));
    }
    let done = rows.iter().filter(|r| r.status == "completed").count();
    out.push_str(&format!(
        "\nSession: {}\nProgress: {}/{}\n",
        chat_id,
        done,
        rows.len()
    ));
    out
}

#[async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &str {
        "todo_write"
    }

    fn description(&self) -> &str {
        "Replace the structured todo list for this chat. Todos are scoped per chat_id (from RUNTIME CONTEXT) and stored in the agent SQLite database (harness_todos) so they survive restarts. If a legacy todos/ JSON folder exists from an older build, it is migrated once on startup. Use for multi-step work tracking."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "Chat/session id from RUNTIME CONTEXT (same as for the message tool)."
                },
                "items": {
                    "type": "array",
                    "description": "Complete new todo list (replaces any previous list for this chat).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Task state"
                            }
                        },
                        "required": ["content", "status"]
                    }
                }
            },
            "required": ["chat_id", "items"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let chat_id = args
            .get("chat_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'chat_id'")?;

        let items_val = args
            .get("items")
            .and_then(|v| v.as_array())
            .ok_or("Missing 'items' array")?;

        if items_val.len() > MAX_TODO_ITEMS {
            return Err(format!(
                "At most {} todo items allowed (got {}).",
                MAX_TODO_ITEMS,
                items_val.len()
            ));
        }

        let mut rows: Vec<TodoRow> = Vec::with_capacity(items_val.len());
        for (i, item) in items_val.iter().enumerate() {
            let content = item
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("items[{}]: missing content", i))?
                .to_string();
            let status_raw = item
                .get("status")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("items[{}]: missing status", i))?;
            normalize_todo_status(status_raw).map_err(|e| format!("items[{}]: {}", i, e))?;
            rows.push(TodoRow {
                content,
                status: status_raw.to_string(),
            });
        }

        let db_path = self.store.db_path().to_path_buf();
        let chat = chat_id.to_string();
        let rows_clone = rows.clone();
        tokio::task::spawn_blocking(move || todo_replace_sqlite(&db_path, &chat, &rows_clone))
            .await
            .map_err(|e| format!("todo_write task: {}", e))?
            .map_err(|e| format!("Failed to save todos: {}", e))?;
        Ok(format_todo_list(chat_id, &rows))
    }
}

/// Search registered tools by keywords (name + description).
pub struct ToolSearchTool {
    pub catalog: Arc<RwLock<Vec<(String, String)>>>,
}

#[async_trait]
impl Tool for ToolSearchTool {
    fn name(&self) -> &str {
        "search_tools"
    }

    fn description(&self) -> &str {
        "Find built-in tools by keyword or short phrase. Use when unsure which tool fits a task. Matches tool names and descriptions."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for (e.g. 'grep', 'schedule', 'memory')."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 12, max 40)."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query'")?;

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(12)
            .clamp(1, 40) as usize;

        let entries = self
            .catalog
            .read()
            .map_err(|e| format!("catalog lock: {}", e))?
            .clone();

        let hits = search_tool_index(&entries, query, limit);
        if hits.is_empty() {
            return Ok("No tools matched that query.".to_string());
        }

        let mut out = String::from("Matching tools:\n\n");
        for (name, score) in hits {
            let desc = entries
                .iter()
                .find(|(n, _)| n == &name)
                .map(|(_, d)| d.as_str())
                .unwrap_or("");
            let snippet: String = desc.chars().take(160).collect();
            let ellipses = if desc.len() > 160 { "…" } else { "" };
            out.push_str(&format!(
                "- **{}** (score {})\n  {}{}\n\n",
                name, score, snippet, ellipses
            ));
        }
        Ok(out.trim_end().to_string())
    }
}

struct ClarificationSlotGuard {
    hub: Arc<ClarificationHub>,
    session_key: String,
    armed: bool,
}

impl ClarificationSlotGuard {
    fn new(hub: Arc<ClarificationHub>, session_key: String) -> Self {
        Self {
            hub,
            session_key,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ClarificationSlotGuard {
    fn drop(&mut self) {
        if self.armed {
            self.hub.cancel_wait(&self.session_key);
        }
    }
}

/// Block until the user sends the next message in this session, then return it as the tool result.
///
/// Requires the agent to wrap tool execution with a [`crate::tool_runtime::ToolExecCtx`]. The
/// channel delivers an [`OutboundMessage`] tagged with [`METADATA_CLARIFICATION`] so terminals and
/// API clients can style the prompt; the following inbound on the same session completes the wait.
pub struct AskUserTool {
    pub clarification_hub: Arc<ClarificationHub>,
    pub outbound_tx: mpsc::Sender<BusMessage>,
}

const ASK_USER_TIMEOUT_SECS_MIN: u64 = 10;
const ASK_USER_TIMEOUT_SECS_MAX: u64 = 86_400;
const ASK_USER_TIMEOUT_SECS_DEFAULT: u64 = 1_800;
const ASK_USER_MAX_CHOICES: usize = 8;

#[async_trait]
impl Tool for AskUserTool {
    fn name(&self) -> &str {
        "ask_user"
    }

    fn description(&self) -> &str {
        "Ask the human a focused question and wait for their next reply in this chat. Use when you need a decision, missing detail, or confirmation before continuing. The user’s following message becomes this tool’s return value (not a new agent turn). Works in terminal and API channels when inbound messages reach the same session."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Clear question for the user (plain text)."
                },
                "choices": {
                    "type": "array",
                    "description": "Optional short list of allowed answers (max 8); shown with the prompt.",
                    "items": { "type": "string" },
                    "maxItems": 8
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Max seconds to wait (10–86400, default 1800)."
                },
                "allow_empty": {
                    "type": "boolean",
                    "description": "If false (default), treat whitespace-only replies as invalid and keep waiting until timeout."
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let ctx = current_tool_exec_ctx().ok_or_else(|| {
            "ask_user is only available during a live agent turn (missing tool runtime context)."
                .to_string()
        })?;

        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'prompt'".to_string())?
            .trim();
        if prompt.is_empty() {
            return Err("prompt must be non-empty".to_string());
        }

        let allow_empty = args
            .get("allow_empty")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(ASK_USER_TIMEOUT_SECS_DEFAULT)
            .clamp(ASK_USER_TIMEOUT_SECS_MIN, ASK_USER_TIMEOUT_SECS_MAX);

        let mut choices: Vec<String> = Vec::new();
        if let Some(arr) = args.get("choices").and_then(|v| v.as_array()) {
            if arr.len() > ASK_USER_MAX_CHOICES {
                return Err(format!(
                    "At most {} choices allowed (got {}).",
                    ASK_USER_MAX_CHOICES,
                    arr.len()
                ));
            }
            for (i, v) in arr.iter().enumerate() {
                let s = v
                    .as_str()
                    .ok_or_else(|| format!("choices[{}]: expected string", i))?
                    .trim();
                if s.is_empty() {
                    return Err(format!("choices[{}]: must be non-empty", i));
                }
                choices.push(s.to_string());
            }
        }

        let rx = self
            .clarification_hub
            .begin_wait(&ctx.session_key)
            .map_err(|e| e.to_string())?;

        let mut guard = ClarificationSlotGuard::new(
            Arc::clone(&self.clarification_hub),
            ctx.session_key.clone(),
        );

        let mut body = String::from("The agent needs your input:\n\n");
        body.push_str(prompt);
        if !choices.is_empty() {
            body.push_str("\n\nOptions:\n");
            for c in &choices {
                body.push_str(&format!("- {}\n", c));
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert(
            METADATA_CLARIFICATION.to_string(),
            serde_json::Value::Bool(true),
        );

        let outbound = OutboundMessage {
            channel: ctx.channel.clone(),
            chat_id: ctx.chat_id.clone(),
            thread_id: ctx.thread_id.clone(),
            content: body,
            metadata,
        };

        self.outbound_tx
            .send(BusMessage::Outbound(outbound))
            .await
            .map_err(|e| format!("failed to send clarification prompt: {}", e))?;

        let wait = tokio::time::Duration::from_secs(timeout_secs);
        let reply = match tokio::time::timeout(wait, rx).await {
            Err(_) => {
                return Err(format!(
                    "Timed out after {}s waiting for a user reply to ask_user.",
                    timeout_secs
                ));
            }
            Ok(Err(_)) => {
                return Err(
                    "Clarification wait ended without a reply (session cancelled or reset)."
                        .to_string(),
                );
            }
            Ok(Ok(text)) => text,
        };

        guard.disarm();

        let trimmed = reply.trim();
        if !allow_empty && trimmed.is_empty() {
            return Err(
                "User reply was empty (allow_empty is false). Call ask_user again if you still need input."
                    .to_string(),
            );
        }

        if !choices.is_empty() && !choices.iter().any(|c| c.as_str() == trimmed) {
            return Ok(format!(
                "User reply (not among listed choices): {}\n\nListed options were: {:?}",
                reply, choices
            ));
        }

        Ok(format!("User reply:\n{}", reply))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::BusMessage;
    use crate::tool_runtime::{with_tool_exec_scope, ToolExecCtx};
    use serde_json::json;
    use std::path::PathBuf;

    #[tokio::test]
    async fn ask_user_outbound_and_reply() {
        let hub = Arc::new(ClarificationHub::new());
        let (ob_tx, mut ob_rx) = mpsc::channel(8);
        let tool = AskUserTool {
            clarification_hub: hub.clone(),
            outbound_tx: ob_tx,
        };
        let hub_signal = hub.clone();
        let join = tokio::spawn(async move {
            with_tool_exec_scope(ToolExecCtx::new("terminal", "u1", None), async move {
                tool.execute(json!({
                    "prompt": "Which?",
                    "timeout_secs": 30,
                    "allow_empty": true
                }))
                .await
            })
            .await
        });

        let ob = ob_rx.recv().await.expect("outbound");
        match ob {
            BusMessage::Outbound(out) => {
                assert_eq!(
                    out.metadata.get(METADATA_CLARIFICATION),
                    Some(&serde_json::Value::Bool(true))
                );
                assert!(out.content.contains("Which?"));
            }
            _ => panic!("expected Outbound"),
        }
        assert!(hub_signal.try_deliver_reply("terminal:u1:", "blue".into()));
        let answer = join.await.expect("join").expect("tool ok");
        assert!(answer.contains("blue"));
    }

    fn temp_todo_db(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "isanagent_todos_{}_{}.sqlite",
            name,
            uuid::Uuid::new_v4()
        ))
    }

    #[tokio::test]
    async fn todo_write_isolated_per_chat_id() {
        let db = temp_todo_db("isolate");
        let store = TodoStore::try_new(db.clone(), None).unwrap();
        let tool = TodoWriteTool {
            store: store.clone(),
        };

        tool.execute(json!({
            "chat_id": "chat-a",
            "items": [
                {"content": "A1", "status": "pending"},
                {"content": "A2", "status": "completed"}
            ]
        }))
        .await
        .unwrap();

        tool.execute(json!({
            "chat_id": "chat-b",
            "items": [{"content": "B1", "status": "in_progress"}]
        }))
        .await
        .unwrap();

        let a = store.load("chat-a").unwrap().expect("chat-a");
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].content, "A1");
        assert_eq!(a[1].status, "completed");

        let b = store.load("chat-b").unwrap().expect("chat-b");
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].content, "B1");

        let _ = fs::remove_file(&db);
    }

    #[tokio::test]
    async fn todo_persists_across_new_store() {
        let db = temp_todo_db("persist");
        {
            let store = TodoStore::try_new(db.clone(), None).unwrap();
            let tool = TodoWriteTool { store };
            tool.execute(json!({
                "chat_id": "session-xyz",
                "items": [{"content": "survive restart", "status": "pending"}]
            }))
            .await
            .unwrap();
        }

        let store2 = TodoStore::try_new(db.clone(), None).unwrap();
        let loaded = store2.load("session-xyz").unwrap().expect("sqlite");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "survive restart");

        let _ = fs::remove_file(&db);
    }

    #[tokio::test]
    async fn legacy_json_migrates_into_sqlite() {
        let db = temp_todo_db("migrate");
        let legacy =
            std::env::temp_dir().join(format!("isanagent_legacy_todos_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&legacy).unwrap();
        let legacy_file = legacy.join("abc.json");
        let payload = TodoFile {
            chat_id: "migrated-chat".to_string(),
            items: vec![TodoRow {
                content: "from json".to_string(),
                status: "pending".to_string(),
            }],
        };
        fs::write(
            &legacy_file,
            serde_json::to_string_pretty(&payload).unwrap(),
        )
        .unwrap();

        let store = TodoStore::try_new(db.clone(), Some(legacy.clone())).unwrap();
        assert!(
            !legacy_file.exists(),
            "legacy file should be removed after migrate"
        );

        let rows = store.load("migrated-chat").unwrap().expect("rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].content, "from json");

        let _ = fs::remove_file(&db);
        let _ = fs::remove_dir_all(&legacy);
    }

    #[tokio::test]
    async fn todo_write_rejects_bad_status() {
        let db = temp_todo_db("badstatus");
        let tool = TodoWriteTool {
            store: TodoStore::try_new(db.clone(), None).unwrap(),
        };
        let err = tool
            .execute(json!({
                "chat_id": "c",
                "items": [{"content": "x", "status": "done"}]
            }))
            .await
            .unwrap_err();
        assert!(err.contains("Invalid status"), "{}", err);
        let _ = fs::remove_file(&db);
    }

    #[tokio::test]
    async fn tool_search_ranks_name_matches() {
        let cat = Arc::new(RwLock::new(vec![
            (
                "read_file".to_string(),
                "Read a local file from disk.".to_string(),
            ),
            (
                "glob_files".to_string(),
                "Find paths by glob pattern.".to_string(),
            ),
            (
                "search_memory".to_string(),
                "Search session memory summaries.".to_string(),
            ),
        ]));
        let tool = ToolSearchTool {
            catalog: Arc::clone(&cat),
        };

        let out = tool
            .execute(json!({"query": "memory", "limit": 5}))
            .await
            .unwrap();
        assert!(
            out.contains("search_memory"),
            "expected search_memory in:\n{}",
            out
        );
        assert!(out.contains("score"));

        let out2 = tool
            .execute(json!({"query": "glob", "limit": 2}))
            .await
            .unwrap();
        assert!(out2.contains("glob_files"));
    }
}
