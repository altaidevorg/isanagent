use async_trait::async_trait;
use chrono::{DateTime, NaiveDateTime, Utc};
use log::debug;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::utils::{ChatMessage, ContentPart, MessageContent, RUNTIME_CONTEXT_END_SUFFIX};
use crate::{ActorError, ActorLogic};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

pub struct SharedReply<T>(pub Arc<Mutex<Option<oneshot::Sender<T>>>>);

impl<T> Clone for SharedReply<T> {
    fn clone(&self) -> Self {
        SharedReply(self.0.clone())
    }
}

impl<T> fmt::Debug for SharedReply<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SharedReply")
    }
}

impl<T> SharedReply<T> {
    pub fn new(tx: oneshot::Sender<T>) -> Self {
        Self(Arc::new(Mutex::new(Some(tx))))
    }

    pub fn send(&self, msg: T) -> Result<(), T> {
        if let Some(tx) = self.0.lock().unwrap().take() {
            tx.send(msg)
        } else {
            Err(msg)
        }
    }
}

/// Stored `messages.content`: JSON array of [`ContentPart`] or legacy plain text.
fn message_content_from_stored(s: String) -> MessageContent {
    if s.trim_start().starts_with('[') {
        match serde_json::from_str::<Vec<ContentPart>>(&s) {
            Ok(parts) => MessageContent::Parts(parts),
            Err(_) => {
                debug!("SqliteMemoryActor: failed to parse content as ContentPart array, treating as plain text");
                MessageContent::Text(s)
            }
        }
    } else {
        MessageContent::Text(s)
    }
}

fn first_user_preview_from_content(s: String) -> Option<String> {
    let message_content = message_content_from_stored(s);
    if let MessageContent::Parts(parts) = &message_content {
        let has_text = parts
            .iter()
            .any(|p| matches!(p, ContentPart::Text { text } if !text.trim().is_empty()));
        let has_image = parts
            .iter()
            .any(|p| matches!(p, ContentPart::ImageUrl { .. }));
        if !has_text && has_image {
            return Some("Image".to_string());
        }
    }
    let text = message_content.text_content();
    let t = text.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

type SessionMessageSinceReflectionRow = (i64, ChatMessage);
type GetMessagesSinceReflectionResult =
    Result<(Vec<SessionMessageSinceReflectionRow>, Option<i64>), String>;

/// How long SQLite waits on `SQLITE_BUSY` before failing (memory + `harness_todos` share one file).
pub const AGENT_SQLITE_BUSY_TIMEOUT_MS: u64 = 5_000;

/// One persisted root session row for a channel (e.g. `terminal` → `thread_id` = `terminal:<chat_uuid>:`).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootThreadListItem {
    /// Full `messages.thread_id` (see [`crate::bus::clarification_session_key`] for root keys).
    pub thread_id: String,
    /// Latest `messages.id` in this thread (for ordering; recency).
    pub last_message_id: i64,
    /// UTC millis from `messages.created_at` on the row with `last_message_id` (`0` if missing).
    pub last_activity_ms: i64,
    /// Truncated first user line, runtime prefix stripped.
    pub preview: String,
}

fn sqlite_datetime_to_unix_ms(s: &str) -> i64 {
    let t = s.trim();
    if t.is_empty() {
        return 0;
    }
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.3f",
        "%Y-%m-%d %H:%M:%S%.6f",
    ] {
        if let Ok(na) = NaiveDateTime::parse_from_str(t, fmt) {
            return na.and_utc().timestamp_millis();
        }
    }
    DateTime::parse_from_rfc3339(t)
        .map(|d| d.timestamp_millis())
        .unwrap_or(0)
}

/// True when `thread_id` is a main session for `channel` (`<channel>:<id>:` with empty sub-thread part).
/// Sub-threads such as `terminal:uuid:subagent-…` return false.
pub fn is_root_session_thread_id(channel: &str, thread_id: &str) -> bool {
    let parts: Vec<&str> = thread_id.split(':').collect();
    if parts.len() != 3 {
        return false;
    }
    if parts[0] != channel {
        return false;
    }
    if uuid::Uuid::parse_str(parts[1]).is_err() {
        return false;
    }
    parts[2].is_empty()
}

/// Parse `chat_id` (UUID) from a root `thread_id`, or return `None`.
pub fn chat_id_from_root_thread_id(channel: &str, thread_id: &str) -> Option<String> {
    if !is_root_session_thread_id(channel, thread_id) {
        return None;
    }
    let parts: Vec<&str> = thread_id.split(':').collect();
    parts.get(1).map(std::string::ToString::to_string)
}

/// Max characters taken from the first line of user content for root-thread list previews.
const THREAD_PREVIEW_MAX_CHARS: usize = 56;

fn strip_user_preview_for_thread_list(s: &str) -> String {
    if let Some(idx) = s.find(RUNTIME_CONTEXT_END_SUFFIX) {
        s[idx + RUNTIME_CONTEXT_END_SUFFIX.len()..]
            .trim_start()
            .to_string()
    } else {
        s.to_string()
    }
}

fn truncate_thread_preview_line(text: &str) -> String {
    let line = text.split('\n').next().unwrap_or(text).trim();
    let mut iter = line.chars();
    let chunk: String = iter.by_ref().take(THREAD_PREVIEW_MAX_CHARS).collect();
    if iter.next().is_some() {
        format!("{}…", chunk)
    } else {
        chunk
    }
}

/// PRAGMAs for file-backed agent DB handles (`SqliteMemoryActor`, harness todos in the same file).
pub fn configure_agent_sqlite_connection(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.busy_timeout(std::time::Duration::from_millis(
        AGENT_SQLITE_BUSY_TIMEOUT_MS,
    ))
}

/// Schema for `harness_todos` (same DB as agent memory; accessed only via [`MemoryMessage`] on [`SqliteMemoryActor`]).
pub fn ensure_harness_todos_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS harness_todos (
            chat_id TEXT PRIMARY KEY NOT NULL,
            items_json TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL
        )",
        [],
    )?;
    Ok(())
}

/// Persisted sub-agent runs for audit (`subagent_tasks` in the agent DB).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SubagentTaskRecord {
    pub task_id: String,
    pub parent_chat_id: String,
    pub child_chat_id: String,
    pub display_name: Option<String>,
    /// Named agent type (e.g. "researcher", "coder"). None for legacy generic spawns.
    pub agent_name: Option<String>,
    pub prompt: String,
    pub status: String,
    pub result: Option<String>,
    pub error: Option<String>,
    pub execution_job_id: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

pub fn ensure_subagent_tasks_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS subagent_tasks (
            task_id TEXT PRIMARY KEY NOT NULL,
            parent_chat_id TEXT NOT NULL,
            child_chat_id TEXT NOT NULL,
            display_name TEXT,
            agent_name TEXT,
            prompt TEXT NOT NULL,
            status TEXT NOT NULL,
            result TEXT,
            error TEXT,
            execution_job_id TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        )",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_subagent_tasks_parent
         ON subagent_tasks(parent_chat_id, updated_at_ms DESC)",
        [],
    )?;
    Ok(())
}

pub fn ensure_cron_jobs_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cron_jobs (
            id TEXT PRIMARY KEY,
            schedule TEXT NOT NULL,
            message TEXT NOT NULL,
            last_run_at_ms INTEGER,
            chat_id TEXT NOT NULL DEFAULT 'unknown',
            channel TEXT NOT NULL DEFAULT 'unknown',
            webhook_token TEXT NOT NULL DEFAULT '',
            trigger_claim_token TEXT NOT NULL DEFAULT '',
            trigger_claimed_at_ms INTEGER,
            completed_at_ms INTEGER
        )",
        [],
    )?;
    Ok(())
}

/// One row in a session todo list (`harness_todos`, via [`SqliteMemoryActor`]).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TodoRow {
    pub content: String,
    pub status: String,
}

fn todo_replace_sqlite_conn(
    conn: &Connection,
    chat_id: &str,
    items: &[TodoRow],
) -> Result<(), String> {
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

fn todo_load_sqlite_conn(conn: &Connection, chat_id: &str) -> Result<Option<Vec<TodoRow>>, String> {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SummaryEntry {
    pub id: i64,
    pub thread_id: String,
    pub summary: String,
    pub key_info: String,
    pub knowledge_gaps: String,
    pub created_at: String,
}

/// Messages sent to the SqliteMemoryActor
#[derive(Clone, Debug)]
pub enum MemoryMessage {
    AddMessage {
        thread_id: String,
        message: ChatMessage,
        reply: SharedReply<Result<(), String>>,
    },
    GetContext {
        thread_id: String,
        reply: SharedReply<Result<Vec<ChatMessage>, String>>,
    },
    /// Plain-text preview from the earliest user turn (for thread list titles).
    FirstUserMessagePreview {
        thread_id: String,
        reply: SharedReply<Result<Option<String>, String>>,
    },
    /// Batch variant: one SQLite round-trip for many `thread_id`s (same order as input).
    FirstUserMessagePreviewsBatch {
        thread_ids: Vec<String>,
        reply: SharedReply<Result<Vec<Option<String>>, String>>,
    },
    Clear {
        thread_id: String,
        keep_last: usize,
        reply: SharedReply<Result<(), String>>,
    },
    // --- Reflection and Summary Messages ---
    AddSummary {
        thread_id: String,
        summary: String,
        key_info: String,
        knowledge_gaps: String,
        reply: SharedReply<Result<(), String>>,
    },
    UpdateSummary {
        id: i64,
        summary: String,
        key_info: String,
        knowledge_gaps: String,
        reply: SharedReply<Result<(), String>>,
    },
    GetRecentSummaries {
        thread_id: String,
        limit: usize,
        reply: SharedReply<Result<Vec<String>, String>>,
    },
    GetSummaries {
        thread_id: String,
        limit: usize,
        reply: SharedReply<Result<Vec<SummaryEntry>, String>>,
    },
    DeleteSummary {
        id: i64,
        reply: SharedReply<Result<(), String>>,
    },
    UpdateThreadMetadata {
        thread_id: String,
        last_reflection_msg_id: Option<i64>,
        reply: SharedReply<Result<(), String>>,
    },
    GetThreadMetadata {
        thread_id: String,
        reply: SharedReply<Result<(Option<i64>, String), String>>, // (last_msg_id, last_reflection_time)
    },
    SearchSummaries {
        query: String,
        reply: SharedReply<Result<Vec<String>, String>>,
    },
    FetchSummariesByTimeRange {
        days_ago: u64,
        limit: usize,
        reply: SharedReply<Result<Vec<String>, String>>,
    },
    GetThreadsNeedingReflection {
        threshold_mins: u64,
        reply: SharedReply<Result<Vec<String>, String>>,
    },
    GetMessagesSinceReflection {
        thread_id: String,
        reply: SharedReply<GetMessagesSinceReflectionResult>,
    },
    GetLongTermReflectionState {
        threshold: usize,
        reply: SharedReply<Result<(bool, String, i64), String>>,
    },
    SetLongTermReflectionState {
        max_id: i64,
        reply: SharedReply<Result<(), String>>,
    },
    /// Replace the structured todo list for `chat_id` (`harness_todos`).
    ReplaceHarnessTodos {
        chat_id: String,
        items: Vec<TodoRow>,
        reply: SharedReply<Result<(), String>>,
    },
    /// Load todos for `chat_id`, if any.
    LoadHarnessTodos {
        chat_id: String,
        reply: SharedReply<Result<Option<Vec<TodoRow>>, String>>,
    },
    /// Insert a `running` row when a sub-agent task starts.
    InsertSubagentTask {
        task_id: String,
        parent_chat_id: String,
        child_chat_id: String,
        display_name: Option<String>,
        agent_name: Option<String>,
        prompt: String,
        reply: SharedReply<Result<(), String>>,
    },
    /// Update row when a sub-agent task finishes (`completed`, `failed`, `cancelled`).
    FinalizeSubagentTask {
        task_id: String,
        parent_chat_id: String,
        status: String,
        result: Option<String>,
        error: Option<String>,
        execution_job_id: Option<String>,
        reply: SharedReply<Result<(), String>>,
    },
    /// Recent persisted sub-agent tasks for a parent chat (newest first).
    ListSubagentTasksForParent {
        parent_chat_id: String,
        limit: usize,
        reply: SharedReply<Result<Vec<SubagentTaskRecord>, String>>,
    },
    /// Root `messages.thread_id` rows for a channel prefix (e.g. `terminal` → `terminal:…`), with previews.
    ListRootThreadsForChannelWithPreviews {
        channel: String,
        limit: u32,
        reply: SharedReply<Result<Vec<RootThreadListItem>, String>>,
    },
    /// Count active cron jobs (completed_at_ms IS NULL)
    GetActiveCronsCount {
        reply: SharedReply<Result<usize, String>>,
    },
}

/// Persistent SQLite-based memory Actor for agents.
pub struct SqliteMemoryActor {
    conn: Connection,
}

impl SqliteMemoryActor {
    /// Create a new SqliteMemory.
    ///
    /// `db_path`: Path to the SQLite DB file. Use `:memory:` for in-memory.
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn = (|| -> Result<Connection, rusqlite::Error> {
            let conn = Connection::open(db_path)?;
            configure_agent_sqlite_connection(&conn)?;

        // Create the messages table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Try adding the native tool calling schema columns dynamically.
        // Failures here are expected on existing databases after the first run.
        let _ = conn.execute("ALTER TABLE messages ADD COLUMN name TEXT", []);
        let _ = conn.execute("ALTER TABLE messages ADD COLUMN tool_calls TEXT", []);
        let _ = conn.execute("ALTER TABLE messages ADD COLUMN tool_call_id TEXT", []);
        let _ = conn.execute("ALTER TABLE messages ADD COLUMN reasoning_content TEXT", []);

        // Create the session_summaries table for reflections
        conn.execute(
            "CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                key_info TEXT NOT NULL,
                knowledge_gaps TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Create the session_metadata table to track reflection progress
        conn.execute(
            "CREATE TABLE IF NOT EXISTS session_metadata (
                thread_id TEXT PRIMARY KEY,
                last_reflection_msg_id INTEGER,
                last_reflection_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_thread_id ON messages (thread_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_thread_role_id ON messages (thread_id, role, id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_summaries_thread ON session_summaries (thread_id)",
            [],
        )?;

        // Create the session_summaries virtual table for FTS5
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS session_summaries_fts USING fts5(
                thread_id, summary, key_info, knowledge_gaps,
                content='session_summaries', content_rowid='id'
            )",
            [],
        )?;

        // FTS Sync Trigger (Insert)
        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS session_summaries_ai AFTER INSERT ON session_summaries BEGIN
                INSERT INTO session_summaries_fts(rowid, thread_id, summary, key_info, knowledge_gaps)
                VALUES (new.id, new.thread_id, new.summary, new.key_info, new.knowledge_gaps);
            END;",
            [],
        )?;

        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS session_summaries_ad AFTER DELETE ON session_summaries BEGIN
                DELETE FROM session_summaries_fts WHERE rowid = old.id;
            END;",
            [],
        )?;

        // global_metadata table (moved here from reflection.rs)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS global_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
            [],
        )?;

        ensure_harness_todos_schema(&conn)?;
        ensure_subagent_tasks_schema(&conn)?;
        ensure_cron_jobs_schema(&conn)?;

            Ok(conn)
        })()
        .map_err(|e| format!("SQLite init ({}): {}", db_path, e))?;

        Ok(Self { conn })
    }
}

#[async_trait]
impl ActorLogic<MemoryMessage> for SqliteMemoryActor {
    fn name(&self) -> String {
        "SqliteMemoryActor".to_string()
    }

    async fn process(
        &mut self,
        packet: MemoryMessage,
    ) -> Result<Option<(String, MemoryMessage)>, ActorError> {
        match packet {
            MemoryMessage::AddMessage {
                thread_id,
                message,
                reply,
            } => {
                let content_str = match &message.content {
                    Some(MessageContent::Text(s)) => Some(s.clone()),
                    Some(MessageContent::Parts(parts)) => serde_json::to_string(parts).ok(),
                    None => None,
                };
                let tool_calls_str = message
                    .tool_calls
                    .map(|tc| serde_json::to_string(&tc).unwrap_or_default());
                let res = self.conn.execute(
                    "INSERT INTO messages (thread_id, role, content, name, tool_calls, tool_call_id, reasoning_content) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![thread_id, message.role, content_str, message.name, tool_calls_str, message.tool_call_id, message.reasoning_content],
                ).map_err(|e| e.to_string()).map(|_| ());

                let _ = reply.send(res);
            }
            MemoryMessage::GetContext { thread_id, reply } => {
                let res = (|| -> Result<Vec<ChatMessage>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT role, content, name, tool_calls, tool_call_id, reasoning_content FROM messages WHERE thread_id = ?1 ORDER BY created_at ASC"
                    ).map_err(|e| e.to_string())?;

                    let message_iter = stmt
                        .query_map(params![thread_id], |row| {
                            let tool_calls_str: Option<String> = row.get(3)?;
                            let tool_calls =
                                tool_calls_str.and_then(|s| serde_json::from_str(&s).ok());
                            let content_raw: Option<String> = row.get(1)?;
                            let content = content_raw.map(message_content_from_stored);
                            Ok(ChatMessage {
                                role: row.get(0)?,
                                content,
                                name: row.get(2)?,
                                tool_calls,
                                tool_call_id: row.get(4)?,
                                reasoning_content: row.get(5)?,
                            })
                        })
                        .map_err(|e| e.to_string())?;

                    let mut messages = Vec::new();
                    for msg_result in message_iter {
                        match msg_result {
                            Ok(msg) => messages.push(msg),
                            Err(e) => return Err(e.to_string()),
                        }
                    }
                    Ok(messages)
                })();

                let _ = reply.send(res);
            }
            MemoryMessage::FirstUserMessagePreview { thread_id, reply } => {
                let res = (|| -> Result<Option<String>, String> {
                    let mut stmt = self
                        .conn
                        .prepare(
                            "SELECT content FROM messages WHERE thread_id = ?1 AND role = 'user' ORDER BY id ASC LIMIT 1",
                        )
                        .map_err(|e| e.to_string())?;
                    let content_raw: Option<String> = stmt
                        .query_row(params![thread_id], |row| row.get(0))
                        .optional()
                        .map_err(|e| e.to_string())?;
                    let Some(s) = content_raw else {
                        return Ok(None);
                    };
                    Ok(first_user_preview_from_content(s))
                })();

                let _ = reply.send(res);
            }
            MemoryMessage::FirstUserMessagePreviewsBatch { thread_ids, reply } => {
                let res = (|| -> Result<Vec<Option<String>>, String> {
                    if thread_ids.is_empty() {
                        return Ok(Vec::new());
                    }
                    let placeholders = thread_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                    let sql = format!(
                        "SELECT m.thread_id, m.content FROM messages m
                         INNER JOIN (
                             SELECT thread_id, MIN(id) AS min_id
                             FROM messages
                             WHERE role = 'user' AND thread_id IN ({placeholders})
                             GROUP BY thread_id
                         ) t ON m.thread_id = t.thread_id AND m.id = t.min_id AND m.role = 'user'"
                    );
                    let mut stmt = self.conn.prepare(&sql).map_err(|e| e.to_string())?;
                    let mut rows = stmt
                        .query(params_from_iter(thread_ids.iter()))
                        .map_err(|e| e.to_string())?;

                    let mut map: HashMap<String, Option<String>> = HashMap::new();
                    while let Some(row) = rows.next().map_err(|e| e.to_string())? {
                        let sid: String = row.get(0).map_err(|e| e.to_string())?;
                        let content: String = row.get(1).map_err(|e| e.to_string())?;
                        map.insert(sid, first_user_preview_from_content(content));
                    }

                    Ok(thread_ids
                        .iter()
                        .map(|id| match map.get(id.as_str()) {
                            None => None,
                            Some(preview) => preview.clone(),
                        })
                        .collect())
                })();

                let _ = reply.send(res);
            }
            MemoryMessage::Clear {
                thread_id,
                keep_last,
                reply,
            } => {
                let res = (|| -> Result<(), String> {
                    let tx = self.conn.transaction().map_err(|e| e.to_string())?;
                    if keep_last == 0 {
                        // Full thread delete (explicit chat removal).
                        tx.execute(
                            "DELETE FROM messages WHERE thread_id = ?1",
                            params![thread_id],
                        )
                        .map_err(|e| e.to_string())?;
                        tx.execute(
                            "DELETE FROM session_summaries WHERE thread_id = ?1",
                            params![thread_id],
                        )
                        .map_err(|e| e.to_string())?;
                        tx.execute(
                            "DELETE FROM session_metadata WHERE thread_id = ?1",
                            params![thread_id],
                        )
                        .map_err(|e| e.to_string())?;
                    } else {
                        // Trim to the last `keep_last` messages by id (see Memory::clear_keep_last).
                        let keep = i64::try_from(keep_last).map_err(|_| {
                            "keep_last is too large for the backing store".to_string()
                        })?;
                        tx.execute(
                            "DELETE FROM messages WHERE thread_id = ?1 AND id NOT IN (
                                SELECT id FROM (
                                    SELECT id FROM messages WHERE thread_id = ?1 ORDER BY id DESC LIMIT ?2
                                )
                            )",
                            params![thread_id, keep],
                        )
                        .map_err(|e| e.to_string())?;
                    }
                    tx.commit().map_err(|e| e.to_string())?;
                    Ok(())
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::AddSummary {
                thread_id,
                summary,
                key_info,
                knowledge_gaps,
                reply,
            } => {
                let res = self.conn.execute(
                    "INSERT INTO session_summaries (thread_id, summary, key_info, knowledge_gaps) 
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(thread_id) DO UPDATE SET 
                        summary=excluded.summary, 
                        key_info=excluded.key_info, 
                        knowledge_gaps=excluded.knowledge_gaps,
                        created_at=CURRENT_TIMESTAMP",
                    params![thread_id, summary, key_info, knowledge_gaps],
                ).map_err(|e| e.to_string()).map(|_| ());

                let _ = reply.send(res);
            }
            MemoryMessage::UpdateSummary {
                id,
                summary,
                key_info,
                knowledge_gaps,
                reply,
            } => {
                let res = self.conn.execute(
                    "UPDATE session_summaries SET summary = ?1, key_info = ?2, knowledge_gaps = ?3 WHERE id = ?4",
                    params![summary, key_info, knowledge_gaps, id],
                ).map_err(|e| e.to_string()).map(|_| ());

                let _ = reply.send(res);
            }
            MemoryMessage::GetRecentSummaries {
                thread_id,
                limit,
                reply,
            } => {
                let res = (|| -> Result<Vec<String>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT summary, key_info, knowledge_gaps, created_at FROM session_summaries 
                         WHERE thread_id LIKE ?1 ORDER BY created_at DESC LIMIT ?2"
                    ).map_err(|e| e.to_string())?;

                    let pattern = format!("{}%", thread_id);
                    let limit_i64 = limit as i64;
                    let rows = stmt
                        .query_map(params![pattern, limit_i64], |row| {
                            let summary: String = row.get(0)?;
                            let key_info: String = row.get(1)?;
                            let knowledge_gaps: String = row.get(2)?;
                            let created_at: String = row.get(3)?;
                            Ok(format!(
                                "[{}] Summary: {}\nKey Info: {}\nGaps: {}",
                                created_at, summary, key_info, knowledge_gaps
                            ))
                        })
                        .map_err(|e| e.to_string())?;

                    let mut summaries = Vec::new();
                    for s in rows {
                        summaries.push(s.map_err(|e| e.to_string())?);
                    }
                    Ok(summaries)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::GetSummaries {
                thread_id,
                limit,
                reply,
            } => {
                let res = (|| -> Result<Vec<SummaryEntry>, String> {
                    let mut stmt = if thread_id.is_empty() {
                        self.conn.prepare(
                            "SELECT id, thread_id, summary, key_info, knowledge_gaps, created_at FROM session_summaries 
                             ORDER BY created_at DESC LIMIT ?1"
                        ).map_err(|e| e.to_string())?
                    } else {
                        self.conn.prepare(
                            "SELECT id, thread_id, summary, key_info, knowledge_gaps, created_at FROM session_summaries 
                             WHERE thread_id LIKE ?1 ORDER BY created_at DESC LIMIT ?2"
                        ).map_err(|e| e.to_string())?
                    };

                    let limit_i64 = limit as i64;
                    let summary_mapper = |row: &rusqlite::Row| {
                        Ok(SummaryEntry {
                            id: row.get(0)?,
                            thread_id: row.get(1)?,
                            summary: row.get(2)?,
                            key_info: row.get(3)?,
                            knowledge_gaps: row.get(4)?,
                            created_at: row.get(5)?,
                        })
                    };

                    let summaries = if thread_id.is_empty() {
                        let rows = stmt
                            .query_map(params![limit_i64], summary_mapper)
                            .map_err(|e| e.to_string())?;
                        rows.collect::<Result<Vec<_>, _>>()
                            .map_err(|e| e.to_string())?
                    } else {
                        let pattern = format!("{}%", thread_id);
                        let rows = stmt
                            .query_map(params![pattern, limit_i64], summary_mapper)
                            .map_err(|e| e.to_string())?;
                        rows.collect::<Result<Vec<_>, _>>()
                            .map_err(|e| e.to_string())?
                    };

                    Ok(summaries)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::DeleteSummary { id, reply } => {
                let res = self
                    .conn
                    .execute("DELETE FROM session_summaries WHERE id = ?1", params![id])
                    .map_err(|e| e.to_string())
                    .map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::UpdateThreadMetadata {
                thread_id,
                last_reflection_msg_id,
                reply,
            } => {
                let res = self.conn.execute(
                    "INSERT INTO session_metadata (thread_id, last_reflection_msg_id, last_reflection_time) 
                     VALUES (?1, ?2, CURRENT_TIMESTAMP) 
                     ON CONFLICT(thread_id) DO UPDATE SET 
                        last_reflection_msg_id=excluded.last_reflection_msg_id,
                        last_reflection_time=CURRENT_TIMESTAMP",
                    params![thread_id, last_reflection_msg_id],
                ).map_err(|e| e.to_string()).map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::GetThreadMetadata { thread_id, reply } => {
                let res = (|| -> Result<(Option<i64>, String), String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT last_reflection_msg_id, last_reflection_time FROM session_metadata WHERE thread_id = ?1"
                    ).map_err(|e| e.to_string())?;

                    let result = stmt.query_row(params![thread_id], |row| {
                        let msg_id: Option<i64> = row.get(0)?;
                        let time: String = row.get(1)?;
                        Ok((msg_id, time))
                    });

                    match result {
                        Ok(data) => Ok(data),
                        Err(rusqlite::Error::QueryReturnedNoRows) => {
                            Ok((None, "Never".to_string()))
                        }
                        Err(e) => Err(e.to_string()),
                    }
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::SearchSummaries { query, reply } => {
                let res = (|| -> Result<Vec<String>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT session_summaries.thread_id, session_summaries.summary, session_summaries.key_info 
                         FROM session_summaries 
                         JOIN session_summaries_fts ON session_summaries.id = session_summaries_fts.rowid
                         WHERE session_summaries_fts MATCH ?1 
                         ORDER BY session_summaries_fts.rank LIMIT 20"
                    ).map_err(|e| e.to_string())?;

                    // Escape quotes for FTS
                    let search_pattern = format!("\"{}\"", query.replace("\"", "\"\""));

                    let rows = stmt
                        .query_map(params![search_pattern], |row| {
                            let sid: String = row.get(0)?;
                            let sum: String = row.get(1)?;
                            let key: String = row.get(2)?;
                            Ok(format!("Thread [{}]: {}\nKey Info: {}", sid, sum, key))
                        })
                        .map_err(|e| e.to_string())?;

                    let mut results = Vec::new();
                    for s in rows {
                        results.push(s.map_err(|e| e.to_string())?);
                    }
                    Ok(results)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::FetchSummariesByTimeRange {
                days_ago,
                limit,
                reply,
            } => {
                let res = (|| -> Result<Vec<String>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT thread_id, summary, key_info, created_at FROM session_summaries 
                         WHERE created_at >= datetime('now', '-' || ?1 || ' days')
                         ORDER BY created_at DESC LIMIT ?2"
                    ).map_err(|e| e.to_string())?;

                    let days_str = days_ago.to_string();
                    let limit_i64 = limit as i64;
                    let rows = stmt
                        .query_map(params![days_str, limit_i64], |row| {
                            let sid: String = row.get(0)?;
                            let sum: String = row.get(1)?;
                            let key: String = row.get(2)?;
                            let created_at: String = row.get(3)?;
                            Ok(format!(
                                "[{}] Thread: {}\nSummary: {}\nKey Info: {}",
                                created_at, sid, sum, key
                            ))
                        })
                        .map_err(|e| e.to_string())?;

                    let mut results = Vec::new();
                    for s in rows {
                        results.push(s.map_err(|e| e.to_string())?);
                    }
                    Ok(results)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::GetThreadsNeedingReflection {
                threshold_mins,
                reply,
            } => {
                let res = (|| -> Result<Vec<String>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT latest.thread_id FROM (
                            SELECT thread_id, max(created_at) as last_msg_time, max(id) as max_id
                            FROM messages GROUP BY thread_id
                        ) as latest
                        LEFT JOIN session_metadata md ON latest.thread_id = md.thread_id
                        WHERE (md.last_reflection_msg_id IS NULL OR latest.max_id > md.last_reflection_msg_id)
                          AND (julianday('now') - julianday(latest.last_msg_time)) * 1440 >= ?1"
                    ).map_err(|e| e.to_string())?;

                    let threshold_f64 = threshold_mins as f64;
                    let ids_iter = stmt
                        .query_map(params![threshold_f64], |row| row.get(0))
                        .map_err(|e| e.to_string())?;
                    let mut ids = Vec::new();
                    for id_res in ids_iter {
                        match id_res {
                            Ok(id) => ids.push(id),
                            Err(e) => return Err(e.to_string()),
                        }
                    }
                    Ok(ids)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::GetMessagesSinceReflection { thread_id, reply } => {
                let res = (|| -> GetMessagesSinceReflectionResult {
                    let last_msg_id: Option<i64> = self.conn.query_row(
                        "SELECT last_reflection_msg_id FROM session_metadata WHERE thread_id = ?1",
                        params![thread_id],
                        |row| row.get(0)
                    ).unwrap_or(None);

                    let mut msg_stmt = self.conn.prepare(
                        "SELECT id, role, content, name, tool_calls, tool_call_id, reasoning_content FROM messages WHERE thread_id = ?1 AND (?2 IS NULL OR id > ?2) ORDER BY id ASC"
                    ).map_err(|e| e.to_string())?;

                    let messages_iter = msg_stmt
                        .query_map(params![thread_id, last_msg_id], |row| {
                            let id: i64 = row.get(0)?;
                            let role: String = row.get(1)?;
                            let content_raw: Option<String> = row.get(2)?;
                            let content = content_raw.map(message_content_from_stored);
                            let name: Option<String> = row.get(3)?;
                            let tool_calls_str: Option<String> = row.get(4)?;
                            let tool_calls =
                                tool_calls_str.and_then(|s| serde_json::from_str(&s).ok());
                            let tool_call_id: Option<String> = row.get(5)?;
                            let reasoning_content: Option<String> = row.get(6)?;
                            Ok((
                                id,
                                ChatMessage {
                                    role,
                                    content,
                                    name,
                                    tool_calls,
                                    tool_call_id,
                                    reasoning_content,
                                },
                            ))
                        })
                        .map_err(|e| e.to_string())?;

                    let mut msgs = Vec::new();
                    for msg_res in messages_iter {
                        match msg_res {
                            Ok(msg) => msgs.push(msg),
                            Err(e) => return Err(e.to_string()),
                        }
                    }
                    Ok((msgs, last_msg_id))
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::GetLongTermReflectionState { threshold, reply } => {
                let res = (|| -> Result<(bool, String, i64), String> {
                    let last_lt_time: Option<String> = self.conn.query_row(
                        "SELECT value FROM global_metadata WHERE key = 'last_long_term_reflection'",
                        [],
                        |row| row.get(0)
                    ).unwrap_or(None);

                    let mut should_run = false;
                    let mut last_id = 0;
                    if let Some(time_str) = last_lt_time {
                        last_id = time_str.parse::<i64>().unwrap_or(0);

                        let count: i64 = self
                            .conn
                            .query_row(
                                "SELECT COUNT(*) FROM session_summaries WHERE id > ?1",
                                params![last_id],
                                |row| row.get(0),
                            )
                            .map_err(|e| e.to_string())?;

                        if count > threshold as i64 {
                            should_run = true;
                        }
                    } else {
                        should_run = true;
                    }

                    if !should_run {
                        return Ok((false, String::new(), last_id));
                    }

                    let mut stmt = self.conn.prepare("SELECT id, summary, key_info FROM session_summaries WHERE id > ?1 ORDER BY id ASC").map_err(|e| e.to_string())?;
                    let rows = stmt
                        .query_map(params![last_id], |row| {
                            let id: i64 = row.get(0)?;
                            let sum: String = row.get(1)?;
                            let key: String = row.get(2)?;
                            Ok((id, sum, key))
                        })
                        .map_err(|e| e.to_string())?;

                    let mut summaries_content = String::new();
                    let mut max_id = last_id;
                    for (id, sum, key) in rows.filter_map(Result::ok) {
                        summaries_content
                            .push_str(&format!("Summary:\n{}\nKey Info:\n{}\n\n", sum, key));
                        max_id = id;
                    }
                    Ok((should_run, summaries_content, max_id))
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::SetLongTermReflectionState { max_id, reply } => {
                let res = self.conn.execute(
                    "INSERT INTO global_metadata (key, value) VALUES ('last_long_term_reflection', ?1)
                     ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    params![max_id.to_string()]
                ).map_err(|e| e.to_string()).map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::ReplaceHarnessTodos {
                chat_id,
                items,
                reply,
            } => {
                let res = todo_replace_sqlite_conn(&self.conn, &chat_id, &items);
                let _ = reply.send(res);
            }
            MemoryMessage::LoadHarnessTodos { chat_id, reply } => {
                let res = todo_load_sqlite_conn(&self.conn, &chat_id);
                let _ = reply.send(res);
            }
            MemoryMessage::InsertSubagentTask {
                task_id,
                parent_chat_id,
                child_chat_id,
                display_name,
                agent_name,
                prompt,
                reply,
            } => {
                let res = (|| -> Result<(), String> {
                    let now = Utc::now().timestamp_millis();
                    self.conn
                        .execute(
                            "INSERT INTO subagent_tasks (
                                task_id, parent_chat_id, child_chat_id, display_name, agent_name, prompt,
                                status, result, error, execution_job_id, created_at_ms, updated_at_ms
                            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'running', NULL, NULL, NULL, ?7, ?7)",
                            params![
                                task_id,
                                parent_chat_id,
                                child_chat_id,
                                display_name,
                                agent_name,
                                prompt,
                                now
                            ],
                        )
                        .map_err(|e| format!("insert subagent_tasks: {}", e))?;
                    Ok(())
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::FinalizeSubagentTask {
                task_id,
                parent_chat_id,
                status,
                result,
                error,
                execution_job_id,
                reply,
            } => {
                let res = (|| -> Result<(), String> {
                    let now = Utc::now().timestamp_millis();
                    let n = self
                        .conn
                        .execute(
                            "UPDATE subagent_tasks SET
                                status = ?1,
                                result = ?2,
                                error = ?3,
                                execution_job_id = COALESCE(?4, execution_job_id),
                                updated_at_ms = ?5
                             WHERE task_id = ?6 AND parent_chat_id = ?7",
                            params![
                                status,
                                result,
                                error,
                                execution_job_id,
                                now,
                                task_id,
                                parent_chat_id
                            ],
                        )
                        .map_err(|e| format!("finalize subagent_tasks: {}", e))?;
                    if n == 0 {
                        return Err("subagent_tasks update: no matching row".to_string());
                    }
                    Ok(())
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::ListSubagentTasksForParent {
                parent_chat_id,
                limit,
                reply,
            } => {
                let res = (|| -> Result<Vec<SubagentTaskRecord>, String> {
                    let lim = limit.clamp(1, 200) as i64;
                    let mut stmt = self
                        .conn
                        .prepare(
                            "SELECT task_id, parent_chat_id, child_chat_id, display_name, agent_name, prompt,
                                    status, result, error, execution_job_id, created_at_ms, updated_at_ms
                             FROM subagent_tasks
                             WHERE parent_chat_id = ?1
                             ORDER BY updated_at_ms DESC
                             LIMIT ?2",
                        )
                        .map_err(|e| e.to_string())?;
                    let rows = stmt
                        .query_map(params![parent_chat_id, lim], |row| {
                            Ok(SubagentTaskRecord {
                                task_id: row.get(0)?,
                                parent_chat_id: row.get(1)?,
                                child_chat_id: row.get(2)?,
                                display_name: row.get(3)?,
                                agent_name: row.get(4)?,
                                prompt: row.get(5)?,
                                status: row.get(6)?,
                                result: row.get(7)?,
                                error: row.get(8)?,
                                execution_job_id: row.get(9)?,
                                created_at_ms: row.get(10)?,
                                updated_at_ms: row.get(11)?,
                            })
                        })
                        .map_err(|e| e.to_string())?;
                    let mut out = Vec::new();
                    for r in rows {
                        out.push(r.map_err(|e| e.to_string())?);
                    }
                    Ok(out)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::ListRootThreadsForChannelWithPreviews {
                channel,
                limit,
                reply,
            } => {
                let res = (|| -> Result<Vec<RootThreadListItem>, String> {
                    let ch = channel.trim();
                    if ch.is_empty() {
                        return Ok(Vec::new());
                    }
                    let lim = limit.clamp(1, 500) as i64;
                    // Over-fetch: sub-threads are filtered out; keep a generous cap.
                    let overfetch: i64 = (lim * 4).min(2_000);
                    let like_pat = format!("{ch}:%");
                    let mut stmt = self
                        .conn
                        .prepare(
                            "SELECT m.thread_id, m.id, COALESCE(m.created_at, '') AS ca
                             FROM messages m
                             INNER JOIN (
                                 SELECT thread_id, MAX(id) AS max_id
                                 FROM messages
                                 WHERE thread_id LIKE ?1
                                 GROUP BY thread_id
                             ) t ON m.thread_id = t.thread_id AND m.id = t.max_id
                             ORDER BY m.id DESC
                             LIMIT ?2",
                        )
                        .map_err(|e| e.to_string())?;
                    let mut candidates: Vec<(String, i64, String)> = Vec::new();
                    let rows = stmt
                        .query_map(params![like_pat, overfetch], |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, i64>(1)?,
                                row.get::<_, String>(2)?,
                            ))
                        })
                        .map_err(|e| e.to_string())?;
                    for r in rows {
                        let (tid, last_id, ca) = r.map_err(|e| e.to_string())?;
                        if is_root_session_thread_id(ch, &tid) {
                            candidates.push((tid, last_id, ca));
                        }
                        if candidates.len() as i64 >= lim {
                            break;
                        }
                    }

                    if candidates.is_empty() {
                        return Ok(Vec::new());
                    }
                    let thread_ids: Vec<String> =
                        candidates.iter().map(|(t, _, _)| t.clone()).collect();

                    if thread_ids.is_empty() {
                        return Ok(Vec::new());
                    }

                    let placeholders = thread_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                    let sql = format!(
                        "SELECT m.thread_id, m.content FROM messages m
                         INNER JOIN (
                             SELECT thread_id, MIN(id) AS min_id
                             FROM messages
                             WHERE role = 'user' AND thread_id IN ({placeholders})
                             GROUP BY thread_id
                         ) t ON m.thread_id = t.thread_id AND m.id = t.min_id AND m.role = 'user'"
                    );
                    let mut pst = self.conn.prepare(&sql).map_err(|e| e.to_string())?;
                    let mut q = pst
                        .query(params_from_iter(thread_ids.iter()))
                        .map_err(|e| e.to_string())?;

                    let mut preview_map: HashMap<String, Option<String>> = HashMap::new();
                    while let Some(row) = q.next().map_err(|e| e.to_string())? {
                        let sid: String = row.get(0).map_err(|e| e.to_string())?;
                        let content: String = row.get(1).map_err(|e| e.to_string())?;
                        preview_map.insert(sid, first_user_preview_from_content(content));
                    }

                    let mut out = Vec::new();
                    for (thread_id, last_message_id, created_at) in
                        candidates.into_iter().take(lim as usize)
                    {
                        let raw_prev = match preview_map.get(&thread_id) {
                            None | Some(None) => String::new(),
                            Some(Some(s)) => truncate_thread_preview_line(
                                strip_user_preview_for_thread_list(s).trim(),
                            ),
                        };
                        let preview = if raw_prev.is_empty() {
                            "(no preview)".into()
                        } else {
                            raw_prev
                        };
                        out.push(RootThreadListItem {
                            thread_id,
                            last_message_id,
                            last_activity_ms: sqlite_datetime_to_unix_ms(&created_at),
                            preview,
                        });
                    }
                    Ok(out)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::GetActiveCronsCount { reply } => {
                let res = (|| -> Result<usize, String> {
                    let mut stmt = self
                        .conn
                        .prepare("SELECT count(*) FROM cron_jobs WHERE completed_at_ms IS NULL")
                        .map_err(|e| e.to_string())?;
                    let count: i64 = stmt
                        .query_row([], |row| row.get(0))
                        .map_err(|e| e.to_string())?;
                    Ok(count as usize)
                })();
                let _ = reply.send(res);
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod root_thread_id_tests {
    use super::is_root_session_thread_id;

    #[test]
    fn root_main_terminal_thread() {
        let tid = "terminal:550e8400-e29b-41d4-a716-446655440000:";
        assert!(is_root_session_thread_id("terminal", tid));
    }

    #[test]
    fn not_root_when_subthread() {
        let tid = "terminal:550e8400-e29b-41d4-a716-446655440000:subagent-abc";
        assert!(!is_root_session_thread_id("terminal", tid));
    }

    #[test]
    fn not_root_short_string() {
        assert!(!is_root_session_thread_id("terminal", "nope"));
    }
}
