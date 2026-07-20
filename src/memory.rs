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
        let mut guard = match self.0.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(tx) = guard.take() {
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
#[non_exhaustive]
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

/// True when `thread_id` is a main session for `channel` (`<channel>:<chat-id>:`).
///
/// Chat IDs are opaque identifiers owned by the host application.  Older hosts
/// commonly used UUIDs, but requiring that format makes persisted sessions from
/// hosts with prefixed IDs (for example `s-…`) undiscoverable.  The delimiter is
/// still reserved for the thread namespace, so IDs may not contain `:`.
/// Sub-threads such as `terminal:s-abc:subagent-…` return false.
const MAX_ROOT_CHAT_ID_LEN: usize = 512;

fn root_chat_id<'a>(channel: &str, thread_id: &'a str) -> Option<&'a str> {
    let chat_id = thread_id
        .strip_prefix(channel)?
        .strip_prefix(':')?
        .strip_suffix(':')?;

    (!chat_id.is_empty()
        && chat_id.len() <= MAX_ROOT_CHAT_ID_LEN
        && !chat_id.contains(':')
        && !chat_id.chars().any(char::is_control))
    .then_some(chat_id)
}

pub fn is_root_session_thread_id(channel: &str, thread_id: &str) -> bool {
    root_chat_id(channel, thread_id).is_some()
}

/// Parse the opaque `chat_id` from a root `thread_id`, or return `None`.
pub fn chat_id_from_root_thread_id(channel: &str, thread_id: &str) -> Option<String> {
    root_chat_id(channel, thread_id).map(std::string::ToString::to_string)
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
#[non_exhaustive]
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub struct BackgroundJobRecord {
    pub job_id: String,
    pub kind: String,
    pub chat_id: String,
    pub channel: String,
    pub thread_id: Option<String>,
    pub state: String,
    pub payload_json: String,
    pub resume_after_restart: bool,
    pub detached: bool,
    pub last_error: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub struct NotificationRecord {
    pub notification_id: String,
    pub chat_id: String,
    pub channel: String,
    pub thread_id: Option<String>,
    pub kind: String,
    pub title: String,
    pub body: String,
    pub action_kind: Option<String>,
    pub action_payload: Option<String>,
    pub seen_at_ms: Option<i64>,
    pub resolved_at_ms: Option<i64>,
    pub created_at_ms: i64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub struct ClarificationTicketRecord {
    pub ticket_id: String,
    pub job_id: String,
    pub chat_id: String,
    pub channel: String,
    pub thread_id: Option<String>,
    pub tool_call_id: Option<String>,
    pub prompt: String,
    pub choices_json: Option<String>,
    pub response: Option<String>,
    pub status: String,
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
    // Migrate existing tables that were created before display_name / agent_name were added.
    let _ = conn.execute(
        "ALTER TABLE subagent_tasks ADD COLUMN display_name TEXT",
        [],
    );
    let _ = conn.execute("ALTER TABLE subagent_tasks ADD COLUMN agent_name TEXT", []);
    Ok(())
}

pub fn ensure_background_runtime_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS background_jobs (
            job_id TEXT PRIMARY KEY NOT NULL,
            kind TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            thread_id TEXT,
            state TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            resume_after_restart INTEGER NOT NULL DEFAULT 1,
            detached INTEGER NOT NULL DEFAULT 1,
            last_error TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        )",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_background_jobs_state_updated
         ON background_jobs(state, updated_at_ms DESC)",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notifications (
            notification_id TEXT PRIMARY KEY NOT NULL,
            chat_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            thread_id TEXT,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            action_kind TEXT,
            action_payload TEXT,
            seen_at_ms INTEGER,
            resolved_at_ms INTEGER,
            created_at_ms INTEGER NOT NULL
        )",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_notifications_chat_created
         ON notifications(chat_id, created_at_ms DESC)",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS clarification_tickets (
            ticket_id TEXT PRIMARY KEY NOT NULL,
            job_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            thread_id TEXT,
            tool_call_id TEXT,
            prompt TEXT NOT NULL,
            choices_json TEXT,
            response TEXT,
            status TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        )",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_clarification_tickets_job
         ON clarification_tickets(job_id, updated_at_ms DESC)",
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
#[non_exhaustive]
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
#[non_exhaustive]
pub struct SummaryEntry {
    pub id: i64,
    pub thread_id: String,
    pub summary: String,
    pub key_info: String,
    pub knowledge_gaps: String,
    pub created_at: String,
    /// PR-2.2: structured sectional JSON written alongside the legacy summary
    /// row by `WriteSectionsJson` (PR-2). `None` for older rows that predate
    /// the migration or were written by paths that bypass the sectional flow.
    /// `#[serde(default)]` so older serialized blobs deserialize cleanly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sections_json: Option<String>,
}

/// Messages sent to the SqliteMemoryActor
#[derive(Clone, Debug)]
#[non_exhaustive]
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
    /// Tail-truncation: delete every message in `thread_id` that comes *after*
    /// the `keep_user_messages`-th user-role row (counting from 1, in insert
    /// order), keeping that user message and everything before it. Returns the
    /// number of deleted rows.
    ///
    /// This is the rewind counterpart to [`MemoryMessage::Clear`]: `Clear`
    /// trims the *head* (keeps the most recent N rows, for compaction), while
    /// `TruncateAfterUserMessage` trims the *tail* (keeps the oldest rows up to
    /// a given user turn). Host apps use it to implement conversation edit /
    /// retry / checkpoint-rollback — because the backend owns the durable
    /// history, the rewind has to happen here, not in the client.
    ///
    /// Semantics:
    /// - `keep_user_messages >= 1`: keep through the N-th user message
    ///   (inclusive); delete everything strictly after it. If the thread has
    ///   fewer than N user messages this is a no-op (returns `0`).
    /// - `keep_user_messages == 0`: delete the entire thread (equivalent to
    ///   `Clear { keep_last: 0 }`).
    ///
    /// On any real truncation the thread's `session_summaries` and
    /// `session_metadata` are cleared (a rewind invalidates the reflection
    /// derived from the discarded turns), and `tool_result_cache` rows for the
    /// dropped tool_call_ids are pruned — all in the same transaction.
    TruncateAfterUserMessage {
        thread_id: String,
        keep_user_messages: usize,
        reply: SharedReply<Result<usize, String>>,
    },
    // --- Reflection and Summary Messages ---
    AddSummary {
        thread_id: String,
        summary: String,
        key_info: String,
        knowledge_gaps: String,
        reply: SharedReply<Result<(), String>>,
    },
    /// PR-2: write the structured sectional JSON for a session into the
    /// `sections_json` column (added by an idempotent ALTER in
    /// [`SqliteMemoryActor::new`]). Independent of [`AddSummary`] so the legacy
    /// reflection path can keep writing without sections; the compaction site
    /// in `AgentLogic::run_reasoning_loop` sends both back-to-back.
    WriteSectionsJson {
        thread_id: String,
        sections_json: String,
        reply: SharedReply<Result<(), String>>,
    },
    /// PR-7: insert (or replace) a tool result in the cache. Populated by the
    /// agent's tool-execution branch after every tool result is added to memory,
    /// so the `recall_tool_result` tool can later restore the full content.
    /// Upsert semantics — re-running the same tool_call_id replaces the row.
    CacheToolResult {
        tool_call_id: String,
        chat_id: String,
        session_key: String,
        tool_name: String,
        full_content: String,
        compact_summary: String,
        reply: SharedReply<Result<(), String>>,
    },
    /// PR-7: fetch the full content of a previously cached tool result by id.
    /// Returns `Ok(None)` when there's no row (id never cached, or cache cleared).
    FetchToolResult {
        tool_call_id: String,
        reply: SharedReply<Result<Option<String>, String>>,
    },
    /// PR-7.1: overwrite the `content` column of a stored message by its DB id.
    /// Used by `do_compaction` to persist the tool-result swap so that future
    /// iterations (and any consumer of `get_context()`) see the compact
    /// placeholder. The original content is preserved in `tool_result_cache`
    /// and recoverable via `recall_tool_result`.
    UpdateMessageContent {
        message_id: i64,
        new_content: String,
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
    UpsertBackgroundJob {
        record: BackgroundJobRecord,
        reply: SharedReply<Result<(), String>>,
    },
    /// List background jobs, optionally scoped to one host channel before the
    /// result limit is applied.
    ListBackgroundJobs {
        chat_id: Option<String>,
        channel: Option<String>,
        limit: usize,
        reply: SharedReply<Result<Vec<BackgroundJobRecord>, String>>,
    },
    UpdateBackgroundJobState {
        job_id: String,
        state: String,
        last_error: Option<String>,
        reply: SharedReply<Result<(), String>>,
    },
    InsertNotification {
        record: NotificationRecord,
        reply: SharedReply<Result<(), String>>,
    },
    /// List notifications, optionally scoped to one host channel before the
    /// result limit is applied.
    ListNotifications {
        chat_id: Option<String>,
        channel: Option<String>,
        limit: usize,
        unseen_only: bool,
        reply: SharedReply<Result<Vec<NotificationRecord>, String>>,
    },
    MarkNotificationSeen {
        notification_id: String,
        reply: SharedReply<Result<(), String>>,
    },
    ResolveNotification {
        notification_id: String,
        reply: SharedReply<Result<(), String>>,
    },
    UpsertClarificationTicket {
        record: ClarificationTicketRecord,
        reply: SharedReply<Result<(), String>>,
    },
    ResolveClarificationTicket {
        ticket_id: String,
        response: String,
        reply: SharedReply<Result<(), String>>,
    },
    GetClarificationTicket {
        ticket_id: String,
        reply: SharedReply<Result<Option<ClarificationTicketRecord>, String>>,
    },
    ResolveClarificationTicketFull {
        ticket_id: String,
        job_id: String,
        response: String,
        reply: SharedReply<Result<(), String>>,
    },
    /// List clarification tickets with optional filters. `channel` is applied
    /// in SQLite before the limit so one host cannot starve another's inbox.
    ListClarificationTickets {
        job_id: Option<String>,
        chat_id: Option<String>,
        channel: Option<String>,
        status: Option<String>,
        limit: usize,
        reply: SharedReply<Result<Vec<ClarificationTicketRecord>, String>>,
    },
    /// Explicitly mark a background job as completed, resolving associated tickets/notifications.
    DismissBackgroundJob {
        job_id: Option<String>,
        ticket_id: Option<String>,
        reply: SharedReply<Result<(), String>>,
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
        // PR-2: structured sectional summary as JSON. Idempotent ALTER —
        // failures on existing databases that already have the column are expected.
        let _ = conn.execute(
            "ALTER TABLE session_summaries ADD COLUMN sections_json TEXT",
            [],
        );
        // PR-7: cache for tool results. Populated on every tool-result add
        // (see agent/mod.rs) and read by the `recall_tool_result` tool when
        // the LLM wants to recover content that was compacted out of the
        // active conversation. `tool_call_id` is the LLM-supplied id used
        // both as the cache key and as the lookup parameter passed by the
        // recall tool. `compact_summary` is the placeholder text inlined
        // into the swapped tool message.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tool_result_cache (
                tool_call_id TEXT PRIMARY KEY NOT NULL,
                chat_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                full_content TEXT NOT NULL,
                compact_summary TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL
            )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_result_cache_session
             ON tool_result_cache(session_key, created_at_ms DESC)",
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
        ensure_background_runtime_schema(&conn)?;

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
                        "SELECT role, content, name, tool_calls, tool_call_id, reasoning_content FROM messages WHERE thread_id = ?1 ORDER BY id ASC"
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
                                // Not persisted (skip_serializing internal field); a reloaded
                                // message keeps only the "Error:" text already in `content`.
                                is_error: None,
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
            MemoryMessage::TruncateAfterUserMessage {
                thread_id,
                keep_user_messages,
                reply,
            } => {
                let res = (|| -> Result<usize, String> {
                    let tx = self.conn.transaction().map_err(|e| e.to_string())?;

                    // Resolve the row id of the keep-th user message (1-based,
                    // insert order). `None` means "nothing to keep": either an
                    // explicit full-delete request (keep == 0) or the thread
                    // has fewer user messages than requested (no-op).
                    let cutoff: Option<i64> = if keep_user_messages == 0 {
                        None
                    } else {
                        let offset = i64::try_from(keep_user_messages.saturating_sub(1))
                            .map_err(|_| "keep_user_messages is too large".to_string())?;
                        tx.query_row(
                            "SELECT id FROM messages
                             WHERE thread_id = ?1 AND role = 'user'
                             ORDER BY id ASC LIMIT 1 OFFSET ?2",
                            params![thread_id, offset],
                            |row| row.get::<_, i64>(0),
                        )
                        .optional()
                        .map_err(|e| e.to_string())?
                    };

                    let (deleted, dropped_tool_call_ids) = match cutoff {
                        Some(cutoff_id) => {
                            let ids = dropped_tool_call_ids(&tx, &thread_id, Some(cutoff_id))?;
                            let n = tx
                                .execute(
                                    "DELETE FROM messages WHERE thread_id = ?1 AND id > ?2",
                                    params![thread_id, cutoff_id],
                                )
                                .map_err(|e| e.to_string())?;
                            (n, ids)
                        }
                        None if keep_user_messages == 0 => {
                            let ids = dropped_tool_call_ids(&tx, &thread_id, None)?;
                            let n = tx
                                .execute(
                                    "DELETE FROM messages WHERE thread_id = ?1",
                                    params![thread_id],
                                )
                                .map_err(|e| e.to_string())?;
                            (n, ids)
                        }
                        None => (0usize, Vec::new()),
                    };

                    for tcid in &dropped_tool_call_ids {
                        tx.execute(
                            "DELETE FROM tool_result_cache WHERE tool_call_id = ?1",
                            params![tcid],
                        )
                        .map_err(|e| e.to_string())?;
                    }

                    // A rewind discards turns the thread's reflection/summary
                    // were derived from, so on any real truncation (partial or
                    // full) clear both — the next reflection rescans from a
                    // consistent state instead of "remembering" deleted turns.
                    // The no-op path (keep >= 1 but the thread has fewer user
                    // messages than requested) touches neither. Errors here
                    // would leave the thread inconsistent, so propagate.
                    if cutoff.is_some() || keep_user_messages == 0 {
                        tx.execute(
                            "DELETE FROM session_metadata WHERE thread_id = ?1",
                            params![thread_id],
                        )
                        .map_err(|e| e.to_string())?;
                        tx.execute(
                            "DELETE FROM session_summaries WHERE thread_id = ?1",
                            params![thread_id],
                        )
                        .map_err(|e| e.to_string())?;
                    }

                    tx.commit().map_err(|e| e.to_string())?;
                    Ok(deleted)
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
            MemoryMessage::WriteSectionsJson {
                thread_id,
                sections_json,
                reply,
            } => {
                // Updates ZERO rows if the matching `AddSummary` hasn't landed
                // yet — that's intentional. The caller (compaction site) sends
                // `AddSummary` first, then this; if the row vanished between the
                // two sends (e.g. the thread was cleared), the no-op is correct.
                let res = self
                    .conn
                    .execute(
                        "UPDATE session_summaries SET sections_json = ?1 WHERE thread_id = ?2",
                        params![sections_json, thread_id],
                    )
                    .map_err(|e| e.to_string())
                    .map(|_| ());

                let _ = reply.send(res);
            }
            MemoryMessage::CacheToolResult {
                tool_call_id,
                chat_id,
                session_key,
                tool_name,
                full_content,
                compact_summary,
                reply,
            } => {
                let now_ms = Utc::now().timestamp_millis();
                let res = self
                    .conn
                    .execute(
                        "INSERT INTO tool_result_cache
                             (tool_call_id, chat_id, session_key, tool_name, full_content, compact_summary, created_at_ms)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                         ON CONFLICT(tool_call_id) DO UPDATE SET
                            chat_id = excluded.chat_id,
                            session_key = excluded.session_key,
                            tool_name = excluded.tool_name,
                            full_content = excluded.full_content,
                            compact_summary = excluded.compact_summary,
                            created_at_ms = excluded.created_at_ms",
                        params![
                            tool_call_id,
                            chat_id,
                            session_key,
                            tool_name,
                            full_content,
                            compact_summary,
                            now_ms
                        ],
                    )
                    .map_err(|e| e.to_string())
                    .map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::FetchToolResult {
                tool_call_id,
                reply,
            } => {
                let res = self
                    .conn
                    .query_row(
                        "SELECT full_content FROM tool_result_cache WHERE tool_call_id = ?1",
                        params![tool_call_id],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()
                    .map_err(|e| e.to_string());
                let _ = reply.send(res);
            }
            MemoryMessage::UpdateMessageContent {
                message_id,
                new_content,
                reply,
            } => {
                // Updates ZERO rows if `message_id` no longer exists (thread
                // cleared between read and write). The no-op is correct — the
                // swap would have been wasted anyway.
                let res = self
                    .conn
                    .execute(
                        "UPDATE messages SET content = ?1 WHERE id = ?2",
                        params![new_content, message_id],
                    )
                    .map_err(|e| e.to_string())
                    .map(|_| ());
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
                            "SELECT id, thread_id, summary, key_info, knowledge_gaps, created_at, sections_json FROM session_summaries
                             ORDER BY created_at DESC LIMIT ?1"
                        ).map_err(|e| e.to_string())?
                    } else {
                        self.conn.prepare(
                            "SELECT id, thread_id, summary, key_info, knowledge_gaps, created_at, sections_json FROM session_summaries
                             WHERE thread_id LIKE ?1 ORDER BY created_at DESC LIMIT ?2"
                        ).map_err(|e| e.to_string())?
                    };

                    let limit_i64 = limit as i64;
                    let summary_mapper = |row: &rusqlite::Row| {
                        // PR-2.2: project the new `sections_json` column. NULL
                        // for older rows; rusqlite maps SQLite NULL to None
                        // through the Option<String> column type.
                        Ok(SummaryEntry {
                            id: row.get(0)?,
                            thread_id: row.get(1)?,
                            summary: row.get(2)?,
                            key_info: row.get(3)?,
                            knowledge_gaps: row.get(4)?,
                            created_at: row.get(5)?,
                            sections_json: row.get::<_, Option<String>>(6)?,
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
                                    is_error: None,
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
            MemoryMessage::UpsertBackgroundJob { record, reply } => {
                let res = (|| -> Result<(), String> {
                    self.conn.execute(
                        "INSERT INTO background_jobs (
                            job_id, kind, chat_id, channel, thread_id, state, payload_json,
                            resume_after_restart, detached, last_error, created_at_ms, updated_at_ms
                        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                        ON CONFLICT(job_id) DO UPDATE SET
                            kind=excluded.kind, chat_id=excluded.chat_id, channel=excluded.channel,
                            thread_id=excluded.thread_id, state=excluded.state, payload_json=excluded.payload_json,
                            resume_after_restart=excluded.resume_after_restart, detached=excluded.detached,
                            last_error=excluded.last_error, updated_at_ms=excluded.updated_at_ms",
                        params![
                            record.job_id,
                            record.kind,
                            record.chat_id,
                            record.channel,
                            record.thread_id,
                            record.state,
                            record.payload_json,
                            if record.resume_after_restart { 1 } else { 0 },
                            if record.detached { 1 } else { 0 },
                            record.last_error,
                            record.created_at_ms,
                            record.updated_at_ms
                        ],
                    ).map_err(|e| format!("upsert background_jobs: {}", e))?;
                    Ok(())
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::ListBackgroundJobs {
                chat_id,
                channel,
                limit,
                reply,
            } => {
                let res = (|| -> Result<Vec<BackgroundJobRecord>, String> {
                    let lim = limit.clamp(1, 500) as i64;
                    let mut sql =
                        "SELECT job_id, kind, chat_id, channel, thread_id, state, payload_json,
                            resume_after_restart, detached, last_error, created_at_ms, updated_at_ms
                         FROM background_jobs "
                            .to_string();
                    let mut filters = Vec::new();
                    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

                    if let Some(cid) = chat_id {
                        filters.push("chat_id = ?");
                        params_vec.push(Box::new(cid));
                    }
                    if let Some(ch) = channel {
                        filters.push("channel = ?");
                        params_vec.push(Box::new(ch));
                    }

                    if !filters.is_empty() {
                        sql.push_str(" WHERE ");
                        sql.push_str(&filters.join(" AND "));
                    }

                    sql.push_str(" ORDER BY updated_at_ms DESC LIMIT ?");
                    params_vec.push(Box::new(lim));

                    let mut stmt = self.conn.prepare(&sql).map_err(|e| e.to_string())?;
                    let rows = stmt
                        .query_map(params_from_iter(params_vec), |row| {
                            Ok(BackgroundJobRecord {
                                job_id: row.get(0)?,
                                kind: row.get(1)?,
                                chat_id: row.get(2)?,
                                channel: row.get(3)?,
                                thread_id: row.get(4)?,
                                state: row.get(5)?,
                                payload_json: row.get(6)?,
                                resume_after_restart: row.get::<_, i64>(7)? != 0,
                                detached: row.get::<_, i64>(8)? != 0,
                                last_error: row.get(9)?,
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
            MemoryMessage::UpdateBackgroundJobState {
                job_id,
                state,
                last_error,
                reply,
            } => {
                let res = (|| -> Result<(), String> {
                    let now = Utc::now().timestamp_millis();
                    self.conn.execute(
                        "UPDATE background_jobs SET state = ?1, last_error = ?2, updated_at_ms = ?3 WHERE job_id = ?4",
                        params![state, last_error, now, job_id],
                    ).map_err(|e| format!("update background_jobs: {}", e))?;
                    Ok(())
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::InsertNotification { record, reply } => {
                let res = self.conn.execute(
                    "INSERT INTO notifications (
                        notification_id, chat_id, channel, thread_id, kind, title, body, action_kind,
                        action_payload, seen_at_ms, resolved_at_ms, created_at_ms
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                    params![
                        record.notification_id, record.chat_id, record.channel, record.thread_id,
                        record.kind, record.title, record.body, record.action_kind, record.action_payload,
                        record.seen_at_ms, record.resolved_at_ms, record.created_at_ms
                    ],
                ).map_err(|e| format!("insert notifications: {}", e)).map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::ListNotifications {
                chat_id,
                channel,
                limit,
                unseen_only,
                reply,
            } => {
                let res = (|| -> Result<Vec<NotificationRecord>, String> {
                    let lim = limit.clamp(1, 500) as i64;
                    let mut sql = "SELECT notification_id, chat_id, channel, thread_id, kind, title, body, action_kind, action_payload, seen_at_ms, resolved_at_ms, created_at_ms FROM notifications".to_string();
                    let mut filters = Vec::new();
                    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

                    if let Some(cid) = chat_id {
                        filters.push("chat_id = ?");
                        params_vec.push(Box::new(cid));
                    }
                    if let Some(ch) = channel {
                        filters.push("channel = ?");
                        params_vec.push(Box::new(ch));
                    }
                    if unseen_only {
                        filters.push("seen_at_ms IS NULL");
                    }

                    if !filters.is_empty() {
                        sql.push_str(" WHERE ");
                        sql.push_str(&filters.join(" AND "));
                    }

                    sql.push_str(" ORDER BY created_at_ms DESC LIMIT ?");
                    params_vec.push(Box::new(lim));

                    let mut stmt = self.conn.prepare(&sql).map_err(|e| e.to_string())?;
                    let rows = stmt
                        .query_map(params_from_iter(params_vec), |row| {
                            Ok(NotificationRecord {
                                notification_id: row.get(0)?,
                                chat_id: row.get(1)?,
                                channel: row.get(2)?,
                                thread_id: row.get(3)?,
                                kind: row.get(4)?,
                                title: row.get(5)?,
                                body: row.get(6)?,
                                action_kind: row.get(7)?,
                                action_payload: row.get(8)?,
                                seen_at_ms: row.get(9)?,
                                resolved_at_ms: row.get(10)?,
                                created_at_ms: row.get(11)?,
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
            MemoryMessage::MarkNotificationSeen {
                notification_id,
                reply,
            } => {
                let now = Utc::now().timestamp_millis();
                let res = self.conn.execute(
                    "UPDATE notifications SET seen_at_ms = COALESCE(seen_at_ms, ?1) WHERE notification_id = ?2",
                    params![now, notification_id],
                ).map_err(|e| format!("mark notification seen: {}", e)).map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::ResolveNotification {
                notification_id,
                reply,
            } => {
                let now = Utc::now().timestamp_millis();
                let res = self.conn.execute(
                    "UPDATE notifications SET resolved_at_ms = COALESCE(resolved_at_ms, ?1) WHERE notification_id = ?2",
                    params![now, notification_id],
                ).map_err(|e| format!("resolve notification: {}", e)).map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::UpsertClarificationTicket { record, reply } => {
                let res = self.conn.execute(
                    "INSERT INTO clarification_tickets (
                        ticket_id, job_id, chat_id, channel, thread_id, tool_call_id, prompt, choices_json, response,
                        status, created_at_ms, updated_at_ms
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                    ON CONFLICT(ticket_id) DO UPDATE SET
                        response = excluded.response, status = excluded.status, updated_at_ms = excluded.updated_at_ms",
                    params![
                        record.ticket_id, record.job_id, record.chat_id, record.channel, record.thread_id,
                        record.tool_call_id, record.prompt, record.choices_json, record.response, record.status,
                        record.created_at_ms, record.updated_at_ms
                    ],
                ).map_err(|e| format!("upsert clarification_tickets: {}", e)).map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::ResolveClarificationTicket {
                ticket_id,
                response,
                reply,
            } => {
                let now = Utc::now().timestamp_millis();
                let res = self.conn.execute(
                    "UPDATE clarification_tickets SET response = ?1, status = 'answered', updated_at_ms = ?2 WHERE ticket_id = ?3",
                    params![response, now, ticket_id],
                ).map_err(|e| format!("resolve clarification ticket: {}", e)).map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::GetClarificationTicket { ticket_id, reply } => {
                let res = self.conn.query_row(
                        "SELECT ticket_id, job_id, chat_id, channel, thread_id, tool_call_id, prompt, choices_json, response,
                                status, created_at_ms, updated_at_ms
                         FROM clarification_tickets WHERE ticket_id = ?1",
                        params![ticket_id],
                        |row| {
                            Ok(ClarificationTicketRecord {
                                ticket_id: row.get(0)?,
                                job_id: row.get(1)?,
                                chat_id: row.get(2)?,
                                channel: row.get(3)?,
                                thread_id: row.get(4)?,
                                tool_call_id: row.get(5)?,
                                prompt: row.get(6)?,
                                choices_json: row.get(7)?,
                                response: row.get(8)?,
                                status: row.get(9)?,
                                created_at_ms: row.get(10)?,
                                updated_at_ms: row.get(11)?,
                            })
                        },
                    ).optional().map_err(|e| e.to_string());
                let _ = reply.send(res);
            }
            MemoryMessage::ResolveClarificationTicketFull {
                ticket_id,
                job_id,
                response,
                reply,
            } => {
                let res = (|| -> Result<(), String> {
                    let now = Utc::now().timestamp_millis();
                    let tx = self.conn.transaction().map_err(|e| e.to_string())?;

                    // Claim the waiting ticket before making any state changes. This is
                    // the single-writer gate for a user reply: a second delivery must not
                    // resume the same background job a second time.
                    let stored_job_id: Option<String> = tx
                        .query_row(
                            "SELECT job_id FROM clarification_tickets
                             WHERE ticket_id = ?1 AND status = 'waiting'",
                            params![ticket_id],
                            |row| row.get(0),
                        )
                        .optional()
                        .map_err(|e| format!("load clarification ticket: {}", e))?;
                    let stored_job_id = stored_job_id.ok_or_else(|| {
                        "clarification ticket was not found or is no longer waiting".to_string()
                    })?;
                    if stored_job_id != job_id {
                        return Err(
                            "clarification ticket does not belong to this background job"
                                .to_string(),
                        );
                    }

                    // 1. Resolve the claimed ticket. Keep the status predicate here as a
                    // defensive compare-and-set in case this handler changes in the future.
                    let ticket_rows = tx
                        .execute(
                            "UPDATE clarification_tickets
                             SET response = ?1, status = 'answered', updated_at_ms = ?2
                             WHERE ticket_id = ?3 AND status = 'waiting'",
                            params![response, now, ticket_id],
                        )
                        .map_err(|e| format!("resolve clarification ticket: {}", e))?;
                    if ticket_rows != 1 {
                        return Err("clarification ticket could not be claimed".to_string());
                    }

                    // 2. Resolve related notifications
                    tx.execute(
                        "UPDATE notifications SET resolved_at_ms = ?1 WHERE action_payload = ?2 AND resolved_at_ms IS NULL",
                        params![now, ticket_id],
                    ).map_err(|e| format!("resolve related notifications: {}", e))?;

                    // 3. Update job state to running
                    let job_rows = tx
                        .execute(
                            "UPDATE background_jobs SET state = 'running', updated_at_ms = ?1 WHERE job_id = ?2",
                            params![now, job_id],
                        )
                        .map_err(|e| format!("update job state to running: {}", e))?;
                    if job_rows != 1 {
                        return Err("background job was not found".to_string());
                    }

                    tx.commit().map_err(|e| e.to_string())?;
                    Ok(())
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::DismissBackgroundJob {
                job_id,
                ticket_id,
                reply,
            } => {
                let res = (|| -> Result<(), String> {
                    let now = Utc::now().timestamp_millis();

                    let mut target_job_id = job_id;
                    if target_job_id.is_none() {
                        if let Some(tid) = ticket_id.as_ref() {
                            let jid: Option<String> = self
                                .conn
                                .query_row(
                                    "SELECT job_id FROM clarification_tickets WHERE ticket_id = ?1",
                                    params![tid],
                                    |row| row.get(0),
                                )
                                .optional()
                                .map_err(|e| e.to_string())?
                                .flatten();
                            target_job_id = jid;
                        }
                    }

                    let jid = match target_job_id {
                        Some(j) => j,
                        None => return Ok(()), // Nothing to dismiss
                    };

                    // 1. Mark job as completed
                    self.conn.execute(
                        "UPDATE background_jobs SET state = 'completed', updated_at_ms = ?1 WHERE job_id = ?2",
                        params![now, jid],
                    ).map_err(|e| format!("dismiss background_job: {}", e))?;

                    // 2. Resolve any associated clarification tickets
                    let mut stmt = self.conn.prepare(
                        "SELECT ticket_id FROM clarification_tickets WHERE job_id = ?1 AND status = 'waiting'"
                    ).map_err(|e| e.to_string())?;
                    let ticket_ids: Vec<String> = stmt
                        .query_map(params![jid], |row| row.get(0))
                        .map_err(|e| e.to_string())?
                        .filter_map(Result::ok)
                        .collect();

                    for tid in ticket_ids {
                        self.conn.execute(
                            "UPDATE clarification_tickets SET response = 'Dismissed', status = 'answered', updated_at_ms = ?1 WHERE ticket_id = ?2",
                            params![now, tid],
                        ).map_err(|e| format!("dismiss ticket: {}", e))?;

                        // Resolve notifications for this ticket
                        self.conn.execute(
                            "UPDATE notifications SET resolved_at_ms = COALESCE(resolved_at_ms, ?1) WHERE action_payload = ?2 AND kind = 'clarification_ticket'",
                            params![now, tid],
                        ).map_err(|e| format!("resolve notifications for ticket: {}", e))?;
                    }

                    Ok(())
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::ListClarificationTickets {
                job_id,
                chat_id,
                channel,
                status,
                limit,
                reply,
            } => {
                let res = (|| -> Result<Vec<ClarificationTicketRecord>, String> {
                    let lim = limit.clamp(1, 500) as i64;
                    let mut sql = "SELECT ticket_id, job_id, chat_id, channel, thread_id, tool_call_id, prompt, choices_json, response, status, created_at_ms, updated_at_ms FROM clarification_tickets".to_string();
                    let mut filters = Vec::new();
                    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

                    if let Some(jid) = job_id {
                        filters.push("job_id = ?");
                        params_vec.push(Box::new(jid));
                    }
                    if let Some(cid) = chat_id {
                        filters.push("chat_id = ?");
                        params_vec.push(Box::new(cid));
                    }
                    if let Some(ch) = channel {
                        filters.push("channel = ?");
                        params_vec.push(Box::new(ch));
                    }
                    if let Some(s) = status {
                        filters.push("status = ?");
                        params_vec.push(Box::new(s));
                    }

                    if !filters.is_empty() {
                        sql.push_str(" WHERE ");
                        sql.push_str(&filters.join(" AND "));
                    }

                    sql.push_str(" ORDER BY updated_at_ms DESC LIMIT ?");
                    params_vec.push(Box::new(lim));

                    let mut stmt = self.conn.prepare(&sql).map_err(|e| e.to_string())?;
                    let rows = stmt
                        .query_map(params_from_iter(params_vec), |row| {
                            Ok(ClarificationTicketRecord {
                                ticket_id: row.get(0)?,
                                job_id: row.get(1)?,
                                chat_id: row.get(2)?,
                                channel: row.get(3)?,
                                thread_id: row.get(4)?,
                                tool_call_id: row.get(5)?,
                                prompt: row.get(6)?,
                                choices_json: row.get(7)?,
                                response: row.get(8)?,
                                status: row.get(9)?,
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
        }
        Ok(None)
    }
}

/// Collect the `tool_call_id`s of messages a rewind is about to remove, so
/// their cached tool results can be pruned in the same transaction. Tool
/// results live on `role='tool'` rows carrying `tool_call_id`. `after =
/// Some(id)` restricts to rows with `id > after`; `None` covers the whole
/// thread (used by the full-wipe path).
fn dropped_tool_call_ids(
    tx: &rusqlite::Transaction,
    thread_id: &str,
    after: Option<i64>,
) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    match after {
        Some(id) => {
            let mut stmt = tx
                .prepare(
                    "SELECT tool_call_id FROM messages
                     WHERE thread_id = ?1 AND id > ?2 AND tool_call_id IS NOT NULL",
                )
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map(params![thread_id, id], |r| r.get::<_, Option<String>>(0))
                .map_err(|e| e.to_string())?;
            for row in rows {
                if let Ok(Some(v)) = row {
                    out.push(v);
                }
            }
        }
        None => {
            let mut stmt = tx
                .prepare(
                    "SELECT tool_call_id FROM messages
                     WHERE thread_id = ?1 AND tool_call_id IS NOT NULL",
                )
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map(params![thread_id], |r| r.get::<_, Option<String>>(0))
                .map_err(|e| e.to_string())?;
            for row in rows {
                if let Ok(Some(v)) = row {
                    out.push(v);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod root_thread_id_tests {
    use super::{
        chat_id_from_root_thread_id, is_root_session_thread_id, BackgroundJobRecord,
        ClarificationTicketRecord, MemoryMessage, NotificationRecord, SharedReply,
        SqliteMemoryActor,
    };
    use crate::session::SessionManager;
    use crate::traits::Memory;
    use crate::NodeHandle;
    use std::time::Duration;

    #[test]
    fn root_main_terminal_thread() {
        let tid = "terminal:550e8400-e29b-41d4-a716-446655440000:";
        assert!(is_root_session_thread_id("terminal", tid));
    }

    #[test]
    fn root_main_thread_accepts_opaque_chat_id() {
        let tid = "tauri:s-01JQ8QBK7T3SZ8RMS4ZK93XMBQ:";
        assert!(is_root_session_thread_id("tauri", tid));
        assert_eq!(
            chat_id_from_root_thread_id("tauri", tid).as_deref(),
            Some("s-01JQ8QBK7T3SZ8RMS4ZK93XMBQ")
        );
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

    #[test]
    fn not_root_when_chat_id_is_empty_or_contains_delimiter() {
        assert!(!is_root_session_thread_id("tauri", "tauri::"));
        assert!(!is_root_session_thread_id("tauri", "tauri:s-abc:extra:"));
    }

    #[tokio::test]
    async fn channel_scoped_inbox_query_filters_before_limit() {
        let actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let node = NodeHandle::new(actor, 16, 1, Duration::from_millis(1));

        for (id, channel, created_at_ms) in [
            ("tauri-notification", "tauri", 1_i64),
            ("terminal-notification", "terminal", 2_i64),
        ] {
            let (tx, rx) = tokio::sync::oneshot::channel();
            node.send_packet(MemoryMessage::InsertNotification {
                record: NotificationRecord {
                    notification_id: id.to_string(),
                    chat_id: "chat-1".to_string(),
                    channel: channel.to_string(),
                    thread_id: None,
                    kind: "test".to_string(),
                    title: "Test".to_string(),
                    body: "Test".to_string(),
                    action_kind: None,
                    action_payload: None,
                    seen_at_ms: None,
                    resolved_at_ms: None,
                    created_at_ms,
                },
                reply: SharedReply::new(tx),
            })
            .await
            .expect("enqueue notification");
            rx.await
                .expect("notification actor reply")
                .expect("insert notification");
        }

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::ListNotifications {
            chat_id: None,
            channel: Some("tauri".to_string()),
            limit: 1,
            unseen_only: false,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue list");
        let rows = rx
            .await
            .expect("notification actor reply")
            .expect("list notifications");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].notification_id, "tauri-notification");

        for (id, channel, updated_at_ms) in [
            ("tauri-job", "tauri", 1_i64),
            ("terminal-job", "terminal", 2_i64),
        ] {
            let (tx, rx) = tokio::sync::oneshot::channel();
            node.send_packet(MemoryMessage::UpsertBackgroundJob {
                record: BackgroundJobRecord {
                    job_id: id.to_string(),
                    kind: "test".to_string(),
                    chat_id: "chat-1".to_string(),
                    channel: channel.to_string(),
                    thread_id: None,
                    state: "waiting".to_string(),
                    payload_json: "{}".to_string(),
                    resume_after_restart: false,
                    detached: false,
                    last_error: None,
                    created_at_ms: updated_at_ms,
                    updated_at_ms,
                },
                reply: SharedReply::new(tx),
            })
            .await
            .expect("enqueue background job");
            rx.await
                .expect("background job actor reply")
                .expect("insert background job");
        }

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::ListBackgroundJobs {
            chat_id: None,
            channel: Some("tauri".to_string()),
            limit: 1,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue background job list");
        let jobs = rx
            .await
            .expect("background job actor reply")
            .expect("list background jobs");
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].job_id, "tauri-job");

        for (id, channel, updated_at_ms) in [
            ("tauri-ticket", "tauri", 1_i64),
            ("terminal-ticket", "terminal", 2_i64),
        ] {
            let (tx, rx) = tokio::sync::oneshot::channel();
            node.send_packet(MemoryMessage::UpsertClarificationTicket {
                record: ClarificationTicketRecord {
                    ticket_id: id.to_string(),
                    job_id: id.to_string(),
                    chat_id: "chat-1".to_string(),
                    channel: channel.to_string(),
                    thread_id: None,
                    tool_call_id: None,
                    prompt: "Test".to_string(),
                    choices_json: None,
                    response: None,
                    status: "waiting".to_string(),
                    created_at_ms: updated_at_ms,
                    updated_at_ms,
                },
                reply: SharedReply::new(tx),
            })
            .await
            .expect("enqueue clarification ticket");
            rx.await
                .expect("clarification ticket actor reply")
                .expect("insert clarification ticket");
        }

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::ListClarificationTickets {
            job_id: None,
            chat_id: None,
            channel: Some("tauri".to_string()),
            status: None,
            limit: 1,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue clarification ticket list");
        let tickets = rx
            .await
            .expect("clarification ticket actor reply")
            .expect("list clarification tickets");
        assert_eq!(tickets.len(), 1);
        assert_eq!(tickets[0].ticket_id, "tauri-ticket");
    }

    #[tokio::test]
    async fn resolving_a_clarification_ticket_claims_it_once() {
        let actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let node = NodeHandle::new(actor, 16, 1, Duration::from_millis(1));

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::UpsertBackgroundJob {
            record: BackgroundJobRecord {
                job_id: "job-1".to_string(),
                kind: "test".to_string(),
                chat_id: "chat-1".to_string(),
                channel: "tauri".to_string(),
                thread_id: None,
                state: "waiting".to_string(),
                payload_json: "{}".to_string(),
                resume_after_restart: false,
                detached: false,
                last_error: None,
                created_at_ms: 1,
                updated_at_ms: 1,
            },
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue background job");
        rx.await
            .expect("background job actor reply")
            .expect("insert background job");

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::UpsertClarificationTicket {
            record: ClarificationTicketRecord {
                ticket_id: "ticket-1".to_string(),
                job_id: "job-1".to_string(),
                chat_id: "chat-1".to_string(),
                channel: "tauri".to_string(),
                thread_id: None,
                tool_call_id: None,
                prompt: "Continue?".to_string(),
                choices_json: None,
                response: None,
                status: "waiting".to_string(),
                created_at_ms: 1,
                updated_at_ms: 1,
            },
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue clarification ticket");
        rx.await
            .expect("clarification ticket actor reply")
            .expect("insert clarification ticket");

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::ResolveClarificationTicketFull {
            ticket_id: "ticket-1".to_string(),
            job_id: "wrong-job".to_string(),
            response: "yes".to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue mismatched resolve");
        let err = rx
            .await
            .expect("mismatched resolve actor reply")
            .expect_err("mismatched job must fail");
        assert!(err.contains("does not belong"));

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::ResolveClarificationTicketFull {
            ticket_id: "ticket-1".to_string(),
            job_id: "job-1".to_string(),
            response: "yes".to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue resolve");
        rx.await
            .expect("resolve actor reply")
            .expect("first resolve succeeds");

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::ResolveClarificationTicketFull {
            ticket_id: "ticket-1".to_string(),
            job_id: "job-1".to_string(),
            response: "duplicate".to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue duplicate resolve");
        let err = rx
            .await
            .expect("duplicate resolve actor reply")
            .expect_err("duplicate resolve must fail");
        assert!(err.contains("no longer waiting"));

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::GetClarificationTicket {
            ticket_id: "ticket-1".to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue ticket lookup");
        let ticket = rx
            .await
            .expect("ticket lookup actor reply")
            .expect("ticket lookup")
            .expect("ticket exists");
        assert_eq!(ticket.status, "answered");
        assert_eq!(ticket.response.as_deref(), Some("yes"));

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::ListBackgroundJobs {
            chat_id: Some("chat-1".to_string()),
            channel: Some("tauri".to_string()),
            limit: 1,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("enqueue job lookup");
        let jobs = rx
            .await
            .expect("job lookup actor reply")
            .expect("job lookup");
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].state, "running");
    }

    #[tokio::test]
    async fn get_context_returns_messages_in_insert_order() {
        let actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let node = NodeHandle::new(actor, 16, 1, Duration::from_millis(1));
        let manager = SessionManager::new(node);
        let mut session = manager
            .get_session("terminal:550e8400-e29b-41d4-a716-446655440000:")
            .await
            .expect("session");

        session
            .add_message(crate::utils::ChatMessage::user("first"))
            .await
            .expect("add first");
        session
            .add_message(crate::utils::ChatMessage::assistant("second"))
            .await
            .expect("add second");

        let msgs = session.get_context().await.expect("context");
        let contents: Vec<String> = msgs
            .iter()
            .map(|m| {
                m.content
                    .as_ref()
                    .map(|c| c.text_content())
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(contents, vec!["first".to_string(), "second".to_string()]);
    }

    #[tokio::test]
    async fn truncate_after_user_message_rewinds_to_nth_user_turn() {
        let actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let node = NodeHandle::new(actor, 16, 1, Duration::from_millis(1));
        let manager = SessionManager::new(node.clone());
        let thread_id = "terminal:550e8400-e29b-41d4-a716-446655440000:";
        let mut session = manager.get_session(thread_id).await.expect("session");

        session
            .add_message(crate::utils::ChatMessage::user("u1"))
            .await
            .expect("add u1");
        session
            .add_message(crate::utils::ChatMessage::assistant("a1"))
            .await
            .expect("add a1");
        session
            .add_message(crate::utils::ChatMessage::user("u2"))
            .await
            .expect("add u2");
        session
            .add_message(crate::utils::ChatMessage::assistant("a2"))
            .await
            .expect("add a2");

        // Rewind to the first user message: drop a1, u2, a2 (3 rows).
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::TruncateAfterUserMessage {
            thread_id: thread_id.to_string(),
            keep_user_messages: 1,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send truncate");
        let deleted = rx.await.expect("reply").expect("truncate ok");
        assert_eq!(deleted, 3);

        let contents: Vec<String> = session
            .get_context()
            .await
            .expect("context")
            .iter()
            .map(|m| {
                m.content
                    .as_ref()
                    .map(|c| c.text_content())
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(contents, vec!["u1".to_string()]);

        // keep_user_messages beyond the user count is a no-op.
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::TruncateAfterUserMessage {
            thread_id: thread_id.to_string(),
            keep_user_messages: 5,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send truncate");
        let deleted = rx.await.expect("reply").expect("truncate ok");
        assert_eq!(deleted, 0);

        // keep_user_messages == 0 wipes the thread.
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::TruncateAfterUserMessage {
            thread_id: thread_id.to_string(),
            keep_user_messages: 0,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send truncate");
        let deleted = rx.await.expect("reply").expect("truncate ok");
        assert_eq!(deleted, 1);
        assert!(session.get_context().await.expect("context").is_empty());
    }

    #[tokio::test]
    async fn truncate_prunes_dropped_tool_cache_and_stale_metadata() {
        let actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let node = NodeHandle::new(actor, 16, 1, Duration::from_millis(1));
        let manager = SessionManager::new(node.clone());
        let thread_id = "terminal:660e8400-e29b-41d4-a716-446655440099:";
        let mut session = manager.get_session(thread_id).await.expect("session");

        // Layout (ids 1..4): u1, a tool-result row carrying tool_call_id, u2, a2.
        // Rewinding to u1 (keep=1) drops ids 2..4 — the tool row feeds the cache
        // prune; a2 (id=4) backs the stale reflection pointer.
        session
            .add_message(crate::utils::ChatMessage::user("u1"))
            .await
            .expect("add u1");
        session
            .add_message(crate::utils::ChatMessage {
                role: "tool".to_string(),
                content: Some(crate::utils::MessageContent::Text(
                    "tool result".to_string(),
                )),
                name: None,
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
                is_error: None,
            })
            .await
            .expect("add tool");
        session
            .add_message(crate::utils::ChatMessage::user("u2"))
            .await
            .expect("add u2");
        session
            .add_message(crate::utils::ChatMessage::assistant("a2"))
            .await
            .expect("add a2");

        // Cache the tool result and point reflection at a2 (id=4, doomed).
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::CacheToolResult {
            tool_call_id: "call_1".to_string(),
            chat_id: thread_id.to_string(),
            session_key: thread_id.to_string(),
            tool_name: "demo".to_string(),
            full_content: "full tool result".to_string(),
            compact_summary: "summary".to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send cache");
        rx.await.expect("reply").expect("cache ok");

        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::UpdateThreadMetadata {
            thread_id: thread_id.to_string(),
            last_reflection_msg_id: Some(4),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send metadata");
        rx.await.expect("reply").expect("metadata ok");

        // Add a thread summary — a rewind must invalidate it (it was derived
        // from the turns being discarded).
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::AddSummary {
            thread_id: thread_id.to_string(),
            summary: "stale summary".to_string(),
            key_info: "k".to_string(),
            knowledge_gaps: "g".to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send summary");
        rx.await.expect("reply").expect("summary ok");

        // Sanity: cache is present before the rewind.
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::FetchToolResult {
            tool_call_id: "call_1".to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send fetch");
        assert_eq!(
            rx.await.expect("reply").expect("fetch ok"),
            Some("full tool result".to_string())
        );

        // Sanity: summary is present before the rewind.
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::GetRecentSummaries {
            thread_id: thread_id.to_string(),
            limit: 10,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send get-summaries");
        let summaries = rx.await.expect("reply").expect("summaries ok");
        assert_eq!(summaries.len(), 1);

        // Rewind to u1 — drops ids 2..4.
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::TruncateAfterUserMessage {
            thread_id: thread_id.to_string(),
            keep_user_messages: 1,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send truncate");
        let deleted = rx.await.expect("reply").expect("truncate ok");
        assert_eq!(deleted, 3);

        // Cached tool result for "call_1" was pruned with its message.
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::FetchToolResult {
            tool_call_id: "call_1".to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send fetch");
        assert_eq!(rx.await.expect("reply").expect("fetch ok"), None);

        // Reflection pointer (id=4) was stale → metadata reset to default.
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::GetThreadMetadata {
            thread_id: thread_id.to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send get-metadata");
        let (last_id, _time) = rx.await.expect("reply").expect("metadata ok");
        assert_eq!(last_id, None);

        // Summary was derived from the discarded turns → cleared.
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::GetRecentSummaries {
            thread_id: thread_id.to_string(),
            limit: 10,
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send get-summaries");
        let summaries = rx.await.expect("reply").expect("summaries ok");
        assert!(summaries.is_empty());
    }
}
