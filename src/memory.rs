use async_trait::async_trait;
use log::debug;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::utils::{ChatMessage, ContentPart, MessageContent};
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

type SessionMessageSinceReflectionRow = (i64, String, String);
type GetMessagesSinceReflectionResult =
    Result<(Vec<SessionMessageSinceReflectionRow>, Option<i64>), String>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SummaryEntry {
    pub id: i64,
    pub session_id: String,
    pub summary: String,
    pub key_info: String,
    pub knowledge_gaps: String,
    pub created_at: String,
}

/// Messages sent to the SqliteMemoryActor
#[derive(Clone, Debug)]
pub enum MemoryMessage {
    AddMessage {
        session_id: String,
        message: ChatMessage,
        reply: SharedReply<Result<(), String>>,
    },
    GetContext {
        session_id: String,
        reply: SharedReply<Result<Vec<ChatMessage>, String>>,
    },
    /// Plain-text preview from the earliest user turn (for session list titles).
    FirstUserMessagePreview {
        session_id: String,
        reply: SharedReply<Result<Option<String>, String>>,
    },
    /// Batch variant: one SQLite round-trip for many `session_id`s (same order as input).
    FirstUserMessagePreviewsBatch {
        session_ids: Vec<String>,
        reply: SharedReply<Result<Vec<Option<String>>, String>>,
    },
    Clear {
        session_id: String,
        keep_last: usize,
        reply: SharedReply<Result<(), String>>,
    },
    // --- Reflection and Summary Messages ---
    AddSummary {
        session_id: String,
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
        session_id: String,
        limit: usize,
        reply: SharedReply<Result<Vec<String>, String>>,
    },
    GetSummaries {
        session_id: String,
        limit: usize,
        reply: SharedReply<Result<Vec<SummaryEntry>, String>>,
    },
    DeleteSummary {
        id: i64,
        reply: SharedReply<Result<(), String>>,
    },
    UpdateSessionMetadata {
        session_id: String,
        last_reflection_msg_id: Option<i64>,
        reply: SharedReply<Result<(), String>>,
    },
    GetSessionMetadata {
        session_id: String,
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
    GetSessionsNeedingReflection {
        threshold_mins: u64,
        reply: SharedReply<Result<Vec<String>, String>>,
    },
    GetMessagesSinceReflection {
        session_id: String,
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
}

/// Persistent SQLite-based memory Actor for agents.
pub struct SqliteMemoryActor {
    conn: Connection,
}

impl SqliteMemoryActor {
    /// Create a new SqliteMemory.
    /// `db_path`: Path to the SQLite DB file. Use ":memory:" for in-memory.
    pub fn new(db_path: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(db_path)?;

        // Create the messages table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
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

        // Create an index to quickly filter by session_id
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_id ON messages (session_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_role_id ON messages (session_id, role, id)",
            [],
        )?;

        // Create the session_summaries table for reflections
        conn.execute(
            "CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                key_info TEXT NOT NULL,
                knowledge_gaps TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Index for session summaries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_summaries_session ON session_summaries (session_id)",
            [],
        )?;

        // Create the session_metadata table to track reflection progress
        conn.execute(
            "CREATE TABLE IF NOT EXISTS session_metadata (
                session_id TEXT PRIMARY KEY,
                last_reflection_msg_id INTEGER,
                last_reflection_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Create the session_summaries virtual table for FTS5
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS session_summaries_fts USING fts5(
                session_id, summary, key_info, knowledge_gaps,
                content='session_summaries', content_rowid='id'
            )",
            [],
        )?;

        // FTS Sync Trigger (Insert)
        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS session_summaries_ai AFTER INSERT ON session_summaries BEGIN
                INSERT INTO session_summaries_fts(rowid, session_id, summary, key_info, knowledge_gaps)
                VALUES (new.id, new.session_id, new.summary, new.key_info, new.knowledge_gaps);
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
                session_id,
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
                    "INSERT INTO messages (session_id, role, content, name, tool_calls, tool_call_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![session_id, message.role, content_str, message.name, tool_calls_str, message.tool_call_id],
                ).map_err(|e| e.to_string()).map(|_| ());

                let _ = reply.send(res);
            }
            MemoryMessage::GetContext { session_id, reply } => {
                let res = (|| -> Result<Vec<ChatMessage>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT role, content, name, tool_calls, tool_call_id FROM messages WHERE session_id = ?1 ORDER BY created_at ASC"
                    ).map_err(|e| e.to_string())?;

                    let message_iter = stmt
                        .query_map(params![session_id], |row| {
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
            MemoryMessage::FirstUserMessagePreview { session_id, reply } => {
                let res = (|| -> Result<Option<String>, String> {
                    let mut stmt = self
                        .conn
                        .prepare(
                            "SELECT content FROM messages WHERE session_id = ?1 AND role = 'user' ORDER BY id ASC LIMIT 1",
                        )
                        .map_err(|e| e.to_string())?;
                    let content_raw: Option<String> = stmt
                        .query_row(params![session_id], |row| row.get(0))
                        .optional()
                        .map_err(|e| e.to_string())?;
                    let Some(s) = content_raw else {
                        return Ok(None);
                    };
                    Ok(first_user_preview_from_content(s))
                })();

                let _ = reply.send(res);
            }
            MemoryMessage::FirstUserMessagePreviewsBatch { session_ids, reply } => {
                let res = (|| -> Result<Vec<Option<String>>, String> {
                    if session_ids.is_empty() {
                        return Ok(Vec::new());
                    }
                    let placeholders = session_ids
                        .iter()
                        .map(|_| "?")
                        .collect::<Vec<_>>()
                        .join(",");
                    let sql = format!(
                        "SELECT m.session_id, m.content FROM messages m
                         INNER JOIN (
                             SELECT session_id, MIN(id) AS min_id
                             FROM messages
                             WHERE role = 'user' AND session_id IN ({placeholders})
                             GROUP BY session_id
                         ) t ON m.session_id = t.session_id AND m.id = t.min_id AND m.role = 'user'"
                    );
                    let mut stmt = self.conn.prepare(&sql).map_err(|e| e.to_string())?;
                    let mut rows = stmt
                        .query(params_from_iter(session_ids.iter()))
                        .map_err(|e| e.to_string())?;

                    let mut map: HashMap<String, Option<String>> = HashMap::new();
                    while let Some(row) = rows.next().map_err(|e| e.to_string())? {
                        let sid: String = row.get(0).map_err(|e| e.to_string())?;
                        let content: String = row.get(1).map_err(|e| e.to_string())?;
                        map.insert(sid, first_user_preview_from_content(content));
                    }

                    Ok(session_ids
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
                session_id,
                keep_last,
                reply,
            } => {
                let res = (|| -> Result<(), String> {
                    // We no longer delete messages here to allow the UI to show full history.
                    // Instead, we just ensure metadata is updated if needed.
                    // If keep_last is 0, it means the user explicitly deleted the chat, so we DO delete.
                    if keep_last == 0 {
                        let tx = self.conn.transaction().map_err(|e| e.to_string())?;
                        // Delete messages
                        tx.execute(
                            "DELETE FROM messages WHERE session_id = ?1",
                            params![session_id],
                        )
                        .map_err(|e| e.to_string())?;

                        // Delete summary
                        tx.execute(
                            "DELETE FROM session_summaries WHERE session_id = ?1",
                            params![session_id],
                        )
                        .map_err(|e| e.to_string())?;

                        // Delete metadata
                        tx.execute(
                            "DELETE FROM session_metadata WHERE session_id = ?1",
                            params![session_id],
                        )
                        .map_err(|e| e.to_string())?;
                        tx.commit().map_err(|e| e.to_string())?;
                    }
                    Ok(())
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::AddSummary {
                session_id,
                summary,
                key_info,
                knowledge_gaps,
                reply,
            } => {
                let res = self.conn.execute(
                    "INSERT INTO session_summaries (session_id, summary, key_info, knowledge_gaps) 
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(session_id) DO UPDATE SET 
                        summary=excluded.summary, 
                        key_info=excluded.key_info, 
                        knowledge_gaps=excluded.knowledge_gaps,
                        created_at=CURRENT_TIMESTAMP",
                    params![session_id, summary, key_info, knowledge_gaps],
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
                session_id,
                limit,
                reply,
            } => {
                let res = (|| -> Result<Vec<String>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT summary, key_info, knowledge_gaps, created_at FROM session_summaries 
                         WHERE session_id LIKE ?1 ORDER BY created_at DESC LIMIT ?2"
                    ).map_err(|e| e.to_string())?;

                    let pattern = format!("{}%", session_id);
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
                session_id,
                limit,
                reply,
            } => {
                let res = (|| -> Result<Vec<SummaryEntry>, String> {
                    let mut stmt = if session_id.is_empty() {
                        self.conn.prepare(
                            "SELECT id, session_id, summary, key_info, knowledge_gaps, created_at FROM session_summaries 
                             ORDER BY created_at DESC LIMIT ?1"
                        ).map_err(|e| e.to_string())?
                    } else {
                        self.conn.prepare(
                            "SELECT id, session_id, summary, key_info, knowledge_gaps, created_at FROM session_summaries 
                             WHERE session_id LIKE ?1 ORDER BY created_at DESC LIMIT ?2"
                        ).map_err(|e| e.to_string())?
                    };

                    let limit_i64 = limit as i64;
                    let summary_mapper = |row: &rusqlite::Row| {
                        Ok(SummaryEntry {
                            id: row.get(0)?,
                            session_id: row.get(1)?,
                            summary: row.get(2)?,
                            key_info: row.get(3)?,
                            knowledge_gaps: row.get(4)?,
                            created_at: row.get(5)?,
                        })
                    };

                    let summaries = if session_id.is_empty() {
                        let rows = stmt
                            .query_map(params![limit_i64], summary_mapper)
                            .map_err(|e| e.to_string())?;
                        rows.collect::<Result<Vec<_>, _>>()
                            .map_err(|e| e.to_string())?
                    } else {
                        let pattern = format!("{}%", session_id);
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
            MemoryMessage::UpdateSessionMetadata {
                session_id,
                last_reflection_msg_id,
                reply,
            } => {
                let res = self.conn.execute(
                    "INSERT INTO session_metadata (session_id, last_reflection_msg_id, last_reflection_time) 
                     VALUES (?1, ?2, CURRENT_TIMESTAMP) 
                     ON CONFLICT(session_id) DO UPDATE SET 
                        last_reflection_msg_id=excluded.last_reflection_msg_id,
                        last_reflection_time=CURRENT_TIMESTAMP",
                    params![session_id, last_reflection_msg_id],
                ).map_err(|e| e.to_string()).map(|_| ());
                let _ = reply.send(res);
            }
            MemoryMessage::GetSessionMetadata { session_id, reply } => {
                let res = (|| -> Result<(Option<i64>, String), String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT last_reflection_msg_id, last_reflection_time FROM session_metadata WHERE session_id = ?1"
                    ).map_err(|e| e.to_string())?;

                    let result = stmt.query_row(params![session_id], |row| {
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
                        "SELECT session_summaries.session_id, session_summaries.summary, session_summaries.key_info 
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
                            Ok(format!("Session [{}]: {}\nKey Info: {}", sid, sum, key))
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
                        "SELECT session_id, summary, key_info, created_at FROM session_summaries 
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
                                "[{}] Session: {}\nSummary: {}\nKey Info: {}",
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
            MemoryMessage::GetSessionsNeedingReflection {
                threshold_mins,
                reply,
            } => {
                let res = (|| -> Result<Vec<String>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT latest.session_id FROM (
                            SELECT session_id, max(created_at) as last_msg_time, max(id) as max_id
                            FROM messages GROUP BY session_id
                        ) as latest
                        LEFT JOIN session_metadata md ON latest.session_id = md.session_id
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
            MemoryMessage::GetMessagesSinceReflection { session_id, reply } => {
                let res = (|| -> Result<(Vec<(i64, String, String)>, Option<i64>), String> {
                    let last_msg_id: Option<i64> = self.conn.query_row(
                        "SELECT last_reflection_msg_id FROM session_metadata WHERE session_id = ?1",
                        params![session_id],
                        |row| row.get(0)
                    ).unwrap_or(None);

                    let mut msg_stmt = self.conn.prepare(
                        "SELECT id, role, content FROM messages WHERE session_id = ?1 AND (?2 IS NULL OR id > ?2) ORDER BY id ASC"
                    ).map_err(|e| e.to_string())?;

                    let messages_iter = msg_stmt
                        .query_map(params![session_id, last_msg_id], |row| {
                            let id: i64 = row.get(0)?;
                            let role: String = row.get(1)?;
                            let content: String = row.get(2)?;
                            Ok((id, role, content))
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
        }
        Ok(None)
    }
}
