use async_trait::async_trait;
use rusqlite::{Connection, params};
use tokio::sync::oneshot;

use crate::{ActorLogic, ActorError};
use crate::utils::ChatMessage;
use std::sync::{Arc, Mutex};
use std::fmt;

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

/// Messages sent to the SqliteMemoryActor
#[derive(Clone, Debug)]
pub enum MemoryMessage {
    AddMessage {
        session_id: String,
        role: String,
        content: String,
        reply: SharedReply<Result<(), String>>,
    },
    GetContext {
        session_id: String,
        reply: SharedReply<Result<Vec<ChatMessage>, String>>,
    },
    Clear {
        session_id: String,
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
    GetRecentSummaries {
        session_id: String,
        limit: usize,
        reply: SharedReply<Result<Vec<String>, String>>,
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
        reply: SharedReply<Result<(Vec<(i64, String, String)>, Option<i64>), String>>,
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
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Create an index to quickly filter by session_id
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_id ON messages (session_id)",
            [],
        )?;

        // Create the session_summaries table for reflections
        conn.execute(
            "CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
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

        // global_metadata table (moved here from reflection.rs)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS global_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
            [],
        )?;

        Ok(Self {
            conn,
        })
    }
}

#[async_trait]
impl ActorLogic<MemoryMessage> for SqliteMemoryActor {
    fn name(&self) -> String {
        "SqliteMemoryActor".to_string()
    }

    async fn process(&mut self, packet: MemoryMessage) -> Result<Option<(String, MemoryMessage)>, ActorError> {
        match packet {
            MemoryMessage::AddMessage { session_id, role, content, reply } => {
                let res = self.conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?1, ?2, ?3)",
                    params![session_id, role, content],
                ).map_err(|e| e.to_string()).map(|_| ());
                
                let _ = reply.send(res);
            }
            MemoryMessage::GetContext { session_id, reply } => {
                let res = (|| -> Result<Vec<ChatMessage>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT role, content FROM messages WHERE session_id = ?1 ORDER BY created_at ASC"
                    ).map_err(|e| e.to_string())?;

                    let message_iter = stmt.query_map(params![session_id], |row| {
                        Ok(ChatMessage {
                            role: row.get(0)?,
                            content: row.get(1)?,
                        })
                    }).map_err(|e| e.to_string())?;

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
            MemoryMessage::Clear { session_id, reply } => {
                let res = self.conn.execute(
                    "DELETE FROM messages WHERE session_id = ?1",
                    params![session_id],
                ).map_err(|e| e.to_string()).map(|_| ());
                
                let _ = reply.send(res);
            }
            MemoryMessage::AddSummary { session_id, summary, key_info, knowledge_gaps, reply } => {
                let res = self.conn.execute(
                    "INSERT INTO session_summaries (session_id, summary, key_info, knowledge_gaps) VALUES (?1, ?2, ?3, ?4)",
                    params![session_id, summary, key_info, knowledge_gaps],
                ).map_err(|e| e.to_string()).map(|_| ());
                
                let _ = reply.send(res);
            }
            MemoryMessage::GetRecentSummaries { session_id, limit, reply } => {
                let res = (|| -> Result<Vec<String>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT summary, key_info, knowledge_gaps, created_at FROM session_summaries 
                         WHERE session_id LIKE ?1 ORDER BY created_at DESC LIMIT ?2"
                    ).map_err(|e| e.to_string())?;

                    let pattern = format!("{}%", session_id);
                    let limit_i64 = limit as i64;
                    let rows = stmt.query_map(params![pattern, limit_i64], |row| {
                        let summary: String = row.get(0)?;
                        let key_info: String = row.get(1)?;
                        let knowledge_gaps: String = row.get(2)?;
                        let created_at: String = row.get(3)?;
                        Ok(format!("[{}] Summary: {}\nKey Info: {}\nGaps: {}", created_at, summary, key_info, knowledge_gaps))
                    }).map_err(|e| e.to_string())?;

                    let mut summaries = Vec::new();
                    for s in rows {
                        summaries.push(s.map_err(|e| e.to_string())?);
                    }
                    Ok(summaries)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::UpdateSessionMetadata { session_id, last_reflection_msg_id, reply } => {
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
                        Err(rusqlite::Error::QueryReturnedNoRows) => Ok((None, "Never".to_string())),
                        Err(e) => Err(e.to_string())
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
                    
                    let rows = stmt.query_map(params![search_pattern], |row| {
                        let sid: String = row.get(0)?;
                        let sum: String = row.get(1)?;
                        let key: String = row.get(2)?;
                        Ok(format!("Session [{}]: {}\nKey Info: {}", sid, sum, key))
                    }).map_err(|e| e.to_string())?;

                    let mut results = Vec::new();
                    for s in rows {
                        results.push(s.map_err(|e| e.to_string())?);
                    }
                    Ok(results)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::FetchSummariesByTimeRange { days_ago, limit, reply } => {
                 let res = (|| -> Result<Vec<String>, String> {
                    let mut stmt = self.conn.prepare(
                        "SELECT session_id, summary, key_info, created_at FROM session_summaries 
                         WHERE created_at >= datetime('now', '-' || ?1 || ' days')
                         ORDER BY created_at DESC LIMIT ?2"
                    ).map_err(|e| e.to_string())?;

                    let days_str = days_ago.to_string();
                    let limit_i64 = limit as i64;
                    let rows = stmt.query_map(params![days_str, limit_i64], |row| {
                        let sid: String = row.get(0)?;
                        let sum: String = row.get(1)?;
                        let key: String = row.get(2)?;
                        let created_at: String = row.get(3)?;
                        Ok(format!("[{}] Session: {}\nSummary: {}\nKey Info: {}", created_at, sid, sum, key))
                    }).map_err(|e| e.to_string())?;

                    let mut results = Vec::new();
                    for s in rows {
                        results.push(s.map_err(|e| e.to_string())?);
                    }
                    Ok(results)
                })();
                let _ = reply.send(res);
            }
            MemoryMessage::GetSessionsNeedingReflection { threshold_mins, reply } => {
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
                    let ids_iter = stmt.query_map(params![threshold_f64], |row| row.get(0)).map_err(|e| e.to_string())?;
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

                    let messages_iter = msg_stmt.query_map(params![session_id, last_msg_id], |row| {
                        let id: i64 = row.get(0)?;
                        let role: String = row.get(1)?;
                        let content: String = row.get(2)?;
                        Ok((id, role, content))
                    }).map_err(|e| e.to_string())?;

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
                        
                        let count: i64 = self.conn.query_row(
                            "SELECT COUNT(*) FROM session_summaries WHERE id > ?1",
                            params![last_id],
                            |row| row.get(0)
                        ).map_err(|e| e.to_string())?;
                        
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
                    let rows = stmt.query_map(params![last_id], |row| {
                        let id: i64 = row.get(0)?;
                        let sum: String = row.get(1)?;
                        let key: String = row.get(2)?;
                        Ok((id, sum, key))
                    }).map_err(|e| e.to_string())?;

                    let mut summaries_content = String::new();
                    let mut max_id = last_id;
                    for row in rows {
                        if let Ok((id, sum, key)) = row {
                            summaries_content.push_str(&format!("Summary:\n{}\nKey Info:\n{}\n\n", sum, key));
                            max_id = id;
                        }
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
