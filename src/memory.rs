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
        }
        Ok(None)
    }
}
