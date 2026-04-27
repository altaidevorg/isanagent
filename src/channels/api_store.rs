use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension};
use tokio::sync::oneshot;

use crate::memory::SharedReply;
use crate::{ActorError, ActorLogic, NodeHandle};

#[derive(Clone, Debug)]
pub(super) struct StoredResponse {
    pub(super) thread_id: String,
    pub(super) sender_id: String,
    pub(super) model: String,
}

/// One row per distinct conversation thread for a given API user (`sender_id`).
#[derive(Clone, Debug)]
pub(super) struct ThreadListRow {
    pub(super) thread_id: String,
    pub(super) updated_at: i64,
    /// Most recent `response_id` for `POST /v1/responses` chaining.
    pub(super) latest_response_id: String,
}

#[derive(Clone, Debug)]
enum ApiStoreMessage {
    InsertResponse {
        response_id: String,
        previous_response_id: Option<String>,
        stored: StoredResponse,
        created_at: i64,
        reply: SharedReply<Result<(), String>>,
    },
    GetResponse {
        response_id: String,
        reply: SharedReply<Result<Option<StoredResponse>, String>>,
    },
    ListThreadsBySender {
        sender_id: String,
        limit: u32,
        reply: SharedReply<Result<Vec<ThreadListRow>, String>>,
    },
    DeleteThreadResponses {
        thread_id: String,
        sender_id: String,
        reply: SharedReply<Result<usize, String>>,
    },
}

struct SqliteApiResponseStoreActor {
    conn: Connection,
}

impl SqliteApiResponseStoreActor {
    fn new(db_path: impl AsRef<Path>) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open API response store: {}", e))?;
        conn.busy_timeout(Duration::from_secs(5))
            .map_err(|e| format!("Failed to configure API response store busy timeout: {}", e))?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| format!("Failed to enable WAL mode for API response store: {}", e))?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| format!("Failed to tune API response store synchronous mode: {}", e))?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS api_responses (
                response_id TEXT PRIMARY KEY,
                previous_response_id TEXT,
                thread_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("Failed to initialize api_responses table: {}", e))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_api_responses_previous_response_id
             ON api_responses(previous_response_id)",
            [],
        )
        .map_err(|e| {
            format!(
                "Failed to initialize api_responses previous_response_id index: {}",
                e
            )
        })?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_api_responses_created_at
             ON api_responses(created_at)",
            [],
        )
        .map_err(|e| format!("Failed to initialize api_responses created_at index: {}", e))?;

        Ok(Self { conn })
    }
}

#[async_trait]
impl ActorLogic<ApiStoreMessage> for SqliteApiResponseStoreActor {
    fn name(&self) -> String {
        "SqliteApiResponseStoreActor".to_string()
    }

    async fn process(
        &mut self,
        packet: ApiStoreMessage,
    ) -> Result<Option<(String, ApiStoreMessage)>, ActorError> {
        match packet {
            ApiStoreMessage::InsertResponse {
                response_id,
                previous_response_id,
                stored,
                created_at,
                reply,
            } => {
                let result = self
                    .conn
                    .execute(
                        "INSERT INTO api_responses
                            (response_id, previous_response_id, thread_id, sender_id, model, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![
                            response_id,
                            previous_response_id,
                            stored.thread_id,
                            stored.sender_id,
                            stored.model,
                            created_at
                        ],
                    )
                    .map_err(|e| format!("Failed to persist response state: {}", e))
                    .map(|_| ());
                let _ = reply.send(result);
            }
            ApiStoreMessage::GetResponse { response_id, reply } => {
                let result = self
                    .conn
                    .query_row(
                        "SELECT thread_id, sender_id, model
                         FROM api_responses
                         WHERE response_id = ?1",
                        params![response_id],
                        |row| {
                            Ok(StoredResponse {
                                thread_id: row.get(0)?,
                                sender_id: row.get(1)?,
                                model: row.get(2)?,
                            })
                        },
                    )
                    .optional()
                    .map_err(|e| format!("Failed to load response state: {}", e));
                let _ = reply.send(result);
            }
            ApiStoreMessage::ListThreadsBySender {
                sender_id,
                limit,
                reply,
            } => {
                let result = (|| -> Result<Vec<ThreadListRow>, String> {
                    let limit_i64 = i64::from(limit);
                    let mut stmt = self
                        .conn
                        .prepare(
                            "WITH RankedResponses AS (
                                SELECT
                                    thread_id,
                                    response_id,
                                    created_at,
                                    ROW_NUMBER() OVER (
                                        PARTITION BY thread_id ORDER BY created_at DESC
                                    ) AS rn
                                FROM api_responses
                                WHERE sender_id = ?1
                            )
                            SELECT
                                thread_id,
                                created_at AS updated_at,
                                response_id AS latest_response_id
                            FROM RankedResponses
                            WHERE rn = 1
                            ORDER BY updated_at DESC
                            LIMIT ?2",
                        )
                        .map_err(|e| format!("Failed to list threads: {}", e))?;
                    let rows = stmt
                        .query_map(params![sender_id, limit_i64], |row| {
                            Ok(ThreadListRow {
                                thread_id: row.get(0)?,
                                updated_at: row.get(1)?,
                                latest_response_id: row.get(2)?,
                            })
                        })
                        .map_err(|e| format!("Failed to list threads: {}", e))?;
                    let mut out = Vec::new();
                    for r in rows {
                        out.push(r.map_err(|e| format!("Failed to read thread row: {}", e))?);
                    }
                    Ok(out)
                })();
                let _ = reply.send(result);
            }
            ApiStoreMessage::DeleteThreadResponses {
                thread_id,
                sender_id,
                reply,
            } => {
                let result = self
                    .conn
                    .execute(
                        "DELETE FROM api_responses
                         WHERE thread_id = ?1 AND sender_id = ?2",
                        params![thread_id, sender_id],
                    )
                    .map_err(|e| format!("Failed to delete thread responses: {}", e));
                let _ = reply.send(result);
            }
        }

        Ok(None)
    }
}

pub(super) struct ResponseStore {
    node: NodeHandle<ApiStoreMessage>,
}

impl ResponseStore {
    pub(super) fn new(db_path: impl AsRef<Path>) -> Result<Self, String> {
        let actor = SqliteApiResponseStoreActor::new(db_path)?;
        let node = NodeHandle::new(actor, 64, 1, Duration::from_millis(5));
        Ok(Self { node })
    }

    pub(super) async fn insert(
        &self,
        response_id: &str,
        previous_response_id: Option<&str>,
        stored: &StoredResponse,
        created_at: i64,
    ) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let msg = ApiStoreMessage::InsertResponse {
            response_id: response_id.to_string(),
            previous_response_id: previous_response_id.map(str::to_string),
            stored: stored.clone(),
            created_at,
            reply: SharedReply::new(tx),
        };
        self.node
            .send_packet(msg)
            .await
            .map_err(|e| format!("Failed to send response store insert request: {}", e))?;
        rx.await
            .map_err(|_| "API response store actor channel closed".to_string())?
    }

    pub(super) async fn get(&self, response_id: &str) -> Result<Option<StoredResponse>, String> {
        let (tx, rx) = oneshot::channel();
        let msg = ApiStoreMessage::GetResponse {
            response_id: response_id.to_string(),
            reply: SharedReply::new(tx),
        };
        self.node
            .send_packet(msg)
            .await
            .map_err(|e| format!("Failed to send response store get request: {}", e))?;
        rx.await
            .map_err(|_| "API response store actor channel closed".to_string())?
    }

    pub(super) async fn list_threads_by_sender(
        &self,
        sender_id: &str,
        limit: u32,
    ) -> Result<Vec<ThreadListRow>, String> {
        let (tx, rx) = oneshot::channel();
        let msg = ApiStoreMessage::ListThreadsBySender {
            sender_id: sender_id.to_string(),
            limit,
            reply: SharedReply::new(tx),
        };
        self.node
            .send_packet(msg)
            .await
            .map_err(|e| format!("Failed to send list threads request: {}", e))?;
        rx.await
            .map_err(|_| "API response store actor channel closed".to_string())?
    }

    /// Deletes persisted response-chain rows for this thread and sender. Returns rows removed.
    pub(super) async fn delete_thread_responses(
        &self,
        thread_id: &str,
        sender_id: &str,
    ) -> Result<usize, String> {
        let (tx, rx) = oneshot::channel();
        let msg = ApiStoreMessage::DeleteThreadResponses {
            thread_id: thread_id.to_string(),
            sender_id: sender_id.to_string(),
            reply: SharedReply::new(tx),
        };
        self.node
            .send_packet(msg)
            .await
            .map_err(|e| format!("Failed to send delete thread request: {}", e))?;
        rx.await
            .map_err(|_| "API response store actor channel closed".to_string())?
    }
}
