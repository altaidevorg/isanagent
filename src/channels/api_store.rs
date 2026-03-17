use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension};
use tokio::sync::oneshot;

use crate::memory::SharedReply;
use crate::{ActorError, ActorLogic, NodeHandle};

#[derive(Clone, Debug)]
pub(super) struct StoredResponse {
    pub(super) internal_chat_id: String,
    pub(super) sender_id: String,
    pub(super) model: String,
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
                internal_chat_id TEXT NOT NULL,
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
                            (response_id, previous_response_id, internal_chat_id, sender_id, model, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![
                            response_id,
                            previous_response_id,
                            stored.internal_chat_id,
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
                        "SELECT internal_chat_id, sender_id, model
                         FROM api_responses
                         WHERE response_id = ?1",
                        params![response_id],
                        |row| {
                            Ok(StoredResponse {
                                internal_chat_id: row.get(0)?,
                                sender_id: row.get(1)?,
                                model: row.get(2)?,
                            })
                        },
                    )
                    .optional()
                    .map_err(|e| format!("Failed to load response state: {}", e));
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
}
