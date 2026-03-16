use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection, OptionalExtension};
use tokio::task;

#[derive(Clone)]
pub(super) struct StoredResponse {
    pub(super) internal_chat_id: String,
    pub(super) sender_id: String,
    pub(super) model: String,
}

pub(super) struct ResponseStore {
    conn: Arc<Mutex<Connection>>,
}

impl ResponseStore {
    pub(super) fn new(db_path: impl AsRef<Path>) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open API response store: {}", e))?;
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

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub(super) async fn insert(
        &self,
        response_id: &str,
        previous_response_id: Option<&str>,
        stored: &StoredResponse,
        created_at: i64,
    ) -> Result<(), String> {
        let conn = self.conn.clone();
        let response_id = response_id.to_string();
        let previous_response_id = previous_response_id.map(str::to_string);
        let stored = stored.clone();
        task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|_| "Failed to lock API response store.".to_string())?;
            conn.execute(
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
            .map_err(|e| format!("Failed to persist response state: {}", e))?;

            Ok(())
        })
        .await
        .map_err(|e| format!("Failed to join API response store task: {}", e))?
    }

    pub(super) async fn get(&self, response_id: &str) -> Result<Option<StoredResponse>, String> {
        let conn = self.conn.clone();
        let response_id = response_id.to_string();
        task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|_| "Failed to lock API response store.".to_string())?;
            conn.query_row(
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
            .map_err(|e| format!("Failed to load response state: {}", e))
        })
        .await
        .map_err(|e| format!("Failed to join API response store task: {}", e))?
    }
}
