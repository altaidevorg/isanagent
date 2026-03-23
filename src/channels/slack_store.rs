use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension};
use tokio::sync::oneshot;

use crate::memory::SharedReply;
use crate::{ActorError, ActorLogic, NodeHandle};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StoredSlackUserProfile {
    pub(crate) display_name: String,
    pub(crate) fetched_at_unix_secs: i64,
}

#[derive(Clone, Debug)]
enum SlackUserStoreMessage {
    UpsertProfile {
        user_id: String,
        stored: StoredSlackUserProfile,
        reply: SharedReply<Result<(), String>>,
    },
    GetProfile {
        user_id: String,
        reply: SharedReply<Result<Option<StoredSlackUserProfile>, String>>,
    },
}

struct SqliteSlackUserProfileStoreActor {
    conn: Connection,
}

impl SqliteSlackUserProfileStoreActor {
    fn new(db_path: impl AsRef<Path>) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open Slack user profile store: {}", e))?;
        conn.busy_timeout(Duration::from_secs(5)).map_err(|e| {
            format!(
                "Failed to configure Slack user profile store busy timeout: {}",
                e
            )
        })?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| {
                format!(
                    "Failed to enable WAL mode for Slack user profile store: {}",
                    e
                )
            })?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| {
                format!(
                    "Failed to tune Slack user profile store synchronous mode: {}",
                    e
                )
            })?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS slack_user_profiles (
                user_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                fetched_at_unix_secs INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("Failed to initialize slack_user_profiles table: {}", e))?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_slack_user_profiles_fetched_at
             ON slack_user_profiles(fetched_at_unix_secs)",
            [],
        )
        .map_err(|e| {
            format!(
                "Failed to initialize slack_user_profiles fetched_at index: {}",
                e
            )
        })?;

        Ok(Self { conn })
    }
}

#[async_trait]
impl ActorLogic<SlackUserStoreMessage> for SqliteSlackUserProfileStoreActor {
    fn name(&self) -> String {
        "SqliteSlackUserProfileStoreActor".to_string()
    }

    async fn process(
        &mut self,
        packet: SlackUserStoreMessage,
    ) -> Result<Option<(String, SlackUserStoreMessage)>, ActorError> {
        match packet {
            SlackUserStoreMessage::UpsertProfile {
                user_id,
                stored,
                reply,
            } => {
                let result = self
                    .conn
                    .execute(
                        "INSERT INTO slack_user_profiles (user_id, display_name, fetched_at_unix_secs)
                         VALUES (?1, ?2, ?3)
                         ON CONFLICT(user_id) DO UPDATE SET
                            display_name = excluded.display_name,
                            fetched_at_unix_secs = excluded.fetched_at_unix_secs",
                        params![user_id, stored.display_name, stored.fetched_at_unix_secs],
                    )
                    .map_err(|e| format!("Failed to persist Slack user profile: {}", e))
                    .map(|_| ());
                let _ = reply.send(result);
            }
            SlackUserStoreMessage::GetProfile { user_id, reply } => {
                let result = self
                    .conn
                    .query_row(
                        "SELECT display_name, fetched_at_unix_secs
                         FROM slack_user_profiles
                         WHERE user_id = ?1",
                        params![user_id],
                        |row| {
                            Ok(StoredSlackUserProfile {
                                display_name: row.get(0)?,
                                fetched_at_unix_secs: row.get(1)?,
                            })
                        },
                    )
                    .optional()
                    .map_err(|e| format!("Failed to load Slack user profile: {}", e));
                let _ = reply.send(result);
            }
        }

        Ok(None)
    }
}

pub(crate) struct SlackUserProfileStore {
    node: NodeHandle<SlackUserStoreMessage>,
}

impl SlackUserProfileStore {
    pub(crate) fn new(db_path: impl AsRef<Path>) -> Result<Self, String> {
        let actor = SqliteSlackUserProfileStoreActor::new(db_path)?;
        let node = NodeHandle::new(actor, 64, 1, Duration::from_millis(5));
        Ok(Self { node })
    }

    pub(crate) async fn upsert(
        &self,
        user_id: &str,
        stored: &StoredSlackUserProfile,
    ) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let msg = SlackUserStoreMessage::UpsertProfile {
            user_id: user_id.to_string(),
            stored: stored.clone(),
            reply: SharedReply::new(tx),
        };
        self.node
            .send_packet(msg)
            .await
            .map_err(|e| format!("Failed to send Slack user profile upsert request: {}", e))?;
        rx.await
            .map_err(|_| "Slack user profile store actor channel closed".to_string())?
    }

    pub(crate) async fn get(
        &self,
        user_id: &str,
    ) -> Result<Option<StoredSlackUserProfile>, String> {
        let (tx, rx) = oneshot::channel();
        let msg = SlackUserStoreMessage::GetProfile {
            user_id: user_id.to_string(),
            reply: SharedReply::new(tx),
        };
        self.node
            .send_packet(msg)
            .await
            .map_err(|e| format!("Failed to send Slack user profile get request: {}", e))?;
        rx.await
            .map_err(|_| "Slack user profile store actor channel closed".to_string())?
    }
}
