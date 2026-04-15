use crate::memory::{MemoryMessage, SharedReply};
use crate::traits::Memory;
use crate::utils::ChatMessage;
use crate::NodeHandle;
use async_trait::async_trait;
use tokio::sync::oneshot;

/// Manages dynamic instantiation of `SessionProxy` interfaces.
#[derive(Clone)]
pub struct SessionManager {
    memory_node: NodeHandle<MemoryMessage>,
}

impl SessionManager {
    pub fn new(memory_node: NodeHandle<MemoryMessage>) -> Self {
        Self { memory_node }
    }

    pub async fn get_session(&self, session_key: &str) -> Result<SessionProxy, String> {
        Ok(SessionProxy::new(session_key, self.memory_node.clone()))
    }

    pub fn get_memory_node(&self) -> NodeHandle<MemoryMessage> {
        self.memory_node.clone()
    }

    pub async fn get_recent_summaries(
        &self,
        session_prefix: &str,
        limit: usize,
    ) -> Result<Vec<String>, String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::GetRecentSummaries {
            session_id: session_prefix.to_string(),
            limit,
            reply: SharedReply::new(tx),
        };
        self.memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;
        rx.await
            .map_err(|_| "Memory Actor Channel Closed".to_string())?
    }
}

/// A lightweight proxy that implements the standard asynchronous `Memory` trait
/// while natively interacting with the `SqliteMemoryActor` over the message bus.
#[derive(Clone)]
pub struct SessionProxy {
    session_id: String,
    memory_node: NodeHandle<MemoryMessage>,
}

impl SessionProxy {
    pub fn new(session_id: &str, memory_node: NodeHandle<MemoryMessage>) -> Self {
        Self {
            session_id: session_id.to_string(),
            memory_node,
        }
    }
}

#[async_trait]
impl Memory for SessionProxy {
    async fn add_message(&mut self, message: ChatMessage) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::AddMessage {
            session_id: self.session_id.clone(),
            message,
            reply: SharedReply::new(tx),
        };
        self.memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;
        rx.await
            .map_err(|_| "Memory Actor Channel Closed".to_string())?
    }

    async fn get_context(&self) -> Result<Vec<ChatMessage>, String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::GetContext {
            session_id: self.session_id.clone(),
            reply: SharedReply::new(tx),
        };
        self.memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;
        rx.await
            .map_err(|_| "Memory Actor Channel Closed".to_string())?
    }

    async fn get_context_since_reflection(&self) -> Result<Vec<ChatMessage>, String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::GetMessagesSinceReflection {
            session_id: self.session_id.clone(),
            reply: SharedReply::new(tx),
        };
        self.memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;
        let (rows, _) = rx
            .await
            .map_err(|_| "Memory Actor Channel Closed".to_string())??;

        let mut messages = Vec::new();
        for (_id, role, content_raw) in rows {
            // We need to parse the content_raw which is stored as JSON or plain text
            // Reusing the logic from memory.rs would be good but it's private there.
            // For now, we'll do a simple check.
            let content = if content_raw.trim_start().starts_with('[') {
                match serde_json::from_str(&content_raw) {
                    Ok(parts) => crate::utils::MessageContent::Parts(parts),
                    Err(_) => crate::utils::MessageContent::Text(content_raw),
                }
            } else {
                crate::utils::MessageContent::Text(content_raw)
            };

            messages.push(ChatMessage {
                role,
                content: Some(content),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        Ok(messages)
    }

    async fn clear(&mut self) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::Clear {
            session_id: self.session_id.clone(),
            keep_last: 0,
            reply: SharedReply::new(tx),
        };
        self.memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;
        rx.await
            .map_err(|_| "Memory Actor Channel Closed".to_string())?
    }

    async fn clear_keep_last(&mut self, keep_last: usize) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::Clear {
            session_id: self.session_id.clone(),
            keep_last,
            reply: SharedReply::new(tx),
        };
        self.memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;
        rx.await
            .map_err(|_| "Memory Actor Channel Closed".to_string())?
    }
}
