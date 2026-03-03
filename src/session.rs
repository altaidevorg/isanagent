use async_trait::async_trait;
use crate::memory::{MemoryMessage, SharedReply};
use crate::traits::Memory;
use crate::utils::ChatMessage;
use crate::NodeHandle;
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
    async fn add_user_message(&mut self, content: &str) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::AddMessage {
            session_id: self.session_id.clone(),
            role: "user".to_string(),
            content: content.to_string(),
            reply: SharedReply::new(tx),
        };
        self.memory_node.send_packet(msg).await.map_err(|e| e.to_string())?;
        rx.await.map_err(|_| "Memory Actor Channel Closed".to_string())?
    }

    async fn add_assistant_message(&mut self, content: &str) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::AddMessage {
            session_id: self.session_id.clone(),
            role: "assistant".to_string(),
            content: content.to_string(),
            reply: SharedReply::new(tx),
        };
        self.memory_node.send_packet(msg).await.map_err(|e| e.to_string())?;
        rx.await.map_err(|_| "Memory Actor Channel Closed".to_string())?
    }

    async fn add_system_message(&mut self, content: &str) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::AddMessage {
            session_id: self.session_id.clone(),
            role: "system".to_string(),
            content: content.to_string(),
            reply: SharedReply::new(tx),
        };
        self.memory_node.send_packet(msg).await.map_err(|e| e.to_string())?;
        rx.await.map_err(|_| "Memory Actor Channel Closed".to_string())?
    }

    async fn get_context(&self) -> Result<Vec<ChatMessage>, String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::GetContext {
            session_id: self.session_id.clone(),
            reply: SharedReply::new(tx),
        };
        self.memory_node.send_packet(msg).await.map_err(|e| e.to_string())?;
        rx.await.map_err(|_| "Memory Actor Channel Closed".to_string())?
    }

    async fn clear(&mut self) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::Clear {
            session_id: self.session_id.clone(),
            reply: SharedReply::new(tx),
        };
        self.memory_node.send_packet(msg).await.map_err(|e| e.to_string())?;
        rx.await.map_err(|_| "Memory Actor Channel Closed".to_string())?
    }
}
