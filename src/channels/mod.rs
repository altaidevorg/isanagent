use async_trait::async_trait;
use crate::bus::{InboundMessage, OutboundMessage};
use tokio::sync::mpsc::Sender;

/// Abstract base trait for chat channel implementations.
#[async_trait]
pub trait Channel: Send + Sync {
    /// The unique name of the channel (e.g. "slack", "email", "terminal").
    fn name(&self) -> &str;

    /// Start the channel and begin listening for messages. 
    /// Messages received should be sent to the `inbound_tx` channel.
    async fn start(&self, inbound_tx: Sender<InboundMessage>) -> Result<(), String>;

    /// Stop the channel and clean up resources.
    async fn stop(&self) -> Result<(), String>;

    /// Send a message out through this channel.
    async fn send(&self, msg: OutboundMessage) -> Result<(), String>;
}

pub mod terminal;
pub mod slack;
pub mod email;
pub mod api;
