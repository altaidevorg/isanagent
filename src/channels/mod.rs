use crate::bus::{BusMessage, OutboundMessage};
use async_trait::async_trait;
use tokio::sync::mpsc::Sender;

use std::any::Any;

/// Abstract base trait for chat channel implementations.
#[async_trait]
pub trait Channel: Send + Sync {
    /// The unique name of the channel (e.g. "slack", "email", "terminal").
    fn name(&self) -> &str;

    /// Start the channel and begin listening for messages.
    /// Messages received should be sent to the `bus_tx` channel.
    async fn start(&self, bus_tx: Sender<BusMessage>) -> Result<(), String>;

    /// Stop the channel and clean up resources.
    async fn stop(&self) -> Result<(), String>;

    /// Send a message out through this channel.
    async fn send(&self, msg: OutboundMessage) -> Result<(), String>;

    /// Downcast to concrete type if needed.
    fn as_any(&self) -> &dyn Any;
}

pub mod api;
pub(crate) mod api_store;
pub mod email;
pub mod oneshot;
pub mod slack;
pub(crate) mod slack_store;
pub mod terminal;
pub(crate) mod terminal_ui;
pub mod tty_prompt;
