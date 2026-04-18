//! User clarification (Phase 3): correlate `ask_user` tool waits with the next inbound message.

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::oneshot;

/// Outbound [`crate::bus::OutboundMessage::metadata`] key: UI / terminal can style clarification prompts.
pub const METADATA_CLARIFICATION: &str = "isanagent_clarification";
/// Optional JSON array of strings (`ask_user` choices); UIs can render as a numbered list without parsing body text.
pub const METADATA_CLARIFICATION_CHOICES: &str = "isanagent_clarification_choices";

/// Routes the next inbound message for a session key to a pending `ask_user` tool call.
#[derive(Debug, Default)]
pub struct ClarificationHub {
    pending: DashMap<String, oneshot::Sender<String>>,
}

impl ClarificationHub {
    pub fn new() -> Self {
        Self {
            pending: DashMap::new(),
        }
    }

    /// Reserve a slot for this session. Fails if a clarification is already pending.
    pub fn begin_wait(&self, session_key: &str) -> Result<oneshot::Receiver<String>, String> {
        let (tx, rx) = oneshot::channel();
        match self.pending.entry(session_key.to_string()) {
            Entry::Occupied(_) => Err(
                "A clarification is already pending for this session; wait for the user reply."
                    .to_string(),
            ),
            Entry::Vacant(v) => {
                v.insert(tx);
                Ok(rx)
            }
        }
    }

    /// Remove a pending wait without notifying the tool (e.g. cooperative cancellation).
    pub fn cancel_wait(&self, session_key: &str) {
        self.pending.remove(session_key);
    }

    /// If a tool is waiting on `session_key`, deliver `text` and return `true`.
    pub fn try_deliver_reply(&self, session_key: &str, text: String) -> bool {
        if let Some((_, tx)) = self.pending.remove(session_key) {
            let _ = tx.send(text);
            true
        } else {
            false
        }
    }

    /// Shared empty hub (tests and default wiring).
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deliver_completes_receiver() {
        let hub = ClarificationHub::new();
        let rx = hub.begin_wait("terminal:cid:").expect("begin");
        assert!(hub.try_deliver_reply("terminal:cid:", "yes".into()));
        assert_eq!(rx.await.expect("recv"), "yes");
    }

    #[tokio::test]
    async fn begin_wait_twice_same_session_errors() {
        let hub = ClarificationHub::new();
        let _rx = hub.begin_wait("api:x:").expect("first");
        assert!(hub.begin_wait("api:x:").is_err());
    }

    #[tokio::test]
    async fn cancel_wait_drops_pending() {
        let hub = ClarificationHub::new();
        let rx = hub.begin_wait("t:1:").expect("begin");
        hub.cancel_wait("t:1:");
        assert!(!hub.try_deliver_reply("t:1:", "late".into()));
        assert!(rx.await.is_err());
    }
}
