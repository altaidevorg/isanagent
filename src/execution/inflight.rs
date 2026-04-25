//! Process-global registry tracking the in-flight synchronous tool call (if any) for each chat.
//!
//! Each chat slot holds the `oneshot::Sender<()>` half of a "promote me to background" channel.
//! The slash-command handler in the terminal UI calls [`InflightSyncRegistry::promote`] when the
//! user types `/background`; the corresponding sync tool wrapper (in `src/tools/execution.rs`)
//! holds the receiver inside [`run_with_auto_promote`](super::auto_promote::run_with_auto_promote).
//!
//! Only one in-flight sync call per chat is tracked. If the same chat starts a second sync run
//! while the first is still racing, the older sender is dropped (its receiver simply won't fire).
//! In practice the agent is sequential per chat, so this is fine.

use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::oneshot;

/// Process-global guard for "auto-promotable" sync tool calls.
///
/// `chat_id` is the slot key (matches the message bus `chat_id`). Use [`Self::register`] to install
/// a fresh sender (returns the matching receiver to wire into `run_with_auto_promote`), and
/// [`Self::promote`] to trigger it from a slash command. The registration is automatically removed
/// when the sync wrapper drops the [`InflightGuard`] returned by `register`.
#[derive(Debug, Default, Clone)]
pub struct InflightSyncRegistry {
    inner: Arc<DashMap<String, oneshot::Sender<()>>>,
}

impl InflightSyncRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }

    /// Register a fresh promote sender for `chat_id` and return the matching receiver plus a guard
    /// that removes the slot when dropped.
    pub fn register(&self, chat_id: &str) -> (oneshot::Receiver<()>, InflightGuard) {
        let (tx, rx) = oneshot::channel::<()>();
        // If an older sender already exists, replace it; the previous tool's receiver simply won't
        // fire, which is fine — that tool will continue racing on its own timer.
        self.inner.insert(chat_id.to_string(), tx);
        (
            rx,
            InflightGuard {
                inner: self.inner.clone(),
                chat_id: chat_id.to_string(),
            },
        )
    }

    /// Fire the promote signal for the in-flight sync call on `chat_id`, if any. Returns whether
    /// a sender was found and consumed.
    pub fn promote(&self, chat_id: &str) -> bool {
        match self.inner.remove(chat_id) {
            Some((_, tx)) => tx.send(()).is_ok(),
            None => false,
        }
    }

    /// Best-effort check (no removal): is there an in-flight sync call on this chat?
    pub fn is_active(&self, chat_id: &str) -> bool {
        self.inner.contains_key(chat_id)
    }
}

/// RAII guard returned by [`InflightSyncRegistry::register`]. Drops the slot on `Drop` so a stale
/// sender never lingers after the sync wrapper returns.
pub struct InflightGuard {
    inner: Arc<DashMap<String, oneshot::Sender<()>>>,
    chat_id: String,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.inner.remove(&self.chat_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn register_and_promote_fires_receiver() {
        let reg = InflightSyncRegistry::new();
        let (rx, _guard) = reg.register("chat-1");
        assert!(reg.is_active("chat-1"));
        assert!(reg.promote("chat-1"));
        rx.await.expect("receiver should fire");
    }

    #[tokio::test]
    async fn promote_without_register_returns_false() {
        let reg = InflightSyncRegistry::new();
        assert!(!reg.promote("nope"));
    }

    #[tokio::test]
    async fn drop_guard_removes_slot() {
        let reg = InflightSyncRegistry::new();
        let (_rx, guard) = reg.register("chat-x");
        assert!(reg.is_active("chat-x"));
        drop(guard);
        assert!(!reg.is_active("chat-x"));
    }
}
