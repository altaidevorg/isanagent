//! Audit X9: mid-run steering inbox, split out of the former
//! `agent/mod.rs` god-file.
//!
//! [`SteeringInbox`] buffers user steer messages accepted while a
//! reasoning turn is in flight; the loop drains it at safe boundaries.

use std::collections::VecDeque;
use std::sync::Mutex;

pub(crate) struct SteeringInbox {
    pub(crate) accepting: bool,
    pub(crate) pending: VecDeque<String>,
}

impl SteeringInbox {
    pub(crate) fn open() -> Self {
        Self {
            accepting: true,
            pending: VecDeque::new(),
        }
    }

    pub(crate) fn push(&mut self, content: String) -> bool {
        if !self.accepting {
            return false;
        }
        self.pending.push_back(content);
        true
    }

    pub(crate) fn drain(&mut self) -> Vec<String> {
        self.pending.drain(..).collect()
    }

    pub(crate) fn close(&mut self) {
        self.accepting = false;
        self.pending.clear();
    }

    pub(crate) fn close_or_drain(&mut self) -> Vec<String> {
        if self.pending.is_empty() {
            self.accepting = false;
            Vec::new()
        } else {
            self.drain()
        }
    }
}

pub(crate) fn steering_guard(
    inbox: &Mutex<SteeringInbox>,
) -> std::sync::MutexGuard<'_, SteeringInbox> {
    inbox
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
