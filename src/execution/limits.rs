//! Audit X15: atomic concurrency budgets for execution providers.
//!
//! `create_session` used to probe `sessions.len() >= max_sessions`, perform
//! fallible setup, and only then insert into the session map — a
//! check-then-act window that let parallel creates overshoot the cap. A
//! permit reserved by a single compare-and-set closes that window: the
//! permit lives inside the session record and is released when the session
//! leaves the map (or its last `Arc` clone drops, which errs on the safe,
//! under-counting side).

use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::error::ExecutionError;

/// Builds the slot budget for a provider: `max_sessions` permits, or an
/// effectively-unlimited budget when `max_sessions == 0` (the documented
/// "unlimited" sentinel).
pub(crate) fn session_slot_semaphore(max_sessions: usize) -> Arc<Semaphore> {
    Arc::new(Semaphore::new(if max_sessions == 0 {
        // `0` is the documented "unlimited" sentinel; clamp to tokio's own
        // permit ceiling instead of `usize::MAX` (which panics).
        Semaphore::MAX_PERMITS
    } else {
        max_sessions
    }))
}

/// Atomically claims one session slot. Fails with the same `limit_exceeded`
/// error shape as before when the budget is exhausted.
pub(crate) fn try_acquire_session_slot(
    slots: &Arc<Semaphore>,
    max_sessions: usize,
) -> Result<OwnedSemaphorePermit, ExecutionError> {
    slots.clone().try_acquire_owned().map_err(|_| {
        ExecutionError::limit_exceeded("sessions", format!("max_sessions={max_sessions} reached"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_budget_rejects_when_exhausted_and_recovers_on_release() {
        let slots = session_slot_semaphore(2);
        let a = try_acquire_session_slot(&slots, 2).expect("first");
        let _b = try_acquire_session_slot(&slots, 2).expect("second");
        let err = try_acquire_session_slot(&slots, 2).expect_err("exhausted");
        assert!(
            err.to_string().contains("max_sessions=2 reached"),
            "unexpected error: {err:?}"
        );
        drop(a);
        assert!(
            try_acquire_session_slot(&slots, 2).is_ok(),
            "released slot must become reservable again"
        );
    }

    #[test]
    fn zero_means_unlimited() {
        let slots = session_slot_semaphore(0);
        let mut held = Vec::new();
        for i in 0..1000 {
            held.push(
                try_acquire_session_slot(&slots, 0)
                    .unwrap_or_else(|e| panic!("unlimited budget rejected reservation {i}: {e:?}")),
            );
        }
    }
}
