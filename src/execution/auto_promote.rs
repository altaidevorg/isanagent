//! Auto-promote primitive: race a synchronous tool's work against a "short bound" timer and an
//! optional user-driven `/background` signal. If either trigger fires before the work completes,
//! the in-flight `tokio` task is handed off (still running) to the caller, who registers it with
//! [`ExecutionJobManager::adopt_inflight`](super::execution_jobs::ExecutionJobManager::adopt_inflight)
//! and returns a `job_id` envelope to the model.
//!
//! Used by `execution_run` to remove the artificial 120s cap on long
//! Colab/ML runs without making the synchronous path useless for short calls.

use std::time::Duration;

use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::sleep;

/// Outcome of an auto-promoted run.
///
/// - [`AutoPromoteOutcome::Completed`] — work finished within the short bound; the inner value is
///   whatever the future produced.
/// - [`AutoPromoteOutcome::Promoted`] — the bound (or `/background` signal) fired first; the
///   `JoinHandle` is still running and the caller has handed it to the job manager. `reason`
///   distinguishes the two paths for telemetry / response shaping.
pub enum AutoPromoteOutcome<T> {
    Completed(T),
    Promoted {
        job_id: String,
        reason: PromoteReason,
    },
}

/// Why the auto-promote fired (used for the `reason` field of the auto-promote response envelope).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromoteReason {
    /// The synchronous run exceeded `auto_promote_after_secs` and was promoted automatically.
    AutoPromoteAfterSecs,
    /// The user pushed the in-flight run to the background (`/background` slash command).
    UserBackground,
}

impl PromoteReason {
    pub fn as_str(self) -> &'static str {
        match self {
            PromoteReason::AutoPromoteAfterSecs => "auto_promote_after_secs",
            PromoteReason::UserBackground => "user_background",
        }
    }
}

/// Race a future against a timer + a user-cancel `oneshot`. On promotion, hand the still-running
/// `JoinHandle` to `on_promote`, which is expected to register it with the
/// [`ExecutionJobManager`](super::execution_jobs::ExecutionJobManager) (typically via
/// `adopt_inflight`) and return the new `job_id`.
///
/// `work` is spawned with [`tokio::spawn`] so the future keeps running even when the
/// synchronous wrapper returns to the caller after promotion. The future therefore needs to be
/// `Send + 'static`, and its output `Send + 'static`.
///
/// `short_bound` of `Duration::ZERO` disables the timer leg; only the `oneshot` and the work
/// itself can complete. (Useful for tests / disabled config.)
pub async fn run_with_auto_promote<T, F, Map>(
    work: F,
    short_bound: Duration,
    promote_signal: Option<oneshot::Receiver<()>>,
    on_promote: Map,
) -> AutoPromoteOutcome<T>
where
    T: Send + 'static,
    F: std::future::Future<Output = T> + Send + 'static,
    Map: FnOnce(JoinHandle<T>, PromoteReason) -> String,
{
    let mut handle = tokio::spawn(work);

    // The timer leg: sleep then fire; if `short_bound` is zero we want a pending future so that
    // only the work / oneshot legs of the select can win.
    let timer = async move {
        if short_bound.is_zero() {
            std::future::pending::<()>().await;
        } else {
            sleep(short_bound).await;
        }
    };
    tokio::pin!(timer);

    // Normalise the optional oneshot into something the `select!` macro can poll uniformly:
    // `Some(rx)` polls the channel, `None` becomes a pending future.
    let mut promote_signal = promote_signal;

    loop {
        tokio::select! {
            biased;

            // (a) work finished first — return the inner value.
            join_res = &mut handle => {
                return match join_res {
                    Ok(value) => AutoPromoteOutcome::Completed(value),
                    Err(join_err) if join_err.is_cancelled() => {
                        // Should be rare on the sync path: someone aborted us via the abort handle.
                        // We surface this as "promoted" with no job, but in practice the caller
                        // never observes this because they own the abort handle. Instead, we return
                        // a synthetic "Completed" by re-panicking the join error to keep types
                        // simple — but cancellation here means there is nothing to return, so we
                        // panic to surface the bug.
                        panic!("auto-promote: spawned work was cancelled before promotion path could observe it");
                    }
                    Err(join_err) => std::panic::resume_unwind(join_err.into_panic()),
                };
            }

            // (b) timer fired — promote.
            _ = &mut timer => {
                let job_id = on_promote(handle, PromoteReason::AutoPromoteAfterSecs);
                return AutoPromoteOutcome::Promoted {
                    job_id,
                    reason: PromoteReason::AutoPromoteAfterSecs,
                };
            }

            // (c) user pushed `/background` — promote.
            promote = async {
                match promote_signal.as_mut() {
                    Some(rx) => rx.await.ok(),
                    None => std::future::pending::<Option<()>>().await,
                }
            } => {
                if promote.is_some() {
                    let job_id = on_promote(handle, PromoteReason::UserBackground);
                    return AutoPromoteOutcome::Promoted {
                        job_id,
                        reason: PromoteReason::UserBackground,
                    };
                } else {
                    // Sender dropped without sending — drop the receiver and continue racing the
                    // remaining legs (work + timer). Without this, we'd busy-loop on a closed
                    // channel.
                    promote_signal = None;
                    continue;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn completed_when_work_finishes_within_bound() {
        let outcome = run_with_auto_promote::<u32, _, _>(
            async { 42u32 },
            Duration::from_secs(5),
            None,
            |_handle, _reason| panic!("should not promote"),
        )
        .await;
        match outcome {
            AutoPromoteOutcome::Completed(v) => assert_eq!(v, 42),
            AutoPromoteOutcome::Promoted { .. } => panic!("expected completed"),
        }
    }

    #[tokio::test]
    async fn promoted_when_timer_fires_first() {
        let outcome = run_with_auto_promote::<u32, _, _>(
            async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                7u32
            },
            Duration::from_millis(50),
            None,
            |handle, reason| {
                assert_eq!(reason, PromoteReason::AutoPromoteAfterSecs);
                drop(handle);
                "test-job-id".to_string()
            },
        )
        .await;
        match outcome {
            AutoPromoteOutcome::Promoted { job_id, reason } => {
                assert_eq!(job_id, "test-job-id");
                assert_eq!(reason, PromoteReason::AutoPromoteAfterSecs);
            }
            AutoPromoteOutcome::Completed(_) => panic!("expected promoted"),
        }
    }

    #[tokio::test]
    async fn promoted_when_oneshot_fires() {
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx.send(());
        });
        let outcome = run_with_auto_promote::<u32, _, _>(
            async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                9u32
            },
            Duration::from_secs(60),
            Some(rx),
            |handle, reason| {
                assert_eq!(reason, PromoteReason::UserBackground);
                drop(handle);
                "user-bg-job".to_string()
            },
        )
        .await;
        match outcome {
            AutoPromoteOutcome::Promoted { job_id, reason } => {
                assert_eq!(job_id, "user-bg-job");
                assert_eq!(reason, PromoteReason::UserBackground);
            }
            AutoPromoteOutcome::Completed(_) => panic!("expected promoted"),
        }
    }

    #[tokio::test]
    async fn dropped_oneshot_does_not_promote() {
        let (tx, rx) = oneshot::channel::<()>();
        drop(tx);
        let outcome = run_with_auto_promote::<u32, _, _>(
            async { 1u32 },
            Duration::from_secs(5),
            Some(rx),
            |_handle, _reason| panic!("should not promote"),
        )
        .await;
        match outcome {
            AutoPromoteOutcome::Completed(v) => assert_eq!(v, 1),
            AutoPromoteOutcome::Promoted { .. } => panic!("expected completed"),
        }
    }
}
