//! Pure, deterministic budget and observable-progress decisions for one reasoning run.
//!
//! The controller deliberately knows nothing about providers, tools, clocks, or persistence.
//! Callers feed cumulative wall time plus typed observations and receive one decision. This keeps
//! the false-positive-sensitive policy replayable without constructing an agent runtime.

use std::collections::HashSet;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::bus::{
    RunBudgetLimit, RunBudgetSnapshot, RunBudgetWarning, RunBudgetWarningReason, RunStuckReason,
};
use crate::traits::ToolErrorCode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WarningKey {
    ApproachingLimit(RunBudgetLimit),
    RepeatedRootCause,
    NoProgress,
}

const DEFAULT_WALL_LIMIT: Duration = Duration::from_secs(2 * 60 * 60);
const DEFAULT_TOKEN_LIMIT: u64 = 5_000_000;
const DEFAULT_PROVIDER_RETRY_LIMIT: u32 = 12;
const DEFAULT_CONTEXT_RECOVERY_LIMIT: u32 = 2;
const DEFAULT_WARNING_NO_PROGRESS_TURNS: usize = 4;
const DEFAULT_STUCK_NO_PROGRESS_TURNS: usize = 8;
const DEFAULT_WARNING_ROOT_CAUSE_FAILURES: usize = 2;
const DEFAULT_STUCK_ROOT_CAUSE_FAILURES: usize = 3;

#[derive(Clone, Debug)]
pub(crate) struct BudgetLimits {
    llm_turns: usize,
    wall_time: Duration,
    tokens: u64,
    provider_retries: u32,
    context_recoveries: u32,
    warning_no_progress_turns: usize,
    stuck_no_progress_turns: usize,
    warning_root_cause_failures: usize,
    stuck_root_cause_failures: usize,
}

impl BudgetLimits {
    pub(crate) fn for_run(emergency_llm_turns: usize) -> Self {
        Self {
            llm_turns: emergency_llm_turns,
            wall_time: DEFAULT_WALL_LIMIT,
            tokens: DEFAULT_TOKEN_LIMIT,
            provider_retries: DEFAULT_PROVIDER_RETRY_LIMIT,
            context_recoveries: DEFAULT_CONTEXT_RECOVERY_LIMIT,
            warning_no_progress_turns: DEFAULT_WARNING_NO_PROGRESS_TURNS,
            stuck_no_progress_turns: DEFAULT_STUCK_NO_PROGRESS_TURNS,
            warning_root_cause_failures: DEFAULT_WARNING_ROOT_CAUSE_FAILURES,
            stuck_root_cause_failures: DEFAULT_STUCK_ROOT_CAUSE_FAILURES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProgressKind {
    /// A materially different, accepted tool intent. It breaks no-progress, but does not prove a
    /// previous failure was fixed.
    NewToolIntent,
    /// A successful tool returned new evidence or changed state.
    NewEvidence,
    /// User steering changes direction without refunding any consumed budget.
    Steering,
    /// Compaction unblocked a provider call. This is progress, but not proof that a tool root cause
    /// was resolved.
    ContextRecovery,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BudgetDecision {
    Continue,
    Warning(RunBudgetWarning),
    Stuck {
        reason: RunStuckReason,
        snapshot: RunBudgetSnapshot,
    },
    BudgetExhausted(RunBudgetSnapshot),
}

impl BudgetDecision {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Stuck { .. } | Self::BudgetExhausted(_))
    }
}

/// State machine for one run. Every method is deterministic for the supplied observation stream.
pub(crate) struct BudgetController {
    limits: BudgetLimits,
    llm_turns: usize,
    elapsed: Duration,
    tokens: u64,
    provider_retries: u32,
    context_recoveries: u32,
    no_progress_turns: usize,
    last_tool_intent: Option<String>,
    last_successful_intent: Option<String>,
    last_root_cause: Option<String>,
    repeated_root_cause_failures: usize,
    emitted_warnings: HashSet<WarningKey>,
    /// Set when a previously emitted non-terminal warning is resolved by progress.
    /// Hosts should drain this via [`Self::take_warning_cleared`] and emit
    /// `RunLifecycleEvent::WarningCleared`.
    warning_cleared: bool,
    terminal: Option<BudgetDecision>,
}

impl BudgetController {
    pub(crate) fn new(limits: BudgetLimits) -> Self {
        Self {
            limits,
            llm_turns: 0,
            elapsed: Duration::ZERO,
            tokens: 0,
            provider_retries: 0,
            context_recoveries: 0,
            no_progress_turns: 0,
            last_tool_intent: None,
            last_successful_intent: None,
            last_root_cause: None,
            repeated_root_cause_failures: 0,
            emitted_warnings: HashSet::new(),
            warning_cleared: false,
            terminal: None,
        }
    }

    /// Drain the latch set when progress resolves a live budget warning.
    pub(crate) fn take_warning_cleared(&mut self) -> bool {
        std::mem::take(&mut self.warning_cleared)
    }

    /// Admit one more LLM turn. The configured turn limit is an absolute emergency ceiling: a
    /// limit of 50 permits 50 turns and rejects the 51st.
    pub(crate) fn start_turn(&mut self, elapsed: Duration) -> BudgetDecision {
        self.elapsed = elapsed;
        if let Some(decision) = self.terminal.clone() {
            return decision;
        }
        if self.llm_turns >= self.limits.llm_turns {
            return self.exhaust(RunBudgetLimit::LlmTurns);
        }
        if let Some(limit) = self.exhausted_non_turn_limit() {
            return self.exhaust(limit);
        }

        self.llm_turns = self.llm_turns.saturating_add(1);
        self.no_progress_turns = self.no_progress_turns.saturating_add(1);
        self.evaluate()
    }

    pub(crate) fn record_elapsed(&mut self, elapsed: Duration) -> BudgetDecision {
        self.elapsed = elapsed;
        self.evaluate()
    }

    pub(crate) fn record_tokens(&mut self, tokens: u64) -> BudgetDecision {
        self.tokens = self.tokens.saturating_add(tokens);
        self.evaluate()
    }

    pub(crate) fn record_provider_retries(&mut self, retries: u32) -> BudgetDecision {
        self.provider_retries = self.provider_retries.saturating_add(retries);
        self.evaluate()
    }

    pub(crate) fn record_context_recovery(&mut self, elapsed: Duration) -> BudgetDecision {
        self.elapsed = elapsed;
        self.context_recoveries = self.context_recoveries.saturating_add(1);
        self.record_progress(ProgressKind::ContextRecovery)
    }

    pub(crate) fn record_tool_call(&mut self, intent: String) -> BudgetDecision {
        if self.last_tool_intent.as_ref() != Some(&intent) {
            self.last_tool_intent = Some(intent);
            return self.record_progress(ProgressKind::NewToolIntent);
        }
        self.evaluate()
    }

    pub(crate) fn record_tool_success(&mut self, intent: String) -> BudgetDecision {
        if self.last_successful_intent.as_ref() != Some(&intent) {
            self.last_successful_intent = Some(intent);
            return self.record_progress(ProgressKind::NewEvidence);
        }
        self.evaluate()
    }

    pub(crate) fn record_tool_failure(&mut self, root_cause: String) -> BudgetDecision {
        if self.last_root_cause.as_ref() == Some(&root_cause) {
            self.repeated_root_cause_failures = self.repeated_root_cause_failures.saturating_add(1);
        } else {
            // A different typed key means the previous repeated-root-cause warning is stale
            // (e.g. intent-scoped NonZeroExit for `pnpm test` then `pnpm lint`).
            if self.emitted_warnings.remove(&WarningKey::RepeatedRootCause) {
                self.warning_cleared = true;
            }
            self.last_root_cause = Some(root_cause);
            self.repeated_root_cause_failures = 1;
        }
        self.evaluate()
    }

    pub(crate) fn record_progress(&mut self, kind: ProgressKind) -> BudgetDecision {
        if let Some(decision) = self.terminal.clone() {
            return decision;
        }
        self.no_progress_turns = 0;
        if self.emitted_warnings.remove(&WarningKey::NoProgress) {
            self.warning_cleared = true;
        }
        if matches!(kind, ProgressKind::NewEvidence) {
            self.last_root_cause = None;
            self.repeated_root_cause_failures = 0;
            if self.emitted_warnings.remove(&WarningKey::RepeatedRootCause) {
                self.warning_cleared = true;
            }
        }
        self.evaluate()
    }

    /// Decide whether a prose-only completion proposal may become terminal. Once the controller
    /// has warned about no progress or a repeated root cause, prose alone is not new evidence and
    /// must not silently convert that warning into `Completed`.
    pub(crate) fn propose_completion(&mut self) -> BudgetDecision {
        if let Some(decision) = self.terminal.clone() {
            return decision;
        }
        if self.repeated_root_cause_failures >= self.limits.warning_root_cause_failures {
            return self.stuck(RunStuckReason::RepeatedRootCause);
        }
        if self.no_progress_turns >= self.limits.warning_no_progress_turns {
            return self.stuck(RunStuckReason::NoProgress);
        }
        BudgetDecision::Continue
    }

    pub(crate) fn snapshot(&self) -> RunBudgetSnapshot {
        RunBudgetSnapshot {
            iterations_used: self.llm_turns,
            iterations_limit: self.limits.llm_turns,
            elapsed_ms: duration_millis(self.elapsed),
            elapsed_limit_ms: duration_millis(self.limits.wall_time),
            tokens_used: self.tokens,
            tokens_limit: self.limits.tokens,
            provider_retries_used: self.provider_retries,
            provider_retries_limit: self.limits.provider_retries,
            context_recoveries_used: self.context_recoveries,
            context_recoveries_limit: self.limits.context_recoveries,
            no_progress_turns: self.no_progress_turns,
            repeated_root_cause_failures: self.repeated_root_cause_failures,
            exhausted_limit: None,
        }
    }

    fn evaluate(&mut self) -> BudgetDecision {
        if let Some(decision) = self.terminal.clone() {
            return decision;
        }
        if let Some(limit) = self.exhausted_non_turn_limit() {
            return self.exhaust(limit);
        }
        if self.repeated_root_cause_failures >= self.limits.stuck_root_cause_failures {
            return self.stuck(RunStuckReason::RepeatedRootCause);
        }
        if self.no_progress_turns >= self.limits.stuck_no_progress_turns {
            return self.stuck(RunStuckReason::NoProgress);
        }

        let warning =
            if self.repeated_root_cause_failures >= self.limits.warning_root_cause_failures {
                Some(RunBudgetWarningReason::RepeatedRootCause {
                    failures: self.repeated_root_cause_failures,
                })
            } else if self.no_progress_turns >= self.limits.warning_no_progress_turns {
                Some(RunBudgetWarningReason::NoProgress {
                    turns: self.no_progress_turns,
                })
            } else {
                self.approaching_limit()
                    .map(|limit| RunBudgetWarningReason::ApproachingLimit { limit })
            };

        if let Some(reason) = warning {
            let key = match &reason {
                RunBudgetWarningReason::ApproachingLimit { limit } => {
                    WarningKey::ApproachingLimit(*limit)
                }
                RunBudgetWarningReason::RepeatedRootCause { .. } => WarningKey::RepeatedRootCause,
                RunBudgetWarningReason::NoProgress { .. } => WarningKey::NoProgress,
            };
            if self.emitted_warnings.insert(key) {
                return BudgetDecision::Warning(RunBudgetWarning {
                    reason,
                    budget: self.snapshot(),
                });
            }
        }
        BudgetDecision::Continue
    }

    fn exhausted_non_turn_limit(&self) -> Option<RunBudgetLimit> {
        if self.elapsed >= self.limits.wall_time {
            Some(RunBudgetLimit::WallTime)
        } else if self.tokens >= self.limits.tokens {
            Some(RunBudgetLimit::Tokens)
        } else if self.provider_retries >= self.limits.provider_retries {
            Some(RunBudgetLimit::ProviderRetries)
        } else if self.context_recoveries >= self.limits.context_recoveries {
            Some(RunBudgetLimit::ContextRecoveries)
        } else {
            None
        }
    }

    fn approaching_limit(&self) -> Option<RunBudgetLimit> {
        let four_fifths = |used: u128, limit: u128| limit > 0 && used * 5 >= limit * 4;
        if four_fifths(self.llm_turns as u128, self.limits.llm_turns as u128) {
            Some(RunBudgetLimit::LlmTurns)
        } else if four_fifths(self.elapsed.as_millis(), self.limits.wall_time.as_millis()) {
            Some(RunBudgetLimit::WallTime)
        } else if four_fifths(self.tokens as u128, self.limits.tokens as u128) {
            Some(RunBudgetLimit::Tokens)
        } else if four_fifths(
            self.provider_retries as u128,
            self.limits.provider_retries as u128,
        ) {
            Some(RunBudgetLimit::ProviderRetries)
        } else if four_fifths(
            self.context_recoveries as u128,
            self.limits.context_recoveries as u128,
        ) {
            Some(RunBudgetLimit::ContextRecoveries)
        } else {
            None
        }
    }

    fn exhaust(&mut self, limit: RunBudgetLimit) -> BudgetDecision {
        let mut snapshot = self.snapshot();
        snapshot.exhausted_limit = Some(limit);
        let decision = BudgetDecision::BudgetExhausted(snapshot);
        debug_assert!(decision.is_terminal());
        self.terminal = Some(decision.clone());
        decision
    }

    fn stuck(&mut self, reason: RunStuckReason) -> BudgetDecision {
        let decision = BudgetDecision::Stuck {
            reason,
            snapshot: self.snapshot(),
        };
        debug_assert!(decision.is_terminal());
        self.terminal = Some(decision.clone());
        decision
    }
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}

/// Build the budget root-cause key for a typed tool failure.
///
/// Policy / allow / invalid-args failures stay **coarse** (`tool:code`): varying arguments rarely
/// fixes them, so the historical doom-loop stop still fires across intents.
///
/// Exit / not-found / execution failures are **intent-scoped** (`tool:code:intent`): a failing
/// `pnpm test` then a failing `pnpm lint` must not count as the same repeated root cause.
pub(crate) fn typed_failure_key(tool_name: &str, code: ToolErrorCode, intent_sig: &str) -> String {
    let tool = tool_name.to_ascii_lowercase();
    let code_label = match code {
        ToolErrorCode::InvalidToolArguments => "invalid_tool_arguments",
        ToolErrorCode::NotFound => "not_found",
        ToolErrorCode::NotAllowed => "not_allowed",
        ToolErrorCode::PolicyDenied => "policy_denied",
        ToolErrorCode::ExecutionFailed => "execution_failed",
        ToolErrorCode::NonZeroExit => "non_zero_exit",
        ToolErrorCode::LegacyReportedFailure => "legacy_reported_failure",
    };
    match code {
        ToolErrorCode::PolicyDenied
        | ToolErrorCode::NotAllowed
        | ToolErrorCode::InvalidToolArguments => {
            format!("{tool}:{code_label}")
        }
        ToolErrorCode::NonZeroExit
        | ToolErrorCode::NotFound
        | ToolErrorCode::ExecutionFailed
        | ToolErrorCode::LegacyReportedFailure => {
            format!("{tool}:{code_label}:{intent_sig}")
        }
    }
}

/// Canonical, content-free fingerprint for comparing accepted tool intents. JSON object key order
/// and insignificant whitespace do not create fake progress; raw malformed arguments still get a
/// stable fingerprint and are later grouped by their typed failure root cause.
pub(crate) fn tool_intent_signature(tool_name: &str, arguments: &str) -> String {
    let canonical_args = serde_json::from_str::<serde_json::Value>(arguments)
        .map(|value| canonical_json(&value))
        .unwrap_or_else(|_| arguments.split_whitespace().collect::<Vec<_>>().join(" "));
    let digest = Sha256::digest(canonical_args.as_bytes());
    format!("{}:{}", tool_name.to_ascii_lowercase(), hex::encode(digest))
}

fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| *key);
            let body = entries
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).unwrap_or_default(),
                        canonical_json(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        serde_json::Value::Array(values) => {
            let body = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_limits(turns: usize) -> BudgetLimits {
        BudgetLimits {
            llm_turns: turns,
            wall_time: Duration::from_secs(60),
            tokens: 10_000,
            provider_retries: 10,
            context_recoveries: 2,
            warning_no_progress_turns: 4,
            stuck_no_progress_turns: 8,
            warning_root_cause_failures: 2,
            stuck_root_cause_failures: 3,
        }
    }

    #[test]
    fn sanitized_historical_fifty_step_failure_stops_on_third_root_cause() {
        let trace = (0..50)
            .map(|step| {
                (
                    format!(r#"{{"command":"write artifact-{step}"}}"#),
                    "exec:policy_denied".to_string(),
                )
            })
            .collect::<Vec<_>>();
        let mut controller = BudgetController::new(test_limits(50));
        let mut terminal = None;

        for (step, (arguments, root_cause)) in trace.into_iter().enumerate() {
            assert!(!controller
                .start_turn(Duration::from_millis(step as u64))
                .is_terminal());
            let intent = tool_intent_signature("exec", &arguments);
            let _ = controller.record_tool_call(intent);
            let decision = controller.record_tool_failure(root_cause);
            if decision.is_terminal() {
                terminal = Some(decision);
                break;
            }
        }

        assert!(matches!(
            terminal,
            Some(BudgetDecision::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
                ref snapshot,
            }) if snapshot.iterations_used == 3 && snapshot.iterations_limit == 50
        ));
    }

    #[test]
    fn productive_long_run_is_not_stopped_for_turn_count_alone() {
        let mut controller = BudgetController::new(test_limits(100));
        for step in 0..75 {
            assert!(!controller
                .start_turn(Duration::from_millis(step))
                .is_terminal());
            let intent = tool_intent_signature("read_file", &format!(r#"{{"path":"{step}"}}"#));
            assert!(!controller.record_tool_call(intent.clone()).is_terminal());
            assert!(!controller.record_tool_success(intent).is_terminal());
        }
        assert_eq!(controller.snapshot().iterations_used, 75);
        assert_eq!(controller.snapshot().no_progress_turns, 0);
    }

    #[test]
    fn repeated_typed_root_cause_warns_then_stops_predictably() {
        let mut controller = BudgetController::new(test_limits(50));
        let _ = controller.start_turn(Duration::ZERO);
        assert_eq!(
            controller.record_tool_failure("exec:non_zero_exit".into()),
            BudgetDecision::Continue
        );
        assert!(matches!(
            controller.record_tool_failure("exec:non_zero_exit".into()),
            BudgetDecision::Warning(RunBudgetWarning {
                reason: RunBudgetWarningReason::RepeatedRootCause { failures: 2 },
                ..
            })
        ));
        assert!(matches!(
            controller.record_tool_failure("exec:non_zero_exit".into()),
            BudgetDecision::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
                ..
            }
        ));
    }

    #[test]
    fn progress_resets_only_progress_related_counters() {
        let mut controller = BudgetController::new(test_limits(50));
        let _ = controller.start_turn(Duration::from_secs(1));
        let _ = controller.record_tokens(200);
        let _ = controller.record_provider_retries(2);
        let _ = controller.record_tool_failure("exec:policy_denied".into());

        let _ = controller.record_progress(ProgressKind::Steering);
        let after_steer = controller.snapshot();
        assert_eq!(after_steer.no_progress_turns, 0);
        assert_eq!(after_steer.repeated_root_cause_failures, 1);
        assert_eq!(after_steer.tokens_used, 200);
        assert_eq!(after_steer.provider_retries_used, 2);

        let _ = controller.record_progress(ProgressKind::NewEvidence);
        let after_evidence = controller.snapshot();
        assert_eq!(after_evidence.repeated_root_cause_failures, 0);
        assert_eq!(after_evidence.tokens_used, 200);
        assert_eq!(after_evidence.elapsed_ms, 1_000);
    }

    #[test]
    fn terminal_decision_is_sticky_and_mutually_exclusive() {
        let mut controller = BudgetController::new(test_limits(1));
        assert!(!controller.start_turn(Duration::ZERO).is_terminal());
        let terminal = controller.start_turn(Duration::from_secs(1));
        assert!(matches!(
            terminal,
            BudgetDecision::BudgetExhausted(RunBudgetSnapshot {
                exhausted_limit: Some(RunBudgetLimit::LlmTurns),
                ..
            })
        ));
        assert_eq!(controller.record_tool_failure("x".into()), terminal);
        assert_eq!(
            controller.record_progress(ProgressKind::NewEvidence),
            terminal
        );
    }

    #[test]
    fn warning_is_non_terminal_and_cannot_masquerade_as_completion() {
        let mut controller = BudgetController::new(test_limits(50));
        for step in 0..4 {
            let decision = controller.start_turn(Duration::from_millis(step));
            if step < 3 {
                assert!(!decision.is_terminal());
            } else {
                assert!(matches!(
                    decision,
                    BudgetDecision::Warning(RunBudgetWarning {
                        reason: RunBudgetWarningReason::NoProgress { turns: 4 },
                        ..
                    })
                ));
            }
        }
        assert!(matches!(
            controller.propose_completion(),
            BudgetDecision::Stuck {
                reason: RunStuckReason::NoProgress,
                ..
            }
        ));
    }

    #[test]
    fn progress_rearms_only_resolved_warning_categories() {
        let mut controller = BudgetController::new(test_limits(50));
        for step in 0..4 {
            let decision = controller.start_turn(Duration::from_millis(step));
            if step == 3 {
                assert!(matches!(
                    decision,
                    BudgetDecision::Warning(RunBudgetWarning {
                        reason: RunBudgetWarningReason::NoProgress { turns: 4 },
                        ..
                    })
                ));
            }
        }
        let _ = controller.record_progress(ProgressKind::Steering);
        for step in 4..8 {
            let decision = controller.start_turn(Duration::from_millis(step));
            if step == 7 {
                assert!(matches!(
                    decision,
                    BudgetDecision::Warning(RunBudgetWarning {
                        reason: RunBudgetWarningReason::NoProgress { turns: 4 },
                        ..
                    })
                ));
            }
        }
    }

    #[test]
    fn independent_non_turn_budgets_are_enforced() {
        let mut token_limits = test_limits(50);
        token_limits.tokens = 100;
        let mut tokens = BudgetController::new(token_limits);
        assert!(matches!(
            tokens.record_tokens(80),
            BudgetDecision::Warning(RunBudgetWarning {
                reason: RunBudgetWarningReason::ApproachingLimit {
                    limit: RunBudgetLimit::Tokens
                },
                ..
            })
        ));
        assert!(matches!(
            tokens.record_tokens(20),
            BudgetDecision::BudgetExhausted(RunBudgetSnapshot {
                exhausted_limit: Some(RunBudgetLimit::Tokens),
                ..
            })
        ));

        let mut wall_limits = test_limits(50);
        wall_limits.wall_time = Duration::from_secs(2);
        let mut wall = BudgetController::new(wall_limits);
        assert!(matches!(
            wall.record_elapsed(Duration::from_secs(2)),
            BudgetDecision::BudgetExhausted(RunBudgetSnapshot {
                exhausted_limit: Some(RunBudgetLimit::WallTime),
                ..
            })
        ));

        let mut retry_limits = test_limits(50);
        retry_limits.provider_retries = 2;
        let mut retries = BudgetController::new(retry_limits);
        assert!(matches!(
            retries.record_provider_retries(2),
            BudgetDecision::BudgetExhausted(RunBudgetSnapshot {
                exhausted_limit: Some(RunBudgetLimit::ProviderRetries),
                ..
            })
        ));

        let mut recovery_limits = test_limits(50);
        recovery_limits.context_recoveries = 1;
        let mut recoveries = BudgetController::new(recovery_limits);
        assert!(matches!(
            recoveries.record_context_recovery(Duration::from_secs(1)),
            BudgetDecision::BudgetExhausted(RunBudgetSnapshot {
                exhausted_limit: Some(RunBudgetLimit::ContextRecoveries),
                ..
            })
        ));
    }

    #[test]
    fn tool_intent_signature_ignores_json_key_order() {
        assert_eq!(
            tool_intent_signature("exec", r#"{"command":"pwd","timeout":10}"#),
            tool_intent_signature("EXEC", r#"{ "timeout": 10, "command": "pwd" }"#)
        );
    }

    #[test]
    fn typed_failure_key_is_coarse_for_policy_and_intent_scoped_for_exits() {
        let intent_a = tool_intent_signature("exec", r#"{"command":"pnpm test"}"#);
        let intent_b = tool_intent_signature("exec", r#"{"command":"pnpm lint"}"#);
        assert_eq!(
            typed_failure_key("exec", ToolErrorCode::PolicyDenied, &intent_a),
            typed_failure_key("exec", ToolErrorCode::PolicyDenied, &intent_b),
        );
        assert_ne!(
            typed_failure_key("exec", ToolErrorCode::NonZeroExit, &intent_a),
            typed_failure_key("exec", ToolErrorCode::NonZeroExit, &intent_b),
        );
    }

    #[test]
    fn different_non_zero_exit_intents_do_not_warn() {
        let mut controller = BudgetController::new(test_limits(50));
        let _ = controller.start_turn(Duration::ZERO);
        let intent_a = tool_intent_signature("exec", r#"{"command":"pnpm test"}"#);
        let intent_b = tool_intent_signature("exec", r#"{"command":"pnpm lint"}"#);
        let key_a = typed_failure_key("exec", ToolErrorCode::NonZeroExit, &intent_a);
        let key_b = typed_failure_key("exec", ToolErrorCode::NonZeroExit, &intent_b);
        assert_eq!(
            controller.record_tool_failure(key_a),
            BudgetDecision::Continue
        );
        assert_eq!(
            controller.record_tool_failure(key_b),
            BudgetDecision::Continue
        );
        assert_eq!(controller.snapshot().repeated_root_cause_failures, 1);
    }

    #[test]
    fn switching_intent_after_warning_latches_warning_cleared() {
        let mut controller = BudgetController::new(test_limits(50));
        let _ = controller.start_turn(Duration::ZERO);
        let intent_a = tool_intent_signature("exec", r#"{"command":"pnpm test"}"#);
        let intent_b = tool_intent_signature("exec", r#"{"command":"pnpm lint"}"#);
        let key_a = typed_failure_key("exec", ToolErrorCode::NonZeroExit, &intent_a);
        let key_b = typed_failure_key("exec", ToolErrorCode::NonZeroExit, &intent_b);
        let _ = controller.record_tool_failure(key_a.clone());
        assert!(matches!(
            controller.record_tool_failure(key_a),
            BudgetDecision::Warning(..)
        ));
        assert!(!controller.take_warning_cleared());
        assert_eq!(
            controller.record_tool_failure(key_b),
            BudgetDecision::Continue
        );
        assert!(controller.take_warning_cleared());
        assert_eq!(controller.snapshot().repeated_root_cause_failures, 1);
    }

    #[test]
    fn identical_non_zero_exit_intent_warns_then_stops() {
        let mut controller = BudgetController::new(test_limits(50));
        let _ = controller.start_turn(Duration::ZERO);
        let intent = tool_intent_signature("exec", r#"{"command":"pnpm test"}"#);
        let key = typed_failure_key("exec", ToolErrorCode::NonZeroExit, &intent);
        assert_eq!(
            controller.record_tool_failure(key.clone()),
            BudgetDecision::Continue
        );
        assert!(matches!(
            controller.record_tool_failure(key.clone()),
            BudgetDecision::Warning(RunBudgetWarning {
                reason: RunBudgetWarningReason::RepeatedRootCause { failures: 2 },
                ..
            })
        ));
        assert!(matches!(
            controller.record_tool_failure(key),
            BudgetDecision::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
                ..
            }
        ));
    }

    #[test]
    fn varied_policy_denied_intents_still_share_one_root_cause() {
        let mut controller = BudgetController::new(test_limits(50));
        let _ = controller.start_turn(Duration::ZERO);
        for step in 0..3 {
            let intent =
                tool_intent_signature("exec", &format!(r#"{{"command":"write artifact-{step}"}}"#));
            let key = typed_failure_key("exec", ToolErrorCode::PolicyDenied, &intent);
            let decision = controller.record_tool_failure(key);
            if step < 1 {
                assert_eq!(decision, BudgetDecision::Continue);
            } else if step == 1 {
                assert!(matches!(
                    decision,
                    BudgetDecision::Warning(RunBudgetWarning {
                        reason: RunBudgetWarningReason::RepeatedRootCause { failures: 2 },
                        ..
                    })
                ));
            } else {
                assert!(matches!(
                    decision,
                    BudgetDecision::Stuck {
                        reason: RunStuckReason::RepeatedRootCause,
                        ..
                    }
                ));
            }
        }
    }

    #[test]
    fn new_evidence_latches_warning_cleared() {
        let mut controller = BudgetController::new(test_limits(50));
        let _ = controller.start_turn(Duration::ZERO);
        let _ = controller.record_tool_failure("exec:policy_denied".into());
        assert!(matches!(
            controller.record_tool_failure("exec:policy_denied".into()),
            BudgetDecision::Warning(..)
        ));
        assert!(!controller.take_warning_cleared());
        let _ = controller.record_progress(ProgressKind::NewEvidence);
        assert!(controller.take_warning_cleared());
        assert!(!controller.take_warning_cleared());
    }
}
