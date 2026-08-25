//! Audit X9: the main reasoning turn pipeline, split out of the former
//! `agent/mod.rs` god-file.
//!
//! Contents: context-hardening/token-budget helpers, background-job
//! notifications, the [`ReasoningLoopExit`] / [`ReasoningLoopError`]
//! terminal states, [`spawn_main_chat_reasoning_turn`], the
//! [`ReasoningLoopCtx`] carrier, and [`AgentLogic::run_reasoning_loop`]
//! itself. Named `reasoning` because `loop` is a Rust keyword.

use std::collections::{HashMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, OnceLock};

use futures::{future::join_all, FutureExt};
use regex::Regex;
use tokio::sync::mpsc;

use super::approval::command_preview;
use super::budget::{
    tool_intent_signature, typed_failure_key, BudgetController, BudgetDecision, BudgetLimits,
    ProgressKind,
};
use super::doom_loop;
use super::failover::{
    build_llm_failed_banner, chat_with_retry, ChatRetryOutcome, FailoverLogCtx, RunProviderContext,
};
use super::steering::{steering_guard, SteeringInbox};
use super::tool_dispatch::{
    execute_tool_call_with_activity, extract_exec_command, hook_observe_telemetry,
    log_tool_invocation_start, parse_tool_arguments, shell_command_uses_grep_like, ToolCallRuntime,
    ToolExecutionFinished,
};
use super::{ActiveRunHandle, AgentLogic, ReasoningSpawnArgs, REDACTED_THINKING_STRIP_RE};
use crate::bus::{
    BusMessage, InboundMessage, LogEvent, OutboundMessage, RunBudgetSnapshot, RunFailureKind,
    RunLifecycleEvent, RunOutcome, RunStuckReason, TelemetryEvent, METADATA_RUN_ID,
};
use crate::clarification::ClarificationHub;
use crate::config::ResolvedShellPolicy;
use crate::hooks::{
    run_user_prompt_hooks, HookSessionInfo, ToolCallHookContext, UserPromptHookOutcome,
};
use crate::logging::LoggerHandle;
use crate::memory::{MemoryMessage, SharedReply, TodoRow};
use crate::session::SessionManager;
use crate::skills::SharedSkillRegistry;
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tool_runtime::ToolExecCtx;
use crate::tools::ToolRegistry;
use crate::traits::{Memory, ToolErrorCode, ToolResult};
use crate::NodeHandle;

pub(crate) async fn load_harness_todos_for_step(
    memory: &NodeHandle<MemoryMessage>,
    chat_id: &str,
) -> Option<Vec<TodoRow>> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    memory
        .send_packet(MemoryMessage::LoadHarnessTodos {
            chat_id: chat_id.to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .ok()?;
    rx.await.ok()?.ok().flatten()
}

pub(crate) fn format_harness_todos_step_block(rows: &[TodoRow]) -> String {
    let mut s = String::from("\n\n--- Harness todos (this step) ---\n");
    for (i, row) in rows.iter().enumerate() {
        let icon = match row.status.as_str() {
            "completed" => "[x]",
            "in_progress" => "[~]",
            _ => "[ ]",
        };
        s.push_str(&format!("{}. {} {}\n", i + 1, icon, row.content));
    }
    s
}

pub(crate) async fn persist_terminal_assistant_message(
    mem: &mut impl Memory,
    logger_tx: &LoggerHandle,
    name: &str,
    chat_id: &str,
    text: &str,
) {
    if let Err(e) = mem
        .add_message(crate::utils::ChatMessage::assistant(text))
        .await
    {
        let _ = logger_tx.send(BusMessage::Log(
            LogEvent::warn(
                name,
                &format!("Failed to persist terminal assistant message: {e}"),
            )
            .with_chat_id(chat_id),
        ));
    }
}

pub(crate) fn metadata_truthy(meta: &HashMap<String, serde_json::Value>, key: &str) -> bool {
    meta.get(key)
        .map(|v| {
            v.as_bool().unwrap_or(false)
                || v.as_str()
                    .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
                    .unwrap_or(false)
        })
        .unwrap_or(false)
}

pub(crate) fn ensure_run_id(inbound: &mut InboundMessage) -> Result<String, String> {
    if let Some(run_id) = inbound
        .metadata
        .get(METADATA_RUN_ID)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|run_id| !run_id.is_empty())
    {
        return Ok(run_id.to_string());
    }

    if inbound.channel.eq_ignore_ascii_case("tauri") {
        return Err("Tauri inbound messages require a non-empty isanagent_run_id".to_string());
    }

    let run_id = format!("legacy-{}", uuid::Uuid::new_v4());
    inbound.metadata.insert(
        METADATA_RUN_ID.to_string(),
        serde_json::Value::String(run_id.clone()),
    );
    Ok(run_id)
}

pub(crate) fn text_looks_like_research_request(content: &str) -> bool {
    static RESEARCH_REQUEST_RE: OnceLock<Option<Regex>> = OnceLock::new();
    RESEARCH_REQUEST_RE
        .get_or_init(|| {
            Regex::new(
                r"(?ix)
                \b(?:
                    research(?:er|ers|ing|ed)? |
                    literature |
                    papers? |
                    state[-\s]+of[-\s]+the[-\s]+art |
                    surveys? |
                    arxiv |
                    evidence |
                    cite |
                    compare\s+methods
                )\b",
            )
            .ok()
        })
        .as_ref()
        .is_some_and(|regex| regex.is_match(content))
}

pub(crate) fn context_has_tool_call(
    context: &[crate::utils::ChatMessage],
    tool_name: &str,
) -> bool {
    context.iter().any(|msg| {
        msg.tool_calls.as_ref().is_some_and(|calls| {
            calls
                .iter()
                .any(|tc| tc.function.name.eq_ignore_ascii_case(tool_name))
        })
    })
}

/// Default context token budget (conservative for most models).
/// Uses char_count / 4 as the token estimate (same heuristic as compaction logic).
const MAX_CONTEXT_TOKENS_DEFAULT: usize = 120_000;

/// After this many *consecutive* doom-loop detections, stop nudging and terminate the run with a
/// "stuck" message. Detection itself already requires 3 repeated calls, so this gives the model
/// two corrective nudges before a hard stop rather than letting it spin to `max_iterations`.
const DOOM_LOOP_HARD_STOP_AFTER: usize = 3;

/// Approximate token count for one message: text content **plus** tool_call argument bytes, /4.
/// Counting tool_call args matters for tool-heavy turns — omitting them under-counts the context
/// and lets the compaction threshold fire late. Shared by context trimming and the
/// auto-compaction threshold so both estimate consistently.
pub(crate) fn estimate_message_tokens(msg: &crate::utils::ChatMessage) -> usize {
    let text_len = msg.content.as_ref().map_or(0, |c| c.text_content().len());
    let args_len = msg.tool_calls.as_ref().map_or(0, |tcs| {
        tcs.iter()
            .map(|t| t.function.arguments.len())
            .sum::<usize>()
    });
    (text_len + args_len) / 4
}

/// Approximate token count for a whole context (sum of [`estimate_message_tokens`]).
pub(crate) fn estimate_context_tokens(context: &[crate::utils::ChatMessage]) -> usize {
    context.iter().map(estimate_message_tokens).sum()
}

/// Best available context-size estimate for the compaction trigger: the larger of the bytes/4
/// heuristic and the last LLM call's exact `usage.prompt_tokens`. The heuristic under-counts the
/// code/JSON/non-English payloads this agent produces, while `prompt_tokens` is the provider's
/// ground-truth input count — so the max guards against silently overflowing the context window.
pub(crate) fn effective_context_tokens(estimate: usize, last_prompt_tokens: Option<u32>) -> usize {
    estimate.max(last_prompt_tokens.unwrap_or(0) as usize)
}

/// Trim context from the front (oldest messages) to stay within a token budget.
/// Preserves the system message at index 0 and never splits tool_call/tool pairs.
/// A marker message is inserted only when messages were actually removed.
pub(crate) fn trim_context_to_budget(
    context: &mut Vec<crate::utils::ChatMessage>,
    max_tokens: usize,
) {
    // Token estimation heuristic: 1 token ≈ 4 characters for English text (text + tool_call args).
    let estimate_msg_tokens = estimate_message_tokens;

    // Quick check: under budget or too small to trim
    if context.len() <= 2 {
        return;
    }
    let total: usize = context.iter().map(&estimate_msg_tokens).sum();
    if total <= max_tokens {
        return;
    }

    // Always remove from index 1 (after system message).
    // Tool call/response pairs are removed atomically.
    // Track remaining tokens to avoid O(N^2) re-computation.
    let mut remaining = total;
    let trim_pos: usize = 1;
    let mut trimmed = false;
    while remaining > max_tokens && trim_pos + 1 < context.len() {
        if context[trim_pos].role == "assistant" && context[trim_pos].tool_calls.is_some() {
            // Find the end of the tool response block
            let mut block_end = trim_pos + 1;
            while block_end < context.len() && context[block_end].role == "tool" {
                block_end += 1;
            }
            // Subtract the tokens for this entire block
            for msg in context[trim_pos..block_end].iter() {
                remaining = remaining.saturating_sub(estimate_msg_tokens(msg));
            }
            context.drain(trim_pos..block_end);
            trimmed = true;
        } else {
            remaining = remaining.saturating_sub(estimate_msg_tokens(&context[trim_pos]));
            context.remove(trim_pos);
            trimmed = true;
        }
    }

    // Only insert marker when we actually removed something
    if trimmed {
        context.insert(
            1,
            crate::utils::ChatMessage::user(
                "[Earlier conversation messages were trimmed to fit context window]",
            ),
        );
    }
}

/// Repair context so every assistant message with `tool_calls` is followed by a tool-role
/// response for each `tool_call_id`. This prevents 400 errors from strict providers (e.g.
/// DeepSeek) when a previous reasoning loop was cancelled mid-tool-execution, leaving
/// orphaned assistant messages in memory without their corresponding tool responses.
pub(crate) fn repair_tool_call_context(context: &mut Vec<crate::utils::ChatMessage>) {
    let mut i = 0;
    while i < context.len() {
        // A tool result without an immediately preceding assistant tool-call
        // block is invalid for strict providers. It can be left behind by
        // legacy/corrupt history, so discard it before context trimming.
        if context[i].role == "tool" {
            context.remove(i);
            continue;
        }

        let tool_call_ids: Vec<String> = match &context[i].tool_calls {
            Some(calls) if context[i].role == "assistant" && !calls.is_empty() => {
                calls.iter().map(|tc| tc.id.clone()).collect()
            }
            _ => {
                i += 1;
                continue;
            }
        };

        // Keep at most one response for each requested id. Mismatched,
        // id-less, and duplicate tool rows are orphaned protocol records and
        // would make strict providers reject the whole request.
        let requested: HashSet<String> = tool_call_ids.iter().cloned().collect();
        let mut responded: HashSet<String> = HashSet::new();
        let mut j = i + 1;
        while j < context.len() && context[j].role == "tool" {
            let keep = context[j]
                .tool_call_id
                .as_deref()
                .is_some_and(|id| requested.contains(id) && responded.insert(id.to_string()));
            if keep {
                j += 1;
            } else {
                context.remove(j);
            }
        }

        // Append placeholder tool responses for any missing tool_call_ids at end of tool block
        let missing: Vec<String> = tool_call_ids
            .into_iter()
            .filter(|id| !responded.contains(id))
            .collect();
        for id in missing {
            context.insert(
                j,
                crate::utils::ChatMessage::tool(
                    "[Cancelled — tool execution interrupted]",
                    &id,
                    None,
                ),
            );
            j += 1;
        }

        i = j;
    }
}

pub(crate) fn should_nudge_research_depth(
    inbound: &crate::bus::InboundMessage,
    context: &[crate::utils::ChatMessage],
) -> bool {
    if !text_looks_like_research_request(&inbound.content) {
        return false;
    }
    let searched = context_has_tool_call(context, "web_search")
        || context_has_tool_call(context, "arxiv_search");
    if !searched {
        return false;
    }
    let has_deep_source_reads = context_has_tool_call(context, "web_fetch")
        || context_has_tool_call(context, "arxiv_fetch")
        || context_has_tool_call(context, "hf_hub_file_fetch")
        || context_has_tool_call(context, "read_file");
    !has_deep_source_reads
}

pub(crate) fn send_background_job_notification(
    outbound_tx: &tokio::sync::mpsc::Sender<BusMessage>,
    channel: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    job_id: &str,
    is_start: bool,
    status: Option<&str>,
) {
    let mut meta = HashMap::new();
    if is_start {
        meta.insert(
            crate::protocol::ISANAGENT_BACKGROUND_JOB_STARTED.to_string(),
            serde_json::json!(true),
        );
        meta.insert(
            crate::protocol::METADATA_BACKGROUND_JOB_TOOL_NAME.to_string(),
            serde_json::json!("background_reasoning"),
        );
    } else {
        meta.insert(
            crate::protocol::ISANAGENT_BACKGROUND_JOB_FINISHED.to_string(),
            serde_json::json!(true),
        );
        if let Some(s) = status {
            meta.insert(
                crate::protocol::METADATA_BACKGROUND_JOB_STATUS.to_string(),
                serde_json::json!(s),
            );
        }
    }
    meta.insert(
        crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
        serde_json::json!(job_id),
    );

    let content = if is_start {
        "Background task started...".to_string()
    } else {
        format!("Background task finished: {}", status.unwrap_or("unknown"))
    };

    let notice = crate::bus::OutboundMessage {
        channel: channel.to_string(),
        chat_id: chat_id.to_string(),
        thread_id: thread_id.map(|s| s.to_string()),
        content,
        metadata: meta,
    };
    let _ = outbound_tx.try_send(BusMessage::Outbound(notice));
}

/// The reasoning loop's terminal state. This deliberately keeps the user-visible
/// assistant text separate from the lifecycle outcome: lifecycle consumers must
/// never infer state from a localized or provider-supplied message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReasoningLoopExit {
    Completed {
        assistant_text: String,
    },
    WaitingForUser {
        ticket_id: String,
    },
    Cancelled {
        assistant_text: String,
    },
    Stuck {
        assistant_text: String,
        reason: RunStuckReason,
    },
    BudgetExhausted {
        assistant_text: String,
        budget: RunBudgetSnapshot,
    },
    Failed {
        assistant_text: String,
        failure: RunFailureKind,
        retryable: bool,
    },
}

impl ReasoningLoopExit {
    pub(crate) fn lifecycle_outcome(&self) -> RunOutcome {
        match self {
            Self::Completed { .. } | Self::WaitingForUser { .. } => RunOutcome::Completed,
            Self::Cancelled { .. } => RunOutcome::Cancelled,
            Self::Stuck { reason, .. } => RunOutcome::Stuck {
                reason: reason.clone(),
            },
            Self::BudgetExhausted { budget, .. } => RunOutcome::BudgetExhausted {
                budget: budget.clone(),
            },
            Self::Failed {
                failure, retryable, ..
            } => RunOutcome::Failed {
                failure: failure.clone(),
                retryable: *retryable,
            },
        }
    }

    pub(crate) fn assistant_text(&self) -> Option<&str> {
        match self {
            Self::Completed { assistant_text }
            | Self::Cancelled { assistant_text }
            | Self::Stuck { assistant_text, .. }
            | Self::BudgetExhausted { assistant_text, .. }
            | Self::Failed { assistant_text, .. } => Some(assistant_text),
            Self::WaitingForUser { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReasoningLoopError {
    pub(crate) message: String,
    failure: RunFailureKind,
    retryable: bool,
}

impl ReasoningLoopError {
    fn persistence(error: impl std::fmt::Display) -> Self {
        Self {
            message: error.to_string(),
            failure: RunFailureKind::Persistence,
            retryable: false,
        }
    }

    fn protocol(message: String) -> Self {
        Self {
            message,
            failure: RunFailureKind::Protocol,
            retryable: false,
        }
    }

    pub(crate) fn lifecycle_outcome(&self) -> RunOutcome {
        RunOutcome::Failed {
            failure: self.failure.clone(),
            retryable: self.retryable,
        }
    }
}

impl From<String> for ReasoningLoopError {
    fn from(message: String) -> Self {
        Self {
            message,
            failure: RunFailureKind::Internal,
            retryable: false,
        }
    }
}

pub(crate) fn spawn_main_chat_reasoning_turn(
    args: ReasoningSpawnArgs,
    inbound: crate::bus::InboundMessage,
    run_id: String,
    run_provider: RunProviderContext,
) {
    let chat_id = inbound.chat_id.clone();
    let cancel_token = Arc::new(tokio_util::sync::CancellationToken::new());
    let steering = Arc::new(Mutex::new(SteeringInbox::open()));
    args.cancellation_tokens.insert(
        chat_id.clone(),
        ActiveRunHandle {
            run_id: run_id.clone(),
            token: cancel_token.clone(),
            steering: steering.clone(),
        },
    );

    let args_for_chain = args.clone();
    let cancellation_tokens = args.cancellation_tokens.clone();
    let pending_inbound = args.pending_inbound.clone();
    let name = args.name.clone();
    let session_manager = args.session_manager.clone();
    let session_manager_for_chain = session_manager.clone();
    let tools = args.tools.clone();
    let skills = args.skills.clone();
    let system_prompt = args.system_prompt.clone();
    let max_iterations = args.max_iterations;
    let max_tool_output_chars = args.max_tool_output_chars;
    let max_recent_summaries = args.max_recent_summaries;
    let short_term_threshold_turns = args.short_term_threshold_turns;
    let short_term_threshold_tokens = args.short_term_threshold_tokens;
    let tool_execution_activity = args.tool_execution_activity.clone();
    let outbound_tx = args.outbound_tx.clone();
    let logger_tx = args.logger_tx.clone();
    let clarification_hub = args.clarification_hub.clone();
    let doom_loop_enabled = args.doom_loop_enabled;
    let harness_runtime_summary = args.harness_runtime_summary.clone();
    let forbid_final_without_tools = args.forbid_final_without_tools;
    let shell_policy = args.shell_policy.clone();
    const BACKGROUND_METADATA_KEYS: &[&str] = &[
        crate::bus::METADATA_SYNTHETIC_CRON_TRIGGER,
        crate::bus::METADATA_SYNTHETIC_JOB_FOLLOWUP,
        crate::bus::METADATA_SYNTHETIC_SUBAGENT_COMPLETION,
        crate::bus::METADATA_SYNTHETIC_BACKGROUND_RESUME,
    ];
    let is_background_turn = BACKGROUND_METADATA_KEYS
        .iter()
        .any(|&key| metadata_truthy(&inbound.metadata, key));
    let background_job_id = crate::bus::get_background_job_id(&inbound.metadata);
    let inbound_metadata = Arc::new(inbound.metadata.clone());
    let tool_exec_ctx = ToolExecCtx::new(
        inbound.channel.clone(),
        inbound.chat_id.clone(),
        inbound.thread_id.clone(),
    )
    .with_background(is_background_turn)
    .with_reasoning_cancel(cancel_token.as_ref().clone())
    .with_metadata(inbound_metadata.clone());
    let inbound_channel = inbound.channel.clone();
    let inbound_thread_id = inbound.thread_id.clone();
    let session_key = crate::bus::clarification_session_key(
        &inbound_channel,
        &chat_id,
        inbound_thread_id.as_deref(),
    );
    let hook_tool_ctx = args.hook_tool_ctx.clone();

    tokio::spawn(async move {
        let task_chat_id = chat_id.clone();
        let task_token_arc = cancel_token.clone();

        let agent_name = name.clone();
        let _ = logger_tx.send(BusMessage::Log(
            LogEvent::debug(
                &agent_name,
                &format!("Spawning reasoning task for chat_id: {task_chat_id}"),
            )
            .with_chat_id(&task_chat_id),
        ));

        if outbound_tx
            .send(BusMessage::RunLifecycle(RunLifecycleEvent::Started {
                run_id: run_id.clone(),
                chat_id: task_chat_id.clone(),
            }))
            .await
            .is_err()
        {
            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::warn(
                    &agent_name,
                    "Could not deliver RunLifecycle::Started; continuing reasoning task.",
                )
                .with_chat_id(&task_chat_id),
            ));
        }

        if let Some(ref jid) = background_job_id {
            send_background_job_notification(
                &outbound_tx,
                &inbound_channel,
                &task_chat_id,
                inbound_thread_id.as_deref(),
                jid,
                true,
                None,
            );
        }

        let res = AssertUnwindSafe(AgentLogic::run_reasoning_loop(ReasoningLoopCtx {
            name,
            run_provider,
            session_manager,
            tools,
            skills,
            system_prompt,
            max_iterations,
            max_tool_output_chars,
            max_recent_summaries,
            short_term_threshold_turns,
            short_term_threshold_tokens,
            tool_execution_activity,
            outbound_tx: outbound_tx.clone(),
            logger_tx: logger_tx.clone(),
            inbound,
            run_id: run_id.clone(),
            steering,
            cancel_token: task_token_arc.as_ref().clone(),
            clarification_hub,
            tool_exec_ctx,
            is_subagent: false,
            subagent_allowlist: None,
            doom_loop_enabled,
            harness_runtime_summary: harness_runtime_summary.clone(),
            forbid_final_without_tools,
            shell_policy: shell_policy.clone(),
            hook_tool_ctx,
            inbound_metadata,
        }))
        .catch_unwind()
        .await;

        match res {
            Ok(Err(ref e)) => {
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::error(
                        "AgentLogic",
                        &format!(
                            "Reasoning loop failed for chat_id {}: {}",
                            task_chat_id, e.message
                        ),
                    )
                    .with_chat_id(&task_chat_id),
                ));
                let notice = crate::protocol::build_channel_error_notice(
                    &inbound_channel,
                    &task_chat_id,
                    inbound_thread_id.as_deref(),
                    &e.message,
                );
                let _ = outbound_tx.send(BusMessage::Outbound(notice)).await;
            }
            Ok(Ok(ReasoningLoopExit::Cancelled { .. })) => {
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::info(
                        &agent_name,
                        &format!(
                            "Reasoning task for chat_id {task_chat_id} finished via cancellation."
                        ),
                    )
                    .with_chat_id(&task_chat_id),
                ));
            }
            Ok(Ok(_)) => {
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::debug(
                        &agent_name,
                        &format!(
                            "Reasoning task for chat_id {task_chat_id} finished successfully."
                        ),
                    )
                    .with_chat_id(&task_chat_id),
                ));
            }
            Err(_) => {
                let panic_msg = "Internal error: reasoning loop panicked and was stopped.";
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::error(
                        "AgentLogic",
                        &format!("Reasoning loop panicked for chat_id {task_chat_id}"),
                    )
                    .with_chat_id(&task_chat_id),
                ));

                if let Ok(mut mem) = session_manager_for_chain.get_session(&session_key).await {
                    persist_terminal_assistant_message(
                        &mut mem,
                        &logger_tx,
                        &agent_name,
                        &task_chat_id,
                        panic_msg,
                    )
                    .await;
                }

                let notice = crate::protocol::build_channel_error_notice(
                    &inbound_channel,
                    &task_chat_id,
                    inbound_thread_id.as_deref(),
                    panic_msg,
                );
                let _ = outbound_tx.send(BusMessage::Outbound(notice)).await;
            }
        }

        let outcome = match &res {
            Ok(Ok(exit)) => exit.lifecycle_outcome(),
            Ok(Err(error)) => error.lifecycle_outcome(),
            Err(_) => RunOutcome::Failed {
                failure: RunFailureKind::Internal,
                retryable: false,
            },
        };
        if outbound_tx
            .send(BusMessage::RunLifecycle(RunLifecycleEvent::Terminated {
                run_id: run_id.clone(),
                chat_id: task_chat_id.clone(),
                outcome,
            }))
            .await
            .is_err()
        {
            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::warn(
                    &agent_name,
                    "Could not deliver RunLifecycle::Terminated after reasoning task completion.",
                )
                .with_chat_id(&task_chat_id),
            ));
        }

        if let Some(job_id) = background_job_id {
            let (state, last_error) = match &res {
                Ok(Ok(ReasoningLoopExit::WaitingForUser { .. })) => ("waiting", None),
                Ok(Ok(ReasoningLoopExit::Cancelled { .. })) => {
                    ("failed", Some("Cancelled".to_string()))
                }
                Ok(Ok(ReasoningLoopExit::Failed { assistant_text, .. })) => {
                    ("failed", Some(assistant_text.clone()))
                }
                Ok(Ok(_)) => ("completed", None),
                Ok(Err(e)) => ("failed", Some(e.message.clone())),
                Err(_) => ("failed", Some("Panic in reasoning loop".to_string())),
            };

            // Send FINISHED notice to TUI
            send_background_job_notification(
                &outbound_tx,
                &inbound_channel,
                &task_chat_id,
                inbound_thread_id.as_deref(),
                &job_id,
                false,
                Some(state),
            );

            let (tx, rx) = tokio::sync::oneshot::channel();
            let memory_node = session_manager_for_chain.get_memory_node();
            let _ = memory_node
                .send_packet(MemoryMessage::UpdateBackgroundJobState {
                    job_id: job_id.clone(),
                    state: state.to_string(),
                    last_error,
                    reply: SharedReply::new(tx),
                })
                .await;
            let _ = rx.await;
        }

        let _ = cancellation_tokens.remove_if(&task_chat_id, |_key, stored| {
            Arc::ptr_eq(&stored.token, &task_token_arc)
        });

        let mut next_inbound = pending_inbound.get(&task_chat_id).and_then(|r| {
            let mut g = match r.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    let _ = logger_tx.send(BusMessage::Log(
                        LogEvent::warn(
                            "AgentLogic",
                            "pending_inbound mutex poisoned after reasoning turn; recovering queue.",
                        )
                        .with_chat_id(&task_chat_id),
                    ));
                    poisoned.into_inner()
                }
            };
            g.pop_front()
        });

        while let Some(mut queued) = next_inbound {
            let next_from_queue = || {
                pending_inbound.get(&task_chat_id).and_then(|r| {
                    let mut g = match r.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => {
                            let _ = logger_tx.send(BusMessage::Log(
                                LogEvent::warn(
                                    "AgentLogic",
                                    "pending_inbound mutex poisoned after reasoning turn; recovering queue.",
                                )
                                .with_chat_id(&task_chat_id),
                            ));
                            poisoned.into_inner()
                        }
                    };
                    g.pop_front()
                })
            };

            match ensure_run_id(&mut queued.inbound) {
                Ok(next_run_id) => {
                    spawn_main_chat_reasoning_turn(
                        args_for_chain,
                        queued.inbound,
                        next_run_id,
                        queued.run_provider,
                    );
                    break;
                }
                Err(error) => {
                    let _ = logger_tx.send(BusMessage::Log(
                        LogEvent::error(
                            "AgentLogic",
                            &format!("Dropping queued inbound without valid run ID: {error}"),
                        )
                        .with_chat_id(&task_chat_id),
                    ));
                    next_inbound = next_from_queue();
                }
            }
        }
    });
}

pub(crate) struct ReasoningLoopCtx {
    pub(crate) name: String,
    pub(crate) run_provider: RunProviderContext,
    pub(crate) session_manager: Arc<SessionManager>,
    pub(crate) tools: Arc<ToolRegistry>,
    pub(crate) skills: SharedSkillRegistry,
    pub(crate) system_prompt: String,
    pub(crate) max_iterations: usize,
    pub(crate) max_tool_output_chars: usize,
    pub(crate) max_recent_summaries: usize,
    pub(crate) short_term_threshold_turns: usize,
    pub(crate) short_term_threshold_tokens: usize,
    pub(crate) tool_execution_activity: Option<SharedToolExecutionActivity>,
    pub(crate) outbound_tx: mpsc::Sender<BusMessage>,
    pub(crate) logger_tx: LoggerHandle,
    pub(crate) inbound: crate::bus::InboundMessage,
    pub(crate) run_id: String,
    pub(crate) steering: Arc<Mutex<SteeringInbox>>,
    pub(crate) cancel_token: tokio_util::sync::CancellationToken,
    pub(crate) clarification_hub: Arc<ClarificationHub>,
    pub(crate) tool_exec_ctx: ToolExecCtx,
    /// When true, tool list / execution use sub-agent allowlist and deny nested spawn/plan tools.
    pub(crate) is_subagent: bool,
    pub(crate) subagent_allowlist: Option<Arc<HashSet<String>>>,
    pub(crate) doom_loop_enabled: bool,
    pub(crate) harness_runtime_summary: String,
    pub(crate) forbid_final_without_tools: bool,
    pub(crate) shell_policy: Arc<ResolvedShellPolicy>,
    pub(crate) hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
    pub(crate) inbound_metadata: Arc<HashMap<String, serde_json::Value>>,
}

impl AgentLogic {
    pub(crate) async fn run_reasoning_loop(
        ctx: ReasoningLoopCtx,
    ) -> Result<ReasoningLoopExit, ReasoningLoopError> {
        let ReasoningLoopCtx {
            name,
            run_provider,
            session_manager,
            tools,
            skills,
            system_prompt,
            max_iterations,
            max_tool_output_chars,
            max_recent_summaries,
            short_term_threshold_turns,
            short_term_threshold_tokens,
            tool_execution_activity,
            outbound_tx,
            logger_tx,
            inbound,
            run_id,
            steering,
            cancel_token,
            clarification_hub,
            tool_exec_ctx,
            is_subagent,
            subagent_allowlist,
            doom_loop_enabled,
            harness_runtime_summary,
            forbid_final_without_tools,
            shell_policy,
            hook_tool_ctx,
            inbound_metadata,
        } = ctx;

        let RunProviderContext {
            provider,
            fallback_providers,
            identity: provider_identity,
        } = run_provider;

        let session_key = tool_exec_ctx.session_key.clone();
        let cancel_notice = "Request cancelled while the agent was processing this turn.";

        let mut mem = session_manager
            .get_session(&session_key)
            .await
            .map_err(ReasoningLoopError::persistence)?;

        let _ = logger_tx.send(BusMessage::Log(
            LogEvent::debug(
                &name,
                &format!(
                    "Run provider snapshot provider={} model={} credential={} fallbacks={}",
                    provider_identity.provider_name,
                    provider_identity.model_name,
                    provider_identity.secret_identity,
                    fallback_providers.len(),
                ),
            )
            .with_chat_id(&inbound.chat_id),
        ));

        let run_started_at = std::time::Instant::now();
        let mut budget = BudgetController::new(BudgetLimits::for_run(max_iterations));

        macro_rules! apply_budget_decision {
            ($decision:expr) => {{
                match $decision {
                    BudgetDecision::Continue => {}
                    BudgetDecision::Warning(warning) => {
                        // A new live warning supersedes any clearance latch from this
                        // observation (e.g. NoProgress cleared then ApproachingLimit raised).
                        let _ = budget.take_warning_cleared();
                        let _ = logger_tx.send(BusMessage::Log(
                            LogEvent::warn(
                                &name,
                                &format!("Run budget warning: {:?}", warning.reason),
                            )
                            .with_chat_id(&inbound.chat_id),
                        ));
                        let _ = outbound_tx
                            .send(BusMessage::RunLifecycle(RunLifecycleEvent::Warning {
                                run_id: run_id.clone(),
                                chat_id: inbound.chat_id.clone(),
                                warning,
                            }))
                            .await;
                    }
                    BudgetDecision::Stuck {
                        reason,
                        snapshot: _,
                    } => {
                        let message = match reason {
                            RunStuckReason::RepeatedRootCause => {
                                "Stopped: the same typed tool failure repeated without measurable \
                                 progress. Change the failing input or policy, steer the run, or \
                                 continue with a new budget segment."
                            }
                            RunStuckReason::NoProgress => {
                                "Stopped: the run continued without observable progress. Steer the \
                                 run, break the task into a smaller step, or continue with a new \
                                 budget segment."
                            }
                            RunStuckReason::DoomLoop => {
                                "Stopped: the agent kept repeating the same action with no progress."
                            }
                        }
                        .to_string();
                        persist_terminal_assistant_message(
                            &mut mem,
                            &logger_tx,
                            &name,
                            &inbound.chat_id,
                            &message,
                        )
                        .await;
                        let _ = outbound_tx
                            .send(BusMessage::Outbound(OutboundMessage {
                                channel: inbound.channel.clone(),
                                chat_id: inbound.chat_id.clone(),
                                thread_id: inbound.thread_id.clone(),
                                content: message.clone(),
                                metadata: HashMap::new(),
                            }))
                            .await;
                        return Ok(ReasoningLoopExit::Stuck {
                            assistant_text: message,
                            reason,
                        });
                    }
                    BudgetDecision::BudgetExhausted(snapshot) => {
                        let limit = match snapshot.exhausted_limit {
                            Some(crate::bus::RunBudgetLimit::LlmTurns) => "LLM-turn",
                            Some(crate::bus::RunBudgetLimit::WallTime) => "wall-time",
                            Some(crate::bus::RunBudgetLimit::Tokens) => "token",
                            Some(crate::bus::RunBudgetLimit::ProviderRetries) => {
                                "provider-retry"
                            }
                            Some(crate::bus::RunBudgetLimit::ContextRecoveries) => {
                                "context-recovery"
                            }
                            None => "run",
                        };
                        let message = format!(
                            "Stopped: the run exhausted its {limit} budget after {} LLM turns. \
                             Continue with a new budget segment or steer the task.",
                            snapshot.iterations_used
                        );
                        persist_terminal_assistant_message(
                            &mut mem,
                            &logger_tx,
                            &name,
                            &inbound.chat_id,
                            &message,
                        )
                        .await;
                        let _ = outbound_tx
                            .send(BusMessage::Outbound(OutboundMessage {
                                channel: inbound.channel.clone(),
                                chat_id: inbound.chat_id.clone(),
                                thread_id: inbound.thread_id.clone(),
                                content: message.clone(),
                                metadata: HashMap::new(),
                            }))
                            .await;
                        return Ok(ReasoningLoopExit::BudgetExhausted {
                            assistant_text: message,
                            budget: snapshot,
                        });
                    }
                }
                if budget.take_warning_cleared() {
                    let _ = outbound_tx
                        .send(BusMessage::RunLifecycle(
                            RunLifecycleEvent::WarningCleared {
                                run_id: run_id.clone(),
                                chat_id: inbound.chat_id.clone(),
                            },
                        ))
                        .await;
                }
            }};
        }

        macro_rules! persist_and_cancel {
            () => {{
                persist_terminal_assistant_message(
                    &mut mem,
                    &logger_tx,
                    &name,
                    &inbound.chat_id,
                    cancel_notice,
                )
                .await;
                return Ok(ReasoningLoopExit::Cancelled {
                    assistant_text: cancel_notice.to_string(),
                });
            }};
        }

        let forbid_final_effective = !is_subagent
            && (forbid_final_without_tools
                || metadata_truthy(
                    &inbound.metadata,
                    crate::bus::METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS,
                ));
        let unattended_session = metadata_truthy(&inbound.metadata, "isanagent_autonomous_session")
            || inbound.metadata.contains_key("isanagent_autonomous_until");

        // 1. Build runtime context and prepend to User message before adding to memory
        let thread_info = inbound
            .thread_id
            .as_deref()
            .map(|t| format!(", thread: '{t}'"))
            .unwrap_or_default();
        let now = chrono::Local::now().to_rfc3339();
        let os_family = std::env::consts::OS;
        let path_sep = std::path::MAIN_SEPARATOR;
        let (shell_family, exec_runner) = if cfg!(windows) {
            match shell_policy.windows_runner {
                crate::config::WindowsShellRunner::Cmd => ("cmd", "cmd.exe /C"),
                crate::config::WindowsShellRunner::PowerShell => {
                    ("powershell", "powershell.exe -Command")
                }
                crate::config::WindowsShellRunner::Pwsh => ("pwsh", "pwsh.exe -Command"),
            }
        } else if std::env::var("SHELL")
            .ok()
            .map(|s| s.contains("bash"))
            .unwrap_or(false)
        {
            ("bash", "sh -c")
        } else {
            ("sh", "sh -c")
        };
        let mut runtime_context = format!(
            "[RUNTIME CONTEXT] Current time is {}. You are navigating and responding in channel: '{}', with chat ID: '{}'{}.",
            now,
            inbound.channel,
            inbound.chat_id,
            thread_info
        );
        runtime_context.push_str(&format!(
            " Host hints: os_family='{}', shell_family='{}', exec_runner='{}', path_separator='{}', windows={}.",
            os_family,
            shell_family,
            exec_runner,
            path_sep,
            cfg!(windows)
        ));
        if let Ok(term) = std::env::var("TERM") {
            if !term.trim().is_empty() {
                runtime_context.push_str(&format!(" terminal='{}'.", term.trim()));
            }
        }
        if let Some(v) = inbound.metadata.get("isanagent_autonomous_until") {
            if let Some(s) = v.as_str() {
                runtime_context
                    .push_str(&format!(" Autonomous session deadline (RFC3339): '{s}'."));
            }
        }
        if forbid_final_effective {
            runtime_context.push_str(
                " This session expects tool use until work is complete — avoid ending on plain text alone.",
            );
        }
        runtime_context.push_str(crate::utils::RUNTIME_CONTEXT_END_SUFFIX);

        let mut contextualized_content = format!("{}{}", runtime_context, inbound.content);
        if let Some(ref hc) = hook_tool_ctx {
            if let Some(st) = &hc.steering {
                let hook_session = HookSessionInfo {
                    channel: inbound.channel.as_str(),
                    chat_id: inbound.chat_id.as_str(),
                    thread_id: inbound.thread_id.as_deref(),
                    metadata: &inbound.metadata,
                    is_subagent,
                };
                match run_user_prompt_hooks(
                    st.as_ref(),
                    contextualized_content.as_str(),
                    hook_session,
                )
                .await
                {
                    UserPromptHookOutcome::Block(msg) => {
                        return Err(ReasoningLoopError::protocol(msg));
                    }
                    UserPromptHookOutcome::InjectPrefix(prefix) => {
                        contextualized_content = format!("{prefix}\n{contextualized_content}");
                    }
                    UserPromptHookOutcome::Proceed => {}
                }
            }
        }

        // Build the user message – multimodal when attachments are present
        let user_msg = if inbound.attachments.is_empty() {
            crate::utils::ChatMessage::user(&contextualized_content)
        } else {
            crate::utils::ChatMessage::user_multimodal(
                &contextualized_content,
                &inbound.attachments,
            )
        };
        mem.add_message(user_msg)
            .await
            .map_err(ReasoningLoopError::persistence)?;

        // Emit an initial thought so the user knows reasoning has started
        let _ = outbound_tx
            .send(BusMessage::Telemetry(TelemetryEvent::AgentThought {
                chat_id: inbound.chat_id.clone(),
                thought: "I am starting to process your request...".to_string(),
                background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
            }))
            .await;

        let thinking_strip_re = REDACTED_THINKING_STRIP_RE.get_or_init(|| {
            Regex::new(crate::utils::REDACTED_THINKING_STRIP_PATTERN)
                .expect("redacted thinking strip regex")
        });

        // 2. Loop until no more tool calls or max iterations reached
        let mut iterations: usize;
        // PR-4.1: hard cap — at most one emergency context-overflow recovery
        // per inbound. A second overflow within the same turn surfaces the
        // failure to the user instead of looping on compact-and-retry.
        let mut overflow_recovery_used = false;
        // P1.4: count consecutive doom-loop detections so we can escalate from an advisory
        // nudge to a hard stop. Reset to 0 on any iteration with no detection.
        let mut consecutive_doom_detections: usize = 0;
        // Ground-truth input size from the most recent LLM call's `usage.prompt_tokens` (exact,
        // server-counted). The bytes/4 heuristic under-counts code/JSON/non-English — exactly what
        // this agent generates — so the compaction trigger uses `max(estimate, last_prompt_tokens)`
        // to avoid silently overflowing the context window. Updated after each provider response.
        let mut last_prompt_tokens: Option<u32> = None;

        // Bounds the `forbid_final_without_tools` push: after this many consecutive
        // text-only replies (no tool call), stop injecting the same nudge. If the
        // model then proposes prose-only completion, the progress controller turns
        // the unresolved warning into `Stuck::NoProgress` instead of claiming the
        // task completed. Any tool call resets the count.
        const MAX_FORBID_FINAL_NUDGES: u32 = 3;
        let mut forbid_final_nudges: u32 = 0;

        loop {
            if cancel_token.is_cancelled() {
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::info(&name, "Reasoning loop cancelled before iteration start.")
                        .with_chat_id(&inbound.chat_id),
                ));
                persist_and_cancel!();
            }
            // Tool paths return to the loop only after their result has been
            // persisted. Consume steering before this next iteration performs
            // compaction, calls another tool, or calls the provider.
            let pending_steering = steering_guard(&steering).drain();
            if !pending_steering.is_empty() {
                for content in pending_steering {
                    mem.add_message(crate::utils::ChatMessage::user(&content))
                        .await
                        .map_err(ReasoningLoopError::persistence)?;
                }
                consecutive_doom_detections = 0;
                forbid_final_nudges = 0;
                apply_budget_decision!(budget.record_progress(ProgressKind::Steering));
            }

            apply_budget_decision!(budget.start_turn(run_started_at.elapsed()));
            iterations = budget.snapshot().iterations_used;

            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::debug(&name, &format!("Iteration {iterations}/{max_iterations}"))
                    .with_chat_id(&inbound.chat_id),
            ));

            // PR-7.2: stale tool-result swap pass. Runs at the top of every
            // iteration (independent of the compaction threshold). For tool
            // results older than `KEEP_RECENT_USER_TURNS_DEFAULT` user turns
            // from the latest, cache the original and replace the stored
            // content with a compact placeholder. The swap is mirrored into
            // the in-memory `messages_with_ids` so the reasoning context is
            // built directly from it below — no second fetch. Idempotent — the
            // helper skips messages already in placeholder form.
            //
            // Cost per iteration: 1 SELECT + N UPDATE pairs (where N = number
            // of newly-stale tool messages, typically 0 except right after a
            // tool-heavy iteration). The helper does no I/O itself; all writes
            // go through the memory actor and serialize.
            let mut messages_with_ids: Vec<(i64, crate::utils::ChatMessage)> = {
                let memory_node = session_manager.get_memory_node();
                let (tx, rx) = tokio::sync::oneshot::channel();
                let _ = memory_node
                    .send_packet(crate::memory::MemoryMessage::GetMessagesSinceReflection {
                        thread_id: session_key.clone(),
                        reply: crate::memory::SharedReply::new(tx),
                    })
                    .await;
                rx.await
                    .ok()
                    .and_then(|r| r.ok())
                    .map(|(rows, _)| rows)
                    .unwrap_or_default()
            };
            {
                let memory_node = session_manager.get_memory_node();
                let stale = crate::agent::compaction::identify_stale_tool_swaps(
                    &messages_with_ids,
                    crate::agent::compaction::KEEP_RECENT_USER_TURNS_DEFAULT,
                );
                for (db_id, tool_call_id, tool_name, full_content, placeholder) in stale {
                    let (tx, rx) = tokio::sync::oneshot::channel();
                    let compact = crate::agent::compaction::build_compact_placeholder(
                        &tool_call_id,
                        &tool_name,
                        &full_content,
                    );
                    let _ = memory_node
                        .send_packet(crate::memory::MemoryMessage::CacheToolResult {
                            tool_call_id: tool_call_id.clone(),
                            chat_id: inbound.chat_id.clone(),
                            session_key: session_key.clone(),
                            tool_name,
                            full_content,
                            compact_summary: compact,
                            reply: crate::memory::SharedReply::new(tx),
                        })
                        .await;
                    let _ = rx.await;

                    let (tx, rx) = tokio::sync::oneshot::channel();
                    let _ = memory_node
                        .send_packet(crate::memory::MemoryMessage::UpdateMessageContent {
                            message_id: db_id,
                            new_content: placeholder.clone(),
                            reply: crate::memory::SharedReply::new(tx),
                        })
                        .await;
                    let _ = rx.await;

                    // Mirror the swap in the already-fetched in-memory copy so
                    // the context built below matches the persisted form.
                    if let Some((_, msg)) =
                        messages_with_ids.iter_mut().find(|(id, _)| *id == db_id)
                    {
                        msg.content = Some(crate::utils::MessageContent::Text(placeholder));
                    }
                }
            }

            // Fetch context — reuse the messages already fetched for the stale
            // swap pass (get_context_since_reflection issues the identical
            // GetMessagesSinceReflection query) instead of a second round-trip.
            let mut context: Vec<crate::utils::ChatMessage> =
                messages_with_ids.into_iter().map(|(_, m)| m).collect();

            // Strip any legacy static system prompts that SQLite may have persisted
            context.retain(|msg| msg.role != "system");

            // Repair any orphaned tool_calls (e.g. from a cancelled previous iteration)
            repair_tool_call_context(&mut context);

            // Fetch short term memory summaries
            let prefix = format!("{}:{}", inbound.channel, inbound.chat_id);
            let summaries = if max_recent_summaries > 0 {
                session_manager
                    .get_recent_summaries(&prefix, max_recent_summaries)
                    .await
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let summaries_text = if !summaries.is_empty() {
                format!(
                    "--- RECENT CONVERSATION SUMMARIES (SHORT-TERM MEMORY) ---\n{}",
                    summaries.join("\n\n")
                )
            } else {
                String::new()
            };

            // Inject the latest static system prompt to the beginning of the context
            let harness_block = if harness_runtime_summary.trim().is_empty() {
                String::new()
            } else {
                format!(
                    "\n\n--- Harness snapshot (this step) ---\n{}\n",
                    harness_runtime_summary.trim()
                )
            };
            let todo_block = if !is_subagent {
                match load_harness_todos_for_step(
                    &session_manager.get_memory_node(),
                    &inbound.chat_id,
                )
                .await
                {
                    Some(rows) if !rows.is_empty() => format_harness_todos_step_block(&rows),
                    _ => String::new(),
                }
            } else {
                String::new()
            };
            let iteration_line = format!(
                "\n--- Reasoning budget ---\nYou are on tool/LLM step {iterations} of {max_iterations} for this user turn.\n"
            );
            let autonomy_line = if forbid_final_effective {
                "\n--- Autonomy ---\nDo not finish this step with assistant text only — call tools (or `ask_user` if blocked). If you believe you are done, still run a verification tool (e.g. read_file or execution_env_info) when appropriate.\n"
            } else {
                ""
            };
            let mut system_body = format!(
                "{}\n\n{}\n{}{}{}{}",
                system_prompt,
                summaries_text,
                skills.read().await.get_capabilities_summary(),
                harness_block,
                todo_block,
                iteration_line
            )
            .trim_end()
            .to_string();
            system_body.push_str(autonomy_line);
            let system_msg = crate::utils::ChatMessage::system(&system_body);
            context.insert(0, system_msg);

            if doom_loop_enabled {
                if let Some(prompt) = doom_loop::check_for_doom_loop_prompt(&context) {
                    // Escalate ONLY while the loop is still active at the tail — a stale run can
                    // linger in the lookback window after the model corrects itself, and counting
                    // those would hard-stop a model that already recovered (see
                    // `doom_loop_active_at_tail`). The advisory nudge still fires on any detection.
                    if doom_loop::doom_loop_active_at_tail(&context) {
                        consecutive_doom_detections += 1;
                        if consecutive_doom_detections >= DOOM_LOOP_HARD_STOP_AFTER {
                            // Nudges didn't break the loop — hard stop so the run doesn't spin to
                            // max_iterations re-receiving the same advice.
                            let _ = logger_tx.send(BusMessage::Log(
                                LogEvent::warn(
                                    &name,
                                    &format!(
                                        "Doom loop still active after {consecutive_doom_detections} consecutive detections — stopping the run."
                                    ),
                                )
                                .with_chat_id(&inbound.chat_id),
                            ));
                            break;
                        }
                    } else {
                        // Detected in the window but not active at the tail — the model varied
                        // this turn, so don't escalate. (A counter *decay* here, to also catch
                        // intermittently-varying loops, is plausible but its benefit is narrow and
                        // hard to test deterministically — left as a deferred follow-up.)
                        consecutive_doom_detections = 0;
                    }
                    let correction = crate::utils::ChatMessage::user(&prompt);
                    mem.add_message(correction.clone())
                        .await
                        .map_err(ReasoningLoopError::persistence)?;
                    context.push(correction);
                    let _ = logger_tx.send(BusMessage::Log(
                        LogEvent::warn(
                            &name,
                            "Doom loop detected — injecting corrective user message.",
                        )
                        .with_chat_id(&inbound.chat_id),
                    ));
                } else {
                    // No loop in the window — reset so only *consecutive active* loops escalate.
                    consecutive_doom_detections = 0;
                }
            }

            // Trim context to stay within token budget before calling the provider
            trim_context_to_budget(&mut context, MAX_CONTEXT_TOKENS_DEFAULT);
            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::debug(
                    &name,
                    &format!("Calling provider.chat (context size: {})", context.len()),
                )
                .with_chat_id(&inbound.chat_id),
            ));

            // Call Provider
            let tools_payload = Some(serde_json::json!(
                tools.list_tools_scoped(subagent_allowlist.as_deref(), is_subagent)
            ));

            let (response, provider_retries) = match chat_with_retry(
                provider.as_ref(),
                &context,
                tools_payload,
                &fallback_providers,
                &cancel_token,
                FailoverLogCtx {
                    logger_tx: &logger_tx,
                    name: &name,
                    chat_id: &inbound.chat_id,
                },
            )
            .await
            {
                ChatRetryOutcome::Ok { response, retries } => (response, retries),
                ChatRetryOutcome::Cancelled => {
                    persist_and_cancel!();
                }
                ChatRetryOutcome::ContextOverflow {
                    tokens_attempted,
                    max,
                } => {
                    // PR-4.1: emergency compact-and-retry. The first overflow per
                    // turn fires `do_compaction` (same code path as the threshold
                    // trigger) and continues to the next iteration — which
                    // refetches a smaller post-compaction context and will retry
                    // the chat call naturally. A second overflow in the same turn
                    // surfaces the failure to the user.
                    let tokens_before_u32 =
                        estimate_context_tokens(&context).min(u32::MAX as usize) as u32;
                    let turns_before_u32 = context
                        .iter()
                        .filter(|m| m.role == "user")
                        .count()
                        .min(u32::MAX as usize) as u32;

                    if !overflow_recovery_used {
                        overflow_recovery_used = true;
                        let memory_node = session_manager.get_memory_node();
                        let outcome = crate::agent::compaction::do_compaction(
                            crate::agent::compaction::DoCompactionArgs {
                                chat_id: &inbound.chat_id,
                                session_key: &session_key,
                                trigger_reason: crate::bus::CompactionTrigger::Overflow400,
                                tokens_before: tokens_before_u32,
                                turns_before: turns_before_u32,
                                current_context: &context,
                                existing_summary: summaries.first().map(|s| s.as_str()),
                                focus_instructions: None,
                                provider: provider.as_ref(),
                                memory_node: &memory_node,
                                outbound_tx: &outbound_tx,
                                cancel_token: &cancel_token,
                            },
                        )
                        .await;
                        match outcome {
                            crate::agent::compaction::CompactionOutcome::Succeeded => {
                                let _ = logger_tx.send(BusMessage::Log(
                                    LogEvent::info(
                                        &name,
                                        &format!(
                                            "Emergency compaction succeeded after context overflow (attempted={} max={}); retrying iteration.",
                                            tokens_attempted,
                                            max.map(|m| m.to_string())
                                                .unwrap_or_else(|| "?".to_string()),
                                        ),
                                    )
                                    .with_chat_id(&inbound.chat_id),
                                ));
                                // The context just shrank, so the pre-compaction `prompt_tokens` is
                                // now stale. Clear it; otherwise, if the retried call returns no
                                // usage stats (mock/local provider, transient gap), the end-of-turn
                                // check would read the old huge value via `effective_context_tokens`
                                // and immediately fire a redundant compaction right after this one.
                                last_prompt_tokens = None;
                                apply_budget_decision!(
                                    budget.record_context_recovery(run_started_at.elapsed())
                                );
                                // Next iteration refetches context (now smaller due
                                // to AddSummary + UpdateThreadMetadata) and re-runs
                                // the chat call. No iteration counter refund — the
                                // existing max_iterations budget absorbs the retry.
                                continue;
                            }
                            crate::agent::compaction::CompactionOutcome::Cancelled => {
                                persist_and_cancel!();
                            }
                            crate::agent::compaction::CompactionOutcome::Failed => {
                                // `do_compaction` already emitted the matched
                                // `CompactionFailed`. Fall through to the
                                // user-facing banner below.
                            }
                        }
                    } else {
                        // Second overflow in the same turn — emit a matched pair
                        // so the eval pipeline still sees the event, then surface
                        // the failure. We do NOT call `do_compaction` again.
                        let _ = outbound_tx
                            .send(BusMessage::Telemetry(TelemetryEvent::CompactionTriggered {
                                chat_id: inbound.chat_id.clone(),
                                reason: crate::bus::CompactionTrigger::Overflow400,
                                tokens_before: tokens_before_u32,
                                turns_before: turns_before_u32,
                                tokens_after_preprocess: 0,
                            }))
                            .await;
                        let _ = outbound_tx
                            .send(BusMessage::Telemetry(TelemetryEvent::CompactionFailed {
                                chat_id: inbound.chat_id.clone(),
                                reason:
                                    "second context overflow in the same turn; recovery cap (1) exhausted"
                                        .to_string(),
                                tokens_at_failure: tokens_before_u32,
                            }))
                            .await;
                    }

                    let err = format!(
                        "Context overflow: input exceeds the model's window \
                         (attempted={} max={}). Reduce the conversation length and retry.",
                        tokens_attempted,
                        max.map(|m| m.to_string())
                            .unwrap_or_else(|| "?".to_string()),
                    );
                    let mut banner = build_llm_failed_banner(
                        &inbound.channel,
                        &inbound.chat_id,
                        inbound.thread_id.as_deref(),
                        &err,
                        false,
                    );
                    let persisted = banner.content.clone();
                    persist_terminal_assistant_message(
                        &mut mem,
                        &logger_tx,
                        &name,
                        &inbound.chat_id,
                        &persisted,
                    )
                    .await;
                    if let Some(job_id) =
                        inbound.metadata.get(crate::bus::METADATA_BACKGROUND_JOB_ID)
                    {
                        banner.metadata.insert(
                            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
                            job_id.clone(),
                        );
                    }
                    let _ = outbound_tx.send(BusMessage::Outbound(banner)).await;
                    return Ok(ReasoningLoopExit::Failed {
                        assistant_text: persisted,
                        failure: RunFailureKind::Provider,
                        retryable: false,
                    });
                }
                ChatRetryOutcome::Failed(err) => {
                    let mut banner = build_llm_failed_banner(
                        &inbound.channel,
                        &inbound.chat_id,
                        inbound.thread_id.as_deref(),
                        &err,
                        true,
                    );
                    let persisted = banner.content.clone();
                    persist_terminal_assistant_message(
                        &mut mem,
                        &logger_tx,
                        &name,
                        &inbound.chat_id,
                        &persisted,
                    )
                    .await;
                    if let Some(job_id) =
                        inbound.metadata.get(crate::bus::METADATA_BACKGROUND_JOB_ID)
                    {
                        banner.metadata.insert(
                            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
                            job_id.clone(),
                        );
                    }
                    let _ = outbound_tx.send(BusMessage::Outbound(banner)).await;
                    return Ok(ReasoningLoopExit::Failed {
                        assistant_text: persisted,
                        failure: RunFailureKind::ProviderRetriesExhausted,
                        retryable: true,
                    });
                }
            };

            apply_budget_decision!(budget.record_provider_retries(provider_retries));
            if let Some(usage) = &response.usage {
                let consumed_tokens = if usage.total_tokens > 0 {
                    usage.total_tokens
                } else {
                    usage.prompt_tokens.saturating_add(usage.completion_tokens)
                };
                apply_budget_decision!(budget.record_tokens(u64::from(consumed_tokens)));
            }

            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::debug(&name, "Provider responded.").with_chat_id(&inbound.chat_id),
            ));

            // A steering request is consumed only at a safe boundary: the
            // provider has returned, but its proposed response has not yet
            // been persisted or allowed to start another tool call.
            let pending_steering = steering_guard(&steering).drain();
            if !pending_steering.is_empty() {
                for content in pending_steering {
                    mem.add_message(crate::utils::ChatMessage::user(&content))
                        .await
                        .map_err(ReasoningLoopError::persistence)?;
                }
                consecutive_doom_detections = 0;
                forbid_final_nudges = 0;
                apply_budget_decision!(budget.record_progress(ProgressKind::Steering));
                continue;
            }

            // Log USAGE telemetry
            if let Some(usage) = &response.usage {
                // Remember the exact server-counted input size for the compaction trigger.
                if usage.prompt_tokens > 0 {
                    last_prompt_tokens = Some(usage.prompt_tokens);
                }
                let usage_evt = TelemetryEvent::AgentUsage {
                    chat_id: inbound.chat_id.clone(),
                    model: "llm_provider".to_string(),
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens,
                    total_tokens: usage.total_tokens,
                    cache_read_tokens: usage.cache_read_tokens,
                    cache_creation_tokens: usage.cache_creation_tokens,
                    background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
                };
                let _ = outbound_tx
                    .send(BusMessage::Telemetry(usage_evt.clone()))
                    .await;
                hook_observe_telemetry(hook_tool_ctx.as_ref(), &inbound, is_subagent, usage_evt);
            }

            // Emit REASONING block as telemetry
            if let Some(reasoning) = &response.reasoning_content {
                let _ = outbound_tx
                    .send(BusMessage::Telemetry(TelemetryEvent::AgentThought {
                        chat_id: inbound.chat_id.clone(),
                        thought: reasoning.clone(),
                        background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
                    }))
                    .await;
            }

            let response_text = response.content.clone();
            let mut tool_invoked = false;

            if let Some(tool_calls) = &response.tool_calls {
                // Record the assistant message that spawned the tool calls
                let assistant_msg = crate::utils::ChatMessage {
                    role: "assistant".to_string(),
                    content: if response_text.is_empty() {
                        None
                    } else {
                        Some(crate::utils::MessageContent::Text(response_text.clone()))
                    },
                    name: None,
                    tool_calls: Some(tool_calls.clone()),
                    tool_call_id: None,
                    reasoning_content: response.reasoning_content.clone(),
                    is_error: None,
                };
                mem.add_message(assistant_msg)
                    .await
                    .map_err(ReasoningLoopError::persistence)?;

                let parallel_ok = !is_subagent
                    && tool_calls.len() > 1
                    && tools
                        .all_parallel_safe(tool_calls.iter().map(|tc| tc.function.name.as_str()));

                let finalize_tool_output = |mut result: ToolResult| -> String {
                    crate::utils::truncate_utf8_safe(
                        &mut result.content,
                        max_tool_output_chars,
                        "\n... [TRUNCATED FOR LENGTH]",
                    );
                    result.content
                };

                if parallel_ok {
                    for tc in tool_calls.iter() {
                        if cancel_token.is_cancelled() {
                            persist_and_cancel!();
                        }
                        apply_budget_decision!(budget.record_tool_call(tool_intent_signature(
                            &tc.function.name,
                            &tc.function.arguments,
                        )));
                        log_tool_invocation_start(
                            &logger_tx,
                            &outbound_tx,
                            hook_tool_ctx.as_ref(),
                            &name,
                            &inbound,
                            tc,
                            is_subagent,
                        )
                        .await;
                    }
                    let mut futures_vec = Vec::with_capacity(tool_calls.len());
                    for tc in tool_calls.iter() {
                        let tools = Arc::clone(&tools);
                        let tool_execution_activity = tool_execution_activity.clone();
                        let hook_for_tool = hook_tool_ctx.clone();
                        let inbound_meta_for_tool = inbound_metadata.clone();
                        let chat_id = inbound.chat_id.clone();
                        let tool_name = tc.function.name.clone();
                        let parsed_args = parse_tool_arguments(&tc.function.arguments);
                        if tool_name == "exec" {
                            if let Ok(args) = &parsed_args {
                                if let Some(cmd) = extract_exec_command(args) {
                                    if shell_command_uses_grep_like(&cmd) {
                                        let _ = outbound_tx
                                            .send(BusMessage::Telemetry(
                                                TelemetryEvent::ShellGrepLikeDetected {
                                                    chat_id: inbound.chat_id.clone(),
                                                    channel: inbound.channel.clone(),
                                                    command_preview: command_preview(&cmd),
                                                },
                                            ))
                                            .await;
                                    }
                                }
                            }
                        }
                        let cancel_token = cancel_token.clone();
                        let tool_exec_ctx = tool_exec_ctx.clone();
                        let clarification_hub = clarification_hub.clone();
                        let subagent_allowlist = subagent_allowlist.clone();
                        let shell_policy_for_call = shell_policy.clone();
                        let outbound_for_call = outbound_tx.clone();
                        let channel_for_call = inbound.channel.clone();
                        let tool_call_id = Some(tc.id.clone());
                        futures_vec.push(async move {
                            let args = match parsed_args {
                                Ok(args) => args,
                                Err(error) => {
                                    return ToolExecutionFinished::Completed(error.to_tool_result())
                                }
                            };
                            execute_tool_call_with_activity(
                                &tools,
                                tool_execution_activity,
                                &chat_id,
                                &channel_for_call,
                                &outbound_for_call,
                                &tool_name,
                                tool_call_id,
                                args,
                                Some(&cancel_token),
                                ToolCallRuntime {
                                    session: tool_exec_ctx,
                                    hub: clarification_hub,
                                    is_subagent,
                                    subagent_allowlist,
                                    shell_policy: shell_policy_for_call,
                                    unattended_session,
                                    hook_tool_ctx: hook_for_tool,
                                    inbound_metadata: inbound_meta_for_tool,
                                },
                            )
                            .await
                        });
                    }
                    let outcomes = join_all(futures_vec).await;
                    for (tc, fin) in tool_calls.iter().zip(outcomes) {
                        if cancel_token.is_cancelled() {
                            persist_and_cancel!();
                        }
                        let tool_result = match fin {
                            ToolExecutionFinished::Completed(result) => result,
                            ToolExecutionFinished::Waiting(ticket_id) => {
                                // Break the iteration loop; the job is now in 'waiting' state.
                                return Ok(ReasoningLoopExit::WaitingForUser { ticket_id });
                            }
                            ToolExecutionFinished::Cancelled => {
                                persist_and_cancel!();
                            }
                        };
                        // Status is authoritative at the executor boundary. Legacy string
                        // classification is isolated inside ToolRegistry and never runs here.
                        let is_error = tool_result.is_error();
                        let tool_name = tc.function.name.clone();
                        let intent = tool_intent_signature(&tool_name, &tc.function.arguments);
                        let budget_decision = if is_error {
                            let code = tool_result
                                .error_code()
                                .unwrap_or(ToolErrorCode::ExecutionFailed);
                            budget.record_tool_failure(typed_failure_key(&tool_name, code, &intent))
                        } else {
                            budget.record_tool_success(intent)
                        };
                        let tool_result_text = finalize_tool_output(tool_result);
                        let tr = TelemetryEvent::ToolResult {
                            chat_id: inbound.chat_id.clone(),
                            channel: inbound.channel.clone(),
                            tool_name: tool_name.clone(),
                            result: tool_result_text.clone(),
                            is_error,
                            tool_call_id: Some(tc.id.clone()),
                            background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
                        };
                        let _ = outbound_tx.send(BusMessage::Telemetry(tr.clone())).await;
                        hook_observe_telemetry(hook_tool_ctx.as_ref(), &inbound, is_subagent, tr);
                        let tfin = TelemetryEvent::ToolCallFinished {
                            chat_id: inbound.chat_id.clone(),
                            tool_name: tool_name.clone(),
                            result: tool_result_text.clone(),
                            is_error,
                            tool_call_id: Some(tc.id.clone()),
                            background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
                        };
                        let _ = outbound_tx.send(BusMessage::Telemetry(tfin.clone())).await;
                        hook_observe_telemetry(hook_tool_ctx.as_ref(), &inbound, is_subagent, tfin);
                        mem.add_message(crate::utils::ChatMessage::tool_with_error(
                            &tool_result_text,
                            &tc.id,
                            Some(tool_name.as_str()),
                            is_error,
                        ))
                        .await
                        .map_err(ReasoningLoopError::persistence)?;
                        apply_budget_decision!(budget_decision);
                        apply_budget_decision!(budget.record_elapsed(run_started_at.elapsed()));
                    }
                    tool_invoked = true;
                    forbid_final_nudges = 0;
                } else {
                    for tc in tool_calls {
                        if cancel_token.is_cancelled() {
                            persist_and_cancel!();
                        }

                        apply_budget_decision!(budget.record_tool_call(tool_intent_signature(
                            &tc.function.name,
                            &tc.function.arguments,
                        )));

                        log_tool_invocation_start(
                            &logger_tx,
                            &outbound_tx,
                            hook_tool_ctx.as_ref(),
                            &name,
                            &inbound,
                            tc,
                            is_subagent,
                        )
                        .await;

                        let tool_name = &tc.function.name;
                        let args_str = &tc.function.arguments;
                        let parsed_args = parse_tool_arguments(args_str);
                        if tool_name == "exec" {
                            if let Ok(args) = &parsed_args {
                                if let Some(cmd) = extract_exec_command(args) {
                                    if shell_command_uses_grep_like(&cmd) {
                                        let _ = outbound_tx
                                            .send(BusMessage::Telemetry(
                                                TelemetryEvent::ShellGrepLikeDetected {
                                                    chat_id: inbound.chat_id.clone(),
                                                    channel: inbound.channel.clone(),
                                                    command_preview: command_preview(&cmd),
                                                },
                                            ))
                                            .await;
                                    }
                                }
                            }
                        }

                        let finished = match parsed_args {
                            Ok(args) => {
                                execute_tool_call_with_activity(
                                    &tools,
                                    tool_execution_activity.clone(),
                                    &inbound.chat_id,
                                    &inbound.channel,
                                    &outbound_tx,
                                    tool_name,
                                    Some(tc.id.clone()),
                                    args,
                                    Some(&cancel_token),
                                    ToolCallRuntime {
                                        session: tool_exec_ctx.clone(),
                                        hub: clarification_hub.clone(),
                                        is_subagent,
                                        subagent_allowlist: subagent_allowlist.clone(),
                                        shell_policy: shell_policy.clone(),
                                        unattended_session,
                                        hook_tool_ctx: hook_tool_ctx.clone(),
                                        inbound_metadata: inbound_metadata.clone(),
                                    },
                                )
                                .await
                            }
                            Err(error) => ToolExecutionFinished::Completed(error.to_tool_result()),
                        };
                        let tool_result = match finished {
                            ToolExecutionFinished::Completed(result) => result,
                            ToolExecutionFinished::Waiting(ticket_id) => {
                                // Break the iteration loop; the job is now in 'waiting' state.
                                return Ok(ReasoningLoopExit::WaitingForUser { ticket_id });
                            }
                            ToolExecutionFinished::Cancelled => {
                                persist_and_cancel!();
                            }
                        };

                        let is_error = tool_result.is_error();
                        let intent = tool_intent_signature(tool_name, &tc.function.arguments);
                        let budget_decision = if is_error {
                            let code = tool_result
                                .error_code()
                                .unwrap_or(ToolErrorCode::ExecutionFailed);
                            budget.record_tool_failure(typed_failure_key(tool_name, code, &intent))
                        } else {
                            budget.record_tool_success(intent)
                        };
                        let tool_result_text = finalize_tool_output(tool_result);

                        let tr = TelemetryEvent::ToolResult {
                            chat_id: inbound.chat_id.clone(),
                            channel: inbound.channel.clone(),
                            tool_name: tool_name.to_string(),
                            result: tool_result_text.clone(),
                            is_error,
                            tool_call_id: Some(tc.id.clone()),
                            background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
                        };
                        let _ = outbound_tx.send(BusMessage::Telemetry(tr.clone())).await;
                        hook_observe_telemetry(hook_tool_ctx.as_ref(), &inbound, is_subagent, tr);
                        let tn = tool_name.to_string();
                        let tfin = TelemetryEvent::ToolCallFinished {
                            chat_id: inbound.chat_id.clone(),
                            tool_name: tn,
                            result: tool_result_text.clone(),
                            is_error,
                            tool_call_id: Some(tc.id.clone()),
                            background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
                        };
                        let _ = outbound_tx.send(BusMessage::Telemetry(tfin.clone())).await;
                        hook_observe_telemetry(hook_tool_ctx.as_ref(), &inbound, is_subagent, tfin);

                        mem.add_message(crate::utils::ChatMessage::tool_with_error(
                            &tool_result_text,
                            &tc.id,
                            Some(tool_name.as_str()),
                            is_error,
                        ))
                        .await
                        .map_err(ReasoningLoopError::persistence)?;
                        apply_budget_decision!(budget_decision);
                        apply_budget_decision!(budget.record_elapsed(run_started_at.elapsed()));
                        tool_invoked = true;
                        forbid_final_nudges = 0;
                    }
                }
            } else {
                let mut assistant_msg = crate::utils::ChatMessage::assistant(&response_text);
                assistant_msg.reasoning_content = response.reasoning_content.clone();
                mem.add_message(assistant_msg)
                    .await
                    .map_err(ReasoningLoopError::persistence)?;
            }

            if !tool_invoked {
                // If the model returned empty text after tool calls, re-prompt once so the
                // user sees an actual response instead of an invisible empty cell.
                if response_text.trim().is_empty() && iterations > 1 && iterations < max_iterations
                {
                    let nudge = "[SYSTEM: You used tools but did not produce a text reply for the user. Please summarize your findings or answer the user's question now.]";
                    let correction = crate::utils::ChatMessage::user(nudge);
                    mem.add_message(correction)
                        .await
                        .map_err(ReasoningLoopError::persistence)?;
                    continue;
                }
                let research_nudge =
                    iterations < max_iterations && should_nudge_research_depth(&inbound, &context);
                if forbid_final_effective
                    && iterations < max_iterations
                    && forbid_final_nudges < MAX_FORBID_FINAL_NUDGES
                {
                    forbid_final_nudges += 1;
                    let nudge = "[SYSTEM: Continue with at least one tool call (or `ask_user` if you are blocked). Plain assistant text alone is not sufficient for this session until the objective is met.]";
                    let correction = crate::utils::ChatMessage::user(nudge);
                    mem.add_message(correction)
                        .await
                        .map_err(ReasoningLoopError::persistence)?;
                    continue;
                }
                if research_nudge {
                    let _ = outbound_tx
                        .send(BusMessage::Telemetry(TelemetryEvent::ResearchDepthNudge {
                            chat_id: inbound.chat_id.clone(),
                            channel: inbound.channel.clone(),
                            reason: "search_without_primary_fetch".to_string(),
                        }))
                        .await;
                    // Audit X4: name only fetch tools that are actually registered so the nudge
                    // never recommends opt-in ML tools gated off by `ml_domain_enabled`.
                    let mut fetch_targets = String::from("`web_fetch`");
                    if tools.get_tool("arxiv_fetch").is_some() {
                        fetch_targets.push_str("/`arxiv_fetch`");
                    }
                    if tools.get_tool("hf_hub_file_fetch").is_some() {
                        fetch_targets.push_str("/`hf_hub_file_fetch`");
                    }
                    let nudge = format!(
    "[SYSTEM: Research depth check - you used discovery search but did not fetch \
primary sources. Before finalizing, use {fetch_targets} on concrete sources, \
cross-verify at least two sources, then synthesize findings with explicit \
uncertainties.]"
);
                    let correction = crate::utils::ChatMessage::user(&nudge);
                    mem.add_message(correction)
                        .await
                        .map_err(ReasoningLoopError::persistence)?;
                    continue;
                }

                // Atomically close steering acceptance before committing the
                // final response. A request racing this boundary is therefore
                // either drained and incorporated, or rejected by `push`; it
                // can never be acknowledged into a stale next-run inbox.
                let final_steering = steering_guard(&steering).close_or_drain();
                if !final_steering.is_empty() {
                    for content in final_steering {
                        mem.add_message(crate::utils::ChatMessage::user(&content))
                            .await
                            .map_err(ReasoningLoopError::persistence)?;
                    }
                    consecutive_doom_detections = 0;
                    forbid_final_nudges = 0;
                    apply_budget_decision!(budget.record_progress(ProgressKind::Steering));
                    continue;
                }
                apply_budget_decision!(budget.propose_completion());
                // Final outbound text
                let final_response = thinking_strip_re
                    .replace_all(&response_text, "")
                    .to_string();

                // Emit outbound response payload.
                let mut metadata = HashMap::new();
                if let Some(job_id) = inbound.metadata.get(crate::bus::METADATA_BACKGROUND_JOB_ID) {
                    metadata.insert(
                        crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
                        job_id.clone(),
                    );
                }

                let outbound = OutboundMessage {
                    channel: inbound.channel.clone(),
                    chat_id: inbound.chat_id.clone(),
                    thread_id: inbound.thread_id.clone(),
                    content: final_response.clone(),
                    metadata,
                };

                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::info(&name, "Sending final response.").with_chat_id(&inbound.chat_id),
                ));

                // Auto-compaction check
                let current_context = mem
                    .get_context_since_reflection()
                    .await
                    .map_err(ReasoningLoopError::persistence)?;
                let user_turns = current_context.iter().filter(|m| m.role == "user").count();
                // Prefer the ground truth: `last_prompt_tokens` is the exact input size the provider
                // counted for the most recent request, which (at this end-of-turn point) covers
                // nearly the entire current context — only the just-produced final assistant message
                // is newer. Taking the max with the bytes/4 estimate corrects the heuristic's
                // under-count on code/JSON-heavy contexts so a real overflow triggers compaction.
                let approx_tokens: usize = effective_context_tokens(
                    estimate_context_tokens(&current_context),
                    last_prompt_tokens,
                );

                // PR-3: pull the model's context window from the provider; if known,
                // tighten the absolute token threshold to whichever is smaller of:
                // 85% of the window, or (window - 16k reserve). With Window=None
                // (provider can't determine), `effective_compaction_threshold`
                // returns the legacy absolute threshold unchanged.
                let effective_token_threshold = {
                    let window = provider.context_window_tokens();
                    crate::agent::compaction::effective_compaction_threshold(
                        short_term_threshold_tokens,
                        window,
                        crate::agent::compaction::TRIGGER_AT_PERCENTAGE_DEFAULT,
                        crate::agent::compaction::RESERVE_TOKENS_DEFAULT,
                    )
                };

                if user_turns >= short_term_threshold_turns
                    || approx_tokens >= effective_token_threshold
                {
                    let tokens_before_u32 = approx_tokens.min(u32::MAX as usize) as u32;
                    let turns_before_u32 = user_turns.min(u32::MAX as usize) as u32;
                    // PR-3: trigger_reason matches the *effective* token threshold,
                    // not the legacy absolute one — otherwise a window-aware
                    // trigger that fires below the absolute would fall through
                    // to the `unreachable!()` arm.
                    let trigger_reason = match (
                        user_turns >= short_term_threshold_turns,
                        approx_tokens >= effective_token_threshold,
                    ) {
                        (true, true) => crate::bus::CompactionTrigger::BothLimits,
                        (true, false) => crate::bus::CompactionTrigger::TurnLimit,
                        (false, true) => crate::bus::CompactionTrigger::TokenLimit,
                        (false, false) => unreachable!("guarded by the outer if"),
                    };

                    // PR-4.1: compaction now goes through the shared `do_compaction`
                    // helper. Used by both this threshold-trigger path and the
                    // overflow-recovery path at the chat call site. The helper emits
                    // the full matched telemetry pair (Triggered + Completed/Failed)
                    // internally; we only need to handle the Cancelled outcome.
                    let memory_node = session_manager.get_memory_node();
                    let outcome = crate::agent::compaction::do_compaction(
                        crate::agent::compaction::DoCompactionArgs {
                            chat_id: &inbound.chat_id,
                            session_key: &session_key,
                            trigger_reason,
                            tokens_before: tokens_before_u32,
                            turns_before: turns_before_u32,
                            current_context: &current_context,
                            existing_summary: summaries.first().map(|s| s.as_str()),
                            focus_instructions: None,
                            provider: provider.as_ref(),
                            memory_node: &memory_node,
                            outbound_tx: &outbound_tx,
                            cancel_token: &cancel_token,
                        },
                    )
                    .await;
                    if matches!(
                        outcome,
                        crate::agent::compaction::CompactionOutcome::Cancelled
                    ) {
                        persist_and_cancel!();
                    }
                }

                let _ = outbound_tx.send(BusMessage::Outbound(outbound)).await;
                return Ok(ReasoningLoopExit::Completed {
                    assistant_text: final_response,
                });
            }
        }

        // `break` is reserved for the legacy doom-loop detector. All budget and progress
        // terminals return through `apply_budget_decision!`, so terminal outcomes cannot race.
        let max_iter_msg =
            "Stopped: the agent kept repeating the same action with no progress and \
                            did not recover after corrective nudges. Try rephrasing the request or \
                            breaking it into smaller steps."
                .to_string();
        persist_terminal_assistant_message(
            &mut mem,
            &logger_tx,
            &name,
            &inbound.chat_id,
            &max_iter_msg,
        )
        .await;
        let fallback = OutboundMessage {
            channel: inbound.channel,
            chat_id: inbound.chat_id,
            thread_id: inbound.thread_id,
            content: max_iter_msg.clone(),
            metadata: HashMap::new(),
        };
        let _ = outbound_tx.send(BusMessage::Outbound(fallback)).await;
        Ok(ReasoningLoopExit::Stuck {
            assistant_text: max_iter_msg,
            reason: RunStuckReason::DoomLoop,
        })
    }
}

#[cfg(test)]
mod context_hardening_tests {
    use super::*;

    fn assistant_with_calls(ids: &[&str]) -> crate::utils::ChatMessage {
        let mut message = crate::utils::ChatMessage::assistant("");
        message.content = None;
        message.tool_calls = Some(
            ids.iter()
                .map(|id| crate::utils::ToolCallRequest {
                    id: (*id).to_string(),
                    tool_type: "function".to_string(),
                    extra_content: None,
                    function: crate::utils::ToolCallFunction {
                        name: "read_file".to_string(),
                        arguments: "{}".to_string(),
                    },
                })
                .collect(),
        );
        message
    }

    #[test]
    fn research_detection_uses_word_and_phrase_boundaries() {
        for positive in [
            "Research this topic",
            "review the papers",
            "give me state-of-the-art evidence",
            "cite primary sources",
            "compare methods",
        ] {
            assert!(text_looks_like_research_request(positive), "{positive}");
        }
        for negative in [
            "I am excited about this",
            "the paperclip is broken",
            "surveying the room",
            "compare methodologies later",
        ] {
            assert!(!text_looks_like_research_request(negative), "{negative}");
        }
    }

    #[test]
    fn repair_removes_orphan_mismatched_and_duplicate_tool_results() {
        let mut context = vec![
            crate::utils::ChatMessage::system("system"),
            crate::utils::ChatMessage::tool("orphan", "orphan", None),
            assistant_with_calls(&["a", "b"]),
            crate::utils::ChatMessage::tool("first", "a", None),
            crate::utils::ChatMessage::tool("duplicate", "a", None),
            crate::utils::ChatMessage::tool("wrong", "other", None),
            crate::utils::ChatMessage::user("next"),
        ];

        repair_tool_call_context(&mut context);

        let tool_ids: Vec<_> = context
            .iter()
            .filter(|message| message.role == "tool")
            .filter_map(|message| message.tool_call_id.as_deref())
            .collect();
        assert_eq!(tool_ids, ["a", "b"]);
        assert!(context.iter().all(|message| {
            message
                .content
                .as_ref()
                .is_none_or(|content| !content.text_content().contains("orphan"))
        }));
    }

    #[test]
    fn trimming_keeps_assistant_tool_blocks_atomic() {
        let mut context = vec![
            crate::utils::ChatMessage::system("system"),
            crate::utils::ChatMessage::user(&"x".repeat(400)),
            assistant_with_calls(&["call"]),
            crate::utils::ChatMessage::tool("result", "call", None),
            crate::utils::ChatMessage::user("recent"),
        ];

        trim_context_to_budget(&mut context, 1);

        assert!(!context.iter().any(|message| message.role == "tool"));
        assert!(!context
            .iter()
            .any(|message| message.role == "assistant" && message.tool_calls.is_some()));
    }
}
