use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::mpsc;

pub mod compaction;
mod doom_loop;
pub mod registry;
mod subagent;
pub use registry::AgentRegistry;
pub use subagent::SubagentHarness;

use crate::clarification::ClarificationHub;
use crate::tool_runtime::{with_tool_exec_and_progress_scope, ToolExecCtx, ToolProgressEmitter};

use crate::bus::{BusMessage, InboundMessage, LogEvent, OutboundMessage, TelemetryEvent};
use crate::config::{ResolvedShellPolicy, ShellPolicyMode};
use crate::hooks::{
    run_post_tool_hooks, run_pre_tool_hooks, run_user_prompt_hooks, HookObservationMeta,
    HookSessionInfo, PreToolOutcome, ToolCallHookContext, UserPromptHookOutcome,
};
use crate::logging::LoggerHandle;
use crate::memory::{MemoryMessage, SharedReply, TodoRow};
use crate::session::SessionManager;
use crate::skills::SkillRegistry;
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tools::ToolRegistry;
use crate::traits::{Memory, Provider, Tool};
use crate::NodeHandle;
use crate::{ActorError, ActorLogic};
use futures::{future::join_all, FutureExt};
use std::panic::AssertUnwindSafe;

static REDACTED_THINKING_STRIP_RE: OnceLock<Regex> = OnceLock::new();

async fn load_harness_todos_for_step(
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

fn format_harness_todos_step_block(rows: &[TodoRow]) -> String {
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

async fn persist_terminal_assistant_message(
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
                &format!("Failed to persist terminal assistant message: {}", e),
            )
            .with_chat_id(chat_id),
        ));
    }
}

fn metadata_truthy(meta: &HashMap<String, serde_json::Value>, key: &str) -> bool {
    meta.get(key)
        .map(|v| {
            v.as_bool().unwrap_or(false)
                || v.as_str()
                    .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
                    .unwrap_or(false)
        })
        .unwrap_or(false)
}

fn text_looks_like_research_request(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    [
        "research",
        "literature",
        "paper",
        "state-of-the-art",
        "state of the art",
        "survey",
        "arxiv",
        "evidence",
        "cite",
        "compare methods",
    ]
    .iter()
    .any(|k| lower.contains(k))
}

fn context_has_tool_call(context: &[crate::utils::ChatMessage], tool_name: &str) -> bool {
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
fn estimate_message_tokens(msg: &crate::utils::ChatMessage) -> usize {
    let text_len = msg.content.as_ref().map_or(0, |c| c.text_content().len());
    let args_len = msg.tool_calls.as_ref().map_or(0, |tcs| {
        tcs.iter()
            .map(|t| t.function.arguments.len())
            .sum::<usize>()
    });
    (text_len + args_len) / 4
}

/// Approximate token count for a whole context (sum of [`estimate_message_tokens`]).
fn estimate_context_tokens(context: &[crate::utils::ChatMessage]) -> usize {
    context.iter().map(estimate_message_tokens).sum()
}

/// Best available context-size estimate for the compaction trigger: the larger of the bytes/4
/// heuristic and the last LLM call's exact `usage.prompt_tokens`. The heuristic under-counts the
/// code/JSON/non-English payloads this agent produces, while `prompt_tokens` is the provider's
/// ground-truth input count — so the max guards against silently overflowing the context window.
fn effective_context_tokens(estimate: usize, last_prompt_tokens: Option<u32>) -> usize {
    estimate.max(last_prompt_tokens.unwrap_or(0) as usize)
}

/// Trim context from the front (oldest messages) to stay within a token budget.
/// Preserves the system message at index 0 and never splits tool_call/tool pairs.
/// A marker message is inserted only when messages were actually removed.
fn trim_context_to_budget(context: &mut Vec<crate::utils::ChatMessage>, max_tokens: usize) {
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
fn repair_tool_call_context(context: &mut Vec<crate::utils::ChatMessage>) {
    let mut i = 0;
    while i < context.len() {
        let tool_call_ids: Vec<String> = match &context[i].tool_calls {
            Some(calls) if context[i].role == "assistant" && !calls.is_empty() => {
                calls.iter().map(|tc| tc.id.clone()).collect()
            }
            _ => {
                i += 1;
                continue;
            }
        };

        // Collect the set of tool_call_ids that have responses immediately following
        let mut responded: HashSet<String> = HashSet::new();
        let mut j = i + 1;
        while j < context.len() && context[j].role == "tool" {
            if let Some(ref id) = context[j].tool_call_id {
                responded.insert(id.clone());
            }
            j += 1;
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

fn should_nudge_research_depth(
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

pub const WAIT_SIGNAL_PREFIX: &str = "ISANAGENT_WAIT_FOR_USER:";
pub const WAITING_FOR_USER_RESULT_PREFIX: &str = "WAITING:";

enum ToolExecutionFinished {
    Completed(Result<String, String>),
    Cancelled,
    Waiting(String), // The ticket ID
}

fn extract_exec_command(args: &Value) -> Option<String> {
    args.get("command")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn should_require_shell_approval(command: &str, patterns: &[String]) -> bool {
    let lower = command.to_ascii_lowercase();
    patterns.iter().any(|p| lower.contains(p))
}

fn shell_policy_mode_for_session(
    policy: &ResolvedShellPolicy,
    unattended_session: bool,
) -> ShellPolicyMode {
    if unattended_session {
        policy.unattended_mode
    } else {
        policy.interactive_mode
    }
}

fn command_preview(command: &str) -> String {
    const MAX_PREVIEW: usize = 160;
    if command.len() <= MAX_PREVIEW {
        command.to_string()
    } else {
        format!("{}...", &command[..MAX_PREVIEW])
    }
}

/// Tools that execute model-authored code/commands on the host or a session. All of these run
/// arbitrary code, so they share the shell-policy approval gate — not just `exec`. Keying the
/// gate on this category (rather than the literal name `"exec"`) is what stops `execution_run`
/// / `execution_run_background` / `python_run` from bypassing approval entirely.
fn is_code_exec_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "exec" | "python_run" | "execution_run" | "execution_run_background"
    )
}

/// Code-exec tools that run *arbitrary* code (Python source / session cells) where the
/// destructive-shell-pattern heuristic does not meaningfully apply, so any such call is
/// treated as approval-worthy in ask/deny mode.
fn is_arbitrary_code_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "python_run" | "execution_run" | "execution_run_background"
    )
}

/// Extract the command/code a code-exec tool will run. `exec` carries it in `command`; the
/// execution / python tools carry it in `code`.
fn extract_code_exec_command(tool_name: &str, args: &Value) -> Option<String> {
    let key = if tool_name == "exec" { "command" } else { "code" };
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Whether a code-exec call needs approval in ask/deny mode. Arbitrary-code tools always do;
/// shell `exec` only when the command matches a destructive pattern (preserves existing UX).
fn code_exec_requires_approval(tool_name: &str, command: &str, patterns: &[String]) -> bool {
    is_arbitrary_code_tool(tool_name) || should_require_shell_approval(command, patterns)
}

#[cfg(test)]
mod code_exec_gate_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn category_covers_all_code_exec_tools() {
        assert!(is_code_exec_tool("exec"));
        assert!(is_code_exec_tool("python_run"));
        assert!(is_code_exec_tool("execution_run"));
        assert!(is_code_exec_tool("execution_run_background"));
        assert!(!is_code_exec_tool("read_file"));
        assert!(!is_code_exec_tool("web_search"));
    }

    #[test]
    fn extracts_command_for_exec_and_code_for_execution_tools() {
        assert_eq!(
            extract_code_exec_command("exec", &json!({"command": " ls -la "})).as_deref(),
            Some("ls -la")
        );
        assert_eq!(
            extract_code_exec_command("execution_run", &json!({"code": "print(1)"})).as_deref(),
            Some("print(1)")
        );
        assert_eq!(
            extract_code_exec_command("python_run", &json!({"code": "import os"})).as_deref(),
            Some("import os")
        );
        // wrong key / empty -> None
        assert!(extract_code_exec_command("execution_run", &json!({"command": "x"})).is_none());
        assert!(extract_code_exec_command("exec", &json!({"command": "  "})).is_none());
    }

    #[test]
    fn arbitrary_code_always_requires_approval_benign_exec_does_not() {
        let patterns = vec!["rm -rf".to_string()];
        // Arbitrary-code tools: even benign code requires approval (closes the bypass).
        assert!(code_exec_requires_approval(
            "execution_run",
            "print('hi')",
            &patterns
        ));
        assert!(code_exec_requires_approval("python_run", "1+1", &patterns));
        // Shell `exec`: benign command does NOT require approval (preserves existing UX)...
        assert!(!code_exec_requires_approval("exec", "ls -la", &patterns));
        // ...but a destructive one does.
        assert!(code_exec_requires_approval("exec", "rm -rf /tmp/x", &patterns));
    }
}

fn hook_observe_telemetry(
    hook_tool_ctx: Option<&Arc<ToolCallHookContext>>,
    inbound: &crate::bus::InboundMessage,
    is_subagent: bool,
    event: TelemetryEvent,
) {
    let Some(hc) = hook_tool_ctx else {
        return;
    };
    let Some(obs) = hc.observation.as_ref() else {
        return;
    };
    let meta = HookObservationMeta {
        channel: inbound.channel.as_str(),
        chat_id: inbound.chat_id.as_str(),
        thread_id: inbound.thread_id.as_deref(),
        is_subagent,
        metadata: &inbound.metadata,
    };
    obs.try_emit(event, meta);
}

fn shell_command_uses_grep_like(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.contains("grep ")
        || lower.contains("| grep")
        || lower.contains("cat ")
        || lower.contains("wc ")
}

async fn log_tool_invocation_start(
    logger_tx: &LoggerHandle,
    outbound_tx: &mpsc::Sender<BusMessage>,
    hook_tool_ctx: Option<&Arc<ToolCallHookContext>>,
    agent_name: &str,
    inbound: &crate::bus::InboundMessage,
    tc: &crate::utils::ToolCallRequest,
    is_subagent: bool,
) {
    let tool_name = &tc.function.name;
    let args_str = &tc.function.arguments;
    let _ = logger_tx.send(BusMessage::Log(
        LogEvent::info(agent_name, &format!("Invoking tool: {}", tool_name))
            .with_chat_id(&inbound.chat_id),
    ));
    let _ = outbound_tx
        .send(BusMessage::Telemetry(TelemetryEvent::ToolCall {
            chat_id: inbound.chat_id.clone(),
            channel: inbound.channel.clone(),
            tool_name: tool_name.to_string(),
            args: args_str.clone(),
            tool_call_id: Some(tc.id.clone()),
            background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
        }))
        .await;
    let _ = outbound_tx
        .send(BusMessage::Telemetry(TelemetryEvent::ToolCallStarted {
            chat_id: inbound.chat_id.clone(),
            tool_name: tool_name.to_string(),
            args: args_str.clone(),
            background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
        }))
        .await;
    hook_observe_telemetry(
        hook_tool_ctx,
        inbound,
        is_subagent,
        TelemetryEvent::ToolCall {
            chat_id: inbound.chat_id.clone(),
            channel: inbound.channel.clone(),
            tool_name: tool_name.to_string(),
            args: args_str.clone(),
            tool_call_id: Some(tc.id.clone()),
            background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
        },
    );
    hook_observe_telemetry(
        hook_tool_ctx,
        inbound,
        is_subagent,
        TelemetryEvent::ToolCallStarted {
            chat_id: inbound.chat_id.clone(),
            tool_name: tool_name.to_string(),
            args: args_str.clone(),
            background_job_id: crate::bus::get_background_job_id(&inbound.metadata),
        },
    );
}

/// Session-scoped wiring for tools that need the active chat (e.g. `ask_user`).
#[derive(Clone)]
struct ToolCallRuntime {
    session: ToolExecCtx,
    hub: Arc<ClarificationHub>,
    is_subagent: bool,
    subagent_allowlist: Option<Arc<HashSet<String>>>,
    shell_policy: Arc<ResolvedShellPolicy>,
    unattended_session: bool,
    hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
    inbound_metadata: Arc<HashMap<String, serde_json::Value>>,
}

/// Runs a tool with optional per-call activity heartbeats and optional cooperative cancellation.
#[allow(clippy::too_many_arguments)] // Central tool-dispatch path; grouping would obscure call sites.
async fn execute_tool_call_with_activity(
    tools: &Arc<ToolRegistry>,
    tool_execution_activity: Option<SharedToolExecutionActivity>,
    chat_id: &str,
    channel: &str,
    outbound_tx: &mpsc::Sender<BusMessage>,
    tool_name: &str,
    tool_call_id: Option<String>,
    args: Value,
    cancel_token: Option<&tokio_util::sync::CancellationToken>,
    runtime: ToolCallRuntime,
) -> ToolExecutionFinished {
    let session_key = runtime.session.session_key.clone();
    let hub = Arc::clone(&runtime.hub);
    let mut tool_exec_ctx = runtime.session;
    tool_exec_ctx.tool_call_id = tool_call_id.clone();
    let thread_id_for_hooks = tool_exec_ctx.thread_id.clone();
    let chat_id = chat_id.to_string();
    let tool_name = tool_name.to_string();
    let channel = channel.to_string();
    let tools = Arc::clone(tools);
    let outbound_tx = outbound_tx.clone();
    let cancel_owned = cancel_token.cloned();
    let tool_call_id_for_hooks = tool_call_id.clone();
    let background_job_id = crate::bus::get_background_job_id(&runtime.inbound_metadata);
    let progress_emitter = ToolProgressEmitter {
        outbound_tx: outbound_tx.clone(),
        channel: channel.clone(),
        chat_id: chat_id.clone(),
        tool_name: tool_name.clone(),
        tool_call_id,
        background_job_id,
    };

    with_tool_exec_and_progress_scope(tool_exec_ctx, progress_emitter, async move {
        let mut args = args;
        let activity_handle = tool_execution_activity
            .as_ref()
            .map(|a| a.start(chat_id.as_str(), tool_name.as_str()));

        let is_subagent = runtime.is_subagent;
        let allow = runtime.subagent_allowlist.clone();
        if is_code_exec_tool(&tool_name) {
            if let Some(command) = extract_code_exec_command(&tool_name, &args) {
                let preview = command_preview(&command);
                let mode =
                    shell_policy_mode_for_session(&runtime.shell_policy, runtime.unattended_session);
                let requires_approval = code_exec_requires_approval(
                    &tool_name,
                    &command,
                    &runtime.shell_policy.approval_patterns,
                );
                match mode {
                    ShellPolicyMode::Allow => {}
                    ShellPolicyMode::Deny => {
                        if requires_approval {
                            let _ = outbound_tx
                                .send(BusMessage::Telemetry(TelemetryEvent::ShellPolicyDecision {
                                    chat_id: chat_id.clone(),
                                    channel: channel.clone(),
                                    mode: "deny".to_string(),
                                    decision: "blocked".to_string(),
                                    command_preview: preview,
                                }))
                                .await;
                            return ToolExecutionFinished::Completed(Err(format!(
                                "Command blocked by shell policy (mode=deny): {}",
                                command
                            )));
                        }
                    }
                    ShellPolicyMode::Ask => {
                        if requires_approval {
                            let _ = outbound_tx
                                .send(BusMessage::Telemetry(TelemetryEvent::ShellPolicyDecision {
                                    chat_id: chat_id.clone(),
                                    channel: channel.clone(),
                                    mode: "ask".to_string(),
                                    decision: "approval_requested".to_string(),
                                    command_preview: preview.clone(),
                                }))
                                .await;
                            let ask_payload = serde_json::json!({
                                "prompt": format!(
                                    "Approve running `{}`?\n\n```\n{}\n```\n\nReply with approve or deny.",
                                    tool_name, command
                                ),
                                "choices": ["approve", "deny"],
                                "timeout_secs": 1800,
                                "allow_empty": false
                            });
                            let ask_result = tools
                                .execute_tool_scoped(
                                    "ask_user",
                                    ask_payload,
                                    allow.as_deref(),
                                    is_subagent,
                                )
                                .await;
                            match ask_result {
                                Ok(reply) => {
                                    let approved = reply.to_ascii_lowercase().contains("approve")
                                        && !reply.to_ascii_lowercase().contains("deny");
                                    if !approved {
                                        let _ = outbound_tx
                                            .send(BusMessage::Telemetry(
                                                TelemetryEvent::ShellPolicyDecision {
                                                    chat_id: chat_id.clone(),
                                                    channel: channel.clone(),
                                                    mode: "ask".to_string(),
                                                    decision: "approval_denied".to_string(),
                                                    command_preview: preview,
                                                },
                                            ))
                                            .await;
                                        return ToolExecutionFinished::Completed(Err(
                                            "Command not approved by user; execution skipped."
                                                .to_string(),
                                        ));
                                    }
                                    let _ = outbound_tx
                                        .send(BusMessage::Telemetry(
                                            TelemetryEvent::ShellPolicyDecision {
                                                chat_id: chat_id.clone(),
                                                channel: channel.clone(),
                                                mode: "ask".to_string(),
                                                decision: "approval_granted".to_string(),
                                                command_preview: preview,
                                            },
                                        ))
                                        .await;
                                }
                                Err(e) => {
                                    return ToolExecutionFinished::Completed(Err(format!(
                                        "Shell policy approval failed: {}",
                                        e
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(ref hc) = runtime.hook_tool_ctx {
            if let Some(st) = &hc.steering {
                let hook_session = HookSessionInfo {
                    channel: channel.as_str(),
                    chat_id: chat_id.as_str(),
                    thread_id: thread_id_for_hooks.as_deref(),
                    metadata: runtime.inbound_metadata.as_ref(),
                    is_subagent: runtime.is_subagent,
                };
                match run_pre_tool_hooks(
                    st.as_ref(),
                    &tool_name,
                    tool_call_id_for_hooks.as_deref(),
                    args.clone(),
                    hook_session,
                )
                .await
                {
                    PreToolOutcome::Block(msg) => {
                        return ToolExecutionFinished::Completed(Err(msg));
                    }
                    PreToolOutcome::Proceed(new_args) => {
                        args = new_args;
                    }
                }
            }
        }

        let args_for_post = args.clone();
        let completed = match cancel_owned.as_ref() {
            None => Some(
                tools
                    .execute_tool_scoped(&tool_name, args, allow.as_deref(), is_subagent)
                    .await,
            ),
            Some(token) => {
                tokio::select! {
                    res = tools.execute_tool_scoped(
                        &tool_name,
                        args,
                        allow.as_deref(),
                        is_subagent,
                    ) => Some(res),
                    _ = token.cancelled() => None,
                }
            }
        };

        if let Some(ref hc) = runtime.hook_tool_ctx {
            if let Some(st) = &hc.steering {
                let res_for_hook = match &completed {
                    Some(r) => r.clone(),
                    None => Err("tool call cancelled".to_string()),
                };
                let hook_session = HookSessionInfo {
                    channel: channel.as_str(),
                    chat_id: chat_id.as_str(),
                    thread_id: thread_id_for_hooks.as_deref(),
                    metadata: runtime.inbound_metadata.as_ref(),
                    is_subagent: runtime.is_subagent,
                };
                run_post_tool_hooks(
                    st.as_ref(),
                    &tool_name,
                    tool_call_id_for_hooks.as_deref(),
                    &args_for_post,
                    &res_for_hook,
                    hook_session,
                )
                .await;
            }
        }

        if let Some(handle) = activity_handle {
            handle.stop().await;
        }

        match completed {
            Some(res) => {
                if let Err(ref e) = res {
                    if let Some(ticket_id) = e.strip_prefix(WAIT_SIGNAL_PREFIX) {
                        return ToolExecutionFinished::Waiting(ticket_id.to_string());
                    }
                }
                ToolExecutionFinished::Completed(res)
            }
            None => {
                hub.cancel_wait(&session_key);
                ToolExecutionFinished::Cancelled
            }
        }
    })
    .await
}

/// Bundles everything needed to run one inbound reasoning task (spawned from `AgentLogic::process`).
/// Cloned into each spawned main-chat reasoning task (and used to chain queued inbounds).
#[derive(Clone)]
struct ReasoningSpawnArgs {
    name: String,
    provider: Box<dyn Provider>,
    session_manager: Arc<SessionManager>,
    tools: Arc<ToolRegistry>,
    skills: Arc<SkillRegistry>,
    system_prompt: String,
    max_iterations: usize,
    max_tool_output_chars: usize,
    max_recent_summaries: usize,
    short_term_threshold_turns: usize,
    short_term_threshold_tokens: usize,
    tool_execution_activity: Option<SharedToolExecutionActivity>,
    outbound_tx: mpsc::Sender<BusMessage>,
    logger_tx: LoggerHandle,
    clarification_hub: Arc<ClarificationHub>,
    doom_loop_enabled: bool,
    cancellation_tokens: Arc<dashmap::DashMap<String, Arc<tokio_util::sync::CancellationToken>>>,
    pending_inbound: Arc<dashmap::DashMap<String, Mutex<VecDeque<crate::bus::InboundMessage>>>>,
    harness_runtime_summary: String,
    forbid_final_without_tools: bool,
    shell_policy: Arc<ResolvedShellPolicy>,
    hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
}

fn send_background_job_notification(
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
            crate::channels::terminal_ui::protocol::ISANAGENT_BACKGROUND_JOB_STARTED.to_string(),
            serde_json::json!(true),
        );
        meta.insert(
            crate::channels::terminal_ui::protocol::METADATA_BACKGROUND_JOB_TOOL_NAME.to_string(),
            serde_json::json!("background_reasoning"),
        );
    } else {
        meta.insert(
            crate::channels::terminal_ui::protocol::ISANAGENT_BACKGROUND_JOB_FINISHED.to_string(),
            serde_json::json!(true),
        );
        if let Some(s) = status {
            meta.insert(
                crate::channels::terminal_ui::protocol::METADATA_BACKGROUND_JOB_STATUS.to_string(),
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

fn spawn_main_chat_reasoning_turn(args: ReasoningSpawnArgs, inbound: crate::bus::InboundMessage) {
    let chat_id = inbound.chat_id.clone();
    let cancel_token = Arc::new(tokio_util::sync::CancellationToken::new());
    args.cancellation_tokens
        .insert(chat_id.clone(), cancel_token.clone());

    let args_for_chain = args.clone();
    let cancellation_tokens = args.cancellation_tokens.clone();
    let pending_inbound = args.pending_inbound.clone();
    let name = args.name.clone();
    let provider = dyn_clone::clone_box(&*args.provider);
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
                &format!("Spawning reasoning task for chat_id: {}", task_chat_id),
            )
            .with_chat_id(&task_chat_id),
        ));

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
            provider,
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
                        &format!("Reasoning loop failed for chat_id {}: {}", task_chat_id, e),
                    )
                    .with_chat_id(&task_chat_id),
                ));
                let notice = crate::channels::terminal::build_channel_error_notice(
                    &inbound_channel,
                    &task_chat_id,
                    inbound_thread_id.as_deref(),
                    e,
                );
                let _ = outbound_tx.send(BusMessage::Outbound(notice)).await;
            }
            Ok(Ok(_)) if task_token_arc.is_cancelled() => {
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::info(
                        &agent_name,
                        &format!(
                            "Reasoning task for chat_id {} finished via cancellation.",
                            task_chat_id
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
                            "Reasoning task for chat_id {} finished successfully.",
                            task_chat_id
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
                        &format!("Reasoning loop panicked for chat_id {}", task_chat_id),
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

                let notice = crate::channels::terminal::build_channel_error_notice(
                    &inbound_channel,
                    &task_chat_id,
                    inbound_thread_id.as_deref(),
                    panic_msg,
                );
                let _ = outbound_tx.send(BusMessage::Outbound(notice)).await;
            }
        }

        if let Some(job_id) = background_job_id {
            let (state, last_error) = match &res {
                Ok(Ok(s)) if s.starts_with(WAITING_FOR_USER_RESULT_PREFIX) => ("waiting", None),
                Ok(Ok(_)) => {
                    if task_token_arc.is_cancelled() {
                        ("failed", Some("Cancelled".to_string()))
                    } else {
                        ("completed", None)
                    }
                }
                Ok(Err(e)) => ("failed", Some(e.to_string())),
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
            Arc::ptr_eq(stored, &task_token_arc)
        });

        let next_inbound = pending_inbound.get(&task_chat_id).and_then(|r| {
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

        if let Some(next_inbound) = next_inbound {
            spawn_main_chat_reasoning_turn(args_for_chain, next_inbound);
        }
    });
}

pub(crate) struct ReasoningLoopCtx {
    pub(crate) name: String,
    pub(crate) provider: Box<dyn Provider>,
    pub(crate) session_manager: Arc<SessionManager>,
    pub(crate) tools: Arc<ToolRegistry>,
    pub(crate) skills: Arc<SkillRegistry>,
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

/// Constructor arguments for [`AgentLogic`], grouped to keep call sites readable.
//
// NOTE: `#[non_exhaustive]` is deliberately NOT applied here. Several fields (`provider`,
// `outbound_tx`, `logger_tx`, `clarification_hub`, …) cannot have a sensible `Default`,
// so the `..Default::default()` workaround that other Phase 0.0b structs use is not
// viable. Adding `#[non_exhaustive]` requires a builder API as a prerequisite — tracked
// as a follow-up to the Phase 0.0b sweep (see docs/public-api-surface.md §9.3).
pub struct AgentLogicParams {
    pub name: String,
    pub provider: Box<dyn Provider>,
    pub session_manager: SessionManager,
    pub tools: ToolRegistry,
    pub skills: SkillRegistry,
    pub system_prompt: String,
    pub max_iterations: usize,
    pub max_tool_output_chars: usize,
    pub max_recent_summaries: usize,
    pub short_term_threshold_turns: usize,
    pub short_term_threshold_tokens: usize,
    pub outbound_tx: mpsc::Sender<BusMessage>,
    pub logger_tx: LoggerHandle,
    pub clarification_hub: Arc<ClarificationHub>,
    /// When set, registers `subagent_*` / `task_*` tools and wires [`SubagentHarness`].
    pub subagent: Option<SubagentHarnessParams>,
    /// When true (default), inject corrective user text if repeated tool calls are detected.
    pub doom_loop_enabled: bool,
    /// Pre-formatted harness lines for system context (execution caps, subagent flags, etc.).
    pub harness_runtime_summary: String,
    /// System prompt used for `subagent_spawn` / plan runs (may include research appendix).
    pub subagent_system_prompt: String,
    /// Config default; inbound metadata `crate::bus::METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS` can override.
    pub forbid_final_without_tools: bool,
    /// Shell command safety policy (`exec`) resolved from config.
    pub shell_policy: ResolvedShellPolicy,
    /// Optional observation + steering hooks (`[harness.hooks]`).
    pub hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
}

/// Build-time options for the Phase 5 sub-agent harness (see `[harness.subagents]`).
//
// NOTE: `#[non_exhaustive]` is deferred — `src/main.rs` (a separate Cargo crate from the lib)
// constructs this struct directly when wiring the sub-agent harness. Adopting the marker
// requires either a builder or a constructor helper; tracked as a Phase 0.0b follow-up
// (see docs/public-api-surface.md §9.3).
#[derive(Clone, Debug)]
pub struct SubagentHarnessParams {
    pub cancel_children_on_parent_cancel: bool,
    pub allowed_tools: Option<Arc<HashSet<String>>>,
    pub max_tasks: usize,
    pub max_wait_secs: u64,
    pub agent_registry: Option<Arc<AgentRegistry>>,
    pub wake_on_completion: bool,
    pub task_history_retention: usize,
    pub bus_tx: Option<tokio::sync::mpsc::Sender<crate::bus::BusMessage>>,
}

/// The central logic for an autonomous Agent running inside an ActorNode.
/// It holds a LLM Provider, a persistent Memory context, and available Tools.
pub struct AgentLogic {
    name: String,
    provider: Arc<tokio::sync::RwLock<Box<dyn Provider>>>,
    session_manager: Arc<SessionManager>,
    tools: Arc<ToolRegistry>,
    skills: Arc<SkillRegistry>,
    system_prompt: String,
    max_iterations: usize,
    max_tool_output_chars: usize,
    max_recent_summaries: usize,
    short_term_threshold_turns: usize,
    short_term_threshold_tokens: usize,
    tool_execution_activity: Option<SharedToolExecutionActivity>,
    outbound_tx: mpsc::Sender<BusMessage>,
    logger_tx: LoggerHandle,
    cancellation_tokens: Arc<dashmap::DashMap<String, Arc<tokio_util::sync::CancellationToken>>>,
    /// FIFO per `chat_id` when a new user inbound arrives while main reasoning is active.
    pending_inbound: Arc<dashmap::DashMap<String, Mutex<VecDeque<crate::bus::InboundMessage>>>>,
    clarification_hub: Arc<ClarificationHub>,
    subagent_harness: Option<Arc<SubagentHarness>>,
    doom_loop_enabled: bool,
    harness_runtime_summary: String,
    forbid_final_without_tools: bool,
    shell_policy: Arc<ResolvedShellPolicy>,
    hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
}

impl AgentLogic {
    pub fn new(params: AgentLogicParams) -> Self {
        let AgentLogicParams {
            name,
            provider,
            session_manager,
            tools,
            skills,
            system_prompt,
            max_iterations,
            max_tool_output_chars,
            max_recent_summaries,
            short_term_threshold_turns,
            short_term_threshold_tokens,
            outbound_tx,
            logger_tx,
            clarification_hub,
            subagent,
            doom_loop_enabled,
            harness_runtime_summary,
            subagent_system_prompt,
            forbid_final_without_tools,
            shell_policy,
            hook_tool_ctx,
        } = params;

        let harness_for_subagent = harness_runtime_summary.clone();
        let session_manager = Arc::new(session_manager);
        let skills = Arc::new(skills);
        let tools = Arc::new(tools);
        let memory_node = session_manager.get_memory_node();
        let shell_policy = Arc::new(shell_policy);

        let provider = Arc::new(tokio::sync::RwLock::new(provider));

        let subagent_harness = subagent.map(|p| {
            Arc::new(SubagentHarness::new(subagent::SubagentSpawnDeps {
                agent_name: name.clone(),
                provider: provider.clone(),
                session_manager: session_manager.clone(),
                skills: skills.clone(),
                system_prompt: subagent_system_prompt,
                max_iterations,
                max_tool_output_chars,
                max_recent_summaries,
                short_term_threshold_turns,
                short_term_threshold_tokens,
                tool_execution_activity: None,
                outbound_tx: outbound_tx.clone(),
                logger_tx: logger_tx.clone(),
                clarification_hub: clarification_hub.clone(),
                cancel_children_on_parent_cancel: p.cancel_children_on_parent_cancel,
                default_allowlist: p.allowed_tools.clone(),
                max_tasks: p.max_tasks,
                max_wait_secs: p.max_wait_secs,
                doom_loop_enabled,
                memory_node: memory_node.clone(),
                harness_runtime_summary: harness_for_subagent.clone(),
                shell_policy: shell_policy.clone(),
                hook_tool_ctx: hook_tool_ctx.clone(),
                agent_registry: p.agent_registry.clone(),
                wake_on_completion: p.wake_on_completion,
                task_history_retention: p.task_history_retention,
                bus_tx: p.bus_tx.clone(),
            }))
        });

        let mut agent = Self {
            name,
            provider,
            session_manager,
            tools,
            skills,
            system_prompt,
            max_iterations,
            max_tool_output_chars,
            max_recent_summaries,
            short_term_threshold_turns,
            short_term_threshold_tokens,
            tool_execution_activity: None,
            outbound_tx,
            logger_tx,
            cancellation_tokens: Arc::new(dashmap::DashMap::new()),
            pending_inbound: Arc::new(dashmap::DashMap::new()),
            clarification_hub,
            subagent_harness: subagent_harness.clone(),
            doom_loop_enabled,
            harness_runtime_summary,
            forbid_final_without_tools,
            shell_policy: shell_policy.clone(),
            hook_tool_ctx,
        };

        let tools_mut = Arc::get_mut(&mut agent.tools)
            .expect("expected unique ownership of tools registry during initialization");
        if let Some(ref h) = subagent_harness {
            subagent::register_subagent_tools(tools_mut, h.clone(), memory_node);
        }
        let skill_reg = agent.skills.clone();
        let loader_tool = LoadSkillTool {
            registry: skill_reg,
        };
        tools_mut.register(Box::new(loader_tool));

        if let Some(ref h) = subagent_harness {
            h.bind_tools(agent.tools.clone())
                .expect("subagent bind_tools after unique registry init");
        }

        agent
    }

    /// Hot-swap the LLM provider at runtime (used by `/model` command).
    pub async fn switch_provider(&self, new_provider: Box<dyn Provider>) {
        *self.provider.write().await = new_provider;
    }

    pub fn with_tool_execution_activity(
        mut self,
        tool_execution_activity: SharedToolExecutionActivity,
    ) -> Self {
        self.tool_execution_activity = Some(tool_execution_activity);
        self
    }

    async fn reasoning_spawn_args(&self) -> ReasoningSpawnArgs {
        let provider_guard = self.provider.read().await;
        ReasoningSpawnArgs {
            name: self.name.clone(),
            provider: dyn_clone::clone_box(&**provider_guard),
            session_manager: self.session_manager.clone(),
            tools: self.tools.clone(),
            skills: self.skills.clone(),
            system_prompt: self.system_prompt.clone(),
            max_iterations: self.max_iterations,
            max_tool_output_chars: self.max_tool_output_chars,
            max_recent_summaries: self.max_recent_summaries,
            short_term_threshold_turns: self.short_term_threshold_turns,
            short_term_threshold_tokens: self.short_term_threshold_tokens,
            tool_execution_activity: self.tool_execution_activity.clone(),
            outbound_tx: self.outbound_tx.clone(),
            logger_tx: self.logger_tx.clone(),
            clarification_hub: self.clarification_hub.clone(),
            doom_loop_enabled: self.doom_loop_enabled,
            cancellation_tokens: self.cancellation_tokens.clone(),
            pending_inbound: self.pending_inbound.clone(),
            harness_runtime_summary: self.harness_runtime_summary.clone(),
            forbid_final_without_tools: self.forbid_final_without_tools,
            shell_policy: self.shell_policy.clone(),
            hook_tool_ctx: self.hook_tool_ctx.clone(),
        }
    }

    #[cfg(test)]
    async fn execute_tool_call(
        &self,
        chat_id: &str,
        tool_name: &str,
        args: Value,
    ) -> Result<String, String> {
        match execute_tool_call_with_activity(
            &self.tools,
            self.tool_execution_activity.clone(),
            chat_id,
            "test",
            &self.outbound_tx,
            tool_name,
            None,
            args,
            None,
            ToolCallRuntime {
                session: ToolExecCtx::new("test", chat_id, None),
                hub: self.clarification_hub.clone(),
                is_subagent: false,
                subagent_allowlist: None,
                shell_policy: self.shell_policy.clone(),
                unattended_session: false,
                hook_tool_ctx: None,
                inbound_metadata: Arc::new(HashMap::new()),
            },
        )
        .await
        {
            ToolExecutionFinished::Completed(res) => res,
            ToolExecutionFinished::Waiting(ticket_id) => Err(format!(
                "tool call waiting for clarification ticket: {}",
                ticket_id
            )),
            ToolExecutionFinished::Cancelled => {
                Err("tool call cancelled without cancellation token".to_string())
            }
        }
    }
}

/// The Agent processes incoming BusMessages, updates memory based on session key,
/// and outputs BusMessages (specifically Outbound) back to the channel.
#[async_trait]
impl ActorLogic<BusMessage> for AgentLogic {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn process(
        &mut self,
        packet: BusMessage,
    ) -> Result<Option<(String, BusMessage)>, ActorError> {
        match packet {
            BusMessage::Cancel(chat_id) => {
                if let Some(h) = &self.subagent_harness {
                    if h.cancel_children_on_parent_cancel() {
                        h.cancel_children_for_parent(&chat_id);
                    }
                }
                if let Some((_, token)) = self.cancellation_tokens.remove(&chat_id) {
                    token.cancel();
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::info(
                            &self.name,
                            &format!("Cancelled reasoning loop for chat_id: {}", chat_id),
                        )
                        .with_chat_id(&chat_id),
                    ));
                }
                self.pending_inbound.remove(&chat_id);
                return Ok(None);
            }
            BusMessage::SwitchModel {
                provider_name,
                model_name,
                base_url,
                api_key,
            } => {
                let new_provider = crate::provider::create_provider(
                    &provider_name,
                    &base_url,
                    &api_key,
                    &model_name,
                );
                self.switch_provider(new_provider).await;
                let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info(
                    &self.name,
                    &format!(
                        "Switched to provider={} model={}",
                        provider_name, model_name
                    ),
                )));
                return Ok(None);
            }
            BusMessage::Inbound(inbound) => {
                let chat_id = inbound.chat_id.clone();
                let session_key = inbound.clarification_session_key();
                if self
                    .clarification_hub
                    .try_deliver_reply(&session_key, inbound.content.clone())
                {
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::debug(
                            &self.name,
                            "Inbound delivered as ask_user clarification reply (same session).",
                        )
                        .with_chat_id(&chat_id),
                    ));
                    return Ok(None);
                }

                // Check for background job resume via explicit clarification ticket UI interaction
                if let Some(res) = self
                    .try_resume_background_job_from_ticket(&inbound, &chat_id, &session_key)
                    .await
                {
                    return res;
                }

                let _ = self.logger_tx.send(BusMessage::Log(
                    LogEvent::info(
                        &self.name,
                        &format!(
                            "Received InboundMessage for chat_id [{}] ({} chars)",
                            chat_id,
                            inbound.content.len(),
                        ),
                    )
                    .with_chat_id(&chat_id),
                ));

                if self.cancellation_tokens.contains_key(&chat_id) {
                    // Check if this is a synthetic cron trigger. If so, drop it if we're already busy.
                    if metadata_truthy(
                        &inbound.metadata,
                        crate::bus::METADATA_SYNTHETIC_CRON_TRIGGER,
                    ) {
                        let _ = self.logger_tx.send(BusMessage::Log(
                            LogEvent::info(
                                &self.name,
                                "Dropping synthetic cron trigger because chat is already active.",
                            )
                            .with_chat_id(&chat_id),
                        ));
                        return Ok(None);
                    }

                    let queue = self
                        .pending_inbound
                        .entry(chat_id.clone())
                        .or_insert_with(|| Mutex::new(VecDeque::new()));
                    let mut guard = match queue.lock() {
                        Ok(g) => g,
                        Err(poisoned) => {
                            let _ = self.logger_tx.send(BusMessage::Log(
                                LogEvent::warn(
                                    &self.name,
                                    "pending_inbound mutex poisoned; recovering queued inbound state.",
                                )
                                .with_chat_id(&chat_id),
                            ));
                            poisoned.into_inner()
                        }
                    };
                    guard.push_back(inbound);
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::debug(
                            &self.name,
                            &format!(
                                "Queued inbound for chat_id {} (FIFO) — reasoning already active.",
                                chat_id
                            ),
                        )
                        .with_chat_id(&chat_id),
                    ));
                    return Ok(None);
                }

                // If not busy, check if there's a waiting background job for this chat
                // to automatically resume it (user replied to the thread instead of via ticket UI).
                if let Some(res) = self
                    .try_auto_resume_waiting_job(&inbound, &chat_id, &session_key)
                    .await
                {
                    return res;
                }

                spawn_main_chat_reasoning_turn(self.reasoning_spawn_args().await, inbound);

                Ok(None)
            }
            BusMessage::TriggerCompaction {
                session_key,
                focus_instructions,
                trigger,
            } => {
                // PR-5 + PR-10: delegate to the internal `trigger_compaction_with_reason`
                // so the carried `trigger` (Manual vs AgentSelf) propagates into the
                // `CompactionTriggered` telemetry event. The per-chat FIFO guard
                // already lives inside that helper; failures are logged and dropped.
                let reason = trigger.unwrap_or(crate::bus::CompactionTrigger::Manual);
                if let Err(e) = self
                    .trigger_compaction_with_reason(session_key.clone(), focus_instructions, reason)
                    .await
                {
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::warn(
                            &self.name,
                            &format!(
                                "TriggerCompaction dropped for session_key={}: {}",
                                session_key, e
                            ),
                        ),
                    ));
                }
                Ok(None)
            }
            BusMessage::Outbound(_)
            | BusMessage::Telemetry(_)
            | BusMessage::LoggerControl(_)
            | BusMessage::Log(_)
            | BusMessage::PromoteSyncToBackground(_)
            | BusMessage::SetTerminalSessionChat { .. } => Ok(None),
        }
    }
}

impl AgentLogic {
    /// PR-5: manually trigger a compaction for `session_key` outside the normal
    /// threshold path. The compaction runs synchronously in the calling task and
    /// emits the full matched `CompactionTriggered { reason: Manual }` + (`Completed`
    /// or `Failed`) telemetry pair.
    ///
    /// Construct `session_key` via [`crate::bus::clarification_session_key`].
    /// `focus_instructions`, when present, is appended to the summarizer prompt
    /// as a `FOCUS:` block so the model can prioritize certain content.
    ///
    /// **Per-chat FIFO.** Returns `Err` if a reasoning turn is currently in flight
    /// for the same `chat_id` — the AGENTS.md invariant requires compaction to
    /// happen *between* turns, not during. Callers that arrive via the bus
    /// (`BusMessage::TriggerCompaction`) should expect drops in that case.
    pub async fn trigger_compaction(
        &self,
        session_key: String,
        focus_instructions: Option<String>,
    ) -> Result<crate::agent::compaction::CompactionOutcome, String> {
        self.trigger_compaction_with_reason(
            session_key,
            focus_instructions,
            crate::bus::CompactionTrigger::Manual,
        )
        .await
    }

    /// Internal entry point shared by the pub [`Self::trigger_compaction`] API
    /// and the PR-10 `compact_context` tool path (via `BusMessage::TriggerCompaction`
    /// with `trigger: Some(AgentSelf)`). Splits the public surface from the
    /// `CompactionTrigger` taxonomy so the eval pipeline can distinguish
    /// caller-driven (`Manual`) from agent-driven (`AgentSelf`) compactions.
    async fn trigger_compaction_with_reason(
        &self,
        session_key: String,
        focus_instructions: Option<String>,
        trigger_reason: crate::bus::CompactionTrigger,
    ) -> Result<crate::agent::compaction::CompactionOutcome, String> {
        // session_key format: `<channel>:<chat_id>:<thread_part>`. The chat_id
        // segment drives the in-flight guard and telemetry labelling.
        let chat_id = session_key
            .split(':')
            .nth(1)
            .ok_or_else(|| {
                format!(
                    "Malformed session_key (expected `channel:chat_id:thread`): {}",
                    session_key
                )
            })?
            .to_string();

        if self.cancellation_tokens.contains_key(&chat_id) {
            return Err(format!(
                "Refusing manual compaction: reasoning turn in flight for chat_id={}",
                chat_id
            ));
        }

        let mem = self
            .session_manager
            .get_session(&session_key)
            .await
            .map_err(|e| format!("get_session({}): {}", session_key, e))?;
        let current_context = mem
            .get_context_since_reflection()
            .await
            .map_err(|e| format!("get_context_since_reflection({}): {}", session_key, e))?;
        let user_turns = current_context
            .iter()
            .filter(|m| m.role == "user")
            .count();
        let approx_tokens: usize = estimate_context_tokens(&current_context);

        // Most recent summary keyed by the same channel:chat_id prefix
        // — same scheme used at the threshold-trigger site.
        let prefix = {
            let mut parts = session_key.splitn(3, ':');
            let channel = parts.next().unwrap_or("");
            format!("{}:{}", channel, chat_id)
        };
        let recent = self
            .session_manager
            .get_recent_summaries(&prefix, self.max_recent_summaries.max(1))
            .await
            .unwrap_or_default();

        let memory_node = self.session_manager.get_memory_node();
        let provider_guard = self.provider.read().await;
        // Manual triggers have no per-call cancellation; a token that never
        // fires keeps `do_compaction`'s `select!` valid without altering behavior.
        let cancel_token = tokio_util::sync::CancellationToken::new();

        let outcome = crate::agent::compaction::do_compaction(
            crate::agent::compaction::DoCompactionArgs {
                chat_id: &chat_id,
                session_key: &session_key,
                trigger_reason,
                tokens_before: approx_tokens.min(u32::MAX as usize) as u32,
                turns_before: user_turns.min(u32::MAX as usize) as u32,
                current_context: &current_context,
                existing_summary: recent.first().map(|s| s.as_str()),
                focus_instructions: focus_instructions.as_deref(),
                provider: provider_guard.as_ref(),
                memory_node: &memory_node,
                outbound_tx: &self.outbound_tx,
                cancel_token: &cancel_token,
            },
        )
        .await;
        Ok(outcome)
    }

    async fn try_resume_background_job_from_ticket(
        &mut self,
        inbound: &InboundMessage,
        chat_id: &str,
        session_key: &str,
    ) -> Option<Result<Option<(String, BusMessage)>, ActorError>> {
        if let Some(ticket_id) = inbound
            .metadata
            .get(crate::bus::METADATA_CLARIFICATION_TICKET_ID)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
        {
            let memory_node = self.session_manager.get_memory_node();
            let (tx, rx) = tokio::sync::oneshot::channel();
            let _ = memory_node
                .send_packet(MemoryMessage::GetClarificationTicket {
                    ticket_id: ticket_id.clone(),
                    reply: SharedReply::new(tx),
                })
                .await;

            if let Ok(Ok(Some(ticket))) = rx.await {
                let _ = self.logger_tx.send(BusMessage::Log(
                    LogEvent::info(
                        &self.name,
                        &format!(
                            "Resuming background job [{}] via clarification ticket [{}]",
                            ticket.job_id, ticket_id
                        ),
                    )
                    .with_chat_id(chat_id),
                ));

                if let Err(e) = self
                    .resolve_and_resume_job(
                        inbound,
                        &ticket.ticket_id,
                        &ticket.job_id,
                        ticket.tool_call_id.as_deref(),
                        session_key,
                    )
                    .await
                {
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::error(
                            &self.name,
                            &format!("Failed to resume job {}: {}", ticket.job_id, e),
                        )
                        .with_chat_id(chat_id),
                    ));
                    let notice = crate::channels::terminal::build_channel_error_notice(
                        &inbound.channel,
                        chat_id,
                        inbound.thread_id.as_deref(),
                        &format!("Failed to resume background job [{}]: {}", ticket.job_id, e),
                    );
                    let _ = self.outbound_tx.try_send(BusMessage::Outbound(notice));
                    return Some(Ok(None));
                }
                return Some(Ok(None));
            }
        }
        None
    }

    async fn try_auto_resume_waiting_job(
        &mut self,
        inbound: &InboundMessage,
        chat_id: &str,
        session_key: &str,
    ) -> Option<Result<Option<(String, BusMessage)>, ActorError>> {
        if metadata_truthy(
            &inbound.metadata,
            crate::bus::METADATA_CLARIFICATION_TICKET_ID,
        ) {
            return None;
        }

        let memory_node = self.session_manager.get_memory_node();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = memory_node
            .send_packet(MemoryMessage::ListBackgroundJobs {
                chat_id: Some(chat_id.to_string()),
                limit: 10,
                reply: SharedReply::new(tx),
            })
            .await;

        if let Ok(Ok(jobs)) = rx.await {
            let waiting_jobs: Vec<_> = jobs.into_iter().filter(|j| j.state == "waiting").collect();
            for job in waiting_jobs {
                // Found a waiting job. Now find the latest ticket for it.
                let (tx2, rx2) = tokio::sync::oneshot::channel();
                let _ = memory_node
                    .send_packet(MemoryMessage::ListClarificationTickets {
                        job_id: Some(job.job_id.clone()),
                        chat_id: Some(chat_id.to_string()),
                        status: Some("waiting".to_string()),
                        limit: 1,
                        reply: SharedReply::new(tx2),
                    })
                    .await;

                if let Ok(Ok(tickets)) = rx2.await {
                    if let Some(ticket) = tickets.into_iter().next() {
                        let _ = self.logger_tx.send(BusMessage::Log(
                            LogEvent::info(
                                &self.name,
                                &format!(
                                    "Auto-resuming waiting background job [{}] via thread reply to ticket [{}]",
                                    job.job_id, ticket.ticket_id
                                ),
                            )
                            .with_chat_id(chat_id),
                        ));

                        if let Err(e) = self
                            .resolve_and_resume_job(
                                inbound,
                                &ticket.ticket_id,
                                &job.job_id,
                                ticket.tool_call_id.as_deref(),
                                session_key,
                            )
                            .await
                        {
                            let _ = self.logger_tx.send(BusMessage::Log(
                                LogEvent::error(
                                    &self.name,
                                    &format!("Failed to auto-resume job {}: {}", job.job_id, e),
                                )
                                .with_chat_id(chat_id),
                            ));
                            let notice = crate::channels::terminal::build_channel_error_notice(
                                &inbound.channel,
                                chat_id,
                                inbound.thread_id.as_deref(),
                                &format!(
                                    "Failed to auto-resume background job [{}]: {}",
                                    job.job_id, e
                                ),
                            );
                            let _ = self.outbound_tx.try_send(BusMessage::Outbound(notice));
                            return Some(Ok(None));
                        }
                        return Some(Ok(None));
                    }
                }
            }
        }
        None
    }

    async fn resolve_and_resume_job(
        &mut self,
        inbound: &InboundMessage,
        ticket_id: &str,
        job_id: &str,
        tool_call_id: Option<&str>,
        session_key: &str,
    ) -> Result<(), String> {
        let memory_node = self.session_manager.get_memory_node();

        // 1. Resolve everything for this ticket in a single go
        let (tx, rx) = tokio::sync::oneshot::channel();
        memory_node
            .send_packet(MemoryMessage::ResolveClarificationTicketFull {
                ticket_id: ticket_id.to_string(),
                job_id: job_id.to_string(),
                response: inbound.content.clone(),
                reply: SharedReply::new(tx),
            })
            .await
            .map_err(|e| format!("Memory actor error: {}", e))?;

        rx.await
            .map_err(|_| "Memory actor channel closed".to_string())?
            .map_err(|e| format!("Memory node failed to resolve ticket fully: {}", e))?;

        // 2. Inject tool response into memory
        if let Some(id) = tool_call_id {
            if let Ok(mut mem) = self.session_manager.get_session(session_key).await {
                // Determine tool name from memory
                let mut tool_name_for_resume = None;
                if let Ok(context) = mem.get_context().await {
                    for msg in context.iter().rev() {
                        if msg.role == "assistant" {
                            if let Some(calls) = &msg.tool_calls {
                                if let Some(tc) = calls.iter().find(|c| c.id == id) {
                                    tool_name_for_resume = Some(tc.function.name.clone());
                                    break;
                                }
                            }
                        }
                    }
                }

                mem.add_message(crate::utils::ChatMessage::tool(
                    &inbound.content,
                    id,
                    tool_name_for_resume.as_deref(),
                ))
                .await
                .map_err(|e| format!("Failed to inject tool response into memory: {}", e))?;
            } else {
                return Err(format!("Failed to get session {}", session_key));
            }
        }

        // 3. Spawn turn with resume metadata
        let mut resumed_inbound = inbound.clone();
        resumed_inbound.metadata.insert(
            crate::bus::METADATA_SYNTHETIC_BACKGROUND_RESUME.to_string(),
            serde_json::json!(true),
        );
        resumed_inbound.metadata.insert(
            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
            serde_json::json!(job_id),
        );

        spawn_main_chat_reasoning_turn(self.reasoning_spawn_args().await, resumed_inbound);
        Ok(())
    }
}

/// Outcome of a `provider.chat` invocation that may be retried for transient errors.
enum ChatRetryOutcome {
    Ok(crate::utils::LLMResponse),
    /// Cancellation token fired during a chat or sleep; caller exits the reasoning loop.
    Cancelled,
    /// Retries exhausted; final user-facing error string. The caller is expected to surface
    /// an LLM-failed banner.
    Failed(String),
    /// PR-4: provider rejected the request because the input exceeded its context
    /// window. Not retried — bouncing the same payload guarantees the same failure.
    /// The reasoning loop is expected to (eventually, PR-4.1) emergency-compact
    /// and retry once.
    ContextOverflow {
        tokens_attempted: u32,
        max: Option<u32>,
    },
}

/// Wrap `provider.chat` with a small retry loop for transient errors (network/5xx/429).
/// Up to 3 total attempts with exponential backoff (1s/2s/4s); the cancel token preempts
/// both the chat and the sleep.
async fn chat_with_retry(
    provider: &dyn crate::traits::Provider,
    context: &[crate::utils::ChatMessage],
    tools_payload: Option<serde_json::Value>,
    cancel_token: &tokio_util::sync::CancellationToken,
    logger_tx: &LoggerHandle,
    name: &str,
    chat_id: &str,
) -> ChatRetryOutcome {
    const MAX_ATTEMPTS: u32 = 3;
    const BACKOFF_BASE_MS: u64 = 1000;
    let mut last_err: Option<crate::utils::LLMError> = None;
    for attempt in 0..MAX_ATTEMPTS {
        let res = tokio::select! {
            r = provider.chat(context, tools_payload.clone()) => r,
            _ = cancel_token.cancelled() => {
                let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                    name,
                    "Reasoning loop cancelled during LLM call.",
                ).with_chat_id(chat_id)));
                return ChatRetryOutcome::Cancelled;
            }
        };
        match res {
            Ok(resp) => return ChatRetryOutcome::Ok(resp),
            Err(crate::utils::LLMError::ContextOverflow {
                tokens_attempted,
                max,
            }) => {
                // PR-4: short-circuit — retrying the identical payload guarantees
                // the same overflow. Caller decides whether to compact and retry.
                return ChatRetryOutcome::ContextOverflow {
                    tokens_attempted,
                    max,
                };
            }
            Err(e) => {
                let transient = e.is_transient();
                let is_last = attempt + 1 >= MAX_ATTEMPTS;
                if !transient || is_last {
                    last_err = Some(e);
                    break;
                }
                let backoff_ms = BACKOFF_BASE_MS * (1u64 << attempt);
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::warn(
                        name,
                        &format!(
                            "LLM call failed (attempt {}/{}): {}. Retrying in {}ms.",
                            attempt + 1,
                            MAX_ATTEMPTS,
                            e,
                            backoff_ms
                        ),
                    )
                    .with_chat_id(chat_id),
                ));
                last_err = Some(e);
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)) => {}
                    _ = cancel_token.cancelled() => {
                        return ChatRetryOutcome::Cancelled;
                    }
                }
            }
        }
    }
    ChatRetryOutcome::Failed(
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown LLM error".to_string()),
    )
}

/// Build the user-facing terminal banner for an exhausted-retry LLM failure.
/// Carries the `isanagent_llm_retry_available` metadata flag so the terminal UI can
/// gate `/retry` on the banner being active.
fn build_llm_failed_banner(
    channel: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    error: &str,
) -> OutboundMessage {
    let content = format!(
        "LLM call failed after 3 attempts: {error}\nPress /retry to try again or /cancel to abandon."
    );
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    if channel == "terminal" {
        metadata.insert(
            crate::channels::terminal_ui::protocol::ISANAGENT_TERMINAL_ERROR.to_string(),
            serde_json::json!(true),
        );
        metadata.insert(
            crate::channels::terminal_ui::protocol::ISANAGENT_LLM_RETRY_AVAILABLE.to_string(),
            serde_json::json!(true),
        );
    }
    OutboundMessage {
        channel: channel.to_string(),
        chat_id: chat_id.to_string(),
        thread_id: thread_id.map(|s| s.to_string()),
        content,
        metadata,
    }
}

impl AgentLogic {
    pub(crate) async fn run_reasoning_loop(ctx: ReasoningLoopCtx) -> Result<String, String> {
        let ReasoningLoopCtx {
            name,
            provider,
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

        let session_key = tool_exec_ctx.session_key.clone();
        let cancel_notice = "Request cancelled while the agent was processing this turn.";

        let mut mem = session_manager.get_session(&session_key).await?;

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
                return Ok(String::new());
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
            .map(|t| format!(", thread: '{}'", t))
            .unwrap_or_default();
        let now = chrono::Local::now().to_rfc3339();
        let os_family = std::env::consts::OS;
        let path_sep = std::path::MAIN_SEPARATOR;
        let shell_family = if cfg!(windows) {
            if std::env::var("PSModulePath").is_ok() {
                "powershell"
            } else {
                "cmd"
            }
        } else if std::env::var("SHELL")
            .ok()
            .map(|s| s.contains("bash"))
            .unwrap_or(false)
        {
            "bash"
        } else {
            "sh"
        };
        let mut runtime_context = format!(
            "[RUNTIME CONTEXT] Current time is {}. You are navigating and responding in channel: '{}', with chat ID: '{}'{}.",
            now,
            inbound.channel,
            inbound.chat_id,
            thread_info
        );
        runtime_context.push_str(&format!(
            " Host hints: os_family='{}', shell_family='{}', path_separator='{}', windows={}.",
            os_family,
            shell_family,
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
                    .push_str(&format!(" Autonomous session deadline (RFC3339): '{}'.", s));
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
                    UserPromptHookOutcome::Block(msg) => return Err(msg),
                    UserPromptHookOutcome::InjectPrefix(prefix) => {
                        contextualized_content = format!("{}\n{}", prefix, contextualized_content);
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
        mem.add_message(user_msg).await?;

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
        let mut iterations = 0;
        // PR-4.1: hard cap — at most one emergency context-overflow recovery
        // per inbound. A second overflow within the same turn surfaces the
        // failure to the user instead of looping on compact-and-retry.
        let mut overflow_recovery_used = false;
        // P1.4: count consecutive doom-loop detections so we can escalate from an advisory
        // nudge to a hard stop. Reset to 0 on any iteration with no detection.
        let mut consecutive_doom_detections: usize = 0;
        // Set when the doom loop persists past the nudge budget; branches the terminal message.
        let mut doom_loop_stuck = false;
        // Ground-truth input size from the most recent LLM call's `usage.prompt_tokens` (exact,
        // server-counted). The bytes/4 heuristic under-counts code/JSON/non-English — exactly what
        // this agent generates — so the compaction trigger uses `max(estimate, last_prompt_tokens)`
        // to avoid silently overflowing the context window. Updated after each provider response.
        let mut last_prompt_tokens: Option<u32> = None;

        while iterations < max_iterations {
            if cancel_token.is_cancelled() {
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::info(&name, "Reasoning loop cancelled before iteration start.")
                        .with_chat_id(&inbound.chat_id),
                ));
                persist_and_cancel!();
            }
            iterations += 1;

            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::debug(
                    &name,
                    &format!("Iteration {}/{}", iterations, max_iterations),
                )
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
                        msg.content =
                            Some(crate::utils::MessageContent::Text(placeholder));
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
                "\n--- Reasoning budget ---\nYou are on tool/LLM step {} of {} for this user turn.\n",
                iterations, max_iterations
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
                skills.get_capabilities_summary(),
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
                            doom_loop_stuck = true;
                            let _ = logger_tx.send(BusMessage::Log(
                                LogEvent::warn(
                                    &name,
                                    &format!(
                                        "Doom loop still active after {} consecutive detections — stopping the run.",
                                        consecutive_doom_detections
                                    ),
                                )
                                .with_chat_id(&inbound.chat_id),
                            ));
                            break;
                        }
                    } else {
                        // Detected in the window but the model has moved on — don't escalate.
                        consecutive_doom_detections = 0;
                    }
                    let correction = crate::utils::ChatMessage::user(&prompt);
                    mem.add_message(correction.clone()).await?;
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

            let response = match chat_with_retry(
                provider.as_ref(),
                &context,
                tools_payload,
                &cancel_token,
                &logger_tx,
                &name,
                &inbound.chat_id,
            )
            .await
            {
                ChatRetryOutcome::Ok(resp) => resp,
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
                        max.map(|m| m.to_string()).unwrap_or_else(|| "?".to_string()),
                    );
                    let persisted = format!(
                        "LLM call failed: {err}\nPress /retry to try again or /cancel to abandon."
                    );
                    persist_terminal_assistant_message(
                        &mut mem,
                        &logger_tx,
                        &name,
                        &inbound.chat_id,
                        &persisted,
                    )
                    .await;
                    let mut banner = build_llm_failed_banner(
                        &inbound.channel,
                        &inbound.chat_id,
                        inbound.thread_id.as_deref(),
                        &err,
                    );
                    if let Some(job_id) =
                        inbound.metadata.get(crate::bus::METADATA_BACKGROUND_JOB_ID)
                    {
                        banner.metadata.insert(
                            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
                            job_id.clone(),
                        );
                    }
                    let _ = outbound_tx.send(BusMessage::Outbound(banner)).await;
                    return Err(err);
                }
                ChatRetryOutcome::Failed(err) => {
                    let persisted = format!(
                        "LLM call failed after 3 attempts: {err}\nPress /retry to try again or /cancel to abandon."
                    );
                    persist_terminal_assistant_message(
                        &mut mem,
                        &logger_tx,
                        &name,
                        &inbound.chat_id,
                        &persisted,
                    )
                    .await;
                    let mut banner = build_llm_failed_banner(
                        &inbound.channel,
                        &inbound.chat_id,
                        inbound.thread_id.as_deref(),
                        &err,
                    );
                    if let Some(job_id) =
                        inbound.metadata.get(crate::bus::METADATA_BACKGROUND_JOB_ID)
                    {
                        banner.metadata.insert(
                            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
                            job_id.clone(),
                        );
                    }
                    let _ = outbound_tx.send(BusMessage::Outbound(banner)).await;
                    return Err(err);
                }
            };

            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::debug(&name, "Provider responded.").with_chat_id(&inbound.chat_id),
            ));

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
                mem.add_message(assistant_msg).await?;

                let parallel_ok = !is_subagent
                    && tool_calls.len() > 1
                    && tool_calls.iter().all(|tc| {
                        crate::tools::ToolRegistry::is_parallel_safe_tool(tc.function.name.as_str())
                    });

                let finalize_tool_output = |res: Result<String, String>| -> String {
                    match res {
                        Ok(mut output) => {
                            crate::utils::truncate_utf8_safe(
                                &mut output,
                                max_tool_output_chars,
                                "\n... [TRUNCATED FOR LENGTH]",
                            );
                            output
                        }
                        Err(e) => format!("Error: {}", e),
                    }
                };

                if parallel_ok {
                    for tc in tool_calls.iter() {
                        if cancel_token.is_cancelled() {
                            persist_and_cancel!();
                        }
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
                        let args =
                            serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                                .unwrap_or_else(|_| serde_json::json!({}));
                        if tool_name == "exec" {
                            if let Some(cmd) = extract_exec_command(&args) {
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
                        let cancel_token = cancel_token.clone();
                        let tool_exec_ctx = tool_exec_ctx.clone();
                        let clarification_hub = clarification_hub.clone();
                        let subagent_allowlist = subagent_allowlist.clone();
                        let shell_policy_for_call = shell_policy.clone();
                        let outbound_for_call = outbound_tx.clone();
                        let channel_for_call = inbound.channel.clone();
                        let tool_call_id = Some(tc.id.clone());
                        futures_vec.push(async move {
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
                            ToolExecutionFinished::Completed(res) => res,
                            ToolExecutionFinished::Waiting(ticket_id) => {
                                // Break the iteration loop; the job is now in 'waiting' state.
                                return Ok(format!(
                                    "{}{}",
                                    WAITING_FOR_USER_RESULT_PREFIX, ticket_id
                                ));
                            }
                            ToolExecutionFinished::Cancelled => {
                                persist_and_cancel!();
                            }
                        };
                        let is_error = tool_result.is_err();
                        let tool_result_text = finalize_tool_output(tool_result);
                        let tool_name = tc.function.name.clone();
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
                        .await?;
                    }
                    tool_invoked = true;
                } else {
                    for tc in tool_calls {
                        if cancel_token.is_cancelled() {
                            persist_and_cancel!();
                        }

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
                        let args = serde_json::from_str::<serde_json::Value>(args_str)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        if tool_name == "exec" {
                            if let Some(cmd) = extract_exec_command(&args) {
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

                        let tool_result = match execute_tool_call_with_activity(
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
                        {
                            ToolExecutionFinished::Completed(res) => res,
                            ToolExecutionFinished::Waiting(ticket_id) => {
                                // Break the iteration loop; the job is now in 'waiting' state.
                                return Ok(format!(
                                    "{}{}",
                                    WAITING_FOR_USER_RESULT_PREFIX, ticket_id
                                ));
                            }
                            ToolExecutionFinished::Cancelled => {
                                persist_and_cancel!();
                            }
                        };

                        let is_error = tool_result.is_err();
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
                        .await?;
                        tool_invoked = true;
                    }
                }
            } else {
                let mut assistant_msg = crate::utils::ChatMessage::assistant(&response_text);
                assistant_msg.reasoning_content = response.reasoning_content.clone();
                mem.add_message(assistant_msg).await?;
            }

            if !tool_invoked {
                // If the model returned empty text after tool calls, re-prompt once so the
                // user sees an actual response instead of an invisible empty cell.
                if response_text.trim().is_empty() && iterations > 1 && iterations < max_iterations
                {
                    let nudge = "[SYSTEM: You used tools but did not produce a text reply for the user. Please summarize your findings or answer the user's question now.]";
                    let correction = crate::utils::ChatMessage::user(nudge);
                    mem.add_message(correction).await?;
                    continue;
                }
                let research_nudge =
                    iterations < max_iterations && should_nudge_research_depth(&inbound, &context);
                if forbid_final_effective && iterations < max_iterations {
                    let nudge = "[SYSTEM: Continue with at least one tool call (or `ask_user` if you are blocked). Plain assistant text alone is not sufficient for this session until the objective is met.]";
                    let correction = crate::utils::ChatMessage::user(nudge);
                    mem.add_message(correction).await?;
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
                    let nudge = "[SYSTEM: Research depth check — you used discovery search but did not fetch primary sources. Before finalizing, use `web_fetch`/`arxiv_fetch` (and/or `hf_hub_file_fetch`) on concrete sources, cross-verify at least two sources, then synthesize findings with explicit uncertainties.]";
                    let correction = crate::utils::ChatMessage::user(nudge);
                    mem.add_message(correction).await?;
                    continue;
                }
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
                let current_context = mem.get_context_since_reflection().await?;
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
                    if matches!(outcome, crate::agent::compaction::CompactionOutcome::Cancelled) {
                        persist_and_cancel!();
                    }
                }

                let _ = outbound_tx.send(BusMessage::Outbound(outbound)).await;
                return Ok(final_response);
            }
        }

        // The loop exits either by hitting max_iterations or by the doom-loop hard stop; surface
        // the matching terminal message (max-iter wording is unchanged for non-doom exits).
        let max_iter_msg = if doom_loop_stuck {
            "Stopped: the agent kept repeating the same action with no progress and did not \
             recover after corrective nudges. Try rephrasing the request or breaking it into \
             smaller steps."
                .to_string()
        } else {
            "Agent reached max reasoning iterations.".to_string()
        };
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
        Ok(max_iter_msg)
    }
}

/// A built-in tool that allows the agent to load the markdown instructions
/// for a skill dynamically from the SkillRegistry.
pub struct LoadSkillTool {
    registry: Arc<SkillRegistry>,
}

#[async_trait]
impl Tool for LoadSkillTool {
    fn name(&self) -> &str {
        "load_skill_instructions"
    }

    fn description(&self) -> &str {
        "Loads the full markdown instructions for a specific Agent Skill. Use this when you need to execute a skill."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["load", "list"],
                    "description": "Use 'list' to enumerate discovered skills. Use 'load' (default) to fetch one skill by name."
                },
                "skill_name": {
                    "type": "string",
                    "description": "Exact skill name when action is load (e.g. 'code_review')."
                },
                "detail": {
                    "type": "string",
                    "enum": ["full", "metadata"],
                    "description": "When action is load: 'full' returns instruction body (default). 'metadata' returns name, availability, description, and body length without the full body."
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("load");

        if action == "list" {
            return Ok(self.registry.format_skill_directory());
        }

        let skill_name = args
            .get("skill_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'skill_name' when action is load (default).".to_string())?;

        let detail = args
            .get("detail")
            .and_then(|v| v.as_str())
            .unwrap_or("full");

        if detail == "metadata" {
            return self.registry.get_skill_metadata(skill_name);
        }

        self.registry.get_skill_instructions(skill_name)
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentLogic, AgentLogicParams, ReasoningLoopCtx};
    use async_trait::async_trait;
    use axum::{
        body::Body,
        extract::State,
        http::{HeaderMap, StatusCode},
        routing::post,
        Router,
    };
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;
    use tower::util::ServiceExt;

    use crate::bus::{clarification_session_key, BusMessage, InboundMessage};
    use crate::clarification::ClarificationHub;
    use crate::logging::create_logger_channel;
    use crate::memory::SqliteMemoryActor;
    use crate::multi_tenant_edge::{ActivityHeartbeatClient, HeartbeatTransport};
    use crate::session::SessionManager;
    use crate::skills::SkillRegistry;
    use crate::tool_activity::SharedToolExecutionActivity;
    use crate::tool_runtime::ToolExecCtx;
    use crate::tools::ToolRegistry;
    use crate::traits::{Memory, Provider, Tool};
    use crate::utils::{ChatMessage, LLMError, LLMResponse};
    use crate::{ActorLogic, NodeHandle};

    struct LocalTempDir {
        path: PathBuf,
    }

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    impl LocalTempDir {
        fn new() -> Self {
            let unique = format!(
                "isanagent-heartbeat-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos(),
                NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).expect("tempdir");
            Self { path }
        }

        fn path(&self) -> &PathBuf {
            &self.path
        }
    }

    impl Drop for LocalTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[derive(Clone)]
    struct RecordedHeartbeat {
        received_at: Instant,
        authorization: Option<String>,
    }

    #[derive(Clone)]
    struct HeartbeatState {
        status: StatusCode,
        records: Arc<Mutex<Vec<RecordedHeartbeat>>>,
    }

    async fn heartbeat_handler(
        State(state): State<HeartbeatState>,
        headers: HeaderMap,
    ) -> StatusCode {
        state.records.lock().unwrap().push(RecordedHeartbeat {
            received_at: Instant::now(),
            authorization: headers
                .get("authorization")
                .and_then(|value| value.to_str().ok())
                .map(|value| value.to_string()),
        });
        state.status
    }

    #[derive(Clone)]
    struct RouterHeartbeatTransport {
        app: Router,
    }

    #[async_trait]
    impl HeartbeatTransport for RouterHeartbeatTransport {
        async fn post_activity(&self, url: &str, token: &str) -> Result<StatusCode, String> {
            let parsed_url = reqwest::Url::parse(url).map_err(|error| error.to_string())?;
            let request = axum::http::Request::builder()
                .method("POST")
                .uri(parsed_url.path())
                .header("authorization", format!("Bearer {}", token))
                .body(Body::empty())
                .map_err(|error| error.to_string())?;
            let response = self
                .app
                .clone()
                .oneshot(request)
                .await
                .map_err(|error| error.to_string())?;
            Ok(response.status())
        }
    }

    #[derive(Clone)]
    struct FailingHeartbeatTransport;

    #[async_trait]
    impl HeartbeatTransport for FailingHeartbeatTransport {
        async fn post_activity(&self, _url: &str, _token: &str) -> Result<StatusCode, String> {
            Err("connection refused".to_string())
        }
    }

    #[derive(Clone)]
    struct SequenceHeartbeatTransport {
        responses: Arc<Mutex<Vec<Result<StatusCode, String>>>>,
        call_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl HeartbeatTransport for SequenceHeartbeatTransport {
        async fn post_activity(&self, _url: &str, _token: &str) -> Result<StatusCode, String> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            let mut responses = self.responses.lock().unwrap();
            if responses.len() > 1 {
                responses.remove(0)
            } else {
                responses
                    .first()
                    .cloned()
                    .unwrap_or(Ok(StatusCode::NO_CONTENT))
            }
        }
    }

    #[derive(Clone)]
    struct SlowHeartbeatTransport {
        delay: Duration,
    }

    #[async_trait]
    impl HeartbeatTransport for SlowHeartbeatTransport {
        async fn post_activity(&self, _url: &str, _token: &str) -> Result<StatusCode, String> {
            tokio::time::sleep(self.delay).await;
            Ok(StatusCode::NO_CONTENT)
        }
    }

    fn build_heartbeat_transport(
        status: StatusCode,
    ) -> (
        Arc<dyn HeartbeatTransport>,
        Arc<Mutex<Vec<RecordedHeartbeat>>>,
    ) {
        let records = Arc::new(Mutex::new(Vec::new()));
        let state = HeartbeatState {
            status,
            records: records.clone(),
        };
        let app = Router::new()
            .route("/_internal/activity", post(heartbeat_handler))
            .with_state(state);

        (
            Arc::new(RouterHeartbeatTransport { app }) as Arc<dyn HeartbeatTransport>,
            records,
        )
    }

    #[derive(Clone)]
    struct DummyProvider;

    #[async_trait]
    impl Provider for DummyProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            unreachable!("DummyProvider is not used in heartbeat tests")
        }
    }

    /// First `chat` waits until the test releases `unblock_rx`; later calls return immediately.
    #[derive(Clone)]
    struct GateFirstChatProvider {
        calls: Arc<AtomicUsize>,
        first_unblock: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
    }

    #[async_trait]
    impl Provider for GateFirstChatProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                let mut slot = self.first_unblock.lock().await;
                if let Some(rx) = slot.take() {
                    let _ = rx.await;
                }
            }
            Ok(LLMResponse {
                content: format!("ok-{n}"),
                tool_calls: None,
                reasoning_content: None,
                usage: None,
            })
        }
    }

    #[derive(Clone)]
    struct LongSleepProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for LongSleepProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Err(LLMError::ApiError(
                "LongSleepProvider should have been cancelled".into(),
            ))
        }
    }

    #[derive(Clone)]
    struct NonTransientErrorProvider;

    #[async_trait]
    impl Provider for NonTransientErrorProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            Err(LLMError::ApiError("Status 400 bad request".into()))
        }
    }

    #[derive(Clone)]
    struct PanicProvider;

    #[async_trait]
    impl Provider for PanicProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            panic!("panic provider exploded")
        }
    }

    /// Always returns the SAME tool call — drives the doom-loop detector to fire repeatedly.
    #[derive(Clone)]
    struct IdenticalToolCallProvider;

    #[async_trait]
    impl Provider for IdenticalToolCallProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            Ok(LLMResponse {
                content: String::new(),
                tool_calls: Some(vec![crate::utils::ToolCallRequest {
                    id: "call_loop".to_string(),
                    tool_type: "function".to_string(),
                    extra_content: None,
                    function: crate::utils::ToolCallFunction {
                        name: "looping_tool".to_string(),
                        arguments: "{\"x\":1}".to_string(),
                    },
                }]),
                reasoning_content: None,
                usage: None,
            })
        }
    }

    /// Loops identically for the first 3 calls (triggers detection + a nudge), then emits
    /// distinct tool calls — simulating a model that corrects itself after the nudge.
    #[derive(Clone)]
    struct CorrectingProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for CorrectingProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            // First 3: identical args (X,X,X → detection). After: distinct args (loop no longer
            // active at the tail), so escalation must reset rather than hard-stop.
            let arguments = if n < 3 {
                "{\"x\":0}".to_string()
            } else {
                format!("{{\"x\":{n}}}")
            };
            Ok(LLMResponse {
                content: String::new(),
                tool_calls: Some(vec![crate::utils::ToolCallRequest {
                    id: format!("call_{n}"),
                    tool_type: "function".to_string(),
                    extra_content: None,
                    function: crate::utils::ToolCallFunction {
                        name: "looping_tool".to_string(),
                        arguments,
                    },
                }]),
                reasoning_content: None,
                usage: None,
            })
        }
    }

    async fn run_loop_once_for_test(
        provider: Box<dyn Provider>,
        max_iterations: usize,
        cancelled_before_start: bool,
        doom_loop_enabled: bool,
    ) -> (Result<String, String>, Vec<ChatMessage>) {
        let memory_actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let memory_node = NodeHandle::new(memory_actor, 16, 1, Duration::from_millis(1));
        let session_manager = Arc::new(SessionManager::new(memory_node));
        let tools = Arc::new(ToolRegistry::new());
        let skills_temp = LocalTempDir::new();
        let skills = Arc::new(SkillRegistry::new(skills_temp.path().clone()));
        let (outbound_tx, mut outbound_rx) = mpsc::channel::<BusMessage>(8);
        // Drain outbound telemetry so the loop's `send().await` never blocks on a full buffer
        // (tool-call-returning providers emit several messages per iteration).
        tokio::spawn(async move { while outbound_rx.recv().await.is_some() {} });
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let cancel_token = tokio_util::sync::CancellationToken::new();
        if cancelled_before_start {
            cancel_token.cancel();
        }
        let inbound = test_inbound("loop-test-chat", "hello");
        let session_key = inbound.clarification_session_key();
        let inbound_metadata = Arc::new(inbound.metadata.clone());
        let result = AgentLogic::run_reasoning_loop(ReasoningLoopCtx {
            name: "LoopTestAgent".to_string(),
            provider,
            session_manager: session_manager.clone(),
            tools,
            skills,
            system_prompt: "test system prompt".to_string(),
            max_iterations,
            max_tool_output_chars: 4_000,
            max_recent_summaries: 0,
            short_term_threshold_turns: 10,
            short_term_threshold_tokens: 10_000,
            tool_execution_activity: None,
            outbound_tx,
            logger_tx,
            inbound,
            cancel_token: cancel_token.clone(),
            clarification_hub: ClarificationHub::shared(),
            tool_exec_ctx: ToolExecCtx::new("terminal", "loop-test-chat", None)
                .with_reasoning_cancel(cancel_token),
            is_subagent: false,
            subagent_allowlist: None,
            doom_loop_enabled,
            harness_runtime_summary: String::new(),
            forbid_final_without_tools: false,
            shell_policy: Arc::new(crate::config::ResolvedShellPolicy {
                interactive_mode: crate::config::ShellPolicyMode::Ask,
                unattended_mode: crate::config::ShellPolicyMode::Deny,
                approval_patterns: Vec::new(),
            }),
            hook_tool_ctx: None,
            inbound_metadata,
        })
        .await;

        let session = session_manager
            .get_session(&session_key)
            .await
            .expect("session");
        let context = session.get_context().await.expect("context");
        (result, context)
    }

    struct SlowTool {
        delay: Duration,
        result: String,
    }

    #[async_trait]
    impl Tool for SlowTool {
        fn name(&self) -> &str {
            "slow_tool"
        }

        fn description(&self) -> &str {
            "Sleeps briefly and returns a static response."
        }

        fn parameters(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }

        async fn execute(&self, _args: Value) -> Result<String, String> {
            tokio::time::sleep(self.delay).await;
            Ok(self.result.clone())
        }
    }

    fn build_agent_with_provider_and_hub(
        provider: Box<dyn Provider>,
        clarification_hub: Arc<ClarificationHub>,
    ) -> (AgentLogic, mpsc::Receiver<BusMessage>) {
        let memory_actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let memory_node = NodeHandle::new(memory_actor, 16, 1, Duration::from_millis(1));
        let session_manager = SessionManager::new(memory_node);

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(SlowTool {
            delay: Duration::from_millis(0),
            result: "tool complete".to_string(),
        }));

        let skills_temp = LocalTempDir::new();
        let skills = SkillRegistry::new(skills_temp.path().clone());

        let (outbound_tx, outbound_rx) = mpsc::channel::<BusMessage>(64);
        let (logger_tx, _logger_rx) = create_logger_channel(32);

        let agent = AgentLogic::new(AgentLogicParams {
            name: "TestAgent".to_string(),
            provider,
            session_manager,
            tools,
            skills,
            system_prompt: "test system prompt".to_string(),
            max_iterations: 4,
            max_tool_output_chars: 4_000,
            max_recent_summaries: 0,
            short_term_threshold_turns: 10,
            short_term_threshold_tokens: 10_000,
            outbound_tx,
            logger_tx,
            clarification_hub,
            subagent: None,
            doom_loop_enabled: false,
            harness_runtime_summary: String::new(),
            subagent_system_prompt: "test system prompt".to_string(),
            forbid_final_without_tools: false,
            shell_policy: crate::config::ResolvedShellPolicy {
                interactive_mode: crate::config::ShellPolicyMode::Ask,
                unattended_mode: crate::config::ShellPolicyMode::Deny,
                approval_patterns: Vec::new(),
            },
            hook_tool_ctx: None,
        });

        (agent, outbound_rx)
    }

    fn build_agent_with_provider(
        provider: Box<dyn Provider>,
    ) -> (AgentLogic, mpsc::Receiver<BusMessage>) {
        build_agent_with_provider_and_hub(provider, ClarificationHub::shared())
    }

    fn build_test_agent(
        tool_execution_activity: Option<SharedToolExecutionActivity>,
        tool_delay: Duration,
    ) -> AgentLogic {
        let memory_actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let memory_node = NodeHandle::new(memory_actor, 16, 1, Duration::from_millis(1));
        let session_manager = SessionManager::new(memory_node);

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(SlowTool {
            delay: tool_delay,
            result: "tool complete".to_string(),
        }));

        let skills_temp = LocalTempDir::new();
        let skills = SkillRegistry::new(skills_temp.path().clone());

        let (outbound_tx, _outbound_rx) = mpsc::channel::<BusMessage>(8);
        let (logger_tx, _logger_rx) = create_logger_channel(32);

        let agent = AgentLogic::new(AgentLogicParams {
            name: "TestAgent".to_string(),
            provider: Box::new(DummyProvider),
            session_manager,
            tools,
            skills,
            system_prompt: "test system prompt".to_string(),
            max_iterations: 4,
            max_tool_output_chars: 4_000,
            max_recent_summaries: 0,
            short_term_threshold_turns: 10,
            short_term_threshold_tokens: 10_000,
            outbound_tx,
            logger_tx,
            clarification_hub: ClarificationHub::shared(),
            subagent: None,
            doom_loop_enabled: false,
            harness_runtime_summary: String::new(),
            subagent_system_prompt: "test system prompt".to_string(),
            forbid_final_without_tools: false,
            shell_policy: crate::config::ResolvedShellPolicy {
                interactive_mode: crate::config::ShellPolicyMode::Ask,
                unattended_mode: crate::config::ShellPolicyMode::Deny,
                approval_patterns: Vec::new(),
            },
            hook_tool_ctx: None,
        });

        if let Some(tool_execution_activity) = tool_execution_activity {
            agent.with_tool_execution_activity(tool_execution_activity)
        } else {
            agent
        }
    }

    #[tokio::test]
    async fn execute_tool_call_sends_immediate_and_repeated_heartbeats() {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let (transport, records) = build_heartbeat_transport(StatusCode::NO_CONTENT);
        let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
            "http://edge.test/_internal/activity".to_string(),
            "edge-token".to_string(),
            Duration::from_millis(30),
            logger_tx,
            transport,
        ));
        let agent = build_test_agent(Some(heartbeat), Duration::from_millis(140));
        let started_at = Instant::now();

        let result = agent
            .execute_tool_call("chat-123", "slow_tool", serde_json::json!({}))
            .await
            .expect("tool result");

        assert_eq!(result, "tool complete");

        let records = records.lock().unwrap().clone();
        assert!(
            records.len() >= 3,
            "expected repeated heartbeats, got {}",
            records.len()
        );
        assert_eq!(
            records[0].authorization.as_deref(),
            Some("Bearer edge-token")
        );
        assert!(
            records[0].received_at.duration_since(started_at) < Duration::from_millis(100),
            "expected immediate heartbeat"
        );
    }

    #[tokio::test]
    async fn execute_tool_call_completes_when_heartbeat_endpoint_returns_statuses() {
        for status in [
            StatusCode::UNAUTHORIZED,
            StatusCode::NOT_FOUND,
            StatusCode::NOT_IMPLEMENTED,
        ] {
            let (logger_tx, _logger_rx) = create_logger_channel(32);
            let (transport, _records) = build_heartbeat_transport(status);
            let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
                "http://edge.test/_internal/activity".to_string(),
                "edge-token".to_string(),
                Duration::from_millis(30),
                logger_tx,
                transport,
            ));
            let agent = build_test_agent(Some(heartbeat), Duration::from_millis(40));

            let result = agent
                .execute_tool_call("chat-123", "slow_tool", serde_json::json!({}))
                .await
                .expect("tool result should succeed even when heartbeat fails");

            assert_eq!(result, "tool complete");
        }
    }

    #[tokio::test]
    async fn execute_tool_call_completes_when_heartbeat_endpoint_is_unavailable() {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
            "http://edge.test/_internal/activity".to_string(),
            "edge-token".to_string(),
            Duration::from_millis(30),
            logger_tx,
            Arc::new(FailingHeartbeatTransport),
        ));
        let agent = build_test_agent(Some(heartbeat), Duration::from_millis(40));

        let result = agent
            .execute_tool_call("chat-123", "slow_tool", serde_json::json!({}))
            .await
            .expect("tool result should succeed without heartbeat server");

        assert_eq!(result, "tool complete");
    }

    #[tokio::test]
    async fn execute_tool_call_retries_after_transient_heartbeat_failures() {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
            "http://edge.test/_internal/activity".to_string(),
            "edge-token".to_string(),
            Duration::from_millis(20),
            logger_tx,
            Arc::new(SequenceHeartbeatTransport {
                responses: Arc::new(Mutex::new(vec![
                    Ok(StatusCode::SERVICE_UNAVAILABLE),
                    Ok(StatusCode::NO_CONTENT),
                    Ok(StatusCode::NO_CONTENT),
                ])),
                call_count: call_count.clone(),
            }),
        ));
        let agent = build_test_agent(Some(heartbeat), Duration::from_millis(75));

        let result = agent
            .execute_tool_call("chat-123", "slow_tool", serde_json::json!({}))
            .await
            .expect("tool result should succeed after transient heartbeat failures");

        assert_eq!(result, "tool complete");
        assert!(
            call_count.load(Ordering::Relaxed) >= 2,
            "expected heartbeat retries after transient failure"
        );
    }

    #[tokio::test]
    async fn execute_tool_call_does_not_wait_for_hung_heartbeat_requests_on_stop() {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
            "http://edge.test/_internal/activity".to_string(),
            "edge-token".to_string(),
            Duration::from_millis(20),
            logger_tx,
            Arc::new(SlowHeartbeatTransport {
                delay: Duration::from_millis(250),
            }),
        ));
        let agent = build_test_agent(Some(heartbeat), Duration::from_millis(10));

        let result = tokio::time::timeout(
            Duration::from_millis(100),
            agent.execute_tool_call("chat-123", "slow_tool", serde_json::json!({})),
        )
        .await
        .expect("tool should not wait for a hung heartbeat request")
        .expect("tool result");

        assert_eq!(result, "tool complete");
    }

    fn test_inbound(chat_id: &str, content: &str) -> InboundMessage {
        InboundMessage {
            channel: "terminal".to_string(),
            sender_id: "local_user".to_string(),
            chat_id: chat_id.to_string(),
            thread_id: None,
            content: content.to_string(),
            attachments: vec![],
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn inbound_queues_while_reasoning_active_second_chat_after_first() {
        let (unblock_tx, unblock_rx) = tokio::sync::oneshot::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let prov = GateFirstChatProvider {
            calls: calls.clone(),
            first_unblock: Arc::new(tokio::sync::Mutex::new(Some(unblock_rx))),
        };
        let (mut agent, _outbound_rx) = build_agent_with_provider(Box::new(prov));
        let cid = "queue-seq-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "first")))
            .await
            .expect("process");
        for _ in 0..200 {
            if calls.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "first reasoning should call provider.chat"
        );
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "second")))
            .await
            .expect("process");
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "second inbound must be queued, not start a concurrent provider.chat"
        );
        let _ = unblock_tx.send(());
        for _ in 0..400 {
            if calls.load(Ordering::SeqCst) >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "after first turn completes, queued inbound should run"
        );
    }

    #[tokio::test]
    async fn cancel_clears_pending_inbound_second_provider_chat_never_starts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let prov = LongSleepProvider {
            calls: calls.clone(),
        };
        let (mut agent, _outbound_rx) = build_agent_with_provider(Box::new(prov));
        let cid = "cancel-q-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "first")))
            .await
            .expect("process");
        for _ in 0..200 {
            if calls.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "second")))
            .await
            .expect("process");
        agent
            .process(BusMessage::Cancel(cid.to_string()))
            .await
            .expect("cancel");
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "queued follow-up must not run provider.chat after Cancel cleared the queue"
        );
    }

    #[tokio::test]
    async fn clarification_inbound_routes_via_hub_before_reasoning_spawn() {
        let hub = Arc::new(ClarificationHub::new());
        let chat_id = "clar-chat";
        let sk = clarification_session_key("terminal", chat_id, None);
        let pending_rx = hub.begin_wait(&sk).expect("begin_wait");
        let (mut agent, _outbound_rx) =
            build_agent_with_provider_and_hub(Box::new(DummyProvider), hub);
        agent
            .process(BusMessage::Inbound(test_inbound(
                chat_id,
                "clarification reply text",
            )))
            .await
            .expect("process");
        let delivered = tokio::time::timeout(Duration::from_secs(2), pending_rx)
            .await
            .expect("timeout waiting clarification")
            .expect("clarification channel closed");
        assert_eq!(delivered, "clarification reply text");
    }

    #[tokio::test]
    async fn run_reasoning_loop_persists_terminal_message_on_llm_failure() {
        let (result, context) =
            run_loop_once_for_test(Box::new(NonTransientErrorProvider), 2, false, false).await;
        assert!(result.is_err(), "expected llm failure");
        let last = context.last().expect("last message");
        assert_eq!(last.role, "assistant");
        let text = last
            .content
            .as_ref()
            .map(|c| c.text_content())
            .unwrap_or_default();
        assert!(
            text.contains("LLM call failed after 3 attempts"),
            "persisted terminal failure not found: {text}"
        );
    }

    #[tokio::test]
    async fn run_reasoning_loop_persists_terminal_message_on_max_iterations() {
        let (result, context) =
            run_loop_once_for_test(Box::new(DummyProvider), 0, false, false).await;
        assert_eq!(
            result.expect("max iterations fallback"),
            "Agent reached max reasoning iterations."
        );
        let last = context.last().expect("last message");
        assert_eq!(last.role, "assistant");
        let text = last
            .content
            .as_ref()
            .map(|c| c.text_content())
            .unwrap_or_default();
        assert_eq!(text, "Agent reached max reasoning iterations.");
    }

    // P1.4: a model that ignores the doom-loop nudges is hard-stopped with a "stuck" message
    // well before max_iterations, instead of spinning to the iteration cap.
    #[tokio::test]
    async fn doom_loop_escalates_to_hard_stop() {
        // max_iterations is high; the doom escalation should terminate the run much earlier.
        let (result, _context) =
            run_loop_once_for_test(Box::new(IdenticalToolCallProvider), 50, false, true).await;
        let msg = result.expect("terminal message");
        assert!(
            msg.starts_with("Stopped:") && msg.contains("repeating"),
            "expected doom-loop stuck message, got: {msg}"
        );
        // Must NOT have run to the iteration cap.
        assert_ne!(msg, "Agent reached max reasoning iterations.");
    }

    // P1.4: when doom detection is disabled, the same repeating provider runs to the cap
    // (escalation must be gated on doom_loop_enabled).
    #[tokio::test]
    async fn doom_loop_disabled_runs_to_max_iterations() {
        let (result, _context) =
            run_loop_once_for_test(Box::new(IdenticalToolCallProvider), 3, false, false).await;
        assert_eq!(
            result.expect("terminal message"),
            "Agent reached max reasoning iterations."
        );
    }

    // P1.4 (review regression): a model that loops then CORRECTS after the nudge must NOT be
    // hard-stopped — escalation counts only loops still active at the tail, so the stale run
    // lingering in the lookback window can't force a stop. Reaches the cap instead.
    #[tokio::test]
    async fn doom_loop_does_not_stop_after_model_corrects() {
        let provider = Box::new(CorrectingProvider {
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let (result, _context) = run_loop_once_for_test(provider, 8, false, true).await;
        assert_eq!(
            result.expect("terminal message"),
            "Agent reached max reasoning iterations.",
            "a model that corrected after a nudge must not be hard-stopped"
        );
    }

    // P1.4: the shared token estimator counts tool_call argument bytes, not just content text
    // (the compaction sites previously under-counted tool-heavy turns).
    #[test]
    fn estimate_tokens_counts_tool_call_args() {
        let mut msg = ChatMessage::assistant("");
        msg.content = None;
        msg.tool_calls = Some(vec![crate::utils::ToolCallRequest {
            id: "1".to_string(),
            tool_type: "function".to_string(),
            extra_content: None,
            function: crate::utils::ToolCallFunction {
                name: "t".to_string(),
                arguments: "x".repeat(400), // 400 bytes / 4 = 100 tokens
            },
        }]);
        assert_eq!(super::estimate_message_tokens(&msg), 100);
        assert_eq!(super::estimate_context_tokens(std::slice::from_ref(&msg)), 100);
    }

    #[test]
    fn effective_context_tokens_prefers_ground_truth() {
        // No usage yet -> fall back to the estimate.
        assert_eq!(super::effective_context_tokens(1000, None), 1000);
        // Provider's exact count exceeds the bytes/4 under-estimate -> use the ground truth so a
        // real overflow still triggers compaction.
        assert_eq!(super::effective_context_tokens(1000, Some(9000)), 9000);
        // Estimate larger (e.g. messages added since the last call) -> keep the estimate.
        assert_eq!(super::effective_context_tokens(9000, Some(1000)), 9000);
        // A zero ground-truth never lowers the estimate.
        assert_eq!(super::effective_context_tokens(1000, Some(0)), 1000);
    }

    #[tokio::test]
    async fn run_reasoning_loop_persists_terminal_message_on_cancel() {
        let (result, context) =
            run_loop_once_for_test(Box::new(DummyProvider), 2, true, false).await;
        assert_eq!(result.expect("cancelled run"), "");
        let last = context.last().expect("last message");
        assert_eq!(last.role, "assistant");
        let text = last
            .content
            .as_ref()
            .map(|c| c.text_content())
            .unwrap_or_default();
        assert!(
            text.contains("Request cancelled while the agent was processing this turn."),
            "persisted cancel marker missing: {text}"
        );
    }

    #[tokio::test]
    async fn panic_in_provider_is_caught_and_surfaces_channel_notice() {
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(PanicProvider));
        let cid = "panic-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "first")))
            .await
            .expect("process");
        let notice = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match outbound_rx.recv().await {
                    Some(BusMessage::Outbound(msg)) if msg.chat_id == cid => break msg,
                    Some(_) => continue,
                    None => panic!("outbound channel closed"),
                }
            }
        })
        .await
        .expect("timeout waiting panic notice");
        assert!(
            notice
                .content
                .contains("Internal error: reasoning loop panicked and was stopped."),
            "unexpected panic notice: {}",
            notice.content
        );
    }

    #[tokio::test]
    async fn load_skill_tool_supports_list_and_metadata() {
        let root = LocalTempDir::new();
        let skills_root = root.path().join("skills");
        let skill_dir = skills_root.join("lint_skill");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: lint_skill\ndescription: Lint helper\n---\n\ndo things\n",
        )
        .unwrap();
        let reg = Arc::new(SkillRegistry::new(skills_root));
        let tool = super::LoadSkillTool {
            registry: reg.clone(),
        };
        let listed = tool
            .execute(serde_json::json!({ "action": "list" }))
            .await
            .unwrap();
        assert!(listed.contains("lint_skill"), "{}", listed);

        let meta = tool
            .execute(serde_json::json!({
                "skill_name": "lint_skill",
                "detail": "metadata"
            }))
            .await
            .unwrap();
        assert!(meta.contains("Instruction length:"));
        assert!(meta.contains("Available: true"));

        let full = tool
            .execute(serde_json::json!({ "skill_name": "lint_skill", "detail": "full" }))
            .await
            .unwrap();
        assert!(full.contains("do things"));
    }
}
