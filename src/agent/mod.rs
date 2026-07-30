use async_trait::async_trait;
use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::mpsc;

mod budget;
pub mod compaction;
mod doom_loop;
pub mod registry;
mod subagent;
pub use registry::AgentRegistry;
pub use subagent::SubagentHarness;

use crate::clarification::ClarificationHub;
use crate::tool_runtime::{with_tool_exec_and_progress_scope, ToolExecCtx, ToolProgressEmitter};

use self::budget::{
    tool_intent_signature, typed_failure_key, BudgetController, BudgetDecision, BudgetLimits,
    ProgressKind,
};

use crate::bus::{
    BusMessage, InboundMessage, LogEvent, OutboundMessage, RunBudgetSnapshot, RunFailureKind,
    RunLifecycleEvent, RunOutcome, RunStuckReason, TelemetryEvent, METADATA_RUN_ID,
};
use crate::config::{ResolvedShellPolicy, ShellPolicyMode};
use crate::hooks::{
    run_post_tool_hooks, run_pre_tool_hooks, run_user_prompt_hooks, HookObservationMeta,
    HookSessionInfo, PreToolOutcome, ToolCallHookContext, UserPromptHookOutcome,
};
use crate::logging::LoggerHandle;
use crate::memory::{MemoryMessage, SharedReply, TodoRow};
use crate::session::SessionManager;
use crate::skills::{SharedSkillRegistry, SkillRegistry};
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tools::ToolRegistry;
use crate::traits::{Memory, Provider, Tool, ToolErrorCode, ToolResult};
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

fn ensure_run_id(inbound: &mut InboundMessage) -> Result<String, String> {
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

fn text_looks_like_research_request(content: &str) -> bool {
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
    Completed(ToolResult),
    Cancelled,
    Waiting(String), // The ticket ID
}

impl ToolExecutionFinished {
    fn error(code: ToolErrorCode, message: impl Into<String>) -> Self {
        Self::Completed(ToolResult::error(code, message))
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct InvalidToolArguments {
    error: InvalidToolArgumentsDetail,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct InvalidToolArgumentsDetail {
    code: &'static str,
    diagnostic: String,
}

impl InvalidToolArguments {
    fn from_json_error(error: serde_json::Error) -> Self {
        Self {
            error: InvalidToolArgumentsDetail {
                code: "invalid_tool_arguments",
                diagnostic: format!(
                    "Malformed JSON at line {} column {} ({:?})",
                    error.line(),
                    error.column(),
                    error.classify()
                ),
            },
        }
    }

    fn to_tool_result(&self) -> ToolResult {
        let content = serde_json::to_string(self).unwrap_or_else(|_| {
            r#"{"error":{"code":"invalid_tool_arguments","diagnostic":"Malformed JSON"}}"#
                .to_string()
        });
        ToolResult::error_with_content(
            ToolErrorCode::InvalidToolArguments,
            self.error.diagnostic.clone(),
            content,
        )
    }
}

fn parse_tool_arguments(raw: &str) -> Result<Value, InvalidToolArguments> {
    serde_json::from_str(raw).map_err(InvalidToolArguments::from_json_error)
}

fn extract_exec_command(args: &Value) -> Option<String> {
    args.get("command")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Append post_tool verification-hook output (build/test/lint results) to a tool result, preserving
/// Ok/Err polarity so the model sees it alongside the tool's own output and can self-correct.
fn append_post_tool_output(mut result: ToolResult, hook_out: &str) -> ToolResult {
    let note = format!("\n\n[post-tool hook]\n{hook_out}");
    // `result` is owned, so append onto the existing buffer in place rather than allocating a
    // fresh string and copying the (potentially large) tool output into it.
    result.content.push_str(&note);
    result
}

/// Lowercase and collapse every run of whitespace (spaces, tabs, newlines) to a single space so a
/// destructive command can't slip past a single-spaced approval pattern via `rm  -rf`, a tab, or a
/// mid-command line break.
fn normalize_command_for_matching(s: &str) -> String {
    let lowercase = s.to_ascii_lowercase();
    let mut result = String::with_capacity(lowercase.len());
    let mut words = lowercase.split_whitespace();
    if let Some(first) = words.next() {
        result.push_str(first);
        for word in words {
            result.push(' ');
            result.push_str(word);
        }
    }
    result
}

fn should_require_shell_approval(command: &str, patterns: &[String]) -> bool {
    // Pad with spaces so matching is on whole-word boundaries: a bare `.contains()` on the
    // normalized command would let a pattern like `rm` match any command containing `terminal`,
    // `platform`, `firmware`, `alarm`, `warm`, `harm`, etc. ("terminal".contains("rm") is true),
    // forcing spurious approval prompts. Padding both sides keeps the whitespace-robustness (a
    // run of spaces/tabs/newlines was already collapsed to one) while only matching real tokens.
    let normalized = format!(" {} ", normalize_command_for_matching(command));
    patterns.iter().any(|p| {
        let np = normalize_command_for_matching(p);
        // Ignore empty/whitespace-only patterns: `contains("")` is always true and would otherwise
        // force approval on *every* command — a silent config footgun.
        if np.is_empty() {
            return false;
        }
        normalized.contains(&format!(" {} ", np))
    })
}

/// Parse a user's reply to a shell-approval prompt. **Deny by default**: the command runs only when
/// the reply is composed *entirely* of affirmative or neutral-filler words AND carries at least one
/// explicit affirmative. Any unrecognized token — a negation ("never", "nope", "can't"), a caveat,
/// or stray prose — forces a deny.
///
/// This allowlist posture (rather than a denylist) is deliberate: it fixes the original
/// `contains("approve") && !contains("deny")` parse that read "do not approve" as APPROVED, and it
/// also closes the broader class a denylist misses, where an affirmative word is buried in a
/// negative sentence ("never approve", "i can't approve", "approve? actually nope"). The prompt
/// constrains the choices to approve/deny with `allow_empty = false`, so the strictness is
/// UX-compatible; an unrecognized reply simply skips execution and the user can re-confirm.
fn shell_approval_reply_is_grant(reply: &str) -> bool {
    const AFFIRM: &[&str] = &[
        "approve",
        "approved",
        "approves",
        "yes",
        "yep",
        "yeah",
        "y",
        "ok",
        "okay",
        "k",
        "allow",
        "allowed",
        "confirm",
        "confirmed",
        "accept",
        "accepted",
        "proceed",
        "go",
        "sure",
    ];
    const FILLER: &[&str] = &["please", "it", "this", "that", "ahead", "now", "run", "do"];

    let r = reply.trim().to_ascii_lowercase();
    if r.is_empty() {
        return false;
    }
    let mut saw_affirmative = false;
    for tok in r
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        if AFFIRM.contains(&tok) {
            saw_affirmative = true;
        } else if !FILLER.contains(&tok) {
            // Negation, caveat, or unmodeled prose -> deny.
            return false;
        }
    }
    saw_affirmative
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

fn edit_policy_mode_for_session(
    policy: &ResolvedShellPolicy,
    unattended_session: bool,
) -> ShellPolicyMode {
    if unattended_session {
        policy.unattended_edit_mode
    } else {
        policy.interactive_edit_mode
    }
}

/// Reason shown when the edit policy blocks a mutation. Unattended sessions
/// default to Deny independently of plan mode, so the message distinguishes the
/// two cases rather than hardcoding "plan mode active" everywhere (PR #62 review #2).
fn edit_policy_block_reason(unattended_session: bool) -> &'static str {
    if unattended_session {
        "File edit blocked by policy: unattended edit mode is active."
    } else {
        "File edit blocked by policy: plan mode active — finalize or apply the plan first."
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

/// Tools that mutate a workspace file and therefore need the edit policy gate.
fn is_file_mutate_tool(tool_name: &str) -> bool {
    matches!(tool_name, "write_file" | "edit_file")
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
    let key = if tool_name == "exec" {
        "command"
    } else {
        "code"
    };
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
    fn file_mutate_category_covers_write_and_edit_only() {
        assert!(is_file_mutate_tool("write_file"));
        assert!(is_file_mutate_tool("edit_file"));
        assert!(!is_file_mutate_tool("read_file"));
        assert!(!is_file_mutate_tool("exec"));
        assert!(!is_file_mutate_tool("list_dir"));
        assert!(!is_file_mutate_tool("search_text"));
    }

    #[test]
    fn edit_block_reason_distinguishes_unattended_and_plan_mode() {
        // PR #62 review #2: the Deny message must match why the edit was blocked.
        let unattended = edit_policy_block_reason(true);
        assert!(unattended.contains("unattended"), "{unattended}");
        assert!(!unattended.contains("plan mode"), "{unattended}");
        let plan = edit_policy_block_reason(false);
        assert!(plan.contains("plan mode"), "{plan}");
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
        assert!(code_exec_requires_approval(
            "exec",
            "rm -rf /tmp/x",
            &patterns
        ));
    }

    #[test]
    fn approval_match_is_whitespace_insensitive() {
        let patterns = vec!["rm -rf".to_string()];
        // Extra spaces, tabs, and a mid-command newline must not bypass the gate.
        assert!(should_require_shell_approval("rm  -rf /tmp/x", &patterns));
        assert!(should_require_shell_approval("rm\t-rf /tmp/x", &patterns));
        assert!(should_require_shell_approval("rm\n-rf /tmp/x", &patterns));
        assert!(should_require_shell_approval("RM -RF /tmp/x", &patterns));
        // A benign command still does not match.
        assert!(!should_require_shell_approval("ls -la", &patterns));
    }

    #[test]
    fn approval_match_is_word_boundary_not_substring() {
        // A bare `rm` pattern must match the real command but NOT words that merely contain "rm"
        // (the substring false positive: "terminal".contains("rm") is true).
        let patterns = vec!["rm".to_string()];
        assert!(should_require_shell_approval("rm -rf /tmp/x", &patterns));
        assert!(should_require_shell_approval(
            "echo hi && rm file",
            &patterns
        ));
        assert!(!should_require_shell_approval("terminal --help", &patterns));
        assert!(!should_require_shell_approval(
            "warm up the cache",
            &patterns
        ));
        assert!(!should_require_shell_approval(
            "npm run platform",
            &patterns
        ));
        assert!(!should_require_shell_approval("check firmware", &patterns));
    }

    #[test]
    fn empty_pattern_does_not_force_approval_on_everything() {
        // `contains("")` is always true; an empty/whitespace pattern must be ignored.
        let patterns = vec!["".to_string(), "   ".to_string()];
        assert!(!should_require_shell_approval("ls -la", &patterns));
        // An empty pattern alongside a real one must not suppress the real match.
        let mixed = vec!["".to_string(), "rm -rf".to_string()];
        assert!(should_require_shell_approval("rm -rf /tmp/x", &mixed));
        assert!(!should_require_shell_approval("ls", &mixed));
    }

    #[test]
    fn approval_reply_is_deny_default() {
        // The regression: "do not approve" must NOT grant.
        assert!(!shell_approval_reply_is_grant("do not approve"));
        assert!(!shell_approval_reply_is_grant("don't approve"));
        assert!(!shell_approval_reply_is_grant("deny"));
        assert!(!shell_approval_reply_is_grant("no"));
        assert!(!shell_approval_reply_is_grant("reject this"));
        assert!(!shell_approval_reply_is_grant(""));
        assert!(!shell_approval_reply_is_grant("   "));
        // Ambiguous / unrelated text is denied (safe default).
        assert!(!shell_approval_reply_is_grant("hmm let me think"));
        // Affirmative word buried in a negative reply must NOT grant (the denylist gap).
        assert!(!shell_approval_reply_is_grant("never approve"));
        assert!(!shell_approval_reply_is_grant("i can't approve"));
        assert!(!shell_approval_reply_is_grant("approve? actually nope"));
        assert!(!shell_approval_reply_is_grant("disapprove"));
        assert!(!shell_approval_reply_is_grant("i approve... no wait, deny"));
        // Explicit grants (incl. affirmative + neutral filler).
        assert!(shell_approval_reply_is_grant("approve"));
        assert!(shell_approval_reply_is_grant("Approved"));
        assert!(shell_approval_reply_is_grant("yes"));
        assert!(shell_approval_reply_is_grant("ok"));
        assert!(shell_approval_reply_is_grant("allow"));
        assert!(shell_approval_reply_is_grant("approve please"));
        assert!(shell_approval_reply_is_grant("approve this"));
        assert!(shell_approval_reply_is_grant("go ahead"));
        // "do" is neutral filler: it rescues genuine affirmatives ("yes do it") without weakening
        // deny-default — a negation always carries another non-filler token (see "do not approve"
        // above), and "do" on its own is not affirmative.
        assert!(shell_approval_reply_is_grant("yes do it"));
        assert!(shell_approval_reply_is_grant("yes, please do"));
        assert!(!shell_approval_reply_is_grant("do it"));
    }

    #[test]
    fn append_post_tool_output_preserves_polarity() {
        // Appends to a success, preserving its typed status.
        let ok = append_post_tool_output(ToolResult::success("applied"), "tests passed");
        assert_eq!(ok.content, "applied\n\n[post-tool hook]\ntests passed");
        assert!(!ok.is_error());
        // Appends to an error without replacing its typed root cause.
        let err = append_post_tool_output(
            ToolResult::error(ToolErrorCode::ExecutionFailed, "boom"),
            "lint output",
        );
        assert_eq!(err.content, "Error: boom\n\n[post-tool hook]\nlint output");
        assert_eq!(err.error_code(), Some(ToolErrorCode::ExecutionFailed));
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
            tool_call_id: Some(tc.id.clone()),
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
            tool_call_id: Some(tc.id.clone()),
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
        let mut approved_mutation_preview = None;
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
                            return ToolExecutionFinished::error(
                                ToolErrorCode::PolicyDenied,
                                format!(
                                    "Command blocked by shell policy (mode=deny): {}",
                                    command
                                ),
                            );
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
                            // System-initiated approval prompt: bypass the sub-agent tool
                            // allowlist so a restricted sub-agent (e.g. allowlist={exec})
                            // can still surface the approval dialog.
                            let ask_result = tools
                                .execute_tool_scoped(
                                    "ask_user",
                                    ask_payload,
                                    None,
                                    is_subagent,
                                )
                                .await;
                            match ask_result {
                                Ok(reply) => {
                                    // Deny-default parse: an explicit grant runs; anything else
                                    // (incl. "do not approve") skips execution.
                                    let approved = shell_approval_reply_is_grant(&reply);
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
                                        return ToolExecutionFinished::error(
                                            ToolErrorCode::PolicyDenied,
                                            "Command not approved by user; execution skipped.",
                                        );
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
                                    return ToolExecutionFinished::error(
                                        ToolErrorCode::ExecutionFailed,
                                        format!("Shell policy approval failed: {}", e),
                                    );
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
                        return ToolExecutionFinished::error(ToolErrorCode::PolicyDenied, msg);
                    }
                    PreToolOutcome::Proceed(new_args) => {
                        args = new_args;
                    }
                }
            }
        }

        // Run this after steering hooks: a hook may rewrite the arguments, and the
        // user must see the exact mutation that will be executed.
        if is_file_mutate_tool(&tool_name) {
            match edit_policy_mode_for_session(&runtime.shell_policy, runtime.unattended_session) {
                ShellPolicyMode::Allow => {}
                ShellPolicyMode::Deny => {
                    return ToolExecutionFinished::error(
                        ToolErrorCode::PolicyDenied,
                        edit_policy_block_reason(runtime.unattended_session),
                    );
                }
                ShellPolicyMode::Ask => {
                    let preview = match tools
                        .preview_mutation_scoped(&tool_name, &args, allow.as_deref(), is_subagent)
                        .await
                    {
                        Ok(preview) => preview,
                        // Invalid/no-op edits retain their ordinary tool result; there is no
                        // mutation to approve in that case.
                        Err(error) => {
                            return ToolExecutionFinished::error(
                                ToolErrorCode::ExecutionFailed,
                                format!("Could not prepare edit approval: {error}"),
                            );
                        }
                    };
                    if let Some(preview) = preview {
                        let ask_payload = serde_json::json!({
                            "prompt": format!(
                                "Approve edit to `{}`? Review the attached diff, then reply with approve or deny.",
                                preview.path
                            ),
                            "choices": ["approve", "deny"],
                            "timeout_secs": 1800,
                            "allow_empty": false,
                            "metadata": {
                                "edit_diff": {
                                    "file": preview.path,
                                    "diff": preview.diff,
                                    "truncated": preview.diff_truncated,
                                }
                            }
                        });
                        // System-initiated approval prompt: bypass the sub-agent tool
                        // allowlist so a restricted sub-agent (e.g. allowlist={write_file})
                        // can still surface the edit approval dialog.
                        let reply = match tools
                            .execute_tool_scoped(
                                "ask_user",
                                ask_payload,
                                None,
                                is_subagent,
                            )
                            .await
                        {
                            Ok(reply) => reply,
                            Err(error) => {
                                return ToolExecutionFinished::error(
                                    ToolErrorCode::ExecutionFailed,
                                    format!("Edit policy approval failed: {error}"),
                                );
                            }
                        };
                        if !shell_approval_reply_is_grant(&reply) {
                            return ToolExecutionFinished::error(
                                ToolErrorCode::PolicyDenied,
                                "Edit not approved by user; mutation skipped.",
                            );
                        }
                        approved_mutation_preview = Some(preview);
                    }
                }
            }
        }

        let args_for_post = args.clone();
        let completed = match cancel_owned.as_ref() {
            None => Some(
                tools
                    .execute_tool_scoped_with_approved_mutation_result(
                        &tool_name,
                        args,
                        approved_mutation_preview.as_ref(),
                        allow.as_deref(),
                        is_subagent,
                    )
                    .await,
            ),
            Some(token) => {
                tokio::select! {
                    res = tools.execute_tool_scoped_with_approved_mutation_result(
                        &tool_name,
                        args,
                        approved_mutation_preview.as_ref(),
                        allow.as_deref(),
                        is_subagent,
                    ) => Some(res),
                    _ = token.cancelled() => None,
                }
            }
        };

        let mut post_tool_output: Option<String> = None;
        if let Some(ref hc) = runtime.hook_tool_ctx {
            if let Some(st) = &hc.steering {
                let res_for_hook = match &completed {
                    Some(result) if result.is_error() => Err(result
                        .error
                        .as_ref()
                        .map(|error| error.message.clone())
                        .unwrap_or_else(|| result.content.clone())),
                    Some(result) => Ok(result.content.clone()),
                    None => Err("tool call cancelled".to_string()),
                };
                let hook_session = HookSessionInfo {
                    channel: channel.as_str(),
                    chat_id: chat_id.as_str(),
                    thread_id: thread_id_for_hooks.as_deref(),
                    metadata: runtime.inbound_metadata.as_ref(),
                    is_subagent: runtime.is_subagent,
                };
                post_tool_output = run_post_tool_hooks(
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
            Some(result) => {
                if let Some(error) = &result.error {
                    if let Some(ticket_id) = error.message.strip_prefix(WAIT_SIGNAL_PREFIX) {
                        return ToolExecutionFinished::Waiting(ticket_id.to_string());
                    }
                }
                // Append any post_tool verification-hook output so the model sees test/lint/build
                // results (including failures) and can self-correct. Ok/Err polarity is preserved.
                let result = match post_tool_output {
                    Some(hook_out) => append_post_tool_output(result, &hook_out),
                    None => result,
                };
                ToolExecutionFinished::Completed(result)
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
    session_manager: Arc<SessionManager>,
    tools: Arc<ToolRegistry>,
    skills: SharedSkillRegistry,
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
    cancellation_tokens: Arc<dashmap::DashMap<String, ActiveRunHandle>>,
    pending_inbound: Arc<dashmap::DashMap<String, Mutex<VecDeque<QueuedInbound>>>>,
    harness_runtime_summary: String,
    forbid_final_without_tools: bool,
    shell_policy: Arc<ResolvedShellPolicy>,
    hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
}

#[derive(Clone)]
struct QueuedInbound {
    inbound: crate::bus::InboundMessage,
    run_provider: RunProviderContext,
}

#[derive(Clone)]
struct ActiveRunHandle {
    run_id: String,
    token: Arc<tokio_util::sync::CancellationToken>,
    steering: Arc<Mutex<SteeringInbox>>,
}

pub(crate) struct SteeringInbox {
    accepting: bool,
    pending: VecDeque<String>,
}

impl SteeringInbox {
    pub(crate) fn open() -> Self {
        Self {
            accepting: true,
            pending: VecDeque::new(),
        }
    }

    fn push(&mut self, content: String) -> bool {
        if !self.accepting {
            return false;
        }
        self.pending.push_back(content);
        true
    }

    fn drain(&mut self) -> Vec<String> {
        self.pending.drain(..).collect()
    }

    fn close(&mut self) {
        self.accepting = false;
        self.pending.clear();
    }

    fn close_or_drain(&mut self) -> Vec<String> {
        if self.pending.is_empty() {
            self.accepting = false;
            Vec::new()
        } else {
            self.drain()
        }
    }
}

fn steering_guard(inbox: &Mutex<SteeringInbox>) -> std::sync::MutexGuard<'_, SteeringInbox> {
    inbox
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
    fn lifecycle_outcome(&self) -> RunOutcome {
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
    message: String,
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

    fn lifecycle_outcome(&self) -> RunOutcome {
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

fn spawn_main_chat_reasoning_turn(
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
                &format!("Spawning reasoning task for chat_id: {}", task_chat_id),
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
                let notice = crate::channels::terminal::build_channel_error_notice(
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
                            &format!("Dropping queued inbound without valid run ID: {}", error),
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
    pub provider_credentials: crate::provider::ProviderCredentials,
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
    /// Canonical project root available to deterministic local worker agents.
    /// This is intentionally separate from IsanAgent's state directory.
    pub workspace_dir: std::path::PathBuf,
}

/// The central logic for an autonomous Agent running inside an ActorNode.
/// It holds a LLM Provider, a persistent Memory context, and available Tools.
pub struct AgentLogic {
    name: String,
    provider_config: Arc<tokio::sync::RwLock<ActiveProviderConfig>>,
    fallback_candidates: Arc<Vec<FallbackProviderSpec>>,
    session_manager: Arc<SessionManager>,
    tools: Arc<ToolRegistry>,
    skills: SharedSkillRegistry,
    system_prompt: String,
    max_iterations: usize,
    max_tool_output_chars: usize,
    max_recent_summaries: usize,
    short_term_threshold_turns: usize,
    short_term_threshold_tokens: usize,
    tool_execution_activity: Option<SharedToolExecutionActivity>,
    outbound_tx: mpsc::Sender<BusMessage>,
    logger_tx: LoggerHandle,
    cancellation_tokens: Arc<dashmap::DashMap<String, ActiveRunHandle>>,
    /// FIFO per `chat_id` when a new user inbound arrives while main reasoning is active.
    pending_inbound: Arc<dashmap::DashMap<String, Mutex<VecDeque<QueuedInbound>>>>,
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
        Self::new_with_fallback_providers(params, Vec::new())
    }

    /// Construct an agent with instance-owned failover candidates. The active primary is removed
    /// when each run snapshots its immutable provider context.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use isanagent::agent::{AgentLogic, AgentLogicParams, FallbackProviderSpec};
    /// use isanagent::provider::{create_provider, ProviderCredentials};
    ///
    /// async fn configure(
    ///     params: AgentLogicParams,
    ///     fallbacks: Vec<FallbackProviderSpec>,
    /// ) {
    ///     let agent = AgentLogic::new_with_fallback_providers(params, fallbacks);
    ///
    ///     let credentials = ProviderCredentials {
    ///         provider_name: "openai".to_string(),
    ///         base_url: "https://api.openai.com/v1".to_string(),
    ///         api_key: "replacement-key".to_string(),
    ///         model_name: "gpt-4o".to_string(),
    ///     };
    ///     let provider = create_provider(
    ///         &credentials.provider_name,
    ///         &credentials.base_url,
    ///         &credentials.api_key,
    ///         &credentials.model_name,
    ///     );
    ///     agent
    ///         .switch_provider_with_credentials(provider, credentials)
    ///         .await;
    /// }
    /// ```
    pub fn new_with_fallback_providers(
        params: AgentLogicParams,
        fallback_providers: Vec<FallbackProviderSpec>,
    ) -> Self {
        let AgentLogicParams {
            name,
            provider,
            provider_credentials,
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
        let skills = Arc::new(tokio::sync::RwLock::new(skills));
        let tools = Arc::new(tools);
        let memory_node = session_manager.get_memory_node();
        let shell_policy = Arc::new(shell_policy);

        let provider_config = Arc::new(tokio::sync::RwLock::new(ActiveProviderConfig {
            provider,
            credentials: provider_credentials,
        }));
        let fallback_candidates = Arc::new(fallback_providers);

        let subagent_harness = subagent.map(|p| {
            Arc::new(SubagentHarness::new(subagent::SubagentSpawnDeps {
                agent_name: name.clone(),
                provider_config: provider_config.clone(),
                fallback_candidates: fallback_candidates.clone(),
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
                workspace_dir: p.workspace_dir.clone(),
            }))
        });

        let mut agent = Self {
            name,
            provider_config,
            fallback_candidates,
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

    /// Atomically replace the provider object and the credentials that created it. Active and
    /// already-queued runs keep their immutable snapshots; subsequent admissions see this pair.
    pub async fn switch_provider_with_credentials(
        &self,
        provider: Box<dyn Provider>,
        credentials: crate::provider::ProviderCredentials,
    ) {
        *self.provider_config.write().await = ActiveProviderConfig {
            provider,
            credentials,
        };
    }

    pub fn with_tool_execution_activity(
        mut self,
        tool_execution_activity: SharedToolExecutionActivity,
    ) -> Self {
        self.tool_execution_activity = Some(tool_execution_activity);
        self
    }

    fn reasoning_spawn_args(&self) -> ReasoningSpawnArgs {
        ReasoningSpawnArgs {
            name: self.name.clone(),
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

    async fn run_provider_context(&self) -> RunProviderContext {
        let active = self.provider_config.read().await;
        RunProviderContext::snapshot(&active, &self.fallback_candidates)
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
            ToolExecutionFinished::Completed(result) => result.into_legacy_result(),
            ToolExecutionFinished::Waiting(ticket_id) => Err(format!(
                "tool call waiting for clarification ticket: {}",
                ticket_id
            )),
            ToolExecutionFinished::Cancelled => {
                Err("tool call cancelled without cancellation token".to_string())
            }
        }
    }

    fn cancel_active_run(&self, chat_id: &str, expected_run_id: Option<&str>) -> bool {
        let active = self
            .cancellation_tokens
            .get(chat_id)
            .map(|entry| entry.value().clone());
        let Some(active) = active else {
            if expected_run_id.is_none() {
                self.pending_inbound.remove(chat_id);
            }
            return false;
        };
        if expected_run_id.is_some_and(|run_id| run_id != active.run_id) {
            let _ = self.logger_tx.send(BusMessage::Log(
                LogEvent::warn(
                    &self.name,
                    &format!(
                        "Ignored cancellation for chat_id {} because run_id did not match the active run.",
                        chat_id
                    ),
                )
                .with_chat_id(chat_id),
            ));
            return false;
        }
        if let Some(harness) = &self.subagent_harness {
            if harness.cancel_children_on_parent_cancel() {
                harness.cancel_children_for_parent(chat_id);
            }
        }
        steering_guard(&active.steering).close();
        active.token.cancel();
        let _ = self.logger_tx.send(BusMessage::Log(
            LogEvent::info(
                &self.name,
                &format!("Cancelled reasoning loop for chat_id: {}", chat_id),
            )
            .with_chat_id(chat_id),
        ));
        self.pending_inbound.remove(chat_id);
        true
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
                // Keep ownership registered until the reasoning task emits its
                // terminal lifecycle event and finalizes. New inbound arriving
                // during cancellation must queue behind that acknowledgement.
                self.cancel_active_run(&chat_id, None);
                return Ok(None);
            }
            BusMessage::CancelRun { chat_id, run_id } => {
                self.cancel_active_run(&chat_id, Some(&run_id));
                return Ok(None);
            }
            BusMessage::Steer {
                chat_id,
                run_id,
                content,
            } => {
                if content.trim().is_empty() {
                    return Ok(None);
                }
                if let Some(active) = self.cancellation_tokens.get(&chat_id) {
                    if active.run_id == run_id {
                        steering_guard(&active.steering).push(content);
                    }
                }
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
                self.switch_provider_with_credentials(
                    new_provider,
                    crate::provider::ProviderCredentials {
                        provider_name: provider_name.clone(),
                        base_url: base_url.clone(),
                        api_key: api_key.clone(),
                        model_name: model_name.clone(),
                    },
                )
                .await;
                let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info(
                    &self.name,
                    &format!(
                        "Switched to provider={} model={}",
                        provider_name, model_name
                    ),
                )));
                return Ok(None);
            }
            BusMessage::InstallSkill {
                repo_url,
                skill_name,
            } => {
                let skills_arc = self.skills.clone();
                let logger_tx = self.logger_tx.clone();
                let name = self.name.clone();

                tokio::spawn(async move {
                    let mut skills_guard = skills_arc.write().await;
                    match skills_guard
                        .install_skills_from_repo(&repo_url, skill_name.as_deref())
                        .await
                    {
                        Ok(installed) => {
                            let msg = if installed.is_empty() {
                                "No skills found in the repository.".to_string()
                            } else {
                                format!("Successfully installed skills: {}", installed.join(", "))
                            };
                            let _ = logger_tx.send(BusMessage::Log(LogEvent::info(&name, &msg)));
                        }
                        Err(e) => {
                            let _ = logger_tx.send(BusMessage::Log(LogEvent::error(
                                &name,
                                &format!("Failed to install skills from {}: {}", repo_url, e),
                            )));
                        }
                    }
                });
                return Ok(None);
            }
            BusMessage::Inbound(mut inbound) => {
                let run_id = match ensure_run_id(&mut inbound) {
                    Ok(run_id) => run_id,
                    Err(error) => {
                        let _ = self.logger_tx.send(BusMessage::Log(
                            LogEvent::error(
                                &self.name,
                                &format!("Rejecting inbound message: {}", error),
                            )
                            .with_chat_id(&inbound.chat_id),
                        ));
                        let notice = crate::channels::terminal::build_channel_error_notice(
                            &inbound.channel,
                            &inbound.chat_id,
                            inbound.thread_id.as_deref(),
                            &error,
                        );
                        let _ = self.outbound_tx.send(BusMessage::Outbound(notice)).await;
                        return Ok(None);
                    }
                };
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

                let run_provider = self.run_provider_context().await;
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
                    guard.push_back(QueuedInbound {
                        inbound,
                        run_provider,
                    });
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

                spawn_main_chat_reasoning_turn(
                    self.reasoning_spawn_args(),
                    inbound,
                    run_id,
                    run_provider,
                );

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
                    let _ = self.logger_tx.send(BusMessage::Log(LogEvent::warn(
                        &self.name,
                        &format!(
                            "TriggerCompaction dropped for session_key={}: {}",
                            session_key, e
                        ),
                    )));
                }
                Ok(None)
            }
            BusMessage::Outbound(_)
            | BusMessage::Telemetry(_)
            | BusMessage::RunLifecycle(_)
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
        let user_turns = current_context.iter().filter(|m| m.role == "user").count();
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
        let provider_guard = self.provider_config.read().await;
        // Manual triggers have no per-call cancellation; a token that never
        // fires keeps `do_compaction`'s `select!` valid without altering behavior.
        let cancel_token = tokio_util::sync::CancellationToken::new();

        let outcome =
            crate::agent::compaction::do_compaction(crate::agent::compaction::DoCompactionArgs {
                chat_id: &chat_id,
                session_key: &session_key,
                trigger_reason,
                tokens_before: approx_tokens.min(u32::MAX as usize) as u32,
                turns_before: user_turns.min(u32::MAX as usize) as u32,
                current_context: &current_context,
                existing_summary: recent.first().map(|s| s.as_str()),
                focus_instructions: focus_instructions.as_deref(),
                provider: provider_guard.provider.as_ref(),
                memory_node: &memory_node,
                outbound_tx: &self.outbound_tx,
                cancel_token: &cancel_token,
            })
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
                // A background job may have been created through a different
                // channel for this same chat. Native recovery deliberately
                // remains chat-scoped; host embedders opt into channel scope
                // when their UI needs an isolated inbox.
                channel: None,
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
                        channel: None,
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

        let run_id = ensure_run_id(&mut resumed_inbound)?;
        let run_provider = self.run_provider_context().await;
        spawn_main_chat_reasoning_turn(
            self.reasoning_spawn_args(),
            resumed_inbound,
            run_id,
            run_provider,
        );
        Ok(())
    }
}

/// A configured alternate LLM provider to fail over to when the primary's retries are exhausted.
/// Holds everything [`crate::provider::create_provider`] needs; resolved once at startup from the
/// `[providers.*]` config.
#[derive(Clone)]
pub struct FallbackProviderSpec {
    pub provider_name: String,
    pub base_url: String,
    pub api_key: String,
    pub model_name: String,
}

// Manual `Debug` so a stray `{:?}` can never dump the API key into a log.
impl std::fmt::Debug for FallbackProviderSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FallbackProviderSpec")
            .field("provider_name", &self.provider_name)
            .field("base_url", &self.base_url)
            .field("api_key", &"[redacted]")
            .field("model_name", &self.model_name)
            .finish()
    }
}

/// Filter `candidates` to genuine fallbacks: drop any whose (provider, base_url, model) matches the
/// active primary, so the primary is never retried as its own fallback. Matching the full identity
/// (not just provider+model) correctly excludes the primary even when it came from the `[provider]`
/// default block rather than the `[providers.*]` map a candidate was built from.
pub fn build_fallback_specs(
    primary_provider: &str,
    primary_base_url: &str,
    primary_model: &str,
    candidates: Vec<FallbackProviderSpec>,
) -> Vec<FallbackProviderSpec> {
    candidates
        .into_iter()
        .filter(|c| {
            // Normalize before comparing so the primary isn't accidentally retried as its own
            // fallback: trailing slashes on base URLs are insignificant, and provider/model names
            // are matched case-insensitively (e.g. `https://api.openai.com/v1/` vs `.../v1`, or
            // `OpenAI` vs `openai`).
            let norm_c_url = c.base_url.trim_end_matches('/');
            let norm_p_url = primary_base_url.trim_end_matches('/');
            !(c.provider_name.eq_ignore_ascii_case(primary_provider)
                && norm_c_url == norm_p_url
                && c.model_name.eq_ignore_ascii_case(primary_model))
        })
        .collect()
}

/// Provider object and the exact credentials that created it. Keeping them behind one lock makes a
/// model switch atomic: a run can never snapshot the old provider with the new credential identity
/// (or vice versa).
pub(crate) struct ActiveProviderConfig {
    pub(crate) provider: Box<dyn Provider>,
    pub(crate) credentials: crate::provider::ProviderCredentials,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProviderRunIdentity {
    provider_name: String,
    model_name: String,
    secret_identity: String,
}

/// Immutable provider/fallback ownership for one accepted run. This value is also stored with a
/// queued inbound, so a later `/model` switch cannot rewrite already-admitted work.
#[derive(Clone)]
pub(crate) struct RunProviderContext {
    provider: Box<dyn Provider>,
    fallback_providers: Vec<FallbackProviderSpec>,
    identity: ProviderRunIdentity,
}

impl RunProviderContext {
    pub(crate) fn snapshot(
        active: &ActiveProviderConfig,
        candidates: &[FallbackProviderSpec],
    ) -> Self {
        let credentials = &active.credentials;
        let fallback_providers = if credentials.is_usable() {
            build_fallback_specs(
                &credentials.provider_name,
                &credentials.base_url,
                &credentials.model_name,
                candidates.to_vec(),
            )
        } else {
            Vec::new()
        };
        Self {
            provider: dyn_clone::clone_box(&*active.provider),
            fallback_providers,
            identity: ProviderRunIdentity {
                provider_name: credentials.provider_name.clone(),
                model_name: credentials.model_name.clone(),
                secret_identity: provider_secret_identity(&credentials.api_key),
            },
        }
    }
}

fn provider_secret_identity(api_key: &str) -> String {
    if api_key.is_empty() {
        return "none".to_string();
    }
    use sha2::Digest;
    let digest = sha2::Sha256::digest(api_key.as_bytes());
    format!("sha256:{}", &hex::encode(digest)[..16])
}

/// Result of attempting the configured fallback providers.
enum FallbackOutcome {
    Ok(crate::utils::LLMResponse),
    Cancelled,
    Exhausted,
}

/// Borrowed logging identity for the failover loop (bundled to keep the arg count in check).
struct FailoverLogCtx<'a> {
    logger_tx: &'a LoggerHandle,
    name: &'a str,
    chat_id: &'a str,
}

/// Try each fallback provider **once**, returning the first successful response. `build` constructs
/// a provider from a spec — real code passes [`crate::provider::create_provider`]; tests inject a
/// mock builder, keeping this loop fully testable without network. Cancellation preempts a
/// fallback chat.
async fn try_fallbacks<F>(
    fallbacks: &[FallbackProviderSpec],
    build: F,
    context: &[crate::utils::ChatMessage],
    tools_payload: &Option<serde_json::Value>,
    cancel_token: &tokio_util::sync::CancellationToken,
    log: FailoverLogCtx<'_>,
) -> FallbackOutcome
where
    F: Fn(&FallbackProviderSpec) -> Box<dyn crate::traits::Provider>,
{
    for spec in fallbacks {
        let _ = log.logger_tx.send(BusMessage::Log(
            LogEvent::warn(
                log.name,
                &format!(
                    "Primary LLM exhausted; failing over to provider={} model={}",
                    spec.provider_name, spec.model_name
                ),
            )
            .with_chat_id(log.chat_id),
        ));
        let provider = build(spec);
        let res = tokio::select! {
            r = provider.chat(context, tools_payload.clone()) => r,
            _ = cancel_token.cancelled() => return FallbackOutcome::Cancelled,
        };
        match res {
            Ok(resp) => {
                let _ = log.logger_tx.send(BusMessage::Log(
                    LogEvent::info(
                        log.name,
                        &format!(
                            "Fallback succeeded: provider={} model={}",
                            spec.provider_name, spec.model_name
                        ),
                    )
                    .with_chat_id(log.chat_id),
                ));
                return FallbackOutcome::Ok(resp);
            }
            Err(e) => {
                let _ = log.logger_tx.send(BusMessage::Log(
                    LogEvent::warn(
                        log.name,
                        &format!("Fallback provider={} failed: {}", spec.provider_name, e),
                    )
                    .with_chat_id(log.chat_id),
                ));
            }
        }
    }
    FallbackOutcome::Exhausted
}

/// Outcome of a `provider.chat` invocation that may be retried for transient errors.
enum ChatRetryOutcome {
    Ok {
        response: crate::utils::LLMResponse,
        retries: u32,
    },
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
    fallback_providers: &[FallbackProviderSpec],
    cancel_token: &tokio_util::sync::CancellationToken,
    log_ctx: FailoverLogCtx<'_>,
) -> ChatRetryOutcome {
    let FailoverLogCtx {
        logger_tx,
        name,
        chat_id,
    } = log_ctx;
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
            Ok(resp) => {
                return ChatRetryOutcome::Ok {
                    response: resp,
                    retries: attempt,
                }
            }
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
    // Primary exhausted. Before surfacing a failure, try each configured fallback provider once, so
    // a transient outage / key rotation / model deprecation on the primary doesn't drop a long
    // unattended turn. The primary stays the active provider — failover is per-call.
    match try_fallbacks(
        fallback_providers,
        |s| {
            crate::provider::create_provider(
                &s.provider_name,
                &s.base_url,
                &s.api_key,
                &s.model_name,
            )
        },
        context,
        &tools_payload,
        cancel_token,
        FailoverLogCtx {
            logger_tx,
            name,
            chat_id,
        },
    )
    .await
    {
        FallbackOutcome::Ok(resp) => {
            return ChatRetryOutcome::Ok {
                response: resp,
                retries: MAX_ATTEMPTS.saturating_sub(1),
            }
        }
        FallbackOutcome::Cancelled => return ChatRetryOutcome::Cancelled,
        FallbackOutcome::Exhausted => {}
    }

    ChatRetryOutcome::Failed(
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown LLM error".to_string()),
    )
}

/// Build the user-facing banner for an LLM failure.
///
/// Only the terminal channel exposes the `/retry` command and its accompanying
/// metadata flag. Other clients receive a channel-neutral recovery hint and use
/// the typed lifecycle outcome to decide whether to offer retry controls.
fn build_llm_failed_banner(
    channel: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    error: &str,
    retryable: bool,
) -> OutboundMessage {
    let content = if channel == "terminal" && retryable {
        format!(
            "LLM call failed after 3 attempts: {error}\nPress /retry to try again or /cancel to abandon."
        )
    } else if retryable {
        format!(
            "LLM call failed after provider retries were exhausted: {error}\nThis run can be retried from the client."
        )
    } else {
        format!("LLM call failed: {error}")
    };
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    if channel == "terminal" {
        metadata.insert(
            crate::channels::terminal_ui::protocol::ISANAGENT_TERMINAL_ERROR.to_string(),
            serde_json::json!(true),
        );
        if retryable {
            metadata.insert(
                crate::channels::terminal_ui::protocol::ISANAGENT_LLM_RETRY_AVAILABLE.to_string(),
                serde_json::json!(true),
            );
        }
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
                    UserPromptHookOutcome::Block(msg) => {
                        return Err(ReasoningLoopError::protocol(msg));
                    }
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
                                        "Doom loop still active after {} consecutive detections — stopping the run.",
                                        consecutive_doom_detections
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
                    && tool_calls.iter().all(|tc| {
                        crate::tools::ToolRegistry::is_parallel_safe_tool(tc.function.name.as_str())
                    });

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
                            budget.record_tool_failure(typed_failure_key(
                                &tool_name, code, &intent,
                            ))
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
                        let intent =
                            tool_intent_signature(tool_name, &tc.function.arguments);
                        let budget_decision = if is_error {
                            let code = tool_result
                                .error_code()
                                .unwrap_or(ToolErrorCode::ExecutionFailed);
                            budget.record_tool_failure(typed_failure_key(
                                tool_name, code, &intent,
                            ))
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
                    let nudge = "[SYSTEM: Research depth check — you used discovery search but did not fetch primary sources. Before finalizing, use `web_fetch`/`arxiv_fetch` (and/or `hf_hub_file_fetch`) on concrete sources, cross-verify at least two sources, then synthesize findings with explicit uncertainties.]";
                    let correction = crate::utils::ChatMessage::user(nudge);
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

/// A built-in tool that allows the agent to load the markdown instructions
/// for a skill dynamically from the SkillRegistry.
pub struct LoadSkillTool {
    registry: SharedSkillRegistry,
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
            return Ok(self.registry.read().await.format_skill_directory());
        }

        let skill_name = args
            .get("skill_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'skill_name' when action is load (default).".to_string())?;

        let detail = args
            .get("detail")
            .and_then(|v| v.as_str())
            .unwrap_or("full");

        // First attempt against the registry as it was last scanned. The read guard is scoped to
        // this block and dropped before the rescan path below takes the write lock, so we never
        // hold a read guard across a write acquisition on the same RwLock (which would deadlock).
        let first = {
            let registry = self.registry.read().await;
            if detail == "metadata" {
                registry.get_skill_metadata(skill_name)
            } else {
                registry.get_skill_instructions(skill_name)
            }
        };
        if first.is_ok() {
            return first;
        }

        // Miss: a SKILL.md may have been dropped into the skills directory since the registry was
        // last scanned (it is scanned once at startup). Rescan once and re-resolve before reporting
        // the skill missing, so a freshly added skill is loadable without restarting the agent. The
        // rescan is paid only on an actual miss, so the common path (skill already present) is
        // unchanged. Hold a single write guard for both the rescan and the follow-up lookup (the
        // guard derefs to the registry for the immutable getters), avoiding a redundant drop-then-
        // reacquire and closing the gap between scan and read.
        let mut registry = self.registry.write().await;
        registry.scan_for_skills();
        if detail == "metadata" {
            registry.get_skill_metadata(skill_name)
        } else {
            registry.get_skill_instructions(skill_name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_llm_failed_banner, steering_guard, ActiveProviderConfig, AgentLogic,
        AgentLogicParams, QueuedInbound, ReasoningLoopCtx, ReasoningLoopExit, RunProviderContext,
        SteeringInbox,
    };
    use async_trait::async_trait;
    use axum::{
        body::Body,
        extract::State,
        http::{HeaderMap, StatusCode},
        routing::post,
        Router,
    };
    use serde_json::Value;
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;
    use tower::util::ServiceExt;

    use crate::bus::{
        clarification_session_key, BusMessage, InboundMessage, RunBudgetSnapshot, RunFailureKind,
        RunLifecycleEvent, RunOutcome, RunStuckReason, METADATA_RUN_ID,
    };
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

    #[test]
    fn tauri_retryable_failure_does_not_advertise_terminal_commands() {
        let banner = build_llm_failed_banner("tauri", "chat-1", None, "provider unavailable", true);

        assert!(!banner.content.contains("/retry"));
        assert!(banner.content.contains("retried from the client"));
        assert!(!banner
            .metadata
            .contains_key(crate::channels::terminal_ui::protocol::ISANAGENT_LLM_RETRY_AVAILABLE));
        assert!(!banner
            .metadata
            .contains_key(crate::channels::terminal_ui::protocol::ISANAGENT_TERMINAL_ERROR));
    }

    #[test]
    fn terminal_retryable_failure_advertises_retry_command() {
        let banner =
            build_llm_failed_banner("terminal", "chat-1", None, "provider unavailable", true);

        assert!(banner.content.contains("Press /retry"));
        assert_eq!(
            banner
                .metadata
                .get(crate::channels::terminal_ui::protocol::ISANAGENT_LLM_RETRY_AVAILABLE),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            banner
                .metadata
                .get(crate::channels::terminal_ui::protocol::ISANAGENT_TERMINAL_ERROR),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn terminal_non_retryable_failure_does_not_offer_retry() {
        let banner = build_llm_failed_banner("terminal", "chat-1", None, "context overflow", false);

        assert!(!banner.content.contains("/retry"));
        assert!(!banner
            .metadata
            .contains_key(crate::channels::terminal_ui::protocol::ISANAGENT_LLM_RETRY_AVAILABLE));
        assert_eq!(
            banner
                .metadata
                .get(crate::channels::terminal_ui::protocol::ISANAGENT_TERMINAL_ERROR),
            Some(&serde_json::json!(true))
        );
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

    /// Returns `Ok` with `content` set to `tag` — a stand-in for a working fallback provider.
    #[derive(Clone)]
    struct RespondingProvider {
        tag: String,
    }

    #[async_trait]
    impl Provider for RespondingProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            Ok(LLMResponse {
                content: self.tag.clone(),
                tool_calls: None,
                reasoning_content: None,
                usage: None,
            })
        }
    }

    fn fb_spec(name: &str) -> super::FallbackProviderSpec {
        super::FallbackProviderSpec {
            provider_name: name.to_string(),
            base_url: String::new(),
            api_key: String::new(),
            model_name: format!("{name}-model"),
        }
    }

    fn fb_full(provider: &str, base: &str, model: &str) -> super::FallbackProviderSpec {
        super::FallbackProviderSpec {
            provider_name: provider.to_string(),
            base_url: base.to_string(),
            api_key: "k".to_string(),
            model_name: model.to_string(),
        }
    }

    #[test]
    fn build_fallback_specs_excludes_primary_by_full_identity() {
        let candidates = vec![
            // Same identity as the primary but with different casing and a trailing slash on the
            // base URL — must still be excluded after normalization.
            fb_full("Anthropic", "https://api.anthropic.com/", "Claude"),
            fb_full("openai", "https://api.openai.com", "gpt-4o"),
        ];
        let out = super::build_fallback_specs(
            "anthropic",
            "https://api.anthropic.com",
            "claude",
            candidates,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].provider_name, "openai");
    }

    #[test]
    fn build_fallback_specs_keeps_same_provider_different_model_or_url() {
        // Same provider but a different model — a legitimate fallback, must be kept.
        let candidates = vec![
            fb_full("openai", "u", "gpt-4o-mini"),
            fb_full("openai", "u2", "gpt-4o"), // same provider+model, different base_url -> kept
        ];
        let out = super::build_fallback_specs("openai", "u", "gpt-4o", candidates);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn fallback_spec_debug_redacts_api_key() {
        let dbg = format!("{:?}", fb_full("openai", "u", "m"));
        assert!(dbg.contains("[redacted]"), "{dbg}");
        assert!(!dbg.contains("\"k\""), "api key must not appear: {dbg}");
    }

    #[tokio::test]
    async fn concurrent_agents_snapshot_distinct_provider_credentials_and_fallbacks() {
        let (agent_a, _rx_a) = build_agent_with_provider_state(
            Box::new(RespondingProvider { tag: "a".into() }),
            provider_credentials("provider-a", "https://a.test/v1/", "secret-a", "model-a"),
            vec![
                fb_full("provider-a", "https://a.test/v1", "model-a"),
                fb_full("fallback-a", "https://fallback-a.test", "fallback-model-a"),
            ],
            ClarificationHub::shared(),
        );
        let (agent_b, _rx_b) = build_agent_with_provider_state(
            Box::new(RespondingProvider { tag: "b".into() }),
            provider_credentials("provider-b", "https://b.test/v1", "secret-b", "model-b"),
            vec![
                fb_full("provider-b", "https://b.test/v1/", "model-b"),
                fb_full("fallback-b", "https://fallback-b.test", "fallback-model-b"),
            ],
            ClarificationHub::shared(),
        );

        let (context_a, context_b) = tokio::join!(
            agent_a.run_provider_context(),
            agent_b.run_provider_context()
        );

        assert_eq!(context_a.identity.provider_name, "provider-a");
        assert_eq!(context_a.identity.model_name, "model-a");
        assert_eq!(context_a.fallback_providers.len(), 1);
        assert_eq!(context_a.fallback_providers[0].provider_name, "fallback-a");
        assert_eq!(context_b.identity.provider_name, "provider-b");
        assert_eq!(context_b.identity.model_name, "model-b");
        assert_eq!(context_b.fallback_providers.len(), 1);
        assert_eq!(context_b.fallback_providers[0].provider_name, "fallback-b");
        assert_ne!(
            context_a.identity.secret_identity,
            context_b.identity.secret_identity
        );
        assert!(!context_a.identity.secret_identity.contains("secret-a"));
        assert!(!context_b.identity.secret_identity.contains("secret-b"));
    }

    #[tokio::test]
    async fn atomic_provider_switch_replaces_the_complete_active_pair() {
        let (agent, _rx) = build_agent_with_provider_state(
            Box::new(RespondingProvider { tag: "old".into() }),
            provider_credentials("provider-a", "https://a.test", "secret-a", "model-a"),
            vec![fb_full(
                "fallback-b",
                "https://fallback-b.test",
                "fallback-model-b",
            )],
            ClarificationHub::shared(),
        );

        agent
            .switch_provider_with_credentials(
                Box::new(RespondingProvider { tag: "new".into() }),
                provider_credentials("provider-b", "https://b.test", "secret-b", "model-b"),
            )
            .await;

        let context = agent.run_provider_context().await;
        assert_eq!(context.identity.provider_name, "provider-b");
        assert_eq!(context.identity.model_name, "model-b");
        assert_ne!(context.identity.secret_identity, "none");
        assert_eq!(context.fallback_providers.len(), 1);
        assert_eq!(context.fallback_providers[0].provider_name, "fallback-b");
    }

    #[tokio::test]
    async fn try_fallbacks_returns_first_success() {
        let (logger, _rx) = crate::logging::create_logger_channel(64);
        let cancel = tokio_util::sync::CancellationToken::new();
        let specs = vec![fb_spec("a"), fb_spec("b")];
        // 'a' fails, 'b' succeeds -> first success wins, 'b' chosen.
        let out = super::try_fallbacks(
            &specs,
            |s| -> Box<dyn Provider> {
                if s.provider_name == "b" {
                    Box::new(RespondingProvider { tag: "b-ok".into() })
                } else {
                    Box::new(NonTransientErrorProvider)
                }
            },
            &[],
            &None,
            &cancel,
            super::FailoverLogCtx {
                logger_tx: &logger,
                name: "agent",
                chat_id: "c1",
            },
        )
        .await;
        match out {
            super::FallbackOutcome::Ok(r) => assert_eq!(r.content, "b-ok"),
            _ => panic!("expected Ok from fallback b"),
        }
    }

    #[tokio::test]
    async fn try_fallbacks_all_fail_is_exhausted() {
        let (logger, _rx) = crate::logging::create_logger_channel(64);
        let cancel = tokio_util::sync::CancellationToken::new();
        let specs = vec![fb_spec("a"), fb_spec("b")];
        let out = super::try_fallbacks(
            &specs,
            |_| -> Box<dyn Provider> { Box::new(NonTransientErrorProvider) },
            &[],
            &None,
            &cancel,
            super::FailoverLogCtx {
                logger_tx: &logger,
                name: "agent",
                chat_id: "c1",
            },
        )
        .await;
        assert!(matches!(out, super::FallbackOutcome::Exhausted));
    }

    #[tokio::test]
    async fn try_fallbacks_empty_is_exhausted() {
        let (logger, _rx) = crate::logging::create_logger_channel(64);
        let cancel = tokio_util::sync::CancellationToken::new();
        let out = super::try_fallbacks(
            &[],
            |_| -> Box<dyn Provider> { Box::new(RespondingProvider { tag: "x".into() }) },
            &[],
            &None,
            &cancel,
            super::FailoverLogCtx {
                logger_tx: &logger,
                name: "agent",
                chat_id: "c1",
            },
        )
        .await;
        assert!(matches!(out, super::FallbackOutcome::Exhausted));
    }

    #[tokio::test]
    async fn try_fallbacks_cancellation_short_circuits() {
        let (logger, _rx) = crate::logging::create_logger_channel(64);
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel(); // pre-cancelled; the slow provider's chat never wins the select
        let specs = vec![fb_spec("a")];
        let out = super::try_fallbacks(
            &specs,
            |_| -> Box<dyn Provider> {
                Box::new(LongSleepProvider {
                    calls: Arc::new(AtomicUsize::new(0)),
                })
            },
            &[],
            &None,
            &cancel,
            super::FailoverLogCtx {
                logger_tx: &logger,
                name: "agent",
                chat_id: "c1",
            },
        )
        .await;
        assert!(matches!(out, super::FallbackOutcome::Cancelled));
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

    #[derive(Clone)]
    struct MalformedToolArgumentsProvider {
        calls: Arc<AtomicUsize>,
        tool_names: Vec<String>,
    }

    #[async_trait]
    impl Provider for MalformedToolArgumentsProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call > 0 {
                return Ok(LLMResponse {
                    content: "recovered after invalid arguments".to_string(),
                    tool_calls: None,
                    reasoning_content: None,
                    usage: None,
                });
            }
            let tool_calls = self
                .tool_names
                .iter()
                .enumerate()
                .map(|(index, name)| crate::utils::ToolCallRequest {
                    id: format!("malformed-{index}"),
                    tool_type: "function".to_string(),
                    extra_content: None,
                    function: crate::utils::ToolCallFunction {
                        name: name.clone(),
                        arguments: "{\"unterminated\":".to_string(),
                    },
                })
                .collect();
            Ok(LLMResponse {
                content: String::new(),
                tool_calls: Some(tool_calls),
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
    ) -> (
        Result<ReasoningLoopExit, super::ReasoningLoopError>,
        Vec<ChatMessage>,
    ) {
        run_loop_once_for_test_with_autonomy(
            provider,
            max_iterations,
            cancelled_before_start,
            doom_loop_enabled,
            false,
        )
        .await
    }

    async fn run_loop_once_for_test_with_autonomy(
        provider: Box<dyn Provider>,
        max_iterations: usize,
        cancelled_before_start: bool,
        doom_loop_enabled: bool,
        forbid_final_without_tools: bool,
    ) -> (
        Result<ReasoningLoopExit, super::ReasoningLoopError>,
        Vec<ChatMessage>,
    ) {
        let memory_actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let memory_node = NodeHandle::new(memory_actor, 16, 1, Duration::from_millis(1));
        let session_manager = Arc::new(SessionManager::new(memory_node));
        let tools = Arc::new(ToolRegistry::new());
        let skills_temp = LocalTempDir::new();
        let skills = Arc::new(tokio::sync::RwLock::new(SkillRegistry::new(
            skills_temp.path().clone(),
        )));
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
        let run_provider = RunProviderContext::snapshot(
            &ActiveProviderConfig {
                provider,
                credentials: crate::provider::ProviderCredentials::empty(),
            },
            &[],
        );
        let result = AgentLogic::run_reasoning_loop(ReasoningLoopCtx {
            name: "LoopTestAgent".to_string(),
            run_provider,
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
            run_id: "test-run-id".to_string(),
            steering: Arc::new(Mutex::new(SteeringInbox::open())),
            cancel_token: cancel_token.clone(),
            clarification_hub: ClarificationHub::shared(),
            tool_exec_ctx: ToolExecCtx::new("terminal", "loop-test-chat", None)
                .with_reasoning_cancel(cancel_token),
            is_subagent: false,
            subagent_allowlist: None,
            doom_loop_enabled,
            harness_runtime_summary: String::new(),
            forbid_final_without_tools,
            shell_policy: Arc::new(crate::config::ResolvedShellPolicy {
                interactive_mode: crate::config::ShellPolicyMode::Ask,
                unattended_mode: crate::config::ShellPolicyMode::Deny,
                interactive_edit_mode: crate::config::ShellPolicyMode::Ask,
                unattended_edit_mode: crate::config::ShellPolicyMode::Deny,
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
        build_agent_with_provider_state(
            provider,
            crate::provider::ProviderCredentials::empty(),
            Vec::new(),
            clarification_hub,
        )
    }

    fn build_agent_with_provider_state(
        provider: Box<dyn Provider>,
        provider_credentials: crate::provider::ProviderCredentials,
        fallback_providers: Vec<super::FallbackProviderSpec>,
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

        let agent = AgentLogic::new_with_fallback_providers(
            AgentLogicParams {
                name: "TestAgent".to_string(),
                provider,
                provider_credentials,
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
                    interactive_edit_mode: crate::config::ShellPolicyMode::Ask,
                    unattended_edit_mode: crate::config::ShellPolicyMode::Deny,
                    approval_patterns: Vec::new(),
                },
                hook_tool_ctx: None,
            },
            fallback_providers,
        );

        (agent, outbound_rx)
    }

    fn build_agent_with_provider(
        provider: Box<dyn Provider>,
    ) -> (AgentLogic, mpsc::Receiver<BusMessage>) {
        build_agent_with_provider_and_hub(provider, ClarificationHub::shared())
    }

    fn provider_credentials(
        provider_name: &str,
        base_url: &str,
        api_key: &str,
        model_name: &str,
    ) -> crate::provider::ProviderCredentials {
        crate::provider::ProviderCredentials {
            provider_name: provider_name.to_string(),
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            model_name: model_name.to_string(),
        }
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
            provider_credentials: crate::provider::ProviderCredentials::empty(),
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
                interactive_edit_mode: crate::config::ShellPolicyMode::Ask,
                unattended_edit_mode: crate::config::ShellPolicyMode::Deny,
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

    #[test]
    fn run_id_is_required_for_tauri_and_backfilled_for_legacy_channels() {
        let mut tauri = test_inbound("run-id-tauri", "hello");
        tauri.channel = "tauri".to_string();
        assert!(super::ensure_run_id(&mut tauri).is_err());

        let mut legacy = test_inbound("run-id-terminal", "hello");
        let generated = super::ensure_run_id(&mut legacy).expect("legacy run id");
        assert!(generated.starts_with("legacy-"));
        assert_eq!(
            legacy
                .metadata
                .get(METADATA_RUN_ID)
                .and_then(|value| value.as_str()),
            Some(generated.as_str())
        );
    }

    #[tokio::test]
    async fn invalid_tauri_inbound_is_rejected_without_stopping_the_actor() {
        let provider = RespondingProvider {
            tag: "done".to_string(),
        };
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(provider));
        let mut invalid = test_inbound("invalid-run-id", "hello");
        invalid.channel = "tauri".to_string();

        assert!(matches!(
            agent.process(BusMessage::Inbound(invalid)).await,
            Ok(None)
        ));
        assert!(matches!(
            outbound_rx.recv().await,
            Some(BusMessage::Outbound(_))
        ));

        let mut valid = test_inbound("valid-after-rejection", "hello");
        valid.channel = "tauri".to_string();
        valid.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("valid-run-id"),
        );
        agent
            .process(BusMessage::Inbound(valid))
            .await
            .expect("actor remains usable after rejecting malformed inbound");
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(2), outbound_rx.recv()).await,
            Ok(Some(BusMessage::RunLifecycle(
                RunLifecycleEvent::Started { .. }
            )))
        ));
    }

    #[tokio::test]
    async fn invalid_queued_inbound_does_not_strand_following_valid_message() {
        let (unblock_tx, unblock_rx) = tokio::sync::oneshot::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = GateFirstChatProvider {
            calls: calls.clone(),
            first_unblock: Arc::new(tokio::sync::Mutex::new(Some(unblock_rx))),
        };
        let (mut agent, _outbound_rx) = build_agent_with_provider(Box::new(provider));
        let chat_id = "skip-invalid-queued";
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start first turn");
        for _ in 0..200 {
            if calls.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let mut invalid = test_inbound(chat_id, "invalid queued");
        invalid.channel = "tauri".to_string();
        let mut valid = test_inbound(chat_id, "valid queued");
        valid.channel = "tauri".to_string();
        valid.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("queued-run-id"),
        );
        let run_provider = agent.run_provider_context().await;
        agent.pending_inbound.insert(
            chat_id.to_string(),
            Mutex::new(VecDeque::from([
                QueuedInbound {
                    inbound: invalid,
                    run_provider: run_provider.clone(),
                },
                QueuedInbound {
                    inbound: valid,
                    run_provider,
                },
            ])),
        );

        unblock_tx.send(()).expect("unblock first turn");
        for _ in 0..400 {
            if calls.load(Ordering::SeqCst) == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "the valid queued message must run after the invalid item is dropped"
        );
    }

    #[tokio::test]
    async fn model_switch_preserves_active_run_and_updates_later_queued_admission() {
        let (unblock_tx, unblock_rx) = tokio::sync::oneshot::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider_a = GateFirstChatProvider {
            calls: calls.clone(),
            first_unblock: Arc::new(tokio::sync::Mutex::new(Some(unblock_rx))),
        };
        let (mut agent, mut outbound_rx) = build_agent_with_provider_state(
            Box::new(provider_a),
            provider_credentials("provider-a", "https://a.test", "secret-a", "model-a"),
            Vec::new(),
            ClarificationHub::shared(),
        );
        let chat_id = "run-provider-admission";

        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start provider-a turn");
        for _ in 0..200 {
            if calls.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        agent
            .switch_provider_with_credentials(
                Box::new(RespondingProvider {
                    tag: "provider-b".to_string(),
                }),
                provider_credentials("provider-b", "https://b.test", "secret-b", "model-b"),
            )
            .await;
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "second")))
            .await
            .expect("queue provider-b turn");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            agent
                .pending_inbound
                .get(chat_id)
                .expect("queued second turn")
                .lock()
                .expect("pending queue lock")
                .len(),
            1
        );

        unblock_tx.send(()).expect("release provider-a turn");
        let (responses, terminal_count) = tokio::time::timeout(Duration::from_secs(5), async {
            let mut responses = Vec::new();
            let mut terminal_count = 0;
            while terminal_count < 2 {
                match outbound_rx.recv().await.expect("outbound channel open") {
                    BusMessage::Outbound(outbound) if outbound.chat_id == chat_id => {
                        responses.push(outbound.content);
                    }
                    BusMessage::RunLifecycle(RunLifecycleEvent::Terminated {
                        chat_id: event_chat,
                        ..
                    }) if event_chat == chat_id => terminal_count += 1,
                    _ => {}
                }
            }
            (responses, terminal_count)
        })
        .await
        .expect("both run snapshots complete");

        assert_eq!(terminal_count, 2);
        assert_eq!(responses, vec!["ok-0", "provider-b"]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn started_lifecycle_event_preserves_caller_run_id() {
        let provider = RespondingProvider {
            tag: "done".to_string(),
        };
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(provider));
        let mut inbound = test_inbound("run-id-chat", "hello");
        inbound.channel = "tauri".to_string();
        inbound.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("caller-run-123"),
        );

        agent
            .process(BusMessage::Inbound(inbound))
            .await
            .expect("process inbound");

        let event = tokio::time::timeout(Duration::from_secs(2), outbound_rx.recv())
            .await
            .expect("started lifecycle event before timeout")
            .expect("outbound event");
        assert!(matches!(
            event,
            BusMessage::RunLifecycle(RunLifecycleEvent::Started { run_id, chat_id })
                if run_id == "caller-run-123" && chat_id == "run-id-chat"
        ));

        let terminal = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(BusMessage::RunLifecycle(
                    event @ RunLifecycleEvent::Terminated { .. },
                )) = outbound_rx.recv().await
                {
                    return event;
                }
            }
        })
        .await
        .expect("terminal lifecycle event before timeout");
        assert!(matches!(
            terminal,
            RunLifecycleEvent::Terminated { run_id, chat_id, outcome: RunOutcome::Completed }
                if run_id == "caller-run-123" && chat_id == "run-id-chat"
        ));
    }

    #[tokio::test]
    async fn provider_retry_exhaustion_emits_one_typed_lifecycle_pair() {
        let (mut agent, mut outbound_rx) =
            build_agent_with_provider(Box::new(NonTransientErrorProvider));
        let mut inbound = test_inbound("provider-terminal-chat", "hello");
        inbound.channel = "tauri".to_string();
        inbound.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("provider-terminal-run"),
        );
        agent
            .process(BusMessage::Inbound(inbound))
            .await
            .expect("process inbound");

        let mut lifecycle_events = Vec::new();
        while lifecycle_events.len() < 2 {
            let event = tokio::time::timeout(Duration::from_secs(2), outbound_rx.recv())
                .await
                .expect("lifecycle event before timeout")
                .expect("outbound channel remains open");
            if let BusMessage::RunLifecycle(event) = event {
                lifecycle_events.push(event);
            }
        }

        assert!(matches!(
            lifecycle_events.as_slice(),
            [
                RunLifecycleEvent::Started { run_id, chat_id },
                RunLifecycleEvent::Terminated {
                    run_id: terminal_run_id,
                    chat_id: terminal_chat_id,
                    outcome: RunOutcome::Failed {
                        failure: RunFailureKind::ProviderRetriesExhausted,
                        retryable: true,
                    },
                },
            ] if run_id == "provider-terminal-run"
                && chat_id == "provider-terminal-chat"
                && terminal_run_id == run_id
                && terminal_chat_id == chat_id
        ));

        let extra_lifecycle = tokio::time::timeout(Duration::from_millis(100), async {
            loop {
                match outbound_rx.recv().await {
                    Some(BusMessage::RunLifecycle(event)) => return Some(event),
                    Some(_) => continue,
                    None => return None,
                }
            }
        })
        .await
        .ok()
        .flatten();
        assert!(
            extra_lifecycle.is_none(),
            "only one lifecycle pair is emitted"
        );
    }

    #[tokio::test]
    async fn repeated_root_cause_emits_warning_then_one_stuck_terminal() {
        let (mut agent, mut outbound_rx) =
            build_agent_with_provider(Box::new(IdenticalToolCallProvider));
        let mut inbound = test_inbound("budget-warning-chat", "hello");
        inbound.channel = "tauri".to_string();
        inbound.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("budget-warning-run"),
        );
        agent
            .process(BusMessage::Inbound(inbound))
            .await
            .expect("process inbound");

        let mut lifecycle_events = Vec::new();
        while lifecycle_events.len() < 3 {
            let event = tokio::time::timeout(Duration::from_secs(2), outbound_rx.recv())
                .await
                .expect("lifecycle event before timeout")
                .expect("outbound channel remains open");
            if let BusMessage::RunLifecycle(event) = event {
                lifecycle_events.push(event);
            }
        }

        assert!(matches!(
            lifecycle_events.as_slice(),
            [
                RunLifecycleEvent::Started { run_id, chat_id },
                RunLifecycleEvent::Warning {
                    run_id: warning_run_id,
                    chat_id: warning_chat_id,
                    warning: crate::bus::RunBudgetWarning {
                        reason: crate::bus::RunBudgetWarningReason::RepeatedRootCause {
                            failures: 2
                        },
                        ..
                    },
                },
                RunLifecycleEvent::Terminated {
                    run_id: terminal_run_id,
                    chat_id: terminal_chat_id,
                    outcome: RunOutcome::Stuck {
                        reason: RunStuckReason::RepeatedRootCause,
                    },
                },
            ] if run_id == "budget-warning-run"
                && chat_id == "budget-warning-chat"
                && warning_run_id == run_id
                && warning_chat_id == chat_id
                && terminal_run_id == run_id
                && terminal_chat_id == chat_id
        ));

        let extra_terminal = tokio::time::timeout(Duration::from_millis(100), async {
            loop {
                match outbound_rx.recv().await {
                    Some(BusMessage::RunLifecycle(
                        event @ RunLifecycleEvent::Terminated { .. },
                    )) => return Some(event),
                    Some(_) => continue,
                    None => return None,
                }
            }
        })
        .await
        .ok()
        .flatten();
        assert!(
            extra_terminal.is_none(),
            "only one terminal event is emitted"
        );
    }

    #[test]
    fn typed_terminal_exits_preserve_budget_and_doom_loop_outcomes() {
        let budget = ReasoningLoopExit::BudgetExhausted {
            assistant_text: "any localized assistant text".to_string(),
            budget: RunBudgetSnapshot {
                iterations_used: 7,
                iterations_limit: 7,
                ..RunBudgetSnapshot::default()
            },
        }
        .lifecycle_outcome();
        assert!(matches!(
            budget,
            RunOutcome::BudgetExhausted { budget }
                if budget.iterations_used == 7 && budget.iterations_limit == 7
        ));

        let stuck = ReasoningLoopExit::Stuck {
            assistant_text: "unrelated assistant text".to_string(),
            reason: RunStuckReason::DoomLoop,
        }
        .lifecycle_outcome();
        assert_eq!(
            stuck,
            RunOutcome::Stuck {
                reason: RunStuckReason::DoomLoop,
            }
        );
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
    async fn inbound_after_cancel_waits_for_old_terminal_before_new_start() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (mut agent, mut outbound_rx) =
            build_agent_with_provider(Box::new(LongSleepProvider { calls }));
        let chat_id = "cancel-serialization-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start first run");

        let first_run_id = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Started {
                    run_id,
                    chat_id: event_chat,
                })) if event_chat == chat_id => break run_id,
                Some(_) => continue,
                None => panic!("outbound channel closed before first start"),
            }
        };

        agent
            .process(BusMessage::Cancel(chat_id.to_string()))
            .await
            .expect("cancel accepted");
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "second")))
            .await
            .expect("queue second run while cancellation unwinds");

        let first_after_cancel = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(event)) => break event,
                Some(_) => continue,
                None => panic!("outbound channel closed during cancellation"),
            }
        };
        assert!(matches!(
            first_after_cancel,
            RunLifecycleEvent::Terminated {
                run_id,
                chat_id: event_chat,
                outcome: RunOutcome::Cancelled,
            } if run_id == first_run_id && event_chat == chat_id
        ));

        let second_start = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(event @ RunLifecycleEvent::Started { .. })) => {
                    break event;
                }
                Some(_) => continue,
                None => panic!("outbound channel closed before second start"),
            }
        };
        assert!(matches!(
            second_start,
            RunLifecycleEvent::Started { run_id, chat_id: event_chat }
                if run_id != first_run_id && event_chat == chat_id
        ));
    }

    #[tokio::test]
    async fn exact_cancel_does_not_interrupt_a_different_run() {
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(LongSleepProvider {
            calls: Arc::new(AtomicUsize::new(0)),
        }));
        let chat_id = "exact-cancel-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start run");
        let run_id = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Started {
                    run_id,
                    chat_id: event_chat,
                })) if event_chat == chat_id => break run_id,
                Some(_) => continue,
                None => panic!("outbound channel closed before start"),
            }
        };

        agent
            .process(BusMessage::CancelRun {
                chat_id: chat_id.to_string(),
                run_id: "wrong-run".to_string(),
            })
            .await
            .expect("wrong cancel is handled");
        assert!(
            tokio::time::timeout(Duration::from_millis(50), async {
                loop {
                    match outbound_rx.recv().await {
                        Some(BusMessage::RunLifecycle(RunLifecycleEvent::Terminated {
                            ..
                        })) => {
                            return;
                        }
                        Some(_) => continue,
                        None => return,
                    }
                }
            })
            .await
            .is_err(),
            "a mismatched cancel must not terminate the active run"
        );

        agent
            .process(BusMessage::CancelRun {
                chat_id: chat_id.to_string(),
                run_id: run_id.clone(),
            })
            .await
            .expect("exact cancel is accepted");
        let terminal = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(event @ RunLifecycleEvent::Terminated { .. })) => {
                    break event;
                }
                Some(_) => continue,
                None => panic!("outbound channel closed before terminal"),
            }
        };
        assert!(matches!(
            terminal,
            RunLifecycleEvent::Terminated {
                run_id: terminal_run_id,
                chat_id: event_chat,
                outcome: RunOutcome::Cancelled,
            } if terminal_run_id == run_id && event_chat == chat_id
        ));
    }

    #[tokio::test]
    async fn steer_is_accepted_only_for_the_exact_active_run() {
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(LongSleepProvider {
            calls: Arc::new(AtomicUsize::new(0)),
        }));
        let chat_id = "steer-run-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start run");
        let run_id = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Started { run_id, .. })) => {
                    break run_id
                }
                Some(_) => continue,
                None => panic!("outbound channel closed before start"),
            }
        };

        agent
            .process(BusMessage::Steer {
                chat_id: chat_id.to_string(),
                run_id: "stale-run".to_string(),
                content: "ignore this".to_string(),
            })
            .await
            .expect("stale steer is handled");
        {
            let active = agent.cancellation_tokens.get(chat_id).expect("active run");
            assert!(steering_guard(&active.steering).pending.is_empty());
        }

        agent
            .process(BusMessage::Steer {
                chat_id: chat_id.to_string(),
                run_id: run_id.clone(),
                content: "change direction".to_string(),
            })
            .await
            .expect("exact steer is handled");
        {
            let active = agent.cancellation_tokens.get(chat_id).expect("active run");
            let inbox = steering_guard(&active.steering);
            assert_eq!(
                inbox.pending.front().map(String::as_str),
                Some("change direction")
            );
        }

        agent
            .process(BusMessage::CancelRun {
                chat_id: chat_id.to_string(),
                run_id,
            })
            .await
            .expect("cancel test run");
    }

    #[test]
    fn steering_final_boundary_is_atomic_and_never_leaks_to_a_later_run() {
        let mut inbox = SteeringInbox::open();
        assert!(inbox.push("first revision".to_string()));
        assert_eq!(inbox.close_or_drain(), vec!["first revision"]);
        assert!(inbox.accepting, "draining a revision keeps this run open");

        assert!(inbox.close_or_drain().is_empty());
        assert!(!inbox.accepting, "empty final boundary closes acceptance");
        assert!(!inbox.push("too late".to_string()));
        assert!(inbox.pending.is_empty());
    }

    #[tokio::test]
    async fn steering_at_provider_boundary_is_persisted_before_the_next_response() {
        let (unblock_tx, unblock_rx) = tokio::sync::oneshot::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = GateFirstChatProvider {
            calls: calls.clone(),
            first_unblock: Arc::new(tokio::sync::Mutex::new(Some(unblock_rx))),
        };
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(provider));
        let chat_id = "steer-provider-boundary";
        agent
            .process(BusMessage::Inbound(test_inbound(
                chat_id,
                "original request",
            )))
            .await
            .expect("start run");
        let run_id = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Started { run_id, .. })) => {
                    break run_id
                }
                Some(_) => continue,
                None => panic!("outbound channel closed before start"),
            }
        };
        agent
            .process(BusMessage::Steer {
                chat_id: chat_id.to_string(),
                run_id,
                content: "use the revised direction".to_string(),
            })
            .await
            .expect("queue steering");
        unblock_tx
            .send(())
            .expect("release first provider response");
        loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Terminated { .. })) => break,
                Some(_) => continue,
                None => panic!("outbound channel closed before terminal"),
            }
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let session_key = clarification_session_key("terminal", chat_id, None);
        let session = agent
            .session_manager
            .get_session(&session_key)
            .await
            .expect("session");
        let context = session.get_context().await.expect("context");
        let text: Vec<_> = context
            .iter()
            .map(|m| {
                m.content
                    .as_ref()
                    .map(|content| content.text_content())
                    .unwrap_or_default()
            })
            .collect();
        assert!(text
            .iter()
            .any(|value| value == "use the revised direction"));
        assert!(text.iter().any(|value| value == "ok-1"));
        assert!(!text.iter().any(|value| value == "ok-0"));
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
        assert!(matches!(
            result,
            Ok(ReasoningLoopExit::Failed {
                failure: RunFailureKind::ProviderRetriesExhausted,
                retryable: true,
                ..
            })
        ));
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
        assert!(matches!(
            result.expect("max iterations fallback"),
            ReasoningLoopExit::BudgetExhausted { .. }
        ));
        let last = context.last().expect("last message");
        assert_eq!(last.role, "assistant");
        let text = last
            .content
            .as_ref()
            .map(|c| c.text_content())
            .unwrap_or_default();
        assert!(text.contains("exhausted its LLM-turn budget"), "{text}");
    }

    #[tokio::test]
    async fn unresolved_no_progress_warning_cannot_complete_from_prose_alone() {
        let provider = Box::new(RespondingProvider {
            tag: "premature completion".to_string(),
        });
        let (result, context) =
            run_loop_once_for_test_with_autonomy(provider, 10, false, false, true).await;
        assert!(matches!(
            result.expect("typed stuck terminal"),
            ReasoningLoopExit::Stuck {
                reason: RunStuckReason::NoProgress,
                ..
            }
        ));
        let terminal_text = context
            .last()
            .and_then(|message| message.content.as_ref())
            .map(|content| content.text_content())
            .unwrap_or_default();
        assert!(terminal_text.contains("without observable progress"));
    }

    fn invalid_argument_codes(context: &[ChatMessage]) -> Vec<String> {
        context
            .iter()
            .filter(|message| message.role == "tool")
            .filter_map(|message| message.content.as_ref())
            .filter_map(|content| serde_json::from_str::<Value>(&content.text_content()).ok())
            .filter_map(|payload| {
                payload
                    .pointer("/error/code")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    #[tokio::test]
    async fn sequential_malformed_tool_arguments_return_typed_error_without_dispatch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = MalformedToolArgumentsProvider {
            calls: calls.clone(),
            tool_names: vec!["slow_tool".to_string()],
        };
        let (result, context) = run_loop_once_for_test(Box::new(provider), 2, false, false).await;

        assert!(matches!(result, Ok(ReasoningLoopExit::Completed { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(invalid_argument_codes(&context), ["invalid_tool_arguments"]);
    }

    #[tokio::test]
    async fn parallel_malformed_tool_arguments_return_typed_errors_without_dispatch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = MalformedToolArgumentsProvider {
            calls: calls.clone(),
            // Both names are classified parallel-safe, forcing the join_all path.
            tool_names: vec!["read_file".to_string(), "list_dir".to_string()],
        };
        let (result, context) = run_loop_once_for_test(Box::new(provider), 2, false, false).await;

        assert!(matches!(result, Ok(ReasoningLoopExit::Completed { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            invalid_argument_codes(&context),
            ["invalid_tool_arguments", "invalid_tool_arguments"]
        );
    }

    // Typed root-cause detection is earlier and more specific than the legacy context-pattern
    // detector for repeated failed calls, so it must stop this trace before the emergency ceiling.
    #[tokio::test]
    async fn repeated_typed_root_cause_preempts_legacy_doom_loop() {
        // max_iterations is high; the doom escalation should terminate the run much earlier.
        let (result, _context) =
            run_loop_once_for_test(Box::new(IdenticalToolCallProvider), 50, false, true).await;
        let exit = result.expect("terminal message");
        assert!(matches!(
            &exit,
            ReasoningLoopExit::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
                ..
            }
        ));
        let msg = exit.assistant_text().expect("stuck assistant text");
        assert!(
            msg.starts_with("Stopped:") && msg.contains("typed tool failure"),
            "expected typed-root-cause stuck message, got: {msg}"
        );
        // Must NOT have run to the iteration cap.
        assert_ne!(msg, "Agent reached max reasoning iterations.");
    }

    // The typed progress controller is independent of the optional legacy doom detector.
    #[tokio::test]
    async fn repeated_typed_root_cause_does_not_depend_on_doom_detection() {
        let (result, _context) =
            run_loop_once_for_test(Box::new(IdenticalToolCallProvider), 3, false, false).await;
        assert!(matches!(
            result.expect("terminal message"),
            ReasoningLoopExit::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
                ..
            }
        ));
    }

    // Merely varying arguments is not measurable progress when every call still reaches the same
    // typed root cause. This is the historical max-iteration failure shape T17 must stop.
    #[tokio::test]
    async fn varied_arguments_do_not_mask_the_same_typed_root_cause() {
        let provider = Box::new(CorrectingProvider {
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let (result, _context) = run_loop_once_for_test(provider, 8, false, true).await;
        assert!(matches!(
            result.expect("terminal message"),
            ReasoningLoopExit::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
                ..
            }
        ));
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
        assert_eq!(
            super::estimate_context_tokens(std::slice::from_ref(&msg)),
            100
        );
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
        assert!(matches!(
            result.expect("cancelled run"),
            ReasoningLoopExit::Cancelled { .. }
        ));
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
        let reg = Arc::new(tokio::sync::RwLock::new(SkillRegistry::new(skills_root)));
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

    #[tokio::test]
    async fn load_skill_rescans_on_miss_to_pick_up_a_new_skill() {
        let root = LocalTempDir::new();
        let skills_root = root.path().join("skills");

        // One skill present when the registry is first scanned (at startup).
        let first = skills_root.join("first_skill");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::write(
            first.join("SKILL.md"),
            "---\nname: first_skill\ndescription: present at startup\n---\n\nalpha body\n",
        )
        .unwrap();

        let reg = Arc::new(tokio::sync::RwLock::new(SkillRegistry::new(
            skills_root.clone(),
        )));
        let tool = super::LoadSkillTool {
            registry: reg.clone(),
        };

        // A skill dropped into the directory AFTER the startup scan: the in-memory registry has
        // never seen it.
        let late = skills_root.join("late_skill");
        std::fs::create_dir_all(&late).unwrap();
        std::fs::write(
            late.join("SKILL.md"),
            "---\nname: late_skill\ndescription: added after scan\n---\n\nomega instructions\n",
        )
        .unwrap();

        // Without rescan-on-miss this would error "skill not found"; the on-miss rescan must pick
        // the new skill up and return its instructions without an agent restart.
        let loaded = tool
            .execute(serde_json::json!({ "skill_name": "late_skill", "detail": "full" }))
            .await
            .expect("late skill should be loadable after the on-miss rescan");
        assert!(loaded.contains("omega instructions"), "{loaded}");

        // metadata path also benefits from the rescan (covers the detail == "metadata" branch).
        let late_meta = tool
            .execute(serde_json::json!({ "skill_name": "late_skill", "detail": "metadata" }))
            .await
            .expect("late skill metadata after rescan");
        assert!(late_meta.contains("Available: true"), "{late_meta}");

        // The originally-present skill still loads.
        let alpha = tool
            .execute(serde_json::json!({ "skill_name": "first_skill", "detail": "full" }))
            .await
            .expect("first skill still loads");
        assert!(alpha.contains("alpha body"), "{alpha}");

        // A genuinely non-existent skill still errors after a rescan turns up nothing new.
        let missing = tool
            .execute(serde_json::json!({ "skill_name": "no_such_skill" }))
            .await;
        assert!(
            missing.is_err(),
            "unknown skill should still error after rescan: {missing:?}"
        );
    }
}
