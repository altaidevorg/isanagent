//! Audit X9: tool dispatch pipeline, split out of the former
//! `agent/mod.rs` god-file.
//!
//! Contents: typed tool-failure values ([`ToolExecutionFinished`],
//! [`InvalidToolArguments`]), argument parsing, pre/post hook plumbing,
//! invocation telemetry, and the central [`execute_tool_call_with_activity`]
//! executor with heartbeats, approvals, and cooperative cancellation.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;

use tokio::sync::mpsc;

use super::approval::{
    approval_already_granted, classify_approval_reply, code_exec_requires_approval,
    command_preview, command_preview_with_flag, edit_policy_block_reason,
    edit_policy_mode_for_session, extract_code_exec_command, is_code_exec_tool,
    is_file_mutate_tool, remember_approval_grant, shell_policy_mode_for_session, ApprovalReply,
};
use super::WAIT_SIGNAL_PREFIX;

use crate::bus::{BusMessage, LogEvent, OutboundMessage, TelemetryEvent};
use crate::clarification::ClarificationHub;
use crate::config::{ResolvedShellPolicy, ShellPolicyMode};
use crate::hooks::{
    run_post_tool_hooks, run_pre_tool_hooks, HookObservationMeta, HookSessionInfo,
    PostToolDecision, PreToolOutcome, ToolCallHookContext,
};
use crate::logging::LoggerHandle;
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tool_runtime::{with_tool_exec_and_progress_scope, ToolExecCtx, ToolProgressEmitter};
use crate::tools::ToolRegistry;
use crate::traits::{ToolErrorCode, ToolResult};

pub(crate) enum ToolExecutionFinished {
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
pub(crate) struct InvalidToolArguments {
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

    pub(crate) fn to_tool_result(&self) -> ToolResult {
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

pub(crate) fn parse_tool_arguments(raw: &str) -> Result<Value, InvalidToolArguments> {
    serde_json::from_str(raw).map_err(InvalidToolArguments::from_json_error)
}

pub(crate) fn extract_exec_command(args: &Value) -> Option<String> {
    args.get("command")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Append post_tool verification-hook output (build/test/lint results) to a tool result, preserving
/// Ok/Err polarity so the model sees it alongside the tool's own output and can self-correct.
pub(crate) fn append_post_tool_output(result: ToolResult, hook_out: &str) -> ToolResult {
    append_hook_note(result, "post-tool hook", hook_out)
}

fn append_hook_note(mut result: ToolResult, label: &str, note: &str) -> ToolResult {
    // `result` is owned, so append onto the existing buffer in place rather than allocating a
    // fresh string and copying the (potentially large) tool output into it.
    result.content.push_str(&format!("\n\n[{label}]\n{note}"));
    result
}

/// Fold steering-hook decisions into the result the model sees, in order: `updatedOutput`
/// replaces the content, a post_tool `block` turns it into an error, then pre_tool notes and
/// post_tool verification output / context are appended.
pub(crate) fn apply_tool_hook_decisions(
    mut result: ToolResult,
    pre_notes: &[String],
    post: PostToolDecision,
) -> ToolResult {
    if let Some(output) = post.updated_output {
        // Keep the error message in step with the rewritten content (it is what later hooks and
        // logs see), and drop any MCP App view: it would still render the un-rewritten data.
        if let Some(error) = result.error.as_mut() {
            error.message = output.clone();
        }
        result.mcp_app = None;
        result.content = output;
    }
    if let Some(reason) = post.block_reason {
        result.status = crate::traits::ToolResultStatus::Error;
        result.error = Some(crate::traits::ToolResultError {
            code: ToolErrorCode::PolicyDenied,
            message: reason.clone(),
        });
        result = append_hook_note(result, "post-tool hook rejected this result", &reason);
    }
    if !pre_notes.is_empty() {
        result = append_hook_note(result, "pre-tool hook", &pre_notes.join("\n"));
    }
    if !post.notes.is_empty() {
        result = append_post_tool_output(result, &post.notes.join("\n\n"));
    }
    result
}

/// A hook asking for approval upgrades `allow` to `ask`; it never loosens `deny`.
fn force_ask_mode(mode: ShellPolicyMode, hook_asked: bool) -> ShellPolicyMode {
    if hook_asked && mode == ShellPolicyMode::Allow {
        ShellPolicyMode::Ask
    } else {
        mode
    }
}

const HOOK_ARGS_PREVIEW_BYTES: usize = 2_000;

fn hook_args_preview(args: &Value) -> String {
    let mut s = serde_json::to_string(args).unwrap_or_default();
    crate::utils::truncate_utf8_safe(&mut s, HOOK_ARGS_PREVIEW_BYTES, "…");
    s
}

fn hook_args_truncated(args: &Value) -> bool {
    serde_json::to_string(args).is_ok_and(|s| s.len() > HOOK_ARGS_PREVIEW_BYTES)
}

/// Approval-grant key for hook-forced prompts: a digest of the FULL arguments, so an "always"
/// grant can't be reused by different arguments that share the displayed (truncated) prefix.
fn hook_args_digest(args: &Value) -> String {
    use sha2::{Digest, Sha256};
    let full = serde_json::to_string(args).unwrap_or_default();
    format!("{:x}", Sha256::digest(full.as_bytes()))
}

pub(crate) fn hook_observe_telemetry(
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

pub(crate) fn shell_command_uses_grep_like(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.contains("grep ")
        || lower.contains("| grep")
        || lower.contains("cat ")
        || lower.contains("wc ")
}

pub(crate) async fn log_tool_invocation_start(
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
        LogEvent::info(agent_name, &format!("Invoking tool: {tool_name}"))
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

/// Cancels the run after an `abort` approval reply and tells the channel so. A cancelled run
/// sends no reply of its own, so a client waiting on the answer (the web UI's stream) would hang.
async fn abort_run(
    cancel_token: Option<&tokio_util::sync::CancellationToken>,
    outbound_tx: &mpsc::Sender<BusMessage>,
    channel: &str,
    chat_id: &str,
    thread_id: Option<String>,
    notice: &str,
) {
    if let Some(token) = cancel_token {
        token.cancel();
    }
    let _ = outbound_tx
        .send(BusMessage::Outbound(OutboundMessage {
            channel: channel.to_string(),
            chat_id: chat_id.to_string(),
            thread_id,
            content: notice.to_string(),
            metadata: HashMap::new(),
        }))
        .await;
}

/// Session-scoped wiring for tools that need the active chat (e.g. `ask_user`).
#[derive(Clone)]
pub(crate) struct ToolCallRuntime {
    pub(crate) session: ToolExecCtx,
    pub(crate) hub: Arc<ClarificationHub>,
    pub(crate) is_subagent: bool,
    pub(crate) subagent_allowlist: Option<Arc<HashSet<String>>>,
    pub(crate) shell_policy: Arc<ResolvedShellPolicy>,
    pub(crate) unattended_session: bool,
    pub(crate) hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
    pub(crate) inbound_metadata: Arc<HashMap<String, serde_json::Value>>,
}

/// Runs a tool with optional per-call activity heartbeats and optional cooperative cancellation.
#[allow(clippy::too_many_arguments)] // Central tool-dispatch path; grouping would obscure call sites.
pub(crate) async fn execute_tool_call_with_activity(
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
        // Steering hooks run FIRST, before any policy check: a hook may rewrite the arguments
        // (`updatedInput`), and the shell / edit policy (and the user's approval prompt) must
        // judge exactly what will be executed, never the pre-rewrite arguments.
        let mut hook_ask_reason: Option<String> = None;
        let mut hook_ask_handled = false;
        let mut pre_hook_notes: Vec<String> = Vec::new();
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
                    PreToolOutcome::Stop(reason) => {
                        abort_run(
                            cancel_owned.as_ref(),
                            &outbound_tx,
                            &channel,
                            &chat_id,
                            thread_id_for_hooks.clone(),
                            &format!("Stopped by hook: {reason}"),
                        )
                        .await;
                        return ToolExecutionFinished::error(
                            ToolErrorCode::PolicyDenied,
                            format!("Run stopped by pre_tool hook: {reason}"),
                        );
                    }
                    PreToolOutcome::Proceed(decision) => {
                        if decision.args_modified {
                            // The persisted assistant message still holds the model's original
                            // arguments; tell the model what actually ran.
                            pre_hook_notes.push(format!(
                                "A pre_tool hook rewrote the arguments; the tool ran with: {}",
                                hook_args_preview(&decision.args)
                            ));
                        }
                        pre_hook_notes.extend(decision.additional_context);
                        hook_ask_reason = decision.ask_reason;
                        args = decision.args;
                    }
                }
            }
        }
        // Nobody can answer an approval prompt in an unattended session: a hook's `ask` is a deny
        // there (before the policy could upgrade `allow` into a prompt that would hang).
        if let Some(reason) = hook_ask_reason.as_deref() {
            if runtime.unattended_session {
                return ToolExecutionFinished::error(
                    ToolErrorCode::PolicyDenied,
                    format!("A pre_tool hook requires approval ({reason}), but this session is unattended; call skipped."),
                );
            }
        }
        let hook_ask_suffix = hook_ask_reason
            .as_deref()
            .map(|r| format!("\n\nA policy hook requested review: {r}"))
            .unwrap_or_default();

        if is_code_exec_tool(&tool_name) {
            if let Some(command) = extract_code_exec_command(&tool_name, &args) {
                let preview = command_preview(&command);
                let mode = force_ask_mode(
                    shell_policy_mode_for_session(&runtime.shell_policy, runtime.unattended_session),
                    hook_ask_reason.is_some(),
                );
                let requires_approval = hook_ask_reason.is_some()
                    || code_exec_requires_approval(
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
                                    "Command blocked by shell policy (mode=deny): {command}"
                                ),
                            );
                        }
                    }
                    ShellPolicyMode::Ask => {
                        if requires_approval {
                            hook_ask_handled = true;
                            let grant_key = format!("shell:{tool_name}:{command}");
                            // A hook's `ask` must reach the user even if this command was
                            // granted "always" earlier: the hook's reason is new information.
                            if hook_ask_reason.is_some()
                                || !approval_already_granted(&grant_key).await
                            {
                            let (preview, preview_truncated) = command_preview_with_flag(&command);
                            let _ = outbound_tx
                                .send(BusMessage::Telemetry(TelemetryEvent::ShellPolicyDecision {
                                    chat_id: chat_id.clone(),
                                    channel: channel.clone(),
                                    mode: "ask".to_string(),
                                    decision: "approval_requested".to_string(),
                                    command_preview: if preview_truncated {
                                        format!("{preview} [truncated]")
                                    } else {
                                        preview.clone()
                                    },
                                }))
                                .await;
                            let ask_payload = serde_json::json!({
                                "prompt": format!(
                                    "Approve running `{}`?\n\n```\n{}\n```{}\n\nReply with approve, deny, always (this run), or abort.{}",
                                    tool_name,
                                    command,
                                    hook_ask_suffix,
                                    if preview_truncated { "\n[command preview truncated in telemetry]" } else { "" }
                                ),
                                "choices": ["approve", "deny", "always", "abort"],
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
                                    let classified = classify_approval_reply(&reply);
                                    match classified {
                                        ApprovalReply::Grant | ApprovalReply::AlwaysThisRun => {
                                            if matches!(classified, ApprovalReply::AlwaysThisRun) {
                                                remember_approval_grant(grant_key).await;
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
                                        ApprovalReply::Abort => {
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
                                            abort_run(
                                                cancel_owned.as_ref(),
                                                &outbound_tx,
                                                &channel,
                                                &chat_id,
                                                thread_id_for_hooks.clone(),
                                                "Aborted: the command was not run.",
                                            )
                                            .await;
                                            return ToolExecutionFinished::error(
                                                ToolErrorCode::PolicyDenied,
                                                "Command approval aborted by user; execution skipped.",
                                            );
                                        }
                                        ApprovalReply::Deny => {
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
                                    }
                                }
                                Err(e) => {
                                    return ToolExecutionFinished::error(
                                        ToolErrorCode::ExecutionFailed,
                                        format!("Shell policy approval failed: {e}"),
                                    );
                                }
                            }
                            } // end if !already_granted
                        }
                    }
                }
            }
        }

        // Run this after steering hooks: a hook may rewrite the arguments, and the
        // user must see the exact mutation that will be executed.
        if is_file_mutate_tool(&tool_name) {
            match force_ask_mode(
                edit_policy_mode_for_session(&runtime.shell_policy, runtime.unattended_session),
                hook_ask_reason.is_some(),
            ) {
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
                        hook_ask_handled = true;
                        let grant_key = format!("edit:{}", preview.path);
                        if hook_ask_reason.is_some()
                            || !approval_already_granted(&grant_key).await
                        {
                        let ask_payload = serde_json::json!({
                            "prompt": format!(
                                "Approve edit to `{}`? Review the attached diff, then reply with approve, deny, always (this run), or abort.{}",
                                preview.path,
                                hook_ask_suffix
                            ),
                            "choices": ["approve", "deny", "always", "abort"],
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
                        match classify_approval_reply(&reply) {
                            ApprovalReply::Grant => {}
                            ApprovalReply::AlwaysThisRun => {
                                remember_approval_grant(grant_key).await;
                            }
                            ApprovalReply::Abort => {
                                abort_run(
                                    cancel_owned.as_ref(),
                                    &outbound_tx,
                                    &channel,
                                    &chat_id,
                                    thread_id_for_hooks.clone(),
                                    "Aborted: the edit was not applied.",
                                )
                                .await;
                                return ToolExecutionFinished::error(
                                    ToolErrorCode::PolicyDenied,
                                    "Edit approval aborted by user; mutation skipped.",
                                );
                            }
                            ApprovalReply::Deny => {
                                return ToolExecutionFinished::error(
                                    ToolErrorCode::PolicyDenied,
                                    "Edit not approved by user; mutation skipped.",
                                );
                            }
                        }
                        } // end if !already_granted
                        approved_mutation_preview = Some(preview);
                    }
                }
            }
        }

        // A hook asked for approval of a tool the shell / edit policy doesn't gate (or that
        // produced no mutation preview): ask with a generic prompt.
        if let Some(reason) = hook_ask_reason.as_deref().filter(|_| !hook_ask_handled) {
            let grant_key = format!("hook:{tool_name}:{}", hook_args_digest(&args));
            if !approval_already_granted(&grant_key).await {
                let ask_payload = serde_json::json!({
                    "prompt": format!(
                        "Approve calling `{}`?\n\n```json\n{}\n```{}\n\nA policy hook requested review: {}\n\nReply with approve, deny, always (this run), or abort.",
                        tool_name,
                        hook_args_preview(&args),
                        if hook_args_truncated(&args) { "\n[arguments truncated for display]" } else { "" },
                        reason
                    ),
                    "choices": ["approve", "deny", "always", "abort"],
                    "timeout_secs": 1800,
                    "allow_empty": false
                });
                let reply = match tools
                    .execute_tool_scoped("ask_user", ask_payload, None, is_subagent)
                    .await
                {
                    Ok(reply) => reply,
                    Err(error) => {
                        return ToolExecutionFinished::error(
                            ToolErrorCode::ExecutionFailed,
                            format!("Hook approval failed: {error}"),
                        );
                    }
                };
                match classify_approval_reply(&reply) {
                    ApprovalReply::Grant => {}
                    ApprovalReply::AlwaysThisRun => remember_approval_grant(grant_key).await,
                    ApprovalReply::Abort => {
                        abort_run(
                            cancel_owned.as_ref(),
                            &outbound_tx,
                            &channel,
                            &chat_id,
                            thread_id_for_hooks.clone(),
                            "Aborted: the tool call was not run.",
                        )
                        .await;
                        return ToolExecutionFinished::error(
                            ToolErrorCode::PolicyDenied,
                            "Tool approval aborted by user; call skipped.",
                        );
                    }
                    ApprovalReply::Deny => {
                        return ToolExecutionFinished::error(
                            ToolErrorCode::PolicyDenied,
                            "Tool call not approved by user; call skipped.",
                        );
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

        let mut post_tool: PostToolDecision = PostToolDecision::default();
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
                post_tool = run_post_tool_hooks(
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
                let result = apply_tool_hook_decisions(result, &pre_hook_notes, post_tool.clone());
                if let Some(reason) = post_tool.stop_reason {
                    abort_run(
                        cancel_owned.as_ref(),
                        &outbound_tx,
                        &channel,
                        &chat_id,
                        thread_id_for_hooks.clone(),
                        &format!("Stopped by hook: {reason}"),
                    )
                    .await;
                }
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

#[cfg(test)]
mod tests {
    use super::abort_run;
    use crate::bus::BusMessage;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn abort_cancels_the_run_and_answers_the_waiting_channel() {
        let token = CancellationToken::new();
        let (tx, mut rx) = mpsc::channel(4);
        abort_run(
            Some(&token),
            &tx,
            "api",
            "chat-1",
            Some("thread-1".to_string()),
            "Aborted: the edit was not applied.",
        )
        .await;

        assert!(token.is_cancelled());
        match rx.recv().await {
            Some(BusMessage::Outbound(out)) => {
                assert_eq!(out.channel, "api");
                assert_eq!(out.chat_id, "chat-1");
                assert_eq!(out.thread_id.as_deref(), Some("thread-1"));
                assert_eq!(out.content, "Aborted: the edit was not applied.");
            }
            _ => panic!("expected an outbound notice"),
        }
    }
}

#[cfg(all(test, unix))]
mod hook_dispatch_tests {
    use super::*;
    use crate::clarification::ClarificationHub;
    use crate::config::{HarnessHookCommandConfig, HarnessHooksSteeringConfig};
    use crate::tool_runtime::ToolExecCtx;
    use crate::traits::Tool;
    use async_trait::async_trait;

    /// Stand-in for `exec`: echoes the command it was asked to run.
    struct EchoExec;

    #[async_trait]
    impl Tool for EchoExec {
        fn name(&self) -> &str {
            "exec"
        }
        fn description(&self) -> &str {
            "echo"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}})
        }
        async fn execute(&self, args: Value) -> Result<String, String> {
            Ok(format!("ran: {}", args["command"].as_str().unwrap_or("")))
        }
    }

    fn hook_ctx(pre_tool_stdout: &str, post_tool_stdout: Option<&str>) -> Arc<ToolCallHookContext> {
        let dir = std::env::temp_dir().join(format!("isanagent_dispatch_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cmd = |out: &str| HarnessHookCommandConfig {
            matcher: None,
            command: format!("cat >/dev/null; printf '%s' '{out}'"),
            timeout_ms: Some(5_000),
            cwd: None,
            on_error: None,
        };
        let cfg = HarnessHooksSteeringConfig {
            enabled: Some(true),
            pre_tool: Some(vec![cmd(pre_tool_stdout)]),
            post_tool: post_tool_stdout.map(|o| vec![cmd(o)]),
            ..Default::default()
        };
        let engine = crate::hooks::build_steering_engine(&cfg, dir.clone(), dir).unwrap();
        Arc::new(ToolCallHookContext {
            observation: None,
            steering: Some(engine),
        })
    }

    async fn run_exec(
        mode: ShellPolicyMode,
        patterns: &[&str],
        hooks: Arc<ToolCallHookContext>,
        command: &str,
    ) -> ToolResult {
        run_exec_with(mode, patterns, hooks, command, false).await
    }

    async fn run_exec_with(
        mode: ShellPolicyMode,
        patterns: &[&str],
        hooks: Arc<ToolCallHookContext>,
        command: &str,
        unattended: bool,
    ) -> ToolResult {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoExec));
        let tools = Arc::new(registry);
        let (tx, _rx) = mpsc::channel(16);
        let runtime = ToolCallRuntime {
            session: ToolExecCtx::new("test", "chat", None),
            hub: ClarificationHub::shared(),
            is_subagent: false,
            subagent_allowlist: None,
            shell_policy: Arc::new(ResolvedShellPolicy {
                interactive_mode: mode,
                unattended_mode: mode,
                interactive_edit_mode: ShellPolicyMode::Allow,
                unattended_edit_mode: ShellPolicyMode::Allow,
                approval_patterns: patterns.iter().map(|s| s.to_string()).collect(),
                windows_runner: Default::default(),
            }),
            unattended_session: unattended,
            hook_tool_ctx: Some(hooks),
            inbound_metadata: Arc::new(HashMap::new()),
        };
        match execute_tool_call_with_activity(
            &tools,
            None,
            "chat",
            "test",
            &tx,
            "exec",
            Some("call_1".into()),
            serde_json::json!({"command": command}),
            None,
            runtime,
        )
        .await
        {
            ToolExecutionFinished::Completed(r) => r,
            _ => panic!("unexpected outcome"),
        }
    }

    #[tokio::test]
    async fn shell_policy_judges_the_hook_rewritten_command() {
        // The model asks for a harmless `ls`; a hook rewrites it into a denied command. The shell
        // policy must evaluate what will actually run, so the call is blocked.
        let hooks = hook_ctx(
            r#"{"hookSpecificOutput":{"updatedInput":{"command":"rm -rf /danger"}}}"#,
            None,
        );
        let r = run_exec(ShellPolicyMode::Deny, &["rm"], hooks, "ls").await;
        assert!(r.is_error(), "{r:?}");
        assert_eq!(r.error_code(), Some(ToolErrorCode::PolicyDenied));
        assert!(r.content.contains("rm -rf /danger"), "{}", r.content);
    }

    #[tokio::test]
    async fn rewritten_args_run_and_model_is_told() {
        let hooks = hook_ctx(
            r#"{"hookSpecificOutput":{"updatedInput":{"command":"ls -la"},"additionalContext":"listing expanded"}}"#,
            Some(r#"{"hookSpecificOutput":{"updatedOutput":"ran: [redacted]"}}"#),
        );
        let r = run_exec(ShellPolicyMode::Allow, &[], hooks, "ls").await;
        assert!(!r.is_error(), "{r:?}");
        assert!(r.content.starts_with("ran: [redacted]"), "{}", r.content);
        assert!(
            r.content
                .contains(r#"the tool ran with: {"command":"ls -la"}"#),
            "{}",
            r.content
        );
        assert!(r.content.contains("listing expanded"), "{}", r.content);
    }

    #[tokio::test]
    async fn hook_ask_cannot_loosen_deny() {
        // `ask` upgrades allow→ask, but under deny it must still block (no prompt is possible).
        let hooks = hook_ctx(
            r#"{"hookSpecificOutput":{"permissionDecision":"ask","permissionDecisionReason":"r"}}"#,
            None,
        );
        let r = run_exec(ShellPolicyMode::Deny, &[], hooks, "ls").await;
        assert_eq!(r.error_code(), Some(ToolErrorCode::PolicyDenied), "{r:?}");
    }

    #[tokio::test]
    async fn hook_ask_blocks_in_unattended_sessions_even_under_allow() {
        let hooks = hook_ctx(
            r#"{"hookSpecificOutput":{"permissionDecision":"ask","permissionDecisionReason":"r"}}"#,
            None,
        );
        let r = run_exec_with(ShellPolicyMode::Allow, &[], hooks, "ls", true).await;
        assert_eq!(r.error_code(), Some(ToolErrorCode::PolicyDenied), "{r:?}");
        assert!(r.content.contains("unattended"), "{}", r.content);
    }

    #[test]
    fn hook_grant_key_covers_full_arguments() {
        let base = "a".repeat(HOOK_ARGS_PREVIEW_BYTES + 10);
        let a = serde_json::json!({"x": format!("{base}1")});
        let b = serde_json::json!({"x": format!("{base}2")});
        assert_eq!(hook_args_preview(&a), hook_args_preview(&b));
        assert_ne!(hook_args_digest(&a), hook_args_digest(&b));
        assert!(hook_args_truncated(&a));
    }

    #[test]
    fn updated_output_rewrites_error_message() {
        let r = apply_tool_hook_decisions(
            ToolResult::error(ToolErrorCode::ExecutionFailed, "secret=abc"),
            &[],
            PostToolDecision {
                updated_output: Some("secret=[REDACTED]".into()),
                ..Default::default()
            },
        );
        assert_eq!(r.content, "secret=[REDACTED]");
        assert_eq!(r.error.unwrap().message, "secret=[REDACTED]");
    }

    #[test]
    fn force_ask_only_upgrades_allow() {
        assert_eq!(
            force_ask_mode(ShellPolicyMode::Allow, true),
            ShellPolicyMode::Ask
        );
        assert_eq!(
            force_ask_mode(ShellPolicyMode::Deny, true),
            ShellPolicyMode::Deny
        );
        assert_eq!(
            force_ask_mode(ShellPolicyMode::Allow, false),
            ShellPolicyMode::Allow
        );
    }

    #[test]
    fn post_tool_block_turns_result_into_error() {
        let r = apply_tool_hook_decisions(
            ToolResult::success("ok"),
            &[],
            PostToolDecision {
                block_reason: Some("lint failed".into()),
                ..Default::default()
            },
        );
        assert!(r.is_error());
        assert!(r.content.contains("lint failed"));
    }
}
