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

use crate::bus::{BusMessage, LogEvent, TelemetryEvent};
use crate::clarification::ClarificationHub;
use crate::config::{ResolvedShellPolicy, ShellPolicyMode};
use crate::hooks::{
    run_post_tool_hooks, run_pre_tool_hooks, HookObservationMeta, HookSessionInfo, PreToolOutcome,
    ToolCallHookContext,
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
pub(crate) fn append_post_tool_output(mut result: ToolResult, hook_out: &str) -> ToolResult {
    let note = format!("\n\n[post-tool hook]\n{hook_out}");
    // `result` is owned, so append onto the existing buffer in place rather than allocating a
    // fresh string and copying the (potentially large) tool output into it.
    result.content.push_str(&note);
    result
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
                                    "Command blocked by shell policy (mode=deny): {command}"
                                ),
                            );
                        }
                    }
                    ShellPolicyMode::Ask => {
                        if requires_approval {
                            let grant_key = format!("shell:{tool_name}:{command}");
                            if !approval_already_granted(&grant_key).await {
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
                                    "Approve running `{}`?\n\n```\n{}\n```\n\nReply with approve, deny, always (this run), or abort.{}",
                                    tool_name,
                                    command,
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
                                            if let Some(token) = cancel_owned.as_ref() {
                                                token.cancel();
                                            }
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
                        let grant_key = format!("edit:{}", preview.path);
                        if !approval_already_granted(&grant_key).await {
                        let ask_payload = serde_json::json!({
                            "prompt": format!(
                                "Approve edit to `{}`? Review the attached diff, then reply with approve, deny, always (this run), or abort.",
                                preview.path
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
                                if let Some(token) = cancel_owned.as_ref() {
                                    token.cancel();
                                }
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
