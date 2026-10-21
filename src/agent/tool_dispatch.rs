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
    extract_code_exec_command, is_code_exec_tool, is_file_mutate_tool, path_already_granted,
    permission_gate, remember_approval_grant, remember_path_grant, ApprovalReply, PathGrant,
    PermissionGate,
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
use crate::traits::{PathAccess, ToolErrorCode, ToolResult};

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

/// Where an approval prompt is raised and how an `abort` reply reaches the client.
struct ApprovalChannel<'a> {
    tools: &'a Arc<ToolRegistry>,
    is_subagent: bool,
    cancel_token: Option<&'a tokio_util::sync::CancellationToken>,
    outbound_tx: &'a mpsc::Sender<BusMessage>,
    channel: &'a str,
    chat_id: &'a str,
    thread_id: Option<String>,
}

/// What an "always" answer remembers.
enum ApprovalGrant<'a> {
    /// An exact key (`tool:<name>:<args digest>`).
    Key(String),
    /// A directory-scoped path grant (see [`PathGrant`]).
    Path {
        grant: PathGrant,
        access: &'a PathAccess,
        protected: &'a [std::path::PathBuf],
    },
}

/// Ask the user (approve / deny / always / abort) through `ask_user`, so web approval cards and
/// TUI prompts work unchanged. `Ok(())` means the call may run; an "always" reply remembers
/// `grant` for the rest of the process. A grant that already covers the call skips the prompt.
async fn request_tool_approval(
    approval: &ApprovalChannel<'_>,
    prompt: String,
    grant: ApprovalGrant<'_>,
    skip_if_granted: bool,
) -> Result<(), ToolExecutionFinished> {
    let already = match &grant {
        ApprovalGrant::Key(key) => approval_already_granted(key).await,
        ApprovalGrant::Path {
            access, protected, ..
        } => path_already_granted(access, protected).await,
    };
    if skip_if_granted && already {
        return Ok(());
    }
    let ask_payload = serde_json::json!({
        "prompt": prompt,
        "choices": ["approve", "deny", "always", "abort"],
        "timeout_secs": 1800,
        "allow_empty": false
    });
    // System-initiated prompt: bypass the sub-agent tool allowlist so a restricted sub-agent
    // can still surface the approval dialog.
    let reply = approval
        .tools
        .execute_tool_scoped("ask_user", ask_payload, None, approval.is_subagent)
        .await
        .map_err(|error| {
            ToolExecutionFinished::error(
                ToolErrorCode::ExecutionFailed,
                format!("Tool approval failed: {error}"),
            )
        })?;
    match classify_approval_reply(&reply) {
        ApprovalReply::Grant => Ok(()),
        ApprovalReply::AlwaysThisRun => {
            match grant {
                ApprovalGrant::Key(key) => remember_approval_grant(key).await,
                ApprovalGrant::Path { grant, .. } => remember_path_grant(grant).await,
            }
            Ok(())
        }
        ApprovalReply::Abort => {
            abort_run(
                approval.cancel_token,
                approval.outbound_tx,
                approval.channel,
                approval.chat_id,
                approval.thread_id.clone(),
                "Aborted: the tool call was not run.",
            )
            .await;
            Err(ToolExecutionFinished::error(
                ToolErrorCode::PolicyDenied,
                "Tool approval aborted by user; call skipped.",
            ))
        }
        ApprovalReply::Deny => Err(ToolExecutionFinished::error(
            ToolErrorCode::PolicyDenied,
            "Tool call not approved by user; call skipped.",
        )),
    }
}

fn location_label(access: &PathAccess) -> &'static str {
    if access.inside_sandbox {
        "inside the workspace sandbox"
    } else {
        "outside the workspace sandbox"
    }
}

/// Deny message for a location rule.
fn location_denied_message(tool_name: &str, access: &PathAccess) -> String {
    let key = if access.inside_sandbox {
        "inside_workspace"
    } else {
        "outside_workspace"
    };
    format!(
        "Permission denied: `{tool_name}` may not {} `{}` ({}; [harness.permissions].{key} = \"deny\" or the session is unattended).",
        access.kind.verb(),
        access.path.display(),
        location_label(access),
    )
}

/// Deny message for a `[harness.permissions.tools]` entry that resolved to `deny` by itself
/// (`deny`, or `ask` in an unattended session); the edit ceiling has its own message.
fn tool_entry_denied_message(tool_name: &str, unattended_session: bool) -> String {
    if unattended_session {
        format!("Permission denied: `{tool_name}` is blocked by [harness.permissions.tools].{tool_name} (deny, or ask in an unattended session where nobody can approve).")
    } else {
        format!("Permission denied: `{tool_name}` is blocked by [harness.permissions.tools].{tool_name} (deny).")
    }
}

/// Deny message for a file tool that reported no paths for this call.
fn unreported_paths_denied_message(tool_name: &str) -> String {
    format!("Permission denied: `{tool_name}` did not report which paths it touches, so the stricter of [harness.permissions].inside_workspace / outside_workspace applies (deny, or ask in an unattended session).")
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

        // `[harness.permissions]`: per-tool entries, file locations (inside / outside the
        // sandbox) and the default for other tools. Judged after hooks (rewritten args).
        let approval = ApprovalChannel {
            tools: &tools,
            is_subagent,
            cancel_token: cancel_owned.as_ref(),
            outbound_tx: &outbound_tx,
            channel: &channel,
            chat_id: &chat_id,
            thread_id: thread_id_for_hooks.clone(),
        };
        let is_edit_tool = is_file_mutate_tool(&tool_name);
        let accesses = tools
            .path_accesses_scoped(&tool_name, &args, allow.as_deref(), is_subagent)
            .ok();
        let gate = permission_gate(
            &runtime.shell_policy.permissions,
            &tool_name,
            accesses.as_deref(),
            tools.reports_path_accesses(&tool_name),
            runtime.unattended_session,
        );
        // Mode for the edit gate below (diff approval) plus what to tell the user.
        // The third element is the deciding path when it lies outside the sandbox.
        let mut edit_gate: Option<(ShellPolicyMode, String, Option<PathAccess>)> = None;
        match gate {
            PermissionGate::Exempt => {}
            PermissionGate::ToolOverride {
                mode,
                ceiling_denied,
            }
            | PermissionGate::UnreportedPaths {
                mode,
                ceiling_denied,
            } if is_edit_tool => {
                let deny_reason = if ceiling_denied {
                    edit_policy_block_reason(runtime.unattended_session).to_string()
                } else if matches!(gate, PermissionGate::UnreportedPaths { .. }) {
                    unreported_paths_denied_message(&tool_name)
                } else {
                    tool_entry_denied_message(&tool_name, runtime.unattended_session)
                };
                edit_gate = Some((mode, deny_reason, None));
            }
            PermissionGate::Location {
                mode,
                access,
                ceiling_denied,
            } if is_edit_tool => {
                let deny_reason = if ceiling_denied {
                    edit_policy_block_reason(runtime.unattended_session).to_string()
                } else {
                    location_denied_message(&tool_name, &access)
                };
                let outside = (!access.inside_sandbox).then_some(access);
                edit_gate = Some((mode, deny_reason, outside));
            }
            PermissionGate::ToolOverride { mode, .. }
            | PermissionGate::UnreportedPaths { mode, .. }
            | PermissionGate::Default(mode) => {
                match force_ask_mode(mode, hook_ask_reason.is_some()) {
                    ShellPolicyMode::Allow => {}
                    ShellPolicyMode::Deny => {
                        let message = match gate {
                            PermissionGate::ToolOverride { .. } => {
                                tool_entry_denied_message(&tool_name, runtime.unattended_session)
                            }
                            PermissionGate::UnreportedPaths { .. } => {
                                unreported_paths_denied_message(&tool_name)
                            }
                            _ => format!("Permission denied: `{tool_name}` is blocked by [harness.permissions].default (deny, or ask in an unattended session)."),
                        };
                        return ToolExecutionFinished::error(ToolErrorCode::PolicyDenied, message);
                    }
                    ShellPolicyMode::Ask => {
                        hook_ask_handled = true;
                        let grant_key =
                            format!("tool:{tool_name}:{}", hook_args_digest(&args));
                        let prompt = format!(
                            "Approve calling `{}`?\n\n```json\n{}\n```{}{}\n\nReply with approve, deny, always (this run), or abort.",
                            tool_name,
                            hook_args_preview(&args),
                            if hook_args_truncated(&args) { "\n[arguments truncated for display]" } else { "" },
                            hook_ask_suffix
                        );
                        if let Err(finished) = request_tool_approval(
                            &approval,
                            prompt,
                            ApprovalGrant::Key(grant_key),
                            hook_ask_reason.is_none(),
                        )
                        .await
                        {
                            return finished;
                        }
                    }
                }
            }
            PermissionGate::Location { mode, access, .. } => match mode {
                // A hook's `ask` on an allowed path falls through to the generic hook prompt.
                ShellPolicyMode::Allow => {}
                ShellPolicyMode::Deny => {
                    return ToolExecutionFinished::error(
                        ToolErrorCode::PolicyDenied,
                        location_denied_message(&tool_name, &access),
                    );
                }
                ShellPolicyMode::Ask => {
                    hook_ask_handled = true;
                    let grant =
                        PathGrant::for_access(&runtime.shell_policy.permissions, &access);
                    let prompt = format!(
                        "Allow `{}` to {} `{}`? This path is {}.{}\n\nReply with approve, deny, always, or abort ({}).",
                        tool_name,
                        access.kind.verb(),
                        access.path.display(),
                        location_label(&access),
                        hook_ask_suffix,
                        grant.describe()
                    );
                    if let Err(finished) = request_tool_approval(
                        &approval,
                        prompt,
                        ApprovalGrant::Path {
                            grant,
                            access: &access,
                            protected: &runtime.shell_policy.permissions.protected_paths,
                        },
                        hook_ask_reason.is_none(),
                    )
                    .await
                    {
                        return finished;
                    }
                }
            },
        }

        if is_code_exec_tool(&tool_name) {
            // A `[harness.permissions.tools]` entry for a code-exec tool sets the shell-policy
            // mode for that tool (interactive; unattended keeps the stricter `unattended_default`)
            // and `deny` turns the tool off entirely.
            let tool_denied = runtime.shell_policy.code_exec_tool_denied(&tool_name);
            if tool_denied {
                let preview = extract_code_exec_command(&tool_name, &args)
                    .map(|command| command_preview(&command))
                    .unwrap_or_default();
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
                    format!("Permission denied: `{tool_name}` is set to \"deny\" in [harness.permissions.tools]; call skipped."),
                );
            }
            if let Some(command) = extract_code_exec_command(&tool_name, &args) {
                let preview = command_preview(&command);
                // A hook's `ask` already answered by the working-directory prompt is not
                // asked a second time.
                let hook_pending = hook_ask_reason.is_some() && !hook_ask_handled;
                let mode = force_ask_mode(
                    runtime
                        .shell_policy
                        .code_exec_mode(&tool_name, runtime.unattended_session),
                    hook_pending,
                );
                let requires_approval = hook_pending
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
                            if hook_pending
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
        if let Some((edit_mode, edit_deny_reason, outside_access)) = edit_gate {
            let protected = &runtime.shell_policy.permissions.protected_paths;
            let outside_grant = outside_access
                .as_ref()
                .map(|access| PathGrant::for_access(&runtime.shell_policy.permissions, access));
            let outside_note = match (outside_access.as_ref(), outside_grant.as_ref()) {
                (Some(access), Some(grant)) => format!(
                    "This file is outside the workspace sandbox (`{}`); {}. ",
                    access.path.display(),
                    grant.describe()
                ),
                _ => String::new(),
            };
            match force_ask_mode(edit_mode, hook_ask_reason.is_some()) {
                ShellPolicyMode::Allow => {}
                ShellPolicyMode::Deny => {
                    return ToolExecutionFinished::error(
                        ToolErrorCode::PolicyDenied,
                        edit_deny_reason,
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
                        let outside_granted = match outside_access.as_ref() {
                            Some(access) => path_already_granted(access, protected).await,
                            None => false,
                        };
                        if hook_ask_reason.is_some()
                            || !(outside_granted || approval_already_granted(&grant_key).await)
                        {
                        let ask_payload = serde_json::json!({
                            "prompt": format!(
                                "Approve edit to `{}`? {}Review the attached diff, then reply with approve, deny, always (this run), or abort.{}",
                                preview.path,
                                outside_note,
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
                                if let Some(grant) = outside_grant.clone() {
                                    remember_path_grant(grant).await;
                                }
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
                approval_patterns: patterns.iter().map(|s| s.to_string()).collect(),
                ..ResolvedShellPolicy::default()
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

#[cfg(test)]
mod permission_dispatch_tests {
    use super::*;
    use crate::config::{AppConfig, ResolvedToolPermissions};
    use crate::tool_runtime::ToolExecCtx;
    use crate::tools::builtin::{ListDirTool, ReadFileTool, WriteFileTool};
    use crate::traits::Tool;
    use async_trait::async_trait;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Stand-in for `ask_user`: records each prompt payload and answers with a fixed reply.
    struct ScriptedAsk {
        reply: String,
        prompts: Arc<Mutex<Vec<Value>>>,
    }

    #[async_trait]
    impl Tool for ScriptedAsk {
        fn name(&self) -> &str {
            "ask_user"
        }
        fn description(&self) -> &str {
            "ask"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, args: Value) -> Result<String, String> {
            if let Ok(mut prompts) = self.prompts.lock() {
                prompts.push(args);
            }
            Ok(self.reply.clone())
        }
    }

    /// A non-file tool (web / MCP / plugin stand-in).
    struct Named(&'static str);

    #[async_trait]
    impl Tool for Named {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "named"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: Value) -> Result<String, String> {
            Ok(format!("{} ran", self.0))
        }
    }

    /// Stand-in for `exec`: reports its working directory like `ShellExecTool` and echoes the
    /// command instead of running it.
    struct FakeExec {
        workspace_dir: PathBuf,
    }

    #[async_trait]
    impl Tool for FakeExec {
        fn name(&self) -> &str {
            "exec"
        }
        fn description(&self) -> &str {
            "exec"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        fn reports_path_accesses(&self) -> bool {
            true
        }
        fn path_accesses(&self, args: &Value) -> Result<Vec<PathAccess>, String> {
            let dir = args
                .get("working_dir")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            let path = crate::tools::builtin::resolve_path(dir, &self.workspace_dir, false)?;
            Ok(vec![PathAccess {
                inside_sandbox: crate::utils::path_is_within(&self.workspace_dir, &path),
                path,
                kind: crate::traits::PathAccessKind::WorkingDir,
            }])
        }
        async fn execute(&self, args: Value) -> Result<String, String> {
            Ok(format!("ran: {}", args["command"].as_str().unwrap_or("")))
        }
    }

    /// A file tool that (wrongly) reports no paths.
    struct SilentFileTool;

    #[async_trait]
    impl Tool for SilentFileTool {
        fn name(&self) -> &str {
            "silent_read"
        }
        fn description(&self) -> &str {
            "silent"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        fn reports_path_accesses(&self) -> bool {
            true
        }
        async fn execute(&self, _args: Value) -> Result<String, String> {
            Ok("silent_read ran".to_string())
        }
    }

    struct Fixture {
        _tmp: tempfile::TempDir,
        sandbox: PathBuf,
        outside: PathBuf,
    }

    fn fixture() -> Fixture {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().canonicalize().expect("canon");
        let sandbox = root.join("workspace");
        let outside = root.join("outside");
        std::fs::create_dir_all(&sandbox).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(sandbox.join("in.txt"), "inside content").unwrap();
        std::fs::write(outside.join("out.txt"), "outside content").unwrap();
        Fixture {
            _tmp: tmp,
            sandbox,
            outside,
        }
    }

    struct Outcome {
        result: ToolResult,
        prompts: Vec<Value>,
        /// `ShellPolicyDecision` telemetry as `mode:decision`.
        shell_decisions: Vec<String>,
    }

    async fn dispatch(
        fx: &Fixture,
        perms: ResolvedToolPermissions,
        unattended: bool,
        reply: &str,
        tool_name: &str,
        args: Value,
    ) -> Outcome {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        // Built exactly like host.rs does.
        let sandboxed = |tool: &str, is_edit: bool| perms.sandbox_enforced_for(tool, is_edit);
        registry.register(Box::new(ReadFileTool {
            workspace_dir: fx.sandbox.clone(),
            restrict_to_workspace: sandboxed("read_file", false),
        }));
        registry.register(Box::new(WriteFileTool {
            workspace_dir: fx.sandbox.clone(),
            restrict_to_workspace: sandboxed("write_file", true),
        }));
        registry.register(Box::new(ListDirTool {
            workspace_dir: fx.sandbox.clone(),
            restrict_to_workspace: sandboxed("list_dir", false),
        }));
        registry.register(Box::new(Named("web_fetch")));
        registry.register(Box::new(Named("mcp__github__create_issue")));
        registry.register(Box::new(Named("execution_run")));
        registry.register(Box::new(FakeExec {
            workspace_dir: fx.sandbox.clone(),
        }));
        registry.register(Box::new(SilentFileTool));
        registry.register(Box::new(ScriptedAsk {
            reply: reply.to_string(),
            prompts: prompts.clone(),
        }));
        let tools = Arc::new(registry);
        let (tx, mut rx) = mpsc::channel(64);
        let runtime = ToolCallRuntime {
            session: ToolExecCtx::new("test", "chat", None),
            hub: ClarificationHub::shared(),
            is_subagent: false,
            subagent_allowlist: None,
            shell_policy: Arc::new(ResolvedShellPolicy {
                permissions: perms,
                ..ResolvedShellPolicy::default()
            }),
            unattended_session: unattended,
            hook_tool_ctx: None,
            inbound_metadata: Arc::new(HashMap::new()),
        };
        let result = match execute_tool_call_with_activity(
            &tools,
            None,
            "chat",
            "test",
            &tx,
            tool_name,
            Some("call_1".into()),
            args,
            None,
            runtime,
        )
        .await
        {
            ToolExecutionFinished::Completed(r) => r,
            _ => panic!("unexpected outcome"),
        };
        let prompts = prompts.lock().map(|p| p.clone()).unwrap_or_default();
        let mut shell_decisions = Vec::new();
        while let Ok(message) = rx.try_recv() {
            if let BusMessage::Telemetry(TelemetryEvent::ShellPolicyDecision {
                mode,
                decision,
                ..
            }) = message
            {
                shell_decisions.push(format!("{mode}:{decision}"));
            }
        }
        Outcome {
            result,
            prompts,
            shell_decisions,
        }
    }

    fn perms(toml_src: &str) -> ResolvedToolPermissions {
        toml::from_str::<AppConfig>(toml_src)
            .expect("config")
            .resolved_tool_permissions()
    }

    fn prompt_text(outcome: &Outcome) -> String {
        outcome.prompts[0]["prompt"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }

    #[tokio::test]
    async fn inside_read_and_write_run_without_a_prompt() {
        let fx = fixture();
        let read = dispatch(
            &fx,
            perms(""),
            false,
            "deny",
            "read_file",
            serde_json::json!({"path": "in.txt"}),
        )
        .await;
        assert!(!read.result.is_error(), "{:?}", read.result);
        assert!(read.result.content.contains("inside content"));
        assert!(read.prompts.is_empty());

        let write = dispatch(
            &fx,
            perms(""),
            false,
            "deny",
            "write_file",
            serde_json::json!({"path": "new/created.txt", "content": "hello"}),
        )
        .await;
        assert!(!write.result.is_error(), "{:?}", write.result);
        assert!(write.prompts.is_empty());
        assert_eq!(
            std::fs::read_to_string(fx.sandbox.join("new/created.txt")).unwrap(),
            "hello"
        );
    }

    #[tokio::test]
    async fn outside_read_asks_with_the_canonical_path_and_approve_proceeds() {
        let fx = fixture();
        let target = fx.outside.join("out.txt");
        let approved = dispatch(
            &fx,
            perms(""),
            false,
            "approve",
            "read_file",
            serde_json::json!({"path": "../outside/./out.txt"}),
        )
        .await;
        assert!(!approved.result.is_error(), "{:?}", approved.result);
        assert!(approved.result.content.contains("outside content"));
        assert_eq!(approved.prompts.len(), 1);
        let prompt = prompt_text(&approved);
        assert!(prompt.contains(&target.display().to_string()), "{prompt}");
        assert!(prompt.contains("to read"), "{prompt}");
        assert!(prompt.contains("outside the workspace sandbox"), "{prompt}");
        assert_eq!(
            approved.prompts[0]["choices"],
            serde_json::json!(["approve", "deny", "always", "abort"])
        );

        let denied = dispatch(
            &fx,
            perms(""),
            false,
            "deny",
            "read_file",
            serde_json::json!({"path": target.to_string_lossy()}),
        )
        .await;
        assert_eq!(
            denied.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert!(!denied.result.content.contains("outside content"));
    }

    #[tokio::test]
    async fn always_grants_the_operation_under_that_directory() {
        let fx = fixture();
        let x = fx.outside.join("x");
        let xy = fx.outside.join("xy");
        std::fs::create_dir_all(x.join("deep")).unwrap();
        std::fs::create_dir_all(&xy).unwrap();
        for path in [
            x.join("a.txt"),
            x.join("b.txt"),
            x.join("deep/d.txt"),
            xy.join("c.txt"),
        ] {
            std::fs::write(&path, "data").unwrap();
        }
        let read = |path: PathBuf, reply: &'static str| {
            let fx = &fx;
            async move {
                dispatch(
                    fx,
                    perms(""),
                    false,
                    reply,
                    "read_file",
                    serde_json::json!({"path": path.to_string_lossy()}),
                )
                .await
            }
        };
        let first = read(x.join("a.txt"), "always").await;
        assert!(!first.result.is_error(), "{:?}", first.result);
        assert_eq!(first.prompts.len(), 1);
        let prompt = prompt_text(&first);
        assert!(
            prompt.contains(&format!("always = allow reads under `{}`", x.display())),
            "{prompt}"
        );
        // Same directory and below: no prompt.
        for path in [x.join("b.txt"), x.join("deep/d.txt")] {
            let again = read(path, "deny").await;
            assert!(!again.result.is_error(), "{:?}", again.result);
            assert!(again.prompts.is_empty());
        }
        // A sibling with the same name prefix still asks.
        let sibling = read(xy.join("c.txt"), "deny").await;
        assert_eq!(sibling.prompts.len(), 1);
        assert_eq!(
            sibling.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        // A read grant does not cover writes.
        let write = dispatch(
            &fx,
            perms(""),
            false,
            "deny",
            "write_file",
            serde_json::json!({"path": x.join("new.txt").to_string_lossy(), "content": "n"}),
        )
        .await;
        assert_eq!(write.prompts.len(), 1);
        assert!(!x.join("new.txt").exists());
        // A directory target grants that directory and below.
        let listed = dispatch(
            &fx,
            perms(""),
            false,
            "always",
            "list_dir",
            serde_json::json!({"path": xy.to_string_lossy()}),
        )
        .await;
        assert_eq!(listed.prompts.len(), 1);
        assert!(prompt_text(&listed).contains(&format!("under `{}`", xy.display())));
        let read_xy = read(xy.join("c.txt"), "deny").await;
        assert!(read_xy.prompts.is_empty());
    }

    #[tokio::test]
    async fn always_on_an_outside_edit_covers_writes_in_that_directory() {
        let fx = fixture();
        let dir = fx.outside.join("edits");
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, reply: &'static str| {
            let path = dir.join(name);
            let fx = &fx;
            async move {
                dispatch(
                    fx,
                    perms(""),
                    false,
                    reply,
                    "write_file",
                    serde_json::json!({"path": path.to_string_lossy(), "content": "v"}),
                )
                .await
            }
        };
        let first = write("one.txt", "always").await;
        assert!(!first.result.is_error(), "{:?}", first.result);
        assert!(prompt_text(&first).contains("always = allow writes and edits under"));
        let second = write("two.txt", "deny").await;
        assert!(!second.result.is_error(), "{:?}", second.result);
        assert!(second.prompts.is_empty());
        assert_eq!(std::fs::read_to_string(dir.join("two.txt")).unwrap(), "v");
    }

    #[tokio::test]
    async fn legacy_restrict_to_workspace_true_now_asks_and_does_not_clamp() {
        let fx = fixture();
        let legacy = perms("restrict_to_workspace = true\n");
        let outcome = dispatch(
            &fx,
            legacy.clone(),
            false,
            "approve",
            "read_file",
            serde_json::json!({"path": fx.outside.join("out.txt").to_string_lossy()}),
        )
        .await;
        assert!(!outcome.result.is_error(), "{:?}", outcome.result);
        assert!(outcome.result.content.contains("outside content"));
        assert_eq!(outcome.prompts.len(), 1);
        // `..` is no longer clamped at the sandbox root: it reaches the real parent and asks.
        let escaped = dispatch(
            &fx,
            legacy.clone(),
            false,
            "deny",
            "read_file",
            serde_json::json!({"path": "../outside/out.txt"}),
        )
        .await;
        assert_eq!(escaped.prompts.len(), 1);
        assert!(prompt_text(&escaped).contains(&fx.outside.join("out.txt").display().to_string()));
        // Unattended: ask -> deny.
        let unattended = dispatch(
            &fx,
            legacy,
            true,
            "approve",
            "read_file",
            serde_json::json!({"path": fx.outside.join("out.txt").to_string_lossy()}),
        )
        .await;
        assert_eq!(
            unattended.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert!(unattended.prompts.is_empty());
    }

    #[tokio::test]
    async fn explicit_outside_deny_keeps_the_hard_sandbox() {
        let fx = fixture();
        let outcome = dispatch(
            &fx,
            perms("[harness.permissions]\noutside_workspace = \"deny\"\n"),
            false,
            "approve",
            "read_file",
            serde_json::json!({"path": fx.outside.join("out.txt").to_string_lossy()}),
        )
        .await;
        assert!(
            outcome.result.content.contains("PermissionError"),
            "{}",
            outcome.result.content
        );
        assert!(outcome.prompts.is_empty());
    }

    #[tokio::test]
    async fn location_deny_rule_blocks_with_a_policy_error() {
        let fx = fixture();
        let outcome = dispatch(
            &fx,
            perms("[harness.permissions]\ninside_workspace = \"deny\"\n"),
            false,
            "approve",
            "read_file",
            serde_json::json!({"path": "in.txt"}),
        )
        .await;
        assert_eq!(
            outcome.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert!(
            outcome.result.content.contains("inside_workspace"),
            "{}",
            outcome.result.content
        );
        assert!(outcome.prompts.is_empty());
    }

    #[tokio::test]
    async fn unattended_sessions_deny_outside_access_instead_of_asking() {
        let fx = fixture();
        let outcome = dispatch(
            &fx,
            perms(""),
            true,
            "approve",
            "read_file",
            serde_json::json!({"path": fx.outside.join("out.txt").to_string_lossy()}),
        )
        .await;
        assert_eq!(
            outcome.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert!(
            outcome
                .result
                .content
                .contains("outside the workspace sandbox"),
            "{}",
            outcome.result.content
        );
        assert!(outcome.prompts.is_empty());
        // Inside stays allowed unattended.
        let inside = dispatch(
            &fx,
            perms(""),
            true,
            "deny",
            "read_file",
            serde_json::json!({"path": "in.txt"}),
        )
        .await;
        assert!(!inside.result.is_error(), "{:?}", inside.result);
    }

    #[tokio::test]
    async fn per_tool_overrides_replace_the_location_and_default_rules() {
        let fx = fixture();
        let p = perms(
            r#"
[harness.permissions]
default = "deny"
[harness.permissions.tools]
read_file = "deny"
list_dir = "allow"
web_fetch = "ask"
"mcp__github__create_issue" = "allow"
"#,
        );
        let read = dispatch(
            &fx,
            p.clone(),
            false,
            "approve",
            "read_file",
            serde_json::json!({"path": "in.txt"}),
        )
        .await;
        assert_eq!(read.result.error_code(), Some(ToolErrorCode::PolicyDenied));
        assert!(
            read.result
                .content
                .contains("[harness.permissions.tools].read_file"),
            "{}",
            read.result.content
        );

        // `allow` lifts the outside prompt for that tool.
        let list = dispatch(
            &fx,
            p.clone(),
            false,
            "deny",
            "list_dir",
            serde_json::json!({"path": fx.outside.to_string_lossy()}),
        )
        .await;
        assert!(!list.result.is_error(), "{:?}", list.result);
        assert!(list.prompts.is_empty());

        let web = dispatch(
            &fx,
            p.clone(),
            false,
            "approve",
            "web_fetch",
            serde_json::json!({"url": "https://example.com"}),
        )
        .await;
        assert_eq!(web.result.content, "web_fetch ran");
        assert_eq!(web.prompts.len(), 1);
        assert!(prompt_text(&web).contains("example.com"));

        let mcp = dispatch(
            &fx,
            p.clone(),
            false,
            "deny",
            "mcp__github__create_issue",
            serde_json::json!({}),
        )
        .await;
        assert_eq!(mcp.result.content, "mcp__github__create_issue ran");
        assert!(mcp.prompts.is_empty());

        // MCP tool without an entry falls under `default = "deny"`.
        let p2 = perms("[harness.permissions]\ndefault = \"deny\"\n");
        let blocked = dispatch(
            &fx,
            p2,
            false,
            "approve",
            "mcp__github__create_issue",
            serde_json::json!({}),
        )
        .await;
        assert_eq!(
            blocked.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert!(blocked
            .result
            .content
            .contains("[harness.permissions].default"));
    }

    #[tokio::test]
    async fn mcp_tool_name_ask_override_prompts_and_unattended_denies() {
        let fx = fixture();
        let p = perms("[harness.permissions.tools]\n\"mcp__github__create_issue\" = \"ask\"\n");
        let asked = dispatch(
            &fx,
            p.clone(),
            false,
            "deny",
            "mcp__github__create_issue",
            serde_json::json!({"title": "x"}),
        )
        .await;
        assert_eq!(asked.prompts.len(), 1);
        assert_eq!(asked.result.error_code(), Some(ToolErrorCode::PolicyDenied));
        let unattended = dispatch(
            &fx,
            p,
            true,
            "approve",
            "mcp__github__create_issue",
            serde_json::json!({}),
        )
        .await;
        assert!(unattended.prompts.is_empty());
        assert_eq!(
            unattended.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
    }

    #[tokio::test]
    async fn outside_write_asks_with_the_diff_and_deny_leaves_the_file() {
        let fx = fixture();
        let target = fx.outside.join("out.txt");
        let denied = dispatch(
            &fx,
            perms(""),
            false,
            "deny",
            "write_file",
            serde_json::json!({"path": target.to_string_lossy(), "content": "rewritten"}),
        )
        .await;
        assert_eq!(
            denied.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert!(
            denied.result.content.contains("not approved"),
            "{}",
            denied.result.content
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "outside content");
        assert_eq!(denied.prompts.len(), 1);
        let prompt = prompt_text(&denied);
        assert!(prompt.contains("outside the workspace sandbox"), "{prompt}");
        assert!(prompt.contains(&target.display().to_string()), "{prompt}");
        let diff = &denied.prompts[0]["metadata"]["edit_diff"];
        assert_eq!(diff["file"], target.display().to_string());
        assert!(diff["diff"]
            .as_str()
            .unwrap_or_default()
            .contains("+rewritten"));

        let approved = dispatch(
            &fx,
            perms(""),
            false,
            "approve",
            "write_file",
            serde_json::json!({"path": target.to_string_lossy(), "content": "rewritten"}),
        )
        .await;
        assert!(!approved.result.is_error(), "{:?}", approved.result);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "rewritten");
    }

    #[tokio::test]
    async fn plan_mode_still_blocks_inside_edits() {
        let fx = fixture();
        let outcome = dispatch(
            &fx,
            perms("[harness.shell_policy]\nedit_mode = \"deny\"\n[harness.permissions.tools]\nwrite_file = \"allow\"\n"),
            false,
            "approve",
            "write_file",
            serde_json::json!({"path": "in.txt", "content": "x"}),
        )
        .await;
        assert_eq!(
            outcome.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert_eq!(
            std::fs::read_to_string(fx.sandbox.join("in.txt")).unwrap(),
            "inside content"
        );
        // The ceiling, not the per-tool entry, blocked it: the message names plan mode.
        assert!(
            outcome.result.content.contains("plan mode"),
            "{}",
            outcome.result.content
        );
        assert!(
            !outcome
                .result
                .content
                .contains("[harness.permissions.tools]"),
            "{}",
            outcome.result.content
        );
        // Without a per-tool entry the message names plan mode too.
        let plan = dispatch(
            &fx,
            perms("[harness.shell_policy]\nedit_mode = \"deny\"\n"),
            false,
            "approve",
            "write_file",
            serde_json::json!({"path": "in.txt", "content": "x"}),
        )
        .await;
        assert!(
            plan.result.content.contains("plan mode"),
            "{}",
            plan.result.content
        );
        // Reads are not edits.
        let read = dispatch(
            &fx,
            perms("[harness.shell_policy]\nedit_mode = \"deny\"\n"),
            false,
            "deny",
            "read_file",
            serde_json::json!({"path": "in.txt"}),
        )
        .await;
        assert!(!read.result.is_error());
    }

    #[tokio::test]
    async fn legacy_edit_mode_ask_prompts_for_inside_edits() {
        let fx = fixture();
        let outcome = dispatch(
            &fx,
            perms("[harness.shell_policy]\nedit_mode = \"ask\"\n"),
            false,
            "approve",
            "write_file",
            serde_json::json!({"path": "in.txt", "content": "edited"}),
        )
        .await;
        assert!(!outcome.result.is_error(), "{:?}", outcome.result);
        assert_eq!(outcome.prompts.len(), 1);
        assert!(!prompt_text(&outcome).contains("outside"));
    }

    #[tokio::test]
    async fn edit_entry_deny_messages_name_the_actual_cause() {
        let fx = fixture();
        let write = |p: ResolvedToolPermissions, unattended: bool| {
            let fx = &fx;
            async move {
                dispatch(
                    fx,
                    p,
                    unattended,
                    "approve",
                    "write_file",
                    serde_json::json!({"path": "in.txt", "content": "x"}),
                )
                .await
            }
        };
        // The entry itself says deny.
        let denied = write(
            perms("[harness.permissions.tools]\nwrite_file = \"deny\"\n"),
            false,
        )
        .await;
        assert_eq!(
            denied.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert!(
            denied
                .result
                .content
                .contains("[harness.permissions.tools].write_file (deny)"),
            "{}",
            denied.result.content
        );
        // `ask` in an unattended session: the entry, with the unattended explanation.
        let unattended_ask = write(
            perms("[harness.permissions.tools]\nwrite_file = \"ask\"\n"),
            true,
        )
        .await;
        assert_eq!(
            unattended_ask.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert!(unattended_ask.prompts.is_empty());
        assert!(
            unattended_ask
                .result
                .content
                .contains("[harness.permissions.tools].write_file")
                && unattended_ask.result.content.contains("unattended"),
            "{}",
            unattended_ask.result.content
        );
        // `allow` capped by the unattended edit ceiling: the ceiling's message.
        let capped = write(
            perms("[harness.shell_policy]\nedit_unattended_default = \"deny\"\n[harness.permissions.tools]\nwrite_file = \"allow\"\n"),
            true,
        )
        .await;
        assert_eq!(
            capped.result.content,
            format!("Error: {}", edit_policy_block_reason(true))
        );
        assert_eq!(
            std::fs::read_to_string(fx.sandbox.join("in.txt")).unwrap(),
            "inside content"
        );
    }

    fn exec_args(command: &str) -> Value {
        serde_json::json!({"command": command})
    }

    #[tokio::test]
    async fn exec_allow_entry_acts_like_shell_mode_allow_interactively() {
        let fx = fixture();
        let p = perms("[harness.permissions.tools]\nexec = \"allow\"\n");
        for command in ["ls -la", "rm -rf build"] {
            let outcome = dispatch(&fx, p.clone(), false, "deny", "exec", exec_args(command)).await;
            assert_eq!(outcome.result.content, format!("ran: {command}"));
            assert!(outcome.prompts.is_empty());
            assert!(outcome.shell_decisions.is_empty());
        }
    }

    #[tokio::test]
    async fn exec_allow_entry_keeps_unattended_default_in_unattended_sessions() {
        let fx = fixture();
        let p = perms("[harness.permissions.tools]\nexec = \"allow\"\n");
        // Default `unattended_default = "deny"` still blocks risky commands.
        let risky = dispatch(
            &fx,
            p.clone(),
            true,
            "approve",
            "exec",
            exec_args("rm -rf build"),
        )
        .await;
        assert_eq!(risky.result.error_code(), Some(ToolErrorCode::PolicyDenied));
        assert!(
            risky.result.content.contains("shell policy (mode=deny)"),
            "{}",
            risky.result.content
        );
        assert!(risky.prompts.is_empty());
        assert_eq!(risky.shell_decisions, vec!["deny:blocked".to_string()]);
        // Safe commands run, exactly as without the entry.
        let safe = dispatch(&fx, p, true, "approve", "exec", exec_args("ls")).await;
        assert_eq!(safe.result.content, "ran: ls");
    }

    #[tokio::test]
    async fn exec_ask_entry_uses_the_shell_prompt_for_risky_commands() {
        let fx = fixture();
        let p = perms("[harness.shell_policy]\nmode = \"allow\"\n[harness.permissions.tools]\nexec = \"ask\"\n");
        let safe = dispatch(&fx, p.clone(), false, "deny", "exec", exec_args("ls")).await;
        assert_eq!(safe.result.content, "ran: ls");
        assert!(safe.prompts.is_empty());
        let risky = dispatch(&fx, p, false, "deny", "exec", exec_args("rm -rf build")).await;
        assert_eq!(risky.prompts.len(), 1);
        let prompt = prompt_text(&risky);
        assert!(prompt.starts_with("Approve running `exec`?"), "{prompt}");
        assert!(prompt.contains("rm -rf build"), "{prompt}");
        assert_eq!(risky.result.error_code(), Some(ToolErrorCode::PolicyDenied));
        assert_eq!(
            risky.shell_decisions,
            vec![
                "ask:approval_requested".to_string(),
                "ask:approval_denied".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn exec_deny_entry_blocks_every_call() {
        let fx = fixture();
        let p = perms("[harness.shell_policy]\nmode = \"allow\"\n[harness.permissions.tools]\nexec = \"deny\"\n");
        for unattended in [false, true] {
            let outcome = dispatch(
                &fx,
                p.clone(),
                unattended,
                "approve",
                "exec",
                exec_args("ls"),
            )
            .await;
            assert_eq!(
                outcome.result.error_code(),
                Some(ToolErrorCode::PolicyDenied)
            );
            assert!(
                outcome
                    .result
                    .content
                    .contains("[harness.permissions.tools]"),
                "{}",
                outcome.result.content
            );
            assert!(outcome.prompts.is_empty());
            assert_eq!(outcome.shell_decisions, vec!["deny:blocked".to_string()]);
        }
    }

    #[tokio::test]
    async fn exec_entry_keeps_working_directory_location_rules() {
        let fx = fixture();
        let p = perms("[harness.permissions.tools]\nexec = \"allow\"\n");
        let outcome = dispatch(
            &fx,
            p,
            false,
            "deny",
            "exec",
            serde_json::json!({"command": "ls", "working_dir": fx.outside.to_string_lossy()}),
        )
        .await;
        assert_eq!(outcome.prompts.len(), 1);
        assert!(prompt_text(&outcome).contains("outside the workspace sandbox"));
        assert_eq!(
            outcome.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
    }

    #[tokio::test]
    async fn execution_run_entry_follows_the_same_shell_rules() {
        let fx = fixture();
        let code = serde_json::json!({"code": "print(1)"});
        let allow = perms("[harness.permissions.tools]\nexecution_run = \"allow\"\n");
        let interactive = dispatch(
            &fx,
            allow.clone(),
            false,
            "deny",
            "execution_run",
            code.clone(),
        )
        .await;
        assert_eq!(interactive.result.content, "execution_run ran");
        assert!(interactive.prompts.is_empty());
        // Unattended: arbitrary code needs approval and `unattended_default = "deny"` blocks it.
        let unattended = dispatch(&fx, allow, true, "approve", "execution_run", code.clone()).await;
        assert_eq!(
            unattended.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert_eq!(unattended.shell_decisions, vec!["deny:blocked".to_string()]);
        let ask = perms("[harness.shell_policy]\nmode = \"allow\"\n[harness.permissions.tools]\nexecution_run = \"ask\"\n");
        let asked = dispatch(&fx, ask, false, "deny", "execution_run", code.clone()).await;
        assert_eq!(asked.prompts.len(), 1);
        assert!(prompt_text(&asked).starts_with("Approve running `execution_run`?"));
        assert_eq!(asked.result.error_code(), Some(ToolErrorCode::PolicyDenied));
        let deny = perms("[harness.permissions.tools]\nexecution_run = \"deny\"\n");
        let denied = dispatch(&fx, deny, false, "approve", "execution_run", code).await;
        assert_eq!(
            denied.result.error_code(),
            Some(ToolErrorCode::PolicyDenied)
        );
        assert!(denied.prompts.is_empty());
    }

    #[tokio::test]
    async fn file_tool_reporting_no_paths_gets_the_stricter_location_rule() {
        let fx = fixture();
        // Default rules: inside allow, outside ask -> ask (not `default = "allow"`).
        let asked = dispatch(
            &fx,
            perms(""),
            false,
            "deny",
            "silent_read",
            serde_json::json!({}),
        )
        .await;
        assert_eq!(asked.prompts.len(), 1);
        assert_eq!(asked.result.error_code(), Some(ToolErrorCode::PolicyDenied));
        // Unattended: ask -> deny, with a message saying why.
        let unattended = dispatch(
            &fx,
            perms(""),
            true,
            "approve",
            "silent_read",
            serde_json::json!({}),
        )
        .await;
        assert!(unattended.prompts.is_empty());
        assert!(
            unattended
                .result
                .content
                .contains("did not report which paths"),
            "{}",
            unattended.result.content
        );
        // Both locations allowed: it runs.
        let open =
            perms("[harness.permissions]\noutside_workspace = \"allow\"\ndefault = \"deny\"\n");
        let ran = dispatch(
            &fx,
            open,
            true,
            "deny",
            "silent_read",
            serde_json::json!({}),
        )
        .await;
        assert_eq!(ran.result.content, "silent_read ran");
    }

    #[test]
    fn gate_classification_matrix() {
        use crate::agent::approval::{permission_gate, PermissionGate};
        use crate::traits::{PathAccess, PathAccessKind};
        let inside = PathAccess {
            path: PathBuf::from("/ws/a"),
            kind: PathAccessKind::Read,
            inside_sandbox: true,
        };
        let outside = PathAccess {
            path: PathBuf::from("/etc/hostname"),
            kind: PathAccessKind::Read,
            inside_sandbox: false,
        };
        let p = ResolvedToolPermissions::default();
        assert!(matches!(
            permission_gate(
                &p,
                "read_file",
                Some(std::slice::from_ref(&inside)),
                true,
                false
            ),
            PermissionGate::Location {
                mode: ShellPolicyMode::Allow,
                ..
            }
        ));
        // The outside path decides, and is the one shown to the user.
        match permission_gate(
            &p,
            "glob_files",
            Some(&[inside.clone(), outside.clone()]),
            true,
            false,
        ) {
            PermissionGate::Location { mode, access, .. } => {
                assert_eq!(mode, ShellPolicyMode::Ask);
                assert_eq!(access, outside);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            permission_gate(&p, "ask_user", Some(&[]), false, false),
            PermissionGate::Exempt
        );
        assert_eq!(
            permission_gate(&p, "execution_run", Some(&[]), false, false),
            PermissionGate::Exempt
        );
        assert_eq!(
            permission_gate(&p, "read_file", None, true, false),
            PermissionGate::Exempt
        );
        assert_eq!(
            permission_gate(&p, "web_fetch", Some(&[]), false, false),
            PermissionGate::Default(ShellPolicyMode::Allow)
        );
        // A path-reporting tool with no paths never falls to `default`.
        assert_eq!(
            permission_gate(&p, "glob_files", Some(&[]), true, false),
            PermissionGate::UnreportedPaths {
                mode: ShellPolicyMode::Ask,
                ceiling_denied: false
            }
        );
        assert_eq!(
            permission_gate(&p, "glob_files", Some(&[]), true, true),
            PermissionGate::UnreportedPaths {
                mode: ShellPolicyMode::Deny,
                ceiling_denied: false
            }
        );
        let plan = perms("[harness.shell_policy]\nedit_mode = \"deny\"\n[harness.permissions]\noutside_workspace = \"allow\"\n");
        assert_eq!(
            permission_gate(&plan, "write_file", Some(&[]), true, false),
            PermissionGate::UnreportedPaths {
                mode: ShellPolicyMode::Deny,
                ceiling_denied: true
            }
        );
        // A code-exec entry is not a rule override: the working directory is location-gated.
        let exec_allow = perms("[harness.permissions.tools]\nexec = \"allow\"\n");
        assert!(matches!(
            permission_gate(
                &exec_allow,
                "exec",
                Some(std::slice::from_ref(&outside)),
                true,
                false
            ),
            PermissionGate::Location {
                mode: ShellPolicyMode::Ask,
                ..
            }
        ));
        // Edit entries: the ceiling is reported separately from the entry's own deny.
        let capped = perms("[harness.shell_policy]\nedit_mode = \"deny\"\n[harness.permissions.tools]\nwrite_file = \"allow\"\n");
        assert_eq!(
            permission_gate(&capped, "write_file", None, true, false),
            PermissionGate::ToolOverride {
                mode: ShellPolicyMode::Deny,
                ceiling_denied: true
            }
        );
        let entry_deny = perms("[harness.shell_policy]\nedit_mode = \"deny\"\n[harness.permissions.tools]\nwrite_file = \"deny\"\n");
        assert_eq!(
            permission_gate(&entry_deny, "write_file", None, true, false),
            PermissionGate::ToolOverride {
                mode: ShellPolicyMode::Deny,
                ceiling_denied: false
            }
        );
        let entry_ask = perms("[harness.permissions.tools]\nwrite_file = \"ask\"\n");
        assert_eq!(
            permission_gate(&entry_ask, "write_file", None, true, true),
            PermissionGate::ToolOverride {
                mode: ShellPolicyMode::Deny,
                ceiling_denied: false
            }
        );
    }
}
