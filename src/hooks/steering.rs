use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::config::{HarnessHookCommandConfig, HarnessHooksSteeringConfig};
use crate::utils::{join_lexically_under_root, normalize_sandbox_relative_input};

const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_MAX_STDOUT: usize = 64 * 1024;
/// Minimum wall-clock budget for a steering hook subprocess (config may not go below this).
const MIN_HOOK_TIMEOUT_MS: u64 = 1_000;
/// Version of the stdin payload. v2 adds `post_model`, `prompt` on `user_prompt`, and the
/// structured (`hookSpecificOutput`) stdout contract; every v1 field is still sent.
const HOOK_SCHEMA_VERSION: u32 = 2;
/// Exit code that means "block, and show stderr to the model" (same convention as Claude Code).
const BLOCKING_EXIT_CODE: i32 = 2;

#[inline]
fn hook_command_timeout_ms(timeout_ms: u64) -> u64 {
    timeout_ms.max(MIN_HOOK_TIMEOUT_MS)
}

#[derive(Debug, Clone)]
pub struct HookHandlerResolved {
    pub matcher: Option<Regex>,
    pub command: String,
    pub timeout_ms: u64,
    pub cwd_relative: Option<String>,
    /// `on_error = "block"`: a failing hook blocks instead of being skipped.
    pub fail_closed: bool,
    /// Where the hook was declared: `"config"` or `"plugin:<name>"` (reported in audit events).
    pub source: Arc<str>,
    /// Extra environment for the hook process (plugin hooks get `ISANAGENT_PLUGIN_ROOT` /
    /// `CLAUDE_PLUGIN_ROOT`), added on top of the scrubbed safe environment.
    pub env: Arc<Vec<(String, String)>>,
}

/// One steering-hook decision that changed something (blocked, rewrote, asked, stopped, …) or a
/// hook failure. Emitted to the observation pipeline as `TelemetryEvent::HookDecision` so
/// operators get an audit trail of what hooks did.
#[derive(Debug, Clone, PartialEq)]
pub struct HookAuditRecord {
    /// `pre_tool`, `post_tool`, `user_prompt`, `post_model`.
    pub hook_event: &'static str,
    /// `"config"` or `"plugin:<name>"`.
    pub source: String,
    pub command: String,
    pub tool_name: Option<String>,
    /// `block`, `stop`, `ask`, `modify`, `retry`, `error_blocked`, `error_ignored`.
    pub decision: &'static str,
    pub reason: Option<String>,
    /// What a `modify` changed: `args`, `prompt`, `output`, `content`, `tool_calls`, `context`.
    pub changed: Vec<&'static str>,
}

/// Receives hook audit records (see [`HookAuditRecord`]).
pub trait HookAuditSink: Send + Sync {
    fn record(&self, session: &HookSessionInfo<'_>, record: HookAuditRecord);
}

const AUDIT_REASON_MAX: usize = 2_000;

/// Per-turn fields passed to steering hook stdin (channel, identity, metadata).
pub struct HookSessionInfo<'a> {
    pub channel: &'a str,
    pub chat_id: &'a str,
    pub thread_id: Option<&'a str>,
    pub metadata: &'a HashMap<String, Value>,
    pub is_subagent: bool,
}

#[derive(Clone)]
pub struct SteeringHooksEngine {
    pub pre_tool: Arc<Vec<HookHandlerResolved>>,
    pub post_tool: Arc<Vec<HookHandlerResolved>>,
    pub user_prompt: Arc<Vec<HookHandlerResolved>>,
    pub post_model: Arc<Vec<HookHandlerResolved>>,
    pub max_stdout_bytes: usize,
    pub default_timeout_ms: u64,
    pub workspace_dir: PathBuf,
    pub sandbox_dir: PathBuf,
    /// Audit trail for hook decisions (wired to observation hooks when they are enabled).
    pub audit: Option<Arc<dyn HookAuditSink>>,
}

// ---------------------------------------------------------------------------------------------
// Outcomes
// ---------------------------------------------------------------------------------------------

/// Combined result of every matching `pre_tool` hook for one call.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PreToolDecision {
    /// Arguments to execute with (possibly rewritten by `updatedInput` / legacy `args`).
    pub args: Value,
    /// True when at least one hook replaced the arguments.
    pub args_modified: bool,
    /// Set when a hook asked for user approval (`permissionDecision: "ask"`); holds its reason.
    /// The dispatcher forces the approval prompt even when the policy would allow the call.
    pub ask_reason: Option<String>,
    /// `additionalContext` strings, appended to the tool result the model sees.
    pub additional_context: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreToolOutcome {
    Proceed(PreToolDecision),
    /// Skip the tool; the message is returned to the model as the tool error.
    Block(String),
    /// `continue: false` — abort the whole run; the reason is shown to the user.
    Stop(String),
}

/// Combined result of every matching `user_prompt` hook.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UserPromptDecision {
    /// Rewritten user prompt (`updatedPrompt`); `None` keeps the original text.
    pub updated_prompt: Option<String>,
    /// Text to prepend to the contextualized message (`inject_prefix` / `additionalContext`).
    pub prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UserPromptHookOutcome {
    Proceed(UserPromptDecision),
    Block(String),
}

/// Combined result of every matching `post_tool` hook.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PostToolDecision {
    /// Plain-text verification output and `additionalContext`, appended to the result.
    pub notes: Vec<String>,
    /// Replacement for the tool output the model sees (`updatedOutput`).
    pub updated_output: Option<String>,
    /// `decision: "block"` — the result is marked as an error carrying this reason.
    pub block_reason: Option<String>,
    /// `continue: false` — abort the run after this tool.
    pub stop_reason: Option<String>,
}

/// A tool call as seen / produced by a `post_model` hook.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct HookToolCall {
    #[serde(default)]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PostModelDecision {
    /// Replacement assistant text (`updatedContent`).
    pub updated_content: Option<String>,
    /// Replacement tool-call list (`updatedToolCalls`); an empty list removes all calls.
    pub updated_tool_calls: Option<Vec<HookToolCall>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PostModelOutcome {
    Proceed(PostModelDecision),
    /// `decision: "block"` — discard this response and send the reason back to the model as
    /// feedback so it produces a new one.
    Retry(String),
    /// `continue: false` — end the turn with this reason.
    Stop(String),
}

// ---------------------------------------------------------------------------------------------
// stdout contract
// ---------------------------------------------------------------------------------------------

/// Hook stdout. Accepts both the v1 shape (`decision` / `message` / `args`) and the v2 shape
/// modelled on Claude Code hooks (`continue`, `stopReason`, `hookSpecificOutput`). camelCase and
/// snake_case spellings are both accepted.
#[derive(Debug, Default, Deserialize)]
struct HookStdoutEnvelope {
    decision: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    args: Option<Value>,
    #[serde(default, rename = "continue")]
    continue_: Option<bool>,
    #[serde(default, rename = "stopReason", alias = "stop_reason")]
    stop_reason: Option<String>,
    #[serde(default, rename = "systemMessage", alias = "system_message")]
    system_message: Option<String>,
    #[serde(default, rename = "hookSpecificOutput", alias = "hook_specific_output")]
    specific: Option<HookSpecificOutput>,
}

#[derive(Debug, Default, Deserialize)]
struct HookSpecificOutput {
    #[serde(default, rename = "permissionDecision", alias = "permission_decision")]
    permission_decision: Option<String>,
    #[serde(
        default,
        rename = "permissionDecisionReason",
        alias = "permission_decision_reason"
    )]
    permission_decision_reason: Option<String>,
    #[serde(default, rename = "updatedInput", alias = "updated_input")]
    updated_input: Option<Value>,
    #[serde(default, rename = "additionalContext", alias = "additional_context")]
    additional_context: Option<String>,
    #[serde(default, rename = "updatedPrompt", alias = "updated_prompt")]
    updated_prompt: Option<String>,
    #[serde(default, rename = "updatedOutput", alias = "updated_output")]
    updated_output: Option<Value>,
    #[serde(default, rename = "updatedContent", alias = "updated_content")]
    updated_content: Option<String>,
    #[serde(default, rename = "updatedToolCalls", alias = "updated_tool_calls")]
    updated_tool_calls: Option<Vec<HookToolCall>>,
}

impl HookStdoutEnvelope {
    fn decision_lc(&self) -> Option<String> {
        self.decision
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
    }

    /// `continue: false` → the reason to stop with.
    fn stop(&self) -> Option<String> {
        (self.continue_ == Some(false)).then(|| {
            self.stop_reason
                .clone()
                .or_else(|| self.reason.clone())
                .or_else(|| self.message.clone())
                .unwrap_or_else(|| "stopped by hook".to_string())
        })
    }

    fn block_message(&self, fallback: &str) -> String {
        self.specific
            .as_ref()
            .and_then(|s| s.permission_decision_reason.clone())
            .or_else(|| self.reason.clone())
            .or_else(|| self.message.clone())
            .unwrap_or_else(|| fallback.to_string())
    }

    fn additional_context(&self) -> Option<String> {
        self.specific
            .as_ref()
            .and_then(|s| s.additional_context.clone())
            .filter(|s| !s.trim().is_empty())
    }

    fn log_system_message(&self, event: &str) {
        if let Some(m) = self
            .system_message
            .as_deref()
            .filter(|m| !m.trim().is_empty())
        {
            log::info!("{event} hook systemMessage: {m}");
        }
    }
}

fn parse_envelope(text: &str) -> Result<HookStdoutEnvelope, String> {
    serde_json::from_str(text).map_err(|e| format!("hook json: {e}"))
}

fn matcher_matches(matcher: &Option<Regex>, text: &str) -> bool {
    match matcher {
        None => true,
        Some(re) => re.is_match(text),
    }
}

/// `sh -c <command>` on Unix; on Windows `cmd /D /S /C "<command>"` with the command appended
/// verbatim so inner double quotes survive (see [`crate::windows_shell`]).
fn shell_command(command: &str) -> Command {
    if cfg!(windows) {
        crate::windows_shell::windows_shell_invocation(
            crate::config::WindowsShellRunner::Cmd,
            command,
        )
        .to_tokio_command()
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    }
}

/// Read a child pipe to EOF, keeping the first `max` bytes (truncating rather than erroring on
/// overflow) so a verbose hook still yields a usable prefix, and lossily decoding so non-UTF-8 output (common from
/// compilers/test runners) doesn't drop the whole capture. Generic over the pipe type so the same
/// reader serves both stdout and stderr.
async fn read_bounded<R: tokio::io::AsyncRead + Unpin>(
    mut reader: R,
    max: usize,
) -> Result<String, String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = reader
            .read(&mut chunk)
            .await
            .map_err(|e| format!("read hook output: {e}"))?;
        if n == 0 {
            break;
        }
        // Keep draining past `max` (discarding the rest) so a verbose hook never blocks on a full
        // pipe — or dies of SIGPIPE — before it exits with its real status.
        let remaining = max.saturating_sub(buf.len());
        buf.extend_from_slice(&chunk[..n.min(remaining)]);
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Raw result of one hook subprocess.
#[derive(Debug)]
struct HookProcessOutput {
    /// `None` when the process was killed by a signal.
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl HookProcessOutput {
    fn success(&self) -> bool {
        self.exit_code == Some(0)
    }
}

/// Spawn a hook, pipe the JSON event to its stdin and capture stdout and stderr (both bounded).
///
/// Both pipes are drained concurrently with `wait()` under one timeout, so a hook that fills a pipe
/// buffer can't deadlock.
async fn spawn_hook(
    command: &str,
    cwd: &Path,
    env: &[(String, String)],
    stdin_json: &Value,
    timeout_ms: u64,
    max_stdout: usize,
) -> Result<HookProcessOutput, String> {
    // Hook commands come from workspace/plugin config (third-party code): run them in the
    // scrubbed safe environment so host master credentials never leak into hook processes.
    let mut hook_cmd = shell_command(command);
    hook_cmd
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    crate::environment::ExecutionEnvironmentPolicy::default_safe()
        .apply_to_tokio_command(&mut hook_cmd);
    for (k, v) in env {
        hook_cmd.env(k, v);
    }
    let mut child = hook_cmd.spawn().map_err(|e| format!("spawn hook: {e}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "hook stdin missing".to_string())?;
    let body = serde_json::to_vec(stdin_json).map_err(|e| format!("hook stdin encode: {e}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "hook stdout missing".to_string())?;
    let stderr = child.stderr.take();

    // The stdin write runs concurrently with draining stdout/stderr and `wait()`, all under one
    // timeout: a hook that ignores a large payload while filling its own output pipe can then
    // neither deadlock us nor outlive the timeout.
    //
    // A hook that ignores its stdin and exits (a bare `cargo build` / `pytest`, or any verify hook
    // that reads the event from argv/env instead) closes its stdin read end before — or while — we
    // write. On Linux that surfaces as `BrokenPipe`; on macOS the small event JSON usually fits the
    // pipe buffer. The hook simply didn't consume the event, which is fine — swallow BrokenPipe and
    // keep its output and exit status.
    let write_fut = async move {
        let res = tokio::io::AsyncWriteExt::write_all(&mut stdin, &body).await;
        drop(stdin);
        match res {
            Err(e) if e.kind() != std::io::ErrorKind::BrokenPipe => {
                Err(format!("hook stdin write: {e}"))
            }
            _ => Ok(()),
        }
    };
    let io = async {
        let out_fut = read_bounded(stdout, max_stdout);
        let err_fut = async {
            match stderr {
                Some(se) => read_bounded(se, max_stdout).await.unwrap_or_default(),
                None => String::new(),
            }
        };
        tokio::join!(write_fut, child.wait(), out_fut, err_fut)
    };
    let (written, status, out, err) = tokio::time::timeout(
        std::time::Duration::from_millis(hook_command_timeout_ms(timeout_ms)),
        io,
    )
    .await
    .map_err(|_| "hook timed out".to_string())?;
    written?;

    let status = status.map_err(|e| format!("hook wait: {e}"))?;
    Ok(HookProcessOutput {
        exit_code: status.code(),
        stdout: out?,
        stderr: err,
    })
}

/// Legacy text contract (see [`legacy_text`]); kept as a test seam.
///
/// When `capture_failure` is true (post_tool *verification* hooks), stderr is joined to stdout and
/// the output is returned **even on a non-zero exit** (prefixed with the exit status) — a failing
/// `cargo build` / `pytest` is exactly what must reach the model. When false, stderr is dropped
/// and a non-zero exit yields `Ok(None)`.
#[cfg(test)]
async fn run_hook_command(
    command: &str,
    cwd: &Path,
    stdin_json: &Value,
    timeout_ms: u64,
    max_stdout: usize,
    capture_failure: bool,
) -> Result<Option<String>, String> {
    let out = spawn_hook(command, cwd, &[], stdin_json, timeout_ms, max_stdout).await?;
    Ok(legacy_text(&out, capture_failure))
}

fn legacy_text(out: &HookProcessOutput, capture_failure: bool) -> Option<String> {
    let mut combined = out.stdout.clone();
    if capture_failure && !out.stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&out.stderr);
    }
    let combined = combined.trim();

    if !out.success() {
        if !capture_failure {
            log::warn!(
                "hook command exited with {} (proceeding)",
                out.exit_code.unwrap_or(-1)
            );
            return None;
        }
        let code = out
            .exit_code
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".to_string());
        return Some(if combined.is_empty() {
            format!("[hook exited {code} with no output]")
        } else {
            format!("[hook exited {code}]\n{combined}")
        });
    }
    (!combined.is_empty()).then(|| combined.to_string())
}

/// How a directive hook (pre_tool / user_prompt / post_model) responded.
enum Directive {
    /// Exit 0 with empty stdout: no opinion.
    Nothing,
    Parsed(Box<HookStdoutEnvelope>),
    /// Exit code 2: block, with stderr (or stdout) as the reason.
    Blocking(String),
    /// The hook itself failed; handled per `on_error`.
    Failed(String),
}

async fn run_directive_hook(
    engine: &SteeringHooksEngine,
    h: &HookHandlerResolved,
    event: &str,
    stdin: &Value,
) -> Directive {
    let cwd = match hook_cwd(&engine.sandbox_dir, h.cwd_relative.as_deref()) {
        Ok(p) => p,
        Err(e) => return Directive::Failed(format!("cwd: {e}")),
    };
    let out = match spawn_hook(
        &h.command,
        &cwd,
        &h.env,
        stdin,
        hook_command_timeout_ms(h.timeout_ms),
        engine.max_stdout_bytes,
    )
    .await
    {
        Ok(o) => o,
        Err(e) => return Directive::Failed(e),
    };
    match out.exit_code {
        Some(0) => {
            let text = out.stdout.trim();
            if text.is_empty() {
                return Directive::Nothing;
            }
            match parse_envelope(text) {
                Ok(env) => {
                    env.log_system_message(event);
                    Directive::Parsed(Box::new(env))
                }
                Err(e) => Directive::Failed(e),
            }
        }
        Some(BLOCKING_EXIT_CODE) => {
            let reason = [out.stderr.trim(), out.stdout.trim()]
                .into_iter()
                .find(|s| !s.is_empty())
                .unwrap_or("blocked by hook")
                .to_string();
            Directive::Blocking(reason)
        }
        other => {
            let code = other
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string());
            let detail = out.stderr.trim();
            Directive::Failed(if detail.is_empty() {
                format!("exited {code}")
            } else {
                format!("exited {code}: {detail}")
            })
        }
    }
}

fn hook_cwd(sandbox_dir: &Path, cwd_relative: Option<&str>) -> Result<PathBuf, String> {
    let rel_str = cwd_relative.unwrap_or(".").trim();
    if rel_str.is_empty() || rel_str == "." {
        return Ok(sandbox_dir.to_path_buf());
    }
    let root = sandbox_dir.to_path_buf();
    let rel_pb = normalize_sandbox_relative_input(&root, rel_str);
    let raw = rel_pb.as_path();
    if raw.is_absolute() {
        return Err("hook cwd must be sandbox-relative".to_string());
    }
    join_lexically_under_root(&root, raw)
}

fn base_payload(engine: &SteeringHooksEngine, event: &str, session: &HookSessionInfo<'_>) -> Value {
    json!({
        "hook_event": event,
        "schema_version": HOOK_SCHEMA_VERSION,
        "channel": session.channel,
        "chat_id": session.chat_id,
        "thread_id": session.thread_id,
        "is_subagent": session.is_subagent,
        "metadata": session.metadata,
        "workspace_dir": engine.workspace_dir.to_string_lossy(),
        "sandbox_dir": engine.sandbox_dir.to_string_lossy(),
    })
}

fn extend(mut base: Value, extra: Value) -> Value {
    if let (Some(b), Value::Object(e)) = (base.as_object_mut(), extra) {
        b.extend(e);
    }
    base
}

// ---------------------------------------------------------------------------------------------
// pre_tool
// ---------------------------------------------------------------------------------------------

fn lc(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}

fn push_reason(slot: &mut Option<String>, reason: String) {
    *slot = Some(match slot.take() {
        Some(prev) => format!("{prev}; {reason}"),
        None => reason,
    });
}

/// Context for one hook invocation, used to report failures and decisions to the audit trail.
struct HookCall<'e, 's, 'a> {
    engine: &'e SteeringHooksEngine,
    session: &'s HookSessionInfo<'a>,
    h: &'e HookHandlerResolved,
    event: &'static str,
    tool_name: Option<&'s str>,
}

impl HookCall<'_, '_, '_> {
    fn audit(&self, decision: &'static str, reason: Option<&str>, changed: Vec<&'static str>) {
        let reason = reason.map(|r| {
            let mut r = r.to_string();
            crate::utils::truncate_utf8_safe(&mut r, AUDIT_REASON_MAX, "…");
            r
        });
        log::info!(
            "hook {} [{}] decision={decision}{}{}",
            self.event,
            self.h.source,
            self.tool_name
                .map(|t| format!(" tool={t}"))
                .unwrap_or_default(),
            if changed.is_empty() {
                String::new()
            } else {
                format!(" changed={}", changed.join(","))
            }
        );
        if let Some(sink) = &self.engine.audit {
            sink.record(
                self.session,
                HookAuditRecord {
                    hook_event: self.event,
                    source: self.h.source.to_string(),
                    command: self.h.command.clone(),
                    tool_name: self.tool_name.map(str::to_string),
                    decision,
                    reason,
                    changed,
                },
            );
        }
    }

    /// Apply `on_error` to a failed hook: `Some(message)` means block, `None` means skip it.
    fn failure(&self, err: &str) -> Option<String> {
        let event = self.event;
        if self.h.fail_closed {
            log::warn!("{event} hook failed (on_error=block): {err}");
            self.audit("error_blocked", Some(err), vec![]);
            Some(format!("{event} hook failed and on_error=block: {err}"))
        } else {
            log::warn!("{event} hook failed (proceeding): {err}");
            self.audit("error_ignored", Some(err), vec![]);
            None
        }
    }
}

/// Fold one `pre_tool` hook response into `decision`. `Ok(Some(_))` short-circuits (block/stop).
/// Nothing is applied when an error is returned.
fn apply_pre_tool_envelope(
    env: HookStdoutEnvelope,
    decision: &mut PreToolDecision,
) -> Result<Option<PreToolOutcome>, String> {
    if let Some(reason) = env.stop() {
        return Ok(Some(PreToolOutcome::Stop(reason)));
    }
    let specific = env.specific.as_ref();
    let updated_input = specific
        .and_then(|s| s.updated_input.clone())
        .or_else(|| env.args.clone())
        .filter(|v| !v.is_null());

    let mut ask = false;
    match env.decision_lc().as_deref() {
        Some("block") | Some("deny") => {
            return Ok(Some(PreToolOutcome::Block(
                env.block_message("blocked by hook"),
            )))
        }
        Some("modify") if updated_input.is_none() => {
            return Err("modify decision requires args".to_string())
        }
        Some("ask") => ask = true,
        Some("proceed") | Some("allow") | Some("modify") | None => {}
        Some(other) => return Err(format!("unknown decision {other:?}")),
    }
    match specific
        .and_then(|s| s.permission_decision.as_deref())
        .map(lc)
        .as_deref()
    {
        Some("deny") => {
            return Ok(Some(PreToolOutcome::Block(
                env.block_message("denied by hook"),
            )))
        }
        Some("ask") => ask = true,
        // `allow` never bypasses isanagent's own shell / edit policy: hooks can tighten
        // permissions, not loosen them.
        Some("allow") | None => {}
        Some(other) => return Err(format!("unknown permissionDecision {other:?}")),
    }
    if let Some(a) = &updated_input {
        if !a.is_object() {
            return Err("updatedInput / args must be a JSON object".to_string());
        }
    }

    if ask {
        push_reason(
            &mut decision.ask_reason,
            env.block_message("a pre_tool hook requested approval"),
        );
    }
    if let Some(a) = updated_input {
        if a != decision.args {
            decision.args = a;
            decision.args_modified = true;
        }
    }
    if let Some(ctx) = env.additional_context() {
        decision.additional_context.push(ctx);
    }
    Ok(None)
}

/// Run all matching `pre_tool` hooks in order. Each hook sees the arguments as rewritten by the
/// hooks before it; the first `block` / `deny` / `continue: false` wins.
pub async fn run_pre_tool_hooks(
    engine: &SteeringHooksEngine,
    tool_name: &str,
    tool_call_id: Option<&str>,
    args: Value,
    session: HookSessionInfo<'_>,
) -> PreToolOutcome {
    let mut decision = PreToolDecision {
        args,
        ..Default::default()
    };
    for h in engine.pre_tool.iter() {
        if !matcher_matches(&h.matcher, tool_name) {
            continue;
        }
        let call = HookCall {
            engine,
            session: &session,
            h,
            event: "pre_tool",
            tool_name: Some(tool_name),
        };
        let stdin = extend(
            base_payload(engine, "pre_tool", &session),
            json!({
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "args": decision.args,
            }),
        );
        let env = match run_directive_hook(engine, h, "pre_tool", &stdin).await {
            Directive::Nothing => continue,
            Directive::Blocking(m) => {
                call.audit("block", Some(&m), vec![]);
                return PreToolOutcome::Block(m);
            }
            Directive::Failed(e) => match call.failure(&e) {
                Some(m) => return PreToolOutcome::Block(m),
                None => continue,
            },
            Directive::Parsed(env) => *env,
        };
        let before_args = decision.args.clone();
        let before_ask = decision.ask_reason.clone();
        let before_ctx = decision.additional_context.len();
        match apply_pre_tool_envelope(env, &mut decision) {
            Ok(None) => {
                if decision.ask_reason != before_ask {
                    call.audit("ask", decision.ask_reason.as_deref(), vec![]);
                }
                let mut changed = Vec::new();
                if decision.args != before_args {
                    changed.push("args");
                }
                if decision.additional_context.len() > before_ctx {
                    changed.push("context");
                }
                if !changed.is_empty() {
                    call.audit("modify", None, changed);
                }
            }
            Ok(Some(outcome)) => {
                match &outcome {
                    PreToolOutcome::Block(m) => call.audit("block", Some(m), vec![]),
                    PreToolOutcome::Stop(m) => call.audit("stop", Some(m), vec![]),
                    PreToolOutcome::Proceed(_) => {}
                }
                return outcome;
            }
            Err(e) => {
                if let Some(m) = call.failure(&e) {
                    return PreToolOutcome::Block(m);
                }
            }
        }
    }
    PreToolOutcome::Proceed(decision)
}

// ---------------------------------------------------------------------------------------------
// post_tool
// ---------------------------------------------------------------------------------------------

const STRUCTURED_KEYS: &[&str] = &[
    "decision",
    "continue",
    "stopReason",
    "stop_reason",
    "systemMessage",
    "system_message",
    "hookSpecificOutput",
    "hook_specific_output",
];

/// A post_tool hook's stdout is a *directive* only when it is a JSON object carrying at least one
/// contract key; anything else (compiler / test output, arbitrary JSON) stays verification text.
fn structured_post_tool_stdout(stdout: &str) -> Option<HookStdoutEnvelope> {
    let text = stdout.trim();
    if !text.starts_with('{') {
        return None;
    }
    let v: Value = serde_json::from_str(text).ok()?;
    let obj = v.as_object()?;
    if !STRUCTURED_KEYS.iter().any(|k| obj.contains_key(*k)) {
        return None;
    }
    serde_json::from_value(v).ok()
}

/// Run the configured post_tool hooks.
///
/// * Plain output (including failures, with the exit status) is collected into `notes` so the
///   caller can append it to the tool result — this closes the verify-into-fix loop for
///   `cargo build` / `pytest` / lint hooks.
/// * A structured JSON response can replace the output the model sees (`updatedOutput`, e.g. to
///   redact secrets), mark the result as failed (`decision: "block"`), add context, or stop the
///   run (`continue: false`). Later hooks see the output as rewritten by earlier ones.
pub async fn run_post_tool_hooks(
    engine: &SteeringHooksEngine,
    tool_name: &str,
    tool_call_id: Option<&str>,
    args: &Value,
    result: &Result<String, String>,
    session: HookSessionInfo<'_>,
) -> PostToolDecision {
    let mut d = PostToolDecision::default();
    let mut current = result.clone();
    for h in engine.post_tool.iter() {
        if !matcher_matches(&h.matcher, tool_name) {
            continue;
        }
        let call = HookCall {
            engine,
            session: &session,
            h,
            event: "post_tool",
            tool_name: Some(tool_name),
        };
        let cwd = match hook_cwd(&engine.sandbox_dir, h.cwd_relative.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("post_tool hook cwd: {e}");
                call.audit("error_ignored", Some(&format!("cwd: {e}")), vec![]);
                continue;
            }
        };
        let stdin = extend(
            base_payload(engine, "post_tool", &session),
            json!({
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "args": args,
                "result_ok": current.is_ok(),
                "result": current.as_ref().ok(),
                "error": current.as_ref().err(),
            }),
        );
        let out = match spawn_hook(
            &h.command,
            &cwd,
            &h.env,
            &stdin,
            hook_command_timeout_ms(h.timeout_ms),
            engine.max_stdout_bytes,
        )
        .await
        {
            Ok(o) => o,
            Err(e) => {
                log::warn!("post_tool hook failed: {e}");
                call.audit("error_ignored", Some(&e), vec![]);
                continue;
            }
        };
        if out.success() {
            if let Some(env) = structured_post_tool_stdout(&out.stdout) {
                env.log_system_message("post_tool");
                if let Some(reason) = env.stop() {
                    call.audit("stop", Some(&reason), vec![]);
                    d.stop_reason = Some(reason);
                    break;
                }
                if matches!(env.decision_lc().as_deref(), Some("block") | Some("deny")) {
                    let reason = env.block_message("rejected by post_tool hook");
                    call.audit("block", Some(&reason), vec![]);
                    d.block_reason = Some(reason);
                }
                let mut changed = Vec::new();
                if let Some(v) = env.specific.as_ref().and_then(|s| s.updated_output.clone()) {
                    let text = match v {
                        Value::String(s) => s,
                        other => other.to_string(),
                    };
                    current = match current {
                        Ok(_) => Ok(text.clone()),
                        Err(_) => Err(text.clone()),
                    };
                    d.updated_output = Some(text);
                    changed.push("output");
                }
                if let Some(ctx) = env.additional_context() {
                    d.notes.push(ctx);
                    changed.push("context");
                }
                if !changed.is_empty() {
                    call.audit("modify", None, changed);
                }
                continue;
            }
        }
        if let Some(text) = legacy_text(&out, true) {
            d.notes.push(text);
        }
    }
    d
}

// ---------------------------------------------------------------------------------------------
// user_prompt
// ---------------------------------------------------------------------------------------------

/// Run all matching `user_prompt` hooks. `prompt` is the raw user text; `runtime_context` is the
/// prefix the agent prepends before storing it (sent to hooks as the v1 `content` field). The
/// matcher tests the (possibly already rewritten) prompt. Hooks can rewrite the prompt
/// (`updatedPrompt`), add context (`inject_prefix` / `additionalContext`) or block the turn.
pub async fn run_user_prompt_hooks(
    engine: &SteeringHooksEngine,
    runtime_context: &str,
    prompt: &str,
    session: HookSessionInfo<'_>,
) -> UserPromptHookOutcome {
    let mut current = prompt.to_string();
    let mut updated = false;
    let mut prefixes: Vec<String> = Vec::new();
    for h in engine.user_prompt.iter() {
        if !matcher_matches(&h.matcher, &current) {
            continue;
        }
        let call = HookCall {
            engine,
            session: &session,
            h,
            event: "user_prompt",
            tool_name: None,
        };
        let stdin = extend(
            base_payload(engine, "user_prompt", &session),
            json!({
                "prompt": current,
                "content": format!("{runtime_context}{current}"),
            }),
        );
        let env = match run_directive_hook(engine, h, "user_prompt", &stdin).await {
            Directive::Nothing => continue,
            Directive::Blocking(m) => {
                call.audit("block", Some(&m), vec![]);
                return UserPromptHookOutcome::Block(m);
            }
            Directive::Failed(e) => match call.failure(&e) {
                Some(m) => return UserPromptHookOutcome::Block(m),
                None => continue,
            },
            Directive::Parsed(env) => *env,
        };
        if let Some(reason) = env.stop() {
            call.audit("stop", Some(&reason), vec![]);
            return UserPromptHookOutcome::Block(reason);
        }
        let parsed: Result<Option<String>, String> = match env.decision_lc().as_deref() {
            Some("block") | Some("deny") => {
                let reason = env.block_message("blocked by hook");
                call.audit("block", Some(&reason), vec![]);
                return UserPromptHookOutcome::Block(reason);
            }
            Some("inject_prefix") => env
                .message
                .clone()
                .map(Some)
                .ok_or_else(|| "inject_prefix requires message".to_string()),
            Some("proceed") | Some("allow") | None => Ok(None),
            Some(other) => Err(format!("unknown decision {other:?}")),
        };
        match parsed {
            Ok(prefix) => {
                let mut changed = Vec::new();
                let before_prefixes = prefixes.len();
                prefixes.extend(prefix);
                if let Some(p) = env.specific.as_ref().and_then(|s| s.updated_prompt.clone()) {
                    if p != current {
                        changed.push("prompt");
                    }
                    current = p;
                    updated = true;
                }
                prefixes.extend(env.additional_context());
                if prefixes.len() > before_prefixes {
                    changed.push("context");
                }
                if !changed.is_empty() {
                    call.audit("modify", None, changed);
                }
            }
            Err(e) => {
                if let Some(m) = call.failure(&e) {
                    return UserPromptHookOutcome::Block(m);
                }
            }
        }
    }
    UserPromptHookOutcome::Proceed(UserPromptDecision {
        updated_prompt: updated.then_some(current),
        prefix: (!prefixes.is_empty()).then(|| prefixes.join("\n")),
    })
}

// ---------------------------------------------------------------------------------------------
// post_model
// ---------------------------------------------------------------------------------------------

/// Run all matching `post_model` hooks on a provider response, before it is persisted or any of
/// its tool calls run. The matcher tests the assistant text. Hooks see the response as rewritten
/// by the hooks before them.
pub async fn run_post_model_hooks(
    engine: &SteeringHooksEngine,
    content: &str,
    tool_calls: &[HookToolCall],
    iteration: u32,
    session: HookSessionInfo<'_>,
) -> PostModelOutcome {
    let mut d = PostModelDecision::default();
    for h in engine.post_model.iter() {
        let cur_content = d.updated_content.as_deref().unwrap_or(content);
        if !matcher_matches(&h.matcher, cur_content) {
            continue;
        }
        let call = HookCall {
            engine,
            session: &session,
            h,
            event: "post_model",
            tool_name: None,
        };
        let cur_calls = d.updated_tool_calls.as_deref().unwrap_or(tool_calls);
        let calls_json: Vec<Value> = cur_calls
            .iter()
            .map(|c| json!({"id": c.id, "name": c.name, "arguments": c.arguments}))
            .collect();
        let stdin = extend(
            base_payload(engine, "post_model", &session),
            json!({
                "content": cur_content,
                "tool_calls": calls_json,
                "iteration": iteration,
            }),
        );
        let env = match run_directive_hook(engine, h, "post_model", &stdin).await {
            Directive::Nothing => continue,
            Directive::Blocking(m) => {
                call.audit("retry", Some(&m), vec![]);
                return PostModelOutcome::Retry(m);
            }
            Directive::Failed(e) => match call.failure(&e) {
                Some(m) => return PostModelOutcome::Stop(m),
                None => continue,
            },
            Directive::Parsed(env) => *env,
        };
        if let Some(reason) = env.stop() {
            call.audit("stop", Some(&reason), vec![]);
            return PostModelOutcome::Stop(reason);
        }
        match env.decision_lc().as_deref() {
            Some("block") | Some("deny") => {
                let reason = env.block_message("response rejected by hook");
                call.audit("retry", Some(&reason), vec![]);
                return PostModelOutcome::Retry(reason);
            }
            Some("proceed") | Some("allow") | None => {}
            Some(other) => {
                if let Some(m) = call.failure(&format!("unknown decision {other:?}")) {
                    return PostModelOutcome::Stop(m);
                }
                continue;
            }
        }
        if let Some(spec) = env.specific {
            let mut changed = Vec::new();
            if let Some(c) = spec.updated_content {
                d.updated_content = Some(c);
                changed.push("content");
            }
            if let Some(calls) = spec.updated_tool_calls {
                d.updated_tool_calls = Some(calls);
                changed.push("tool_calls");
            }
            if !changed.is_empty() {
                call.audit("modify", None, changed);
            }
        }
    }
    PostModelOutcome::Proceed(d)
}

// ---------------------------------------------------------------------------------------------
// Engine construction
// ---------------------------------------------------------------------------------------------

fn compile_hook_handlers(
    raw: &[HarnessHookCommandConfig],
    default_timeout_ms: u64,
    source: &Arc<str>,
    env: &Arc<Vec<(String, String)>>,
) -> Vec<HookHandlerResolved> {
    raw.iter()
        .filter(|h| !h.command.trim().is_empty())
        .filter_map(|h| {
            let matcher = match h
                .matcher
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                // `*` is the Claude Code spelling of "match everything".
                None | Some("") | Some("*") => None,
                Some(pat) => match Regex::new(pat) {
                    Ok(re) => Some(re),
                    Err(e) => {
                        log::warn!("hooks: skip invalid matcher {pat:?}: {e}");
                        None
                    }
                },
            };
            if h.matcher
                .as_deref()
                .map(str::trim)
                .is_some_and(|m| !m.is_empty() && m != "*")
                && matcher.is_none()
            {
                return None;
            }
            let fail_closed = match h.on_error.as_deref().map(lc).as_deref() {
                None | Some("") | Some("proceed") | Some("allow") => false,
                Some("block") | Some("deny") => true,
                Some(other) => {
                    // Unknown value on a policy knob: be safe.
                    log::warn!("hooks: unknown on_error {other:?}; treating as \"block\"");
                    true
                }
            };
            Some(HookHandlerResolved {
                matcher,
                command: h.command.trim().to_string(),
                timeout_ms: h
                    .timeout_ms
                    .unwrap_or(default_timeout_ms)
                    .clamp(MIN_HOOK_TIMEOUT_MS, 3_600_000),
                cwd_relative: h
                    .cwd
                    .as_ref()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
                fail_closed,
                source: Arc::clone(source),
                env: Arc::clone(env),
            })
        })
        .collect()
}

/// Hooks contributed by one Agent Plugin (`hooks.json`), already parsed per event.
#[derive(Debug, Clone)]
pub struct PluginHookSet {
    pub plugin: String,
    /// Absolute plugin directory, exported to the hook as `ISANAGENT_PLUGIN_ROOT`,
    /// `CLAUDE_PLUGIN_ROOT` and `PLUGIN_ROOT`.
    pub root: PathBuf,
    /// Only the per-event lists are used.
    pub events: HarnessHooksSteeringConfig,
}

pub fn build_steering_engine(
    cfg: &HarnessHooksSteeringConfig,
    workspace_dir: PathBuf,
    sandbox_dir: PathBuf,
) -> Result<Arc<SteeringHooksEngine>, String> {
    build_steering_engine_with(cfg, workspace_dir, sandbox_dir, &[], None)
}

type HookSource<'a> = (
    &'a HarnessHooksSteeringConfig,
    Arc<str>,
    Arc<Vec<(String, String)>>,
);

/// Build the engine from `[harness.hooks.steering]` plus allowed plugin hooks. Config hooks run
/// first, then plugin hooks in plugin-name order.
pub fn build_steering_engine_with(
    cfg: &HarnessHooksSteeringConfig,
    workspace_dir: PathBuf,
    sandbox_dir: PathBuf,
    plugins: &[PluginHookSet],
    audit: Option<Arc<dyn HookAuditSink>>,
) -> Result<Arc<SteeringHooksEngine>, String> {
    let default_timeout_ms = cfg
        .default_timeout_ms
        .unwrap_or(DEFAULT_TIMEOUT_MS)
        .clamp(MIN_HOOK_TIMEOUT_MS, 3_600_000);
    let max_stdout_bytes = cfg
        .max_stdout_bytes
        .unwrap_or(DEFAULT_MAX_STDOUT)
        .clamp(1024, 512 * 1024);

    let mut sources: Vec<HookSource<'_>> = vec![(cfg, Arc::from("config"), Arc::new(Vec::new()))];
    let mut sorted: Vec<&PluginHookSet> = plugins.iter().collect();
    sorted.sort_by(|a, b| a.plugin.cmp(&b.plugin));
    for p in sorted {
        let root = p.root.to_string_lossy().into_owned();
        let env = Arc::new(vec![
            ("ISANAGENT_PLUGIN_ROOT".to_string(), root.clone()),
            ("CLAUDE_PLUGIN_ROOT".to_string(), root.clone()),
            ("PLUGIN_ROOT".to_string(), root),
        ]);
        sources.push((&p.events, Arc::from(format!("plugin:{}", p.plugin)), env));
    }

    let compile =
        |pick: fn(&HarnessHooksSteeringConfig) -> &Option<Vec<HarnessHookCommandConfig>>| {
            let mut all = Vec::new();
            for (c, source, env) in &sources {
                all.extend(compile_hook_handlers(
                    pick(c).as_deref().unwrap_or(&[]),
                    default_timeout_ms,
                    source,
                    env,
                ));
            }
            Arc::new(all)
        };
    Ok(Arc::new(SteeringHooksEngine {
        pre_tool: compile(|c| &c.pre_tool),
        post_tool: compile(|c| &c.post_tool),
        user_prompt: compile(|c| &c.user_prompt),
        post_model: compile(|c| &c.post_model),
        max_stdout_bytes,
        default_timeout_ms,
        workspace_dir,
        sandbox_dir,
        audit,
    }))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn handler(cmd: &str, matcher: Option<&str>, fail_closed: bool) -> HookHandlerResolved {
        HookHandlerResolved {
            matcher: matcher.map(|m| Regex::new(m).unwrap()),
            command: cmd.to_string(),
            timeout_ms: 5_000,
            cwd_relative: None,
            fail_closed,
            source: Arc::from("config"),
            env: Arc::new(Vec::new()),
        }
    }

    #[derive(Default)]
    struct Hooks {
        pre_tool: Vec<HookHandlerResolved>,
        post_tool: Vec<HookHandlerResolved>,
        user_prompt: Vec<HookHandlerResolved>,
        post_model: Vec<HookHandlerResolved>,
    }

    fn engine_with(h: Hooks) -> SteeringHooksEngine {
        let dir = std::env::temp_dir().join(format!("isanagent_hook_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        SteeringHooksEngine {
            pre_tool: Arc::new(h.pre_tool),
            post_tool: Arc::new(h.post_tool),
            user_prompt: Arc::new(h.user_prompt),
            post_model: Arc::new(h.post_model),
            max_stdout_bytes: 64 * 1024,
            default_timeout_ms: 5_000,
            workspace_dir: dir.clone(),
            sandbox_dir: dir,
            audit: None,
        }
    }

    fn post_engine(post_cmd: &str) -> SteeringHooksEngine {
        engine_with(Hooks {
            post_tool: vec![handler(post_cmd, None, false)],
            ..Default::default()
        })
    }

    fn pre_engine(cmds: &[&str]) -> SteeringHooksEngine {
        engine_with(Hooks {
            pre_tool: cmds.iter().map(|c| handler(c, None, false)).collect(),
            ..Default::default()
        })
    }

    fn session(meta: &HashMap<String, Value>) -> HookSessionInfo<'_> {
        HookSessionInfo {
            channel: "terminal",
            chat_id: "c1",
            thread_id: None,
            metadata: meta,
            is_subagent: false,
        }
    }

    /// Shell snippet that prints `json` to stdout (single-quoted for sh).
    fn echo_json(json: &str) -> String {
        format!("cat >/dev/null; printf '%s' '{json}'")
    }

    fn cleanup(eng: &SteeringHooksEngine) {
        let _ = std::fs::remove_dir_all(&eng.sandbox_dir);
    }

    // ---------------- post_tool: legacy verification contract ----------------

    #[tokio::test]
    async fn post_tool_captures_failure_stdout_and_stderr() {
        // A verify hook that fails: its stdout AND stderr must reach the model, with the exit code.
        let eng = post_engine("echo build-out; echo build-err 1>&2; exit 1");
        let meta = HashMap::new();
        let d = run_post_tool_hooks(
            &eng,
            "edit_file",
            Some("id"),
            &json!({}),
            &Ok("applied".to_string()),
            session(&meta),
        )
        .await;
        let out = d.notes.join("\n");
        assert!(out.contains("[hook exited 1]"), "{out}");
        assert!(out.contains("build-out"), "{out}");
        assert!(out.contains("build-err"), "{out}");
        assert!(d.updated_output.is_none());
        cleanup(&eng);
    }

    #[tokio::test]
    async fn post_tool_captures_success_stdout() {
        let eng = post_engine("echo all-green");
        let meta = HashMap::new();
        let d = run_post_tool_hooks(
            &eng,
            "edit_file",
            None,
            &json!({}),
            &Ok("applied".to_string()),
            session(&meta),
        )
        .await;
        let out = d.notes.join("\n");
        assert!(out.contains("all-green"), "{out}");
        assert!(!out.contains("hook exited"), "{out}");
        cleanup(&eng);
    }

    #[tokio::test]
    async fn capture_failure_false_preserves_legacy_directive_contract() {
        let dir = std::env::temp_dir().join(format!("isanagent_hook_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let nonzero = run_hook_command(
            "echo out; echo err 1>&2; exit 3",
            &dir,
            &json!({}),
            5_000,
            64 * 1024,
            false,
        )
        .await
        .unwrap();
        assert!(
            nonzero.is_none(),
            "non-zero exit must be Ok(None): {nonzero:?}"
        );

        let ok = run_hook_command(
            "echo only-stdout; echo hidden 1>&2",
            &dir,
            &json!({}),
            5_000,
            64 * 1024,
            false,
        )
        .await
        .unwrap();
        assert_eq!(ok.as_deref(), Some("only-stdout")); // stderr ("hidden") dropped
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn post_tool_no_output_returns_empty_decision() {
        let eng = post_engine("true");
        let meta = HashMap::new();
        let d = run_post_tool_hooks(
            &eng,
            "edit_file",
            None,
            &json!({}),
            &Ok("x".to_string()),
            session(&meta),
        )
        .await;
        assert_eq!(d, PostToolDecision::default());
        cleanup(&eng);
    }

    #[tokio::test]
    async fn post_tool_plain_json_without_contract_keys_is_verification_text() {
        let eng = post_engine(&echo_json(r#"{"tests":3,"failed":0}"#));
        let meta = HashMap::new();
        let d = run_post_tool_hooks(
            &eng,
            "exec",
            None,
            &json!({}),
            &Ok("x".to_string()),
            session(&meta),
        )
        .await;
        assert_eq!(d.notes, vec![r#"{"tests":3,"failed":0}"#.to_string()]);
        assert!(d.updated_output.is_none());
        cleanup(&eng);
    }

    // ---------------- post_tool: v2 directives ----------------

    #[tokio::test]
    async fn post_tool_updated_output_replaces_result_and_chains() {
        // Hook 1 redacts; hook 2 must see the redacted output on stdin.
        let eng = engine_with(Hooks {
            post_tool: vec![
                handler(
                    &echo_json(
                        r#"{"hookSpecificOutput":{"hookEventName":"post_tool","updatedOutput":"token=[REDACTED]","additionalContext":"redacted 1 secret"}}"#,
                    ),
                    None,
                    false,
                ),
                handler(
                    r#"python3 -c 'import json,sys; e=json.load(sys.stdin); print(json.dumps({"hookSpecificOutput":{"additionalContext":"saw:"+e["result"]}}))'"#,
                    None,
                    false,
                ),
            ],
            ..Default::default()
        });
        let meta = HashMap::new();
        let d = run_post_tool_hooks(
            &eng,
            "read_file",
            None,
            &json!({}),
            &Ok("token=abc123".to_string()),
            session(&meta),
        )
        .await;
        assert_eq!(d.updated_output.as_deref(), Some("token=[REDACTED]"));
        assert_eq!(
            d.notes,
            vec![
                "redacted 1 secret".to_string(),
                "saw:token=[REDACTED]".to_string()
            ]
        );
        cleanup(&eng);
    }

    #[tokio::test]
    async fn post_tool_block_and_stop() {
        let eng = post_engine(&echo_json(r#"{"decision":"block","reason":"lint failed"}"#));
        let meta = HashMap::new();
        let d = run_post_tool_hooks(
            &eng,
            "edit_file",
            None,
            &json!({}),
            &Ok("ok".to_string()),
            session(&meta),
        )
        .await;
        assert_eq!(d.block_reason.as_deref(), Some("lint failed"));
        cleanup(&eng);

        let eng = post_engine(&echo_json(r#"{"continue":false,"stopReason":"budget"}"#));
        let d = run_post_tool_hooks(
            &eng,
            "edit_file",
            None,
            &json!({}),
            &Ok("ok".to_string()),
            session(&meta),
        )
        .await;
        assert_eq!(d.stop_reason.as_deref(), Some("budget"));
        cleanup(&eng);
    }

    // ---------------- pre_tool ----------------

    fn proceed(o: PreToolOutcome) -> PreToolDecision {
        match o {
            PreToolOutcome::Proceed(d) => d,
            other => panic!("expected Proceed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pre_tool_updated_input_chains_between_hooks() {
        let eng = pre_engine(&[
            &echo_json(r#"{"hookSpecificOutput":{"updatedInput":{"command":"ls -la"}}}"#),
            r#"python3 -c 'import json,sys; e=json.load(sys.stdin); a=e["args"]; a["command"]+=" /tmp"; print(json.dumps({"hookSpecificOutput":{"updatedInput":a,"additionalContext":"scoped to /tmp"}}))'"#,
        ]);
        let meta = HashMap::new();
        let d = proceed(
            run_pre_tool_hooks(&eng, "exec", None, json!({"command": "ls"}), session(&meta)).await,
        );
        assert_eq!(d.args, json!({"command": "ls -la /tmp"}));
        assert!(d.args_modified);
        assert_eq!(d.additional_context, vec!["scoped to /tmp".to_string()]);
        assert!(d.ask_reason.is_none());
        cleanup(&eng);
    }

    #[tokio::test]
    async fn pre_tool_legacy_modify_and_block_still_work() {
        let meta = HashMap::new();
        let eng = pre_engine(&[&echo_json(
            r#"{"decision":"modify","args":{"path":"README.md"}}"#,
        )]);
        let d = proceed(
            run_pre_tool_hooks(
                &eng,
                "read_file",
                None,
                json!({"path": "x"}),
                session(&meta),
            )
            .await,
        );
        assert_eq!(d.args, json!({"path": "README.md"}));
        cleanup(&eng);

        let eng = pre_engine(&[&echo_json(r#"{"decision":"block","message":"nope"}"#)]);
        let o = run_pre_tool_hooks(&eng, "exec", None, json!({}), session(&meta)).await;
        assert_eq!(o, PreToolOutcome::Block("nope".to_string()));
        cleanup(&eng);
    }

    #[tokio::test]
    async fn pre_tool_permission_decisions() {
        let meta = HashMap::new();
        let eng = pre_engine(&[&echo_json(
            r#"{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"prod is read-only"}}"#,
        )]);
        let o = run_pre_tool_hooks(&eng, "exec", None, json!({}), session(&meta)).await;
        assert_eq!(o, PreToolOutcome::Block("prod is read-only".to_string()));
        cleanup(&eng);

        let eng = pre_engine(&[&echo_json(
            r#"{"hookSpecificOutput":{"permissionDecision":"ask","permissionDecisionReason":"touches billing"}}"#,
        )]);
        let d = proceed(run_pre_tool_hooks(&eng, "exec", None, json!({}), session(&meta)).await);
        assert_eq!(d.ask_reason.as_deref(), Some("touches billing"));
        cleanup(&eng);

        let eng = pre_engine(&[&echo_json(
            r#"{"continue":false,"stopReason":"quota exceeded"}"#,
        )]);
        let o = run_pre_tool_hooks(&eng, "exec", None, json!({}), session(&meta)).await;
        assert_eq!(o, PreToolOutcome::Stop("quota exceeded".to_string()));
        cleanup(&eng);
    }

    #[tokio::test]
    async fn pre_tool_exit_2_blocks_with_stderr() {
        let meta = HashMap::new();
        let eng = pre_engine(&["echo 'rm is not allowed' 1>&2; exit 2"]);
        let o = run_pre_tool_hooks(&eng, "exec", None, json!({}), session(&meta)).await;
        assert_eq!(o, PreToolOutcome::Block("rm is not allowed".to_string()));
        cleanup(&eng);
    }

    #[tokio::test]
    async fn pre_tool_on_error_controls_fail_open_vs_closed() {
        let meta = HashMap::new();
        // Fail-open (default): a crashing hook is skipped.
        let eng = pre_engine(&["exit 1"]);
        let d =
            proceed(run_pre_tool_hooks(&eng, "exec", None, json!({"a": 1}), session(&meta)).await);
        assert_eq!(d.args, json!({"a": 1}));
        assert!(!d.args_modified);
        cleanup(&eng);

        // Fail-closed: the same failure blocks; so does unparseable stdout.
        for cmd in ["exit 1", "echo not-json"] {
            let eng = engine_with(Hooks {
                pre_tool: vec![handler(cmd, None, true)],
                ..Default::default()
            });
            let o = run_pre_tool_hooks(&eng, "exec", None, json!({}), session(&meta)).await;
            assert!(
                matches!(o, PreToolOutcome::Block(ref m) if m.contains("on_error=block")),
                "{cmd}: {o:?}"
            );
            cleanup(&eng);
        }
    }

    #[tokio::test]
    async fn pre_tool_rejects_non_object_updated_input() {
        let meta = HashMap::new();
        let eng = engine_with(Hooks {
            pre_tool: vec![handler(
                &echo_json(r#"{"hookSpecificOutput":{"updatedInput":"rm -rf /"}}"#),
                None,
                true,
            )],
            ..Default::default()
        });
        let o =
            run_pre_tool_hooks(&eng, "exec", None, json!({"command": "ls"}), session(&meta)).await;
        assert!(matches!(o, PreToolOutcome::Block(_)), "{o:?}");
        cleanup(&eng);
    }

    #[tokio::test]
    async fn pre_tool_matcher_filters_by_tool_name() {
        let meta = HashMap::new();
        let eng = engine_with(Hooks {
            pre_tool: vec![handler(
                &echo_json(r#"{"decision":"block"}"#),
                Some("^exec$"),
                false,
            )],
            ..Default::default()
        });
        let d =
            proceed(run_pre_tool_hooks(&eng, "read_file", None, json!({}), session(&meta)).await);
        assert!(!d.args_modified);
        cleanup(&eng);
    }

    // ---------------- user_prompt ----------------

    #[tokio::test]
    async fn user_prompt_rewrite_and_context() {
        let meta = HashMap::new();
        let eng = engine_with(Hooks {
            user_prompt: vec![
                handler(
                    r#"python3 -c 'import json,sys; e=json.load(sys.stdin); assert e["content"].startswith("CTX "); print(json.dumps({"hookSpecificOutput":{"updatedPrompt":e["prompt"].replace("4111-1111","[CARD]"),"additionalContext":"PII masked"}}))'"#,
                    None,
                    false,
                ),
                handler(
                    &echo_json(r#"{"decision":"inject_prefix","message":"tenant=acme"}"#),
                    Some(r"\[CARD\]"), // matches the rewritten prompt
                    false,
                ),
            ],
            ..Default::default()
        });
        let o = run_user_prompt_hooks(&eng, "CTX ", "pay with 4111-1111", session(&meta)).await;
        assert_eq!(
            o,
            UserPromptHookOutcome::Proceed(UserPromptDecision {
                updated_prompt: Some("pay with [CARD]".to_string()),
                prefix: Some("PII masked\ntenant=acme".to_string()),
            })
        );
        cleanup(&eng);
    }

    #[tokio::test]
    async fn user_prompt_block_and_noop() {
        let meta = HashMap::new();
        let eng = engine_with(Hooks {
            user_prompt: vec![handler("echo 'off-topic' 1>&2; exit 2", None, false)],
            ..Default::default()
        });
        let o = run_user_prompt_hooks(&eng, "", "hi", session(&meta)).await;
        assert_eq!(o, UserPromptHookOutcome::Block("off-topic".to_string()));
        cleanup(&eng);

        let eng = engine_with(Hooks {
            user_prompt: vec![handler("true", None, false)],
            ..Default::default()
        });
        let o = run_user_prompt_hooks(&eng, "", "hi", session(&meta)).await;
        assert_eq!(
            o,
            UserPromptHookOutcome::Proceed(UserPromptDecision::default())
        );
        cleanup(&eng);
    }

    // ---------------- post_model ----------------

    #[tokio::test]
    async fn post_model_rewrites_content_and_tool_calls() {
        let meta = HashMap::new();
        let eng = engine_with(Hooks {
            post_model: vec![handler(
                r#"python3 -c 'import json,sys; e=json.load(sys.stdin); c=[t for t in e["tool_calls"] if t["name"]!="exec"]; print(json.dumps({"hookSpecificOutput":{"updatedContent":e["content"].upper(),"updatedToolCalls":c}}))'"#,
                None,
                false,
            )],
            ..Default::default()
        });
        let calls = vec![
            HookToolCall {
                id: Some("a".into()),
                name: "exec".into(),
                arguments: json!({"command": "rm -rf /"}),
            },
            HookToolCall {
                id: Some("b".into()),
                name: "read_file".into(),
                arguments: json!({"path": "x"}),
            },
        ];
        let o = run_post_model_hooks(&eng, "hello", &calls, 1, session(&meta)).await;
        match o {
            PostModelOutcome::Proceed(d) => {
                assert_eq!(d.updated_content.as_deref(), Some("HELLO"));
                assert_eq!(d.updated_tool_calls, Some(vec![calls[1].clone()]));
            }
            other => panic!("{other:?}"),
        }
        cleanup(&eng);
    }

    #[tokio::test]
    async fn post_model_retry_and_stop() {
        let meta = HashMap::new();
        let eng = engine_with(Hooks {
            post_model: vec![handler(
                &echo_json(r#"{"decision":"block","reason":"cite your sources"}"#),
                None,
                false,
            )],
            ..Default::default()
        });
        let o = run_post_model_hooks(&eng, "x", &[], 1, session(&meta)).await;
        assert_eq!(o, PostModelOutcome::Retry("cite your sources".to_string()));
        cleanup(&eng);

        let eng = engine_with(Hooks {
            post_model: vec![handler(
                &echo_json(r#"{"continue":false,"stopReason":"policy"}"#),
                None,
                false,
            )],
            ..Default::default()
        });
        let o = run_post_model_hooks(&eng, "x", &[], 1, session(&meta)).await;
        assert_eq!(o, PostModelOutcome::Stop("policy".to_string()));
        cleanup(&eng);
    }

    #[tokio::test]
    async fn large_payload_and_verbose_hook_do_not_deadlock() {
        // The hook ignores a 256 KiB stdin payload and floods stderr/stdout well past the capture
        // limit before exiting 2. Before the fix the stdin write blocked outside the timeout.
        let eng = pre_engine(&[
            "head -c 300000 /dev/zero | tr '\\0' x 1>&2; head -c 300000 /dev/zero; exit 2",
        ]);
        let meta = HashMap::new();
        let big = "y".repeat(256 * 1024);
        let o = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            run_pre_tool_hooks(
                &eng,
                "write_file",
                None,
                json!({"content": big}),
                session(&meta),
            ),
        )
        .await
        .expect("hook must not hang");
        match o {
            PreToolOutcome::Block(m) => assert!(m.len() <= 64 * 1024 && m.starts_with('x')),
            other => panic!("exit 2 must block: {other:?}"),
        }
        cleanup(&eng);
    }

    #[test]
    fn on_error_parsing() {
        let mk = |v: Option<&str>| HarnessHookCommandConfig {
            matcher: None,
            command: "true".into(),
            timeout_ms: None,
            cwd: None,
            on_error: v.map(str::to_string),
        };
        let compiled = compile_hook_handlers(
            &[
                mk(None),
                mk(Some("proceed")),
                mk(Some("block")),
                mk(Some("bogus")),
            ],
            5_000,
            &Arc::from("config"),
            &Arc::new(Vec::new()),
        );
        let flags: Vec<bool> = compiled.iter().map(|h| h.fail_closed).collect();
        assert_eq!(flags, vec![false, false, true, true]);
    }
}

#[cfg(all(test, unix))]
mod audit_tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Collect(Mutex<Vec<HookAuditRecord>>);
    impl HookAuditSink for Collect {
        fn record(&self, _s: &HookSessionInfo<'_>, r: HookAuditRecord) {
            self.0.lock().unwrap().push(r);
        }
    }

    fn engine(cmds: &[(&str, bool)], sink: Arc<Collect>) -> SteeringHooksEngine {
        let dir = std::env::temp_dir().join(format!("isanagent_audit_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let handlers = cmds
            .iter()
            .map(|(c, fail_closed)| HookHandlerResolved {
                matcher: None,
                command: c.to_string(),
                timeout_ms: 5_000,
                cwd_relative: None,
                fail_closed: *fail_closed,
                source: Arc::from("config"),
                env: Arc::new(Vec::new()),
            })
            .collect::<Vec<_>>();
        SteeringHooksEngine {
            pre_tool: Arc::new(handlers),
            post_tool: Arc::new(vec![]),
            user_prompt: Arc::new(vec![]),
            post_model: Arc::new(vec![]),
            max_stdout_bytes: 64 * 1024,
            default_timeout_ms: 5_000,
            workspace_dir: dir.clone(),
            sandbox_dir: dir,
            audit: Some(sink),
        }
    }

    #[tokio::test]
    async fn records_each_effective_decision_but_not_noops() {
        let sink = Arc::new(Collect::default());
        let eng = engine(
            &[
                ("true", false), // no-op: not recorded
                (
                    r#"cat >/dev/null; printf '%s' '{"hookSpecificOutput":{"updatedInput":{"command":"ls -la"},"additionalContext":"c"}}'"#,
                    false,
                ),
                (
                    r#"cat >/dev/null; printf '%s' '{"hookSpecificOutput":{"permissionDecision":"ask","permissionDecisionReason":"why"}}'"#,
                    false,
                ),
                ("exit 7", false),
                ("echo nope 1>&2; exit 2", false),
            ],
            sink.clone(),
        );
        let meta = HashMap::new();
        let o = run_pre_tool_hooks(
            &eng,
            "exec",
            Some("id"),
            json!({"command": "ls"}),
            HookSessionInfo {
                channel: "t",
                chat_id: "c",
                thread_id: None,
                metadata: &meta,
                is_subagent: false,
            },
        )
        .await;
        assert_eq!(o, PreToolOutcome::Block("nope".into()));
        let recs = sink.0.lock().unwrap().clone();
        let summary: Vec<(&str, Vec<&str>)> = recs
            .iter()
            .map(|r| (r.decision, r.changed.clone()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("modify", vec!["args", "context"]),
                ("ask", vec![]),
                ("error_ignored", vec![]),
                ("block", vec![]),
            ]
        );
        assert!(recs.iter().all(|r| r.tool_name.as_deref() == Some("exec")));
        assert_eq!(recs[1].reason.as_deref(), Some("why"));
        let _ = std::fs::remove_dir_all(&eng.sandbox_dir);
    }
}
