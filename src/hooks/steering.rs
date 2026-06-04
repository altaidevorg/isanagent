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
}

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
    pub max_stdout_bytes: usize,
    pub default_timeout_ms: u64,
    pub workspace_dir: PathBuf,
    pub sandbox_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreToolOutcome {
    Proceed(Value),
    Block(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserPromptHookOutcome {
    Proceed,
    Block(String),
    InjectPrefix(String),
}

#[derive(Debug, Deserialize)]
struct HookStdoutEnvelope {
    decision: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    args: Option<Value>,
}

fn matches_tool(matcher: &Option<Regex>, tool_name: &str) -> bool {
    match matcher {
        None => true,
        Some(re) => re.is_match(tool_name),
    }
}

fn shell_invocation() -> (&'static str, &'static str) {
    if cfg!(windows) {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    }
}

/// Read up to `max` bytes from a child pipe, truncating (rather than erroring) on overflow so a
/// verbose hook still yields a usable prefix, and lossily decoding so non-UTF-8 output (common from
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
            .map_err(|e| format!("read hook output: {}", e))?;
        if n == 0 {
            break;
        }
        let remaining = max.saturating_sub(buf.len());
        if n >= remaining {
            buf.extend_from_slice(&chunk[..remaining]);
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Run a hook subprocess, piping the JSON event to its stdin and capturing stdout.
///
/// When `capture_failure` is true (post_tool *verification* hooks), stderr is also captured and the
/// output is returned **even on a non-zero exit** (prefixed with the exit status) — a failing
/// `cargo build` / `pytest` is exactly what must reach the model. When false (pre_tool /
/// user_prompt directive hooks), the legacy contract holds: stderr is dropped and a non-zero exit
/// yields `Ok(None)` (proceed without the hook's directive).
///
/// Both pipes are drained concurrently with `wait()` under one timeout, so a hook that fills a pipe
/// buffer can't deadlock.
async fn run_hook_command(
    command: &str,
    cwd: &Path,
    stdin_json: &Value,
    timeout_ms: u64,
    max_stdout: usize,
    capture_failure: bool,
) -> Result<Option<String>, String> {
    let (shell, arg) = shell_invocation();
    let mut child = Command::new(shell)
        .arg(arg)
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(if capture_failure {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("spawn hook: {}", e))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "hook stdin missing".to_string())?;
    let body = serde_json::to_vec(stdin_json).map_err(|e| format!("hook stdin encode: {}", e))?;
    // A hook that ignores its stdin and exits (a bare `cargo build` / `pytest`, or any verify hook
    // that reads the event from argv/env instead) closes its stdin read end before — or while — we
    // write. On Linux that surfaces here as `BrokenPipe`; on macOS the small event JSON usually fits
    // the pipe buffer and the write completes before the child exits. Treating BrokenPipe as fatal
    // therefore dropped the hook's captured output non-deterministically (the failure reached the
    // model on macOS but vanished on Linux). The hook simply didn't consume the event, which is
    // fine — swallow BrokenPipe and proceed to capture its output and exit status.
    if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut stdin, &body).await {
        if e.kind() != std::io::ErrorKind::BrokenPipe {
            return Err(format!("hook stdin write: {}", e));
        }
    }
    drop(stdin);

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "hook stdout missing".to_string())?;
    let stderr = child.stderr.take();

    let io = async {
        let out_fut = read_bounded(stdout, max_stdout);
        let err_fut = async {
            match stderr {
                Some(se) => read_bounded(se, max_stdout).await.unwrap_or_default(),
                None => String::new(),
            }
        };
        let (status, out, err) = tokio::join!(child.wait(), out_fut, err_fut);
        (status, out, err)
    };
    let (status, out, err) = tokio::time::timeout(
        std::time::Duration::from_millis(hook_command_timeout_ms(timeout_ms)),
        io,
    )
    .await
    .map_err(|_| "hook timed out".to_string())?;

    let status = status.map_err(|e| format!("hook wait: {}", e))?;
    let out = out?;

    // Join stdout + stderr (stderr is empty unless `capture_failure` piped it).
    let mut combined = out;
    if !err.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&err);
    }
    let combined = combined.trim();

    if !status.success() {
        if !capture_failure {
            log::warn!(
                "hook command exited with {} (proceeding)",
                status.code().unwrap_or(-1)
            );
            return Ok(None);
        }
        let code = status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".to_string());
        return Ok(Some(if combined.is_empty() {
            format!("[hook exited {code} with no output]")
        } else {
            format!("[hook exited {code}]\n{combined}")
        }));
    }

    if combined.is_empty() {
        return Ok(None);
    }
    Ok(Some(combined.to_string()))
}

fn parse_pre_tool_stdout(text: &str) -> Result<PreToolOutcome, String> {
    let v: HookStdoutEnvelope =
        serde_json::from_str(text).map_err(|e| format!("hook json: {}", e))?;
    match v
        .decision
        .as_deref()
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("block") | Some("deny") => {
            let msg = v.message.unwrap_or_else(|| "blocked by hook".to_string());
            Ok(PreToolOutcome::Block(msg))
        }
        Some("modify") => {
            let args = v
                .args
                .ok_or_else(|| "modify decision requires args".to_string())?;
            Ok(PreToolOutcome::Proceed(args))
        }
        Some("proceed") | Some("allow") | None => {
            if let Some(a) = v.args {
                Ok(PreToolOutcome::Proceed(a))
            } else {
                Ok(PreToolOutcome::Proceed(Value::Null))
            }
        }
        Some(other) => Err(format!("unknown decision {:?}", other)),
    }
}

fn parse_user_prompt_stdout(text: &str) -> Result<UserPromptHookOutcome, String> {
    let v: HookStdoutEnvelope =
        serde_json::from_str(text).map_err(|e| format!("hook json: {}", e))?;
    match v
        .decision
        .as_deref()
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("block") | Some("deny") => {
            let msg = v.message.unwrap_or_else(|| "blocked by hook".to_string());
            Ok(UserPromptHookOutcome::Block(msg))
        }
        Some("inject_prefix") => {
            let msg = v
                .message
                .ok_or_else(|| "inject_prefix requires message".to_string())?;
            Ok(UserPromptHookOutcome::InjectPrefix(msg))
        }
        Some("proceed") | Some("allow") | None => Ok(UserPromptHookOutcome::Proceed),
        Some(other) => Err(format!("unknown decision {:?}", other)),
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

/// Run all matching `pre_tool` hooks in order. `modify` replaces args for subsequent hooks and execution.
pub async fn run_pre_tool_hooks(
    engine: &SteeringHooksEngine,
    tool_name: &str,
    tool_call_id: Option<&str>,
    mut args: Value,
    session: HookSessionInfo<'_>,
) -> PreToolOutcome {
    for h in engine.pre_tool.iter() {
        if !matches_tool(&h.matcher, tool_name) {
            continue;
        }
        let cwd = match hook_cwd(&engine.sandbox_dir, h.cwd_relative.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("pre_tool hook cwd: {}", e);
                continue;
            }
        };
        let stdin = json!({
            "hook_event": "pre_tool",
            "schema_version": 1,
            "channel": session.channel,
            "chat_id": session.chat_id,
            "thread_id": session.thread_id,
            "is_subagent": session.is_subagent,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "args": args,
            "metadata": session.metadata,
            "workspace_dir": engine.workspace_dir.to_string_lossy(),
            "sandbox_dir": engine.sandbox_dir.to_string_lossy(),
        });
        let timeout = hook_command_timeout_ms(h.timeout_ms);
        let out = match run_hook_command(
            &h.command,
            &cwd,
            &stdin,
            timeout,
            engine.max_stdout_bytes,
            false,
        )
        .await
        {
            Ok(o) => o,
            Err(e) => {
                log::warn!("pre_tool hook failed (proceeding): {}", e);
                continue;
            }
        };
        let Some(text) = out else { continue };
        match parse_pre_tool_stdout(&text) {
            Ok(PreToolOutcome::Block(m)) => return PreToolOutcome::Block(m),
            Ok(PreToolOutcome::Proceed(new_args)) => {
                if !new_args.is_null() {
                    args = new_args;
                }
            }
            Err(e) => log::warn!("pre_tool hook stdout parse (ignored): {}", e),
        }
    }
    PreToolOutcome::Proceed(args)
}

/// Run the configured post_tool hooks. Returns their combined stdout/stderr (verification output)
/// so the caller can append it to the tool result the model sees — this is what closes the
/// verify-into-fix loop: a `cargo build` / `pytest` / lint hook's output (including failures) now
/// reaches the model for self-correction instead of being discarded.
pub async fn run_post_tool_hooks(
    engine: &SteeringHooksEngine,
    tool_name: &str,
    tool_call_id: Option<&str>,
    args: &Value,
    result: &Result<String, String>,
    session: HookSessionInfo<'_>,
) -> Option<String> {
    let mut outputs: Vec<String> = Vec::new();
    for h in engine.post_tool.iter() {
        if !matches_tool(&h.matcher, tool_name) {
            continue;
        }
        let cwd = match hook_cwd(&engine.sandbox_dir, h.cwd_relative.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("post_tool hook cwd: {}", e);
                continue;
            }
        };
        let stdin = json!({
            "hook_event": "post_tool",
            "schema_version": 1,
            "channel": session.channel,
            "chat_id": session.chat_id,
            "thread_id": session.thread_id,
            "is_subagent": session.is_subagent,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "args": args,
            "result_ok": result.is_ok(),
            "result": result.as_ref().ok(),
            "error": result.as_ref().err(),
            "metadata": session.metadata,
            "workspace_dir": engine.workspace_dir.to_string_lossy(),
            "sandbox_dir": engine.sandbox_dir.to_string_lossy(),
        });
        let timeout = hook_command_timeout_ms(h.timeout_ms);
        match run_hook_command(
            &h.command,
            &cwd,
            &stdin,
            timeout,
            engine.max_stdout_bytes,
            true,
        )
        .await
        {
            Ok(Some(out)) => outputs.push(out),
            Ok(None) => {}
            Err(e) => log::warn!("post_tool hook failed: {}", e),
        }
    }
    if outputs.is_empty() {
        None
    } else {
        Some(outputs.join("\n\n"))
    }
}

pub async fn run_user_prompt_hooks(
    engine: &SteeringHooksEngine,
    content: &str,
    session: HookSessionInfo<'_>,
) -> UserPromptHookOutcome {
    let mut inject: Option<String> = None;
    for h in engine.user_prompt.iter() {
        let cwd = match hook_cwd(&engine.sandbox_dir, h.cwd_relative.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("user_prompt hook cwd: {}", e);
                continue;
            }
        };
        let stdin = json!({
            "hook_event": "user_prompt",
            "schema_version": 1,
            "channel": session.channel,
            "chat_id": session.chat_id,
            "thread_id": session.thread_id,
            "is_subagent": session.is_subagent,
            "content": content,
            "metadata": session.metadata,
            "workspace_dir": engine.workspace_dir.to_string_lossy(),
            "sandbox_dir": engine.sandbox_dir.to_string_lossy(),
        });
        let timeout = hook_command_timeout_ms(h.timeout_ms);
        let out = match run_hook_command(
            &h.command,
            &cwd,
            &stdin,
            timeout,
            engine.max_stdout_bytes,
            false,
        )
        .await
        {
            Ok(o) => o,
            Err(e) => {
                log::warn!("user_prompt hook failed (proceeding): {}", e);
                continue;
            }
        };
        let Some(text) = out else { continue };
        match parse_user_prompt_stdout(&text) {
            Ok(UserPromptHookOutcome::Block(m)) => return UserPromptHookOutcome::Block(m),
            Ok(UserPromptHookOutcome::InjectPrefix(prefix)) => {
                inject = Some(match inject.take() {
                    Some(prev) => format!("{}\n{}", prev, prefix),
                    None => prefix,
                });
            }
            Ok(UserPromptHookOutcome::Proceed) => {}
            Err(e) => log::warn!("user_prompt hook stdout parse (ignored): {}", e),
        }
    }
    match inject {
        Some(prefix) => UserPromptHookOutcome::InjectPrefix(prefix),
        None => UserPromptHookOutcome::Proceed,
    }
}

fn compile_hook_handlers(
    raw: &[HarnessHookCommandConfig],
    default_timeout_ms: u64,
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
                None | Some("") => None,
                Some(pat) => match Regex::new(pat) {
                    Ok(re) => Some(re),
                    Err(e) => {
                        log::warn!("hooks: skip invalid matcher {:?}: {}", pat, e);
                        None
                    }
                },
            };
            if h.matcher.as_deref().is_some_and(|m| !m.trim().is_empty()) && matcher.is_none() {
                return None;
            }
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
            })
        })
        .collect()
}

pub fn build_steering_engine(
    cfg: &HarnessHooksSteeringConfig,
    workspace_dir: PathBuf,
    sandbox_dir: PathBuf,
) -> Result<Arc<SteeringHooksEngine>, String> {
    let default_timeout_ms = cfg
        .default_timeout_ms
        .unwrap_or(DEFAULT_TIMEOUT_MS)
        .clamp(MIN_HOOK_TIMEOUT_MS, 3_600_000);
    let max_stdout_bytes = cfg
        .max_stdout_bytes
        .unwrap_or(DEFAULT_MAX_STDOUT)
        .clamp(1024, 512 * 1024);
    let pre_tool =
        compile_hook_handlers(cfg.pre_tool.as_deref().unwrap_or(&[]), default_timeout_ms);
    let post_tool =
        compile_hook_handlers(cfg.post_tool.as_deref().unwrap_or(&[]), default_timeout_ms);
    let user_prompt = compile_hook_handlers(
        cfg.user_prompt.as_deref().unwrap_or(&[]),
        default_timeout_ms,
    );
    Ok(Arc::new(SteeringHooksEngine {
        pre_tool: Arc::new(pre_tool),
        post_tool: Arc::new(post_tool),
        user_prompt: Arc::new(user_prompt),
        max_stdout_bytes,
        default_timeout_ms,
        workspace_dir,
        sandbox_dir,
    }))
}

#[cfg(all(test, unix))]
mod post_tool_capture_tests {
    use super::*;

    fn engine(post_cmd: &str) -> SteeringHooksEngine {
        let dir = std::env::temp_dir().join(format!("isanagent_hook_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        SteeringHooksEngine {
            pre_tool: Arc::new(vec![]),
            post_tool: Arc::new(vec![HookHandlerResolved {
                matcher: None,
                command: post_cmd.to_string(),
                timeout_ms: 5_000,
                cwd_relative: None,
            }]),
            user_prompt: Arc::new(vec![]),
            max_stdout_bytes: 64 * 1024,
            default_timeout_ms: 5_000,
            workspace_dir: dir.clone(),
            sandbox_dir: dir,
        }
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

    #[tokio::test]
    async fn post_tool_captures_failure_stdout_and_stderr() {
        // A verify hook that fails: its stdout AND stderr must reach the model, with the exit code.
        let eng = engine("echo build-out; echo build-err 1>&2; exit 1");
        let meta = HashMap::new();
        let out = run_post_tool_hooks(
            &eng,
            "edit_file",
            Some("id"),
            &json!({}),
            &Ok("applied".to_string()),
            session(&meta),
        )
        .await
        .expect("failure output captured");
        assert!(out.contains("[hook exited 1]"), "{out}");
        assert!(out.contains("build-out"), "{out}");
        assert!(out.contains("build-err"), "{out}");
        let _ = std::fs::remove_dir_all(&eng.sandbox_dir);
    }

    #[tokio::test]
    async fn post_tool_captures_success_stdout() {
        let eng = engine("echo all-green");
        let meta = HashMap::new();
        let out = run_post_tool_hooks(
            &eng,
            "edit_file",
            None,
            &json!({}),
            &Ok("applied".to_string()),
            session(&meta),
        )
        .await
        .expect("success output captured");
        assert!(out.contains("all-green"), "{out}");
        assert!(!out.contains("hook exited"), "{out}");
        let _ = std::fs::remove_dir_all(&eng.sandbox_dir);
    }

    #[tokio::test]
    async fn capture_failure_false_preserves_legacy_directive_contract() {
        // pre_tool / user_prompt hooks pass capture_failure=false: stderr is dropped and a non-zero
        // exit yields Ok(None) (proceed without the directive) — unchanged from before this PR.
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
    async fn post_tool_no_output_returns_none() {
        let eng = engine("true"); // succeeds, no output
        let meta = HashMap::new();
        let out = run_post_tool_hooks(
            &eng,
            "edit_file",
            None,
            &json!({}),
            &Ok("x".to_string()),
            session(&meta),
        )
        .await;
        assert!(out.is_none());
        let _ = std::fs::remove_dir_all(&eng.sandbox_dir);
    }
}
