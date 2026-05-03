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

async fn read_bounded(
    mut reader: tokio::process::ChildStdout,
    max: usize,
) -> Result<String, String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = reader
            .read(&mut chunk)
            .await
            .map_err(|e| format!("read stdout: {}", e))?;
        if n == 0 {
            break;
        }
        if buf.len() + n > max {
            return Err(format!("hook stdout exceeded {} bytes", max));
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    String::from_utf8(buf).map_err(|e| format!("hook stdout utf8: {}", e))
}

async fn run_hook_command(
    command: &str,
    cwd: &Path,
    stdin_json: &Value,
    timeout_ms: u64,
    max_stdout: usize,
) -> Result<Option<String>, String> {
    let (shell, arg) = shell_invocation();
    let mut child = Command::new(shell)
        .arg(arg)
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("spawn hook: {}", e))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "hook stdin missing".to_string())?;
    let body = serde_json::to_vec(stdin_json).map_err(|e| format!("hook stdin encode: {}", e))?;
    tokio::io::AsyncWriteExt::write_all(&mut stdin, &body)
        .await
        .map_err(|e| format!("hook stdin write: {}", e))?;
    drop(stdin);

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "hook stdout missing".to_string())?;
    let read_out = read_bounded(stdout, max_stdout);

    let status = tokio::time::timeout(
        std::time::Duration::from_millis(hook_command_timeout_ms(timeout_ms)),
        child.wait(),
    )
    .await
    .map_err(|_| "hook timed out".to_string())?
    .map_err(|e| format!("hook wait: {}", e))?;

    let out = read_out.await?;

    if !status.success() {
        log::warn!(
            "hook command exited with {} (proceeding)",
            status.code().unwrap_or(-1)
        );
        return Ok(None);
    }

    let trimmed = out.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(trimmed.to_string()))
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
        let out = match run_hook_command(&h.command, &cwd, &stdin, timeout, engine.max_stdout_bytes)
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

pub async fn run_post_tool_hooks(
    engine: &SteeringHooksEngine,
    tool_name: &str,
    tool_call_id: Option<&str>,
    args: &Value,
    result: &Result<String, String>,
    session: HookSessionInfo<'_>,
) {
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
        if let Err(e) =
            run_hook_command(&h.command, &cwd, &stdin, timeout, engine.max_stdout_bytes).await
        {
            log::warn!("post_tool hook failed: {}", e);
        }
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
        let out = match run_hook_command(&h.command, &cwd, &stdin, timeout, engine.max_stdout_bytes)
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
