use crate::bus::{BusMessage, InboundMessage, LogEvent, OutboundMessage};
use crate::channels::Channel;
use crate::config::AppConfig;
use crate::logging::LoggerHandle;
use crate::memory::MemoryMessage;
use crate::NodeHandle;
use async_trait::async_trait;
use log::error;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::Sender;

const ISANAGENT_TOOL_NOTIFY: &str = "isanagent_tool_notify";
const ISANAGENT_TOOL_PHASE: &str = "isanagent_tool_phase";

use crate::channels::terminal_ui::protocol::{
    ISANAGENT_EXECUTION_JOB, ISANAGENT_EXECUTION_JOB_STARTED, ISANAGENT_EXECUTION_STREAM,
    ISANAGENT_SUBAGENT_TASK_FINISHED, ISANAGENT_SUBAGENT_TASK_STARTED, ISANAGENT_TOOL_PROGRESS,
    METADATA_EXECUTION_DESCRIPTION, METADATA_EXECUTION_JOB_ID, METADATA_EXECUTION_JOB_STATUS,
    METADATA_EXECUTION_JOB_TOOL_NAME, METADATA_EXECUTION_RUN_ID, METADATA_EXECUTION_SESSION_ID,
    METADATA_SUBAGENT_AGENT_NAME, METADATA_SUBAGENT_CHILD_CHAT_ID, METADATA_SUBAGENT_DISPLAY_NAME,
    METADATA_SUBAGENT_STATUS, METADATA_SUBAGENT_TASK_ID, METADATA_TOOL_CALL_ID,
    METADATA_TOOL_CALL_PREVIEW, METADATA_TOOL_NAME, METADATA_TOOL_RESULT_CHAR_COUNT,
    METADATA_TOOL_RESULT_PREVIEW,
};

/// When true, `main` skips the large colored stdout banner (Ratatui alternate screen owns the TTY).
pub fn terminal_startup_suppresses_plain_banner(cfg: &AppConfig) -> bool {
    use std::io::{self, IsTerminal};
    cfg.terminal_enabled() && io::stdin().is_terminal() && io::stdout().is_terminal()
}

fn truncate_display(s: &str, max_chars: usize) -> String {
    let t = s.trim();
    let n = t.chars().count();
    if n <= max_chars {
        return t.to_string();
    }
    let shortened: String = t.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{shortened}…")
}

fn truncate_leading_ellipsis(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        return s.to_string();
    }
    if max_chars <= 1 {
        return "…".to_string();
    }
    let tail: String = s.chars().skip(n - (max_chars - 1)).collect();
    format!("…{tail}")
}

fn summarize_message_tool_result(t: &str) -> Option<String> {
    const PREFIX: &str = "Message sent to ";
    let rest = t.strip_prefix(PREFIX)?;
    let (channel, chat_id) = rest.split_once(':')?;
    let chat_id = chat_id.trim();
    if channel.eq_ignore_ascii_case("terminal") && uuid::Uuid::parse_str(chat_id).is_ok() {
        return Some("Message delivered".to_string());
    }
    let id_short = if chat_id.chars().count() > 12 {
        let head: String = chat_id.chars().take(8).collect();
        format!("{head}…")
    } else {
        chat_id.to_string()
    };
    Some(format!("Delivered ({channel}: {id_short})"))
}

fn summarize_tool_result_for_terminal(tool_name: &str, result: &str) -> String {
    let t = result.trim();
    if t.is_empty() {
        return "(empty output)".to_string();
    }
    if tool_name == "message" {
        if let Some(s) = summarize_message_tool_result(t) {
            return s;
        }
    }
    if t.starts_with("Error:") {
        let line = t.lines().next().unwrap_or(t);
        return truncate_display(line, 160);
    }
    truncate_display(t, 120)
}

fn tool_args_preview_message(args: &str) -> String {
    let v: Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return truncate_display(args, 220),
    };
    v.get("content")
        .and_then(|x| x.as_str())
        .map(|c| {
            let t = c.trim();
            if t.is_empty() {
                "(empty)".to_string()
            } else {
                truncate_display(t, 80)
            }
        })
        .unwrap_or_else(|| truncate_display(args, 220))
}

fn tool_args_preview_execution(args: &str) -> String {
    let v: Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return truncate_display(args, 220),
    };
    if let Some(d) = v
        .get("description")
        .and_then(|x| x.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return truncate_display(d, 120);
    }
    if let Some(t) = v.get("timeout_secs").and_then(|x| x.as_u64()) {
        return format!("timeout {t}s");
    }
    if let Some(t) = v.get("timeout_secs").and_then(|x| x.as_i64()) {
        return format!("timeout {t}s");
    }
    if let Some(c) = v.get("code").and_then(|x| x.as_str()) {
        return truncate_display(c.trim(), 80);
    }
    truncate_display(args, 220)
}

fn tool_call_preview_for_terminal(tool_name: &str, args: &str) -> String {
    match tool_name {
        "message" => tool_args_preview_message(args),
        "execution_run" | "execution_run_background" => tool_args_preview_execution(args),
        _ => tool_args_preview_generic_description(args)
            .unwrap_or_else(|| truncate_display(args, 220)),
    }
}

/// Generic fallback: if any tool's args carry a top-level `description` (free-form short
/// human-facing intent), surface it instead of dumping raw JSON. Tools opt in by accepting
/// a `description` arg in their JSON schema.
fn tool_args_preview_generic_description(args: &str) -> Option<String> {
    let v: Value = serde_json::from_str(args).ok()?;
    v.get("description")
        .and_then(|x| x.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| truncate_display(s, 160))
}

/// Decide whether to skip emitting a synthetic tool-call/tool-result notice on the terminal
/// for a given telemetry event. Today this only applies to `MessageTool` whose destination is
/// the terminal: that tool already emits its own `BusMessage::Outbound` carrying the full
/// user-facing text, so a second tool-notify cell would print the same content twice.
///
/// `payload` is the tool args (for ToolCall) or the tool result string (for ToolResult).
pub fn should_suppress_tool_notice_for_terminal(tool_name: &str, payload: &str) -> bool {
    if tool_name != "message" {
        return false;
    }
    if let Ok(v) = serde_json::from_str::<Value>(payload) {
        if let Some(ch) = v.get("channel").and_then(|x| x.as_str()) {
            return ch == "terminal";
        }
    }
    // Result string format is `Message sent to {channel}:{chat_id}` (see MessageTool::execute).
    // Anything else (errors, future formats) keeps the notice visible.
    payload.starts_with("Message sent to terminal:")
}

/// Live `execution_run` stream line for Ratatui (`content` is usually JSON for [`RunEvent`](crate::execution::RunEvent)).
pub fn build_execution_stream_notice(
    chat_id: &str,
    channel: &str,
    session_id: &str,
    run_id: &str,
    content: &str,
    description: Option<&str>,
) -> OutboundMessage {
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_EXECUTION_STREAM.to_string(), json!(true));
    metadata.insert(METADATA_EXECUTION_SESSION_ID.to_string(), json!(session_id));
    metadata.insert(METADATA_EXECUTION_RUN_ID.to_string(), json!(run_id));
    if let Some(d) = description.filter(|s| !s.trim().is_empty()) {
        metadata.insert(METADATA_EXECUTION_DESCRIPTION.to_string(), json!(d.trim()));
    }
    OutboundMessage {
        channel: channel.to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content: content.to_string(),
        metadata,
    }
}

/// Short user-visible line when a background execution job is registered with the manager.
///
/// Surfaces the new job in the multi-job execution strip immediately (before any output is
/// available). Paired with [`build_execution_job_notice`] which fires on completion / failure.
pub fn build_execution_job_started_notice(
    chat_id: &str,
    channel: &str,
    job_id: &str,
    session_id: &str,
    tool_name: &str,
    description: Option<&str>,
) -> OutboundMessage {
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_EXECUTION_JOB_STARTED.to_string(), json!(true));
    metadata.insert(METADATA_EXECUTION_JOB_ID.to_string(), json!(job_id));
    metadata.insert(METADATA_EXECUTION_SESSION_ID.to_string(), json!(session_id));
    metadata.insert(
        METADATA_EXECUTION_JOB_TOOL_NAME.to_string(),
        json!(tool_name),
    );
    if let Some(d) = description.filter(|s| !s.trim().is_empty()) {
        metadata.insert(METADATA_EXECUTION_DESCRIPTION.to_string(), json!(d.trim()));
    }
    let summary = match description.map(str::trim).filter(|s| !s.is_empty()) {
        Some(d) => format!("{tool_name} started: {d}"),
        None => format!("{tool_name} started"),
    };
    OutboundMessage {
        channel: channel.to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content: summary,
        metadata,
    }
}

/// Short user-visible line when a background execution job finishes (or fails).
#[allow(clippy::too_many_arguments)] // Outbound metadata builder; each field maps to protocol keys.
pub fn build_execution_job_notice(
    chat_id: &str,
    channel: &str,
    job_id: &str,
    session_id: &str,
    status: &str,
    summary: &str,
    description: Option<&str>,
    tool_name: Option<&str>,
) -> OutboundMessage {
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_EXECUTION_JOB.to_string(), json!(true));
    metadata.insert(METADATA_EXECUTION_JOB_ID.to_string(), json!(job_id));
    metadata.insert(METADATA_EXECUTION_SESSION_ID.to_string(), json!(session_id));
    metadata.insert(METADATA_EXECUTION_JOB_STATUS.to_string(), json!(status));
    if let Some(d) = description.filter(|s| !s.trim().is_empty()) {
        metadata.insert(METADATA_EXECUTION_DESCRIPTION.to_string(), json!(d.trim()));
    }
    if let Some(t) = tool_name.filter(|s| !s.trim().is_empty()) {
        metadata.insert(
            METADATA_EXECUTION_JOB_TOOL_NAME.to_string(),
            json!(t.trim()),
        );
    }
    OutboundMessage {
        channel: channel.to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content: summary.to_string(),
        metadata,
    }
}

/// Short user-visible line when a sub-agent task is spawned.
pub fn build_subagent_task_started_notice(
    chat_id: &str,
    task_id: &str,
    child_chat_id: &str,
    agent_name: Option<&str>,
    display_name: Option<&str>,
) -> OutboundMessage {
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_SUBAGENT_TASK_STARTED.to_string(), json!(true));
    metadata.insert(METADATA_SUBAGENT_TASK_ID.to_string(), json!(task_id));
    metadata.insert(
        METADATA_SUBAGENT_CHILD_CHAT_ID.to_string(),
        json!(child_chat_id),
    );
    if let Some(a) = agent_name.filter(|s| !s.is_empty()) {
        metadata.insert(METADATA_SUBAGENT_AGENT_NAME.to_string(), json!(a));
    }
    if let Some(d) = display_name.filter(|s| !s.is_empty()) {
        metadata.insert(METADATA_SUBAGENT_DISPLAY_NAME.to_string(), json!(d));
    }
    let label = match (agent_name, display_name) {
        (Some(a), Some(d)) => format!("{a}: {d}"),
        (Some(a), None) => a.to_string(),
        (None, Some(d)) => d.to_string(),
        (None, None) => {
            let short = &task_id[..8.min(task_id.len())];
            format!("task-{short}")
        }
    };
    let content = format!("Sub-agent started: {label}");
    OutboundMessage {
        channel: "terminal".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content,
        metadata,
    }
}

/// Short user-visible line when a sub-agent task finishes (completed / failed / cancelled).
pub fn build_subagent_task_finished_notice(
    chat_id: &str,
    task_id: &str,
    child_chat_id: &str,
    status: &str,
    summary: &str,
    agent_name: Option<&str>,
    display_name: Option<&str>,
) -> OutboundMessage {
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_SUBAGENT_TASK_FINISHED.to_string(), json!(true));
    metadata.insert(METADATA_SUBAGENT_TASK_ID.to_string(), json!(task_id));
    metadata.insert(
        METADATA_SUBAGENT_CHILD_CHAT_ID.to_string(),
        json!(child_chat_id),
    );
    metadata.insert(METADATA_SUBAGENT_STATUS.to_string(), json!(status));
    if let Some(a) = agent_name.filter(|s| !s.is_empty()) {
        metadata.insert(METADATA_SUBAGENT_AGENT_NAME.to_string(), json!(a));
    }
    if let Some(d) = display_name.filter(|s| !s.is_empty()) {
        metadata.insert(METADATA_SUBAGENT_DISPLAY_NAME.to_string(), json!(d));
    }
    let label = match (agent_name, display_name) {
        (Some(a), Some(d)) => format!("{a}: {d}"),
        (Some(a), None) => a.to_string(),
        (None, Some(d)) => d.to_string(),
        (None, None) => {
            let short = &task_id[..8.min(task_id.len())];
            format!("task-{short}")
        }
    };
    let summary = summary.trim();
    let content = if summary.is_empty() {
        format!("Sub-agent finished ({status}): {label}")
    } else {
        format!("Sub-agent finished ({status}): {label} — {summary}")
    };
    OutboundMessage {
        channel: "terminal".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content,
        metadata,
    }
}

/// Ephemeral mid–tool status for the Ratatui tool strip (no transcript cell).
pub fn build_tool_progress_terminal_notice(
    chat_id: &str,
    tool_name: &str,
    message: &str,
    tool_call_id: Option<&str>,
    background_job_id: Option<&str>,
) -> OutboundMessage {
    let detail = message.trim();
    let content = if detail.is_empty() {
        tool_name.to_string()
    } else {
        format!("{tool_name} — {detail}")
    };
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_TOOL_PROGRESS.to_string(), json!(true));
    metadata.insert(METADATA_TOOL_NAME.to_string(), json!(tool_name));
    if let Some(id) = tool_call_id.filter(|s| !s.is_empty()) {
        metadata.insert(METADATA_TOOL_CALL_ID.to_string(), json!(id));
    }
    if let Some(id) = background_job_id.filter(|s| !s.is_empty()) {
        metadata.insert(
            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
            json!(id),
        );
    }
    OutboundMessage {
        channel: "terminal".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content,
        metadata,
    }
}

/// Live terminal line when a tool is invoked (mirrors telemetry, user-visible).
pub fn build_tool_call_terminal_notice(
    chat_id: &str,
    tool_name: &str,
    args: &str,
    tool_call_id: Option<&str>,
    background_job_id: Option<&str>,
) -> OutboundMessage {
    let preview = tool_call_preview_for_terminal(tool_name, args);
    let content = if preview.is_empty() {
        tool_name.to_string()
    } else {
        format!("{tool_name} {preview}")
    };
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_TOOL_NOTIFY.to_string(), json!(true));
    metadata.insert(ISANAGENT_TOOL_PHASE.to_string(), json!("call"));
    metadata.insert(METADATA_TOOL_NAME.to_string(), json!(tool_name));
    metadata.insert(METADATA_TOOL_CALL_PREVIEW.to_string(), json!(preview));
    if let Some(id) = tool_call_id.filter(|s| !s.is_empty()) {
        metadata.insert(METADATA_TOOL_CALL_ID.to_string(), json!(id));
    }
    if let Some(id) = background_job_id.filter(|s| !s.is_empty()) {
        metadata.insert(
            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
            json!(id),
        );
    }
    OutboundMessage {
        channel: "terminal".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content,
        metadata,
    }
}

/// Live terminal row for model reasoning / thought telemetry (Ratatui → `Cell::Thinking`).
pub fn build_agent_thought_terminal_notice(
    chat_id: &str,
    thought: &str,
    background_job_id: Option<&str>,
) -> OutboundMessage {
    let mut metadata = HashMap::new();
    metadata.insert(
        crate::channels::terminal_ui::protocol::ISANAGENT_AGENT_THOUGHT.to_string(),
        json!(true),
    );
    if let Some(id) = background_job_id.filter(|s| !s.is_empty()) {
        metadata.insert(
            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
            json!(id),
        );
    }
    OutboundMessage {
        channel: "terminal".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content: thought.to_string(),
        metadata,
    }
}

/// Live terminal line when a tool finishes (short summary; avoids flooding the TTY).
pub fn build_tool_result_terminal_notice(
    chat_id: &str,
    tool_name: &str,
    result: &str,
    is_error: bool,
    tool_call_id: Option<&str>,
    background_job_id: Option<&str>,
) -> OutboundMessage {
    let t = result.trim();
    let summary = summarize_tool_result_for_terminal(tool_name, result);
    let content = format!("{tool_name} → {summary}");
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_TOOL_NOTIFY.to_string(), json!(true));
    let phase = if is_error { "fail" } else { "result" };
    metadata.insert(ISANAGENT_TOOL_PHASE.to_string(), json!(phase));
    metadata.insert(METADATA_TOOL_NAME.to_string(), json!(tool_name));
    metadata.insert(METADATA_TOOL_RESULT_PREVIEW.to_string(), json!(summary));
    if t.chars().count() > 120 {
        metadata.insert(
            METADATA_TOOL_RESULT_CHAR_COUNT.to_string(),
            json!(t.chars().count()),
        );
    }
    if let Some(id) = tool_call_id.filter(|s| !s.is_empty()) {
        metadata.insert(METADATA_TOOL_CALL_ID.to_string(), json!(id));
    }
    if let Some(id) = background_job_id.filter(|s| !s.is_empty()) {
        metadata.insert(
            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
            json!(id),
        );
    }
    OutboundMessage {
        channel: "terminal".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content,
        metadata,
    }
}

/// User-visible error notice for reasoning / provider failures, routed to `channel`.
///
/// Terminal UI metadata (`isanagent_terminal_error`) is attached only when `channel` is
/// `"terminal"` so other channels are not mis-tagged.
pub fn build_channel_error_notice(
    channel: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    message: &str,
) -> OutboundMessage {
    let mut metadata = HashMap::new();
    if channel == "terminal" {
        metadata.insert(
            crate::channels::terminal_ui::protocol::ISANAGENT_TERMINAL_ERROR.to_string(),
            json!(true),
        );
    }
    OutboundMessage {
        channel: channel.to_string(),
        chat_id: chat_id.to_string(),
        thread_id: thread_id.map(|s| s.to_string()),
        content: message.to_string(),
        metadata,
    }
}

/// Constructor parameters for `TerminalChannel`.
pub struct TerminalChannelConfig {
    pub chat_id: String,
    pub logger_tx: LoggerHandle,
    pub shutdown_tx: tokio::sync::mpsc::UnboundedSender<()>,
    pub workspace_dir: PathBuf,
    pub sandbox_dir: PathBuf,
    pub status_model: String,
    /// Short permission label for the status bar (`ask`, `plan`, …).
    pub status_permission: String,
    pub memory_node: NodeHandle<MemoryMessage>,
    pub providers: std::collections::HashMap<String, crate::config::ProviderConfig>,
    /// Whether the TUI should render ANSI foreground colors.
    pub color_enabled: bool,
    /// Host-selected ALTAI theme (resolved with `color_enabled` / NO_COLOR).
    pub theme: crate::channels::terminal_ui::HostThemeMode,
    /// Load the configured chat's persisted transcript before accepting input.
    pub resume_session: bool,
    /// File references composed into the first user message.
    pub initial_files: Vec<PathBuf>,
    pub mode: TerminalMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalMode {
    Tui,
    Line,
}

/// Stdin/stdout terminal: always Ratatui (alternate screen). Requires an interactive TTY.
pub struct TerminalChannel {
    chat_id: String,
    logger_tx: LoggerHandle,
    shutdown_tx: tokio::sync::mpsc::UnboundedSender<()>,
    /// Workspace root (`config.toml`, `.system_generated/`, execution journals).
    workspace_dir: PathBuf,
    /// All user-supplied `@<filepath>` references are resolved relative to this
    /// directory.  Paths that escape the sandbox boundary are silently rejected.
    sandbox_dir: PathBuf,
    /// Provider model id for the status line (e.g. from config).
    status_model: String,
    status_permission: String,
    /// Workspace memory actor (for past-session list + transcript load in the TUI thread).
    memory_node: NodeHandle<MemoryMessage>,
    /// Outbound messages for the Ratatui thread (set when `start` succeeds).
    outbound_ui_tx: Arc<Mutex<Option<std::sync::mpsc::Sender<OutboundMessage>>>>,
    /// Named alternative providers for `/model` switching.
    providers: std::collections::HashMap<String, crate::config::ProviderConfig>,
    color_enabled: bool,
    theme: crate::channels::terminal_ui::HostThemeMode,
    resume_session: bool,
    initial_files: Vec<PathBuf>,
    mode: TerminalMode,
}

impl TerminalChannel {
    pub fn new(config: TerminalChannelConfig) -> Self {
        Self {
            chat_id: config.chat_id,
            logger_tx: config.logger_tx,
            shutdown_tx: config.shutdown_tx,
            workspace_dir: config.workspace_dir,
            sandbox_dir: config.sandbox_dir,
            status_model: config.status_model,
            status_permission: config.status_permission,
            memory_node: config.memory_node,
            outbound_ui_tx: Arc::new(Mutex::new(None)),
            providers: config.providers,
            color_enabled: config.color_enabled,
            theme: config.theme,
            resume_session: config.resume_session,
            initial_files: config.initial_files,
            mode: config.mode,
        }
    }
}

#[async_trait]
impl Channel for TerminalChannel {
    fn name(&self) -> &str {
        "terminal"
    }

    async fn start(&self, bus_tx: Sender<BusMessage>) -> Result<(), String> {
        let tty_in = io::stdin().is_terminal();
        let tty_out = io::stdout().is_terminal();
        if !tty_in || !tty_out {
            return Err(
                "Terminal channel requires an interactive terminal (stdin and stdout must be TTYs). \
For headless or piped runs, set [terminal] enabled = false in config.toml (requires another inbound channel such as API, Slack, or Email)."
                    .to_string(),
            );
        }

        let channel_name = self.name().to_string();
        if self.mode == TerminalMode::Line {
            crate::channels::terminal_ui::init_from_host(self.theme, !self.color_enabled);
            let (tx, rx) = std::sync::mpsc::channel::<OutboundMessage>();
            *self
                .outbound_ui_tx
                .lock()
                .map_err(|_| "terminal outbound bridge poisoned".to_string())? = Some(tx);
            let chat_id = self.chat_id.clone();
            let shutdown = self.shutdown_tx.clone();
            let status_model = self.status_model.clone();
            let status_permission = self.status_permission.clone();
            let sandbox_label = self
                .sandbox_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("workspace")
                .to_string();
            let session_short = truncate_leading_ellipsis(&chat_id, 13);
            std::thread::spawn(move || {
                for message in rx {
                    let is_clarification = message
                        .metadata
                        .get(crate::clarification::METADATA_CLARIFICATION)
                        .and_then(|v| v.as_bool())
                        == Some(true);
                    let prefix = if is_clarification {
                        "approval"
                    } else if message
                        .metadata
                        .get(ISANAGENT_TOOL_NOTIFY)
                        .and_then(|v| v.as_bool())
                        == Some(true)
                    {
                        "tool"
                    } else {
                        "assistant"
                    };
                    println!("[{prefix}] {}", message.content);
                    if let Some(edit) = message.metadata.get("edit_diff") {
                        let file = edit
                            .get("file")
                            .and_then(|v| v.as_str())
                            .unwrap_or("(unknown)");
                        let truncated = edit
                            .get("truncated")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        let badge = if truncated { " [truncated]" } else { "" };
                        println!("[edit_diff] {file}{badge}");
                        if let Some(diff) = edit.get("diff").and_then(|v| v.as_str()) {
                            println!("{diff}");
                        }
                    }
                    if is_clarification {
                        println!(
                            "[choices] 1=approve 2=deny 3=always 4=abort  (type the word or number)"
                        );
                    }
                }
            });
            let sandbox_dir = self.sandbox_dir.clone();
            let memory_node = self.memory_node.clone();
            let channel_name_for_line = channel_name.clone();
            let mut pending_host_files = self.initial_files.clone();
            std::thread::spawn(move || {
                use std::io::BufRead;
                println!(
                    "ALTAI line mode · {sandbox_label} · {status_model} · {status_permission} · session {session_short}"
                );
                println!(
                    "Commands: /exit · /context · /compact [focus] · @file attachments. Color: {}",
                    if crate::channels::terminal_ui::uses_ansi_color() {
                        "on"
                    } else {
                        "off (plain)"
                    }
                );
                if !pending_host_files.is_empty() {
                    let refs = pending_host_files
                        .iter()
                        .map(|path| format!("@{}", path.display()))
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!(
                        "Pending --file attachments ({refs}) will load with your first message."
                    );
                }
                print!("> ");
                let _ = std::io::Write::flush(&mut std::io::stdout());
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(error) => {
                        eprintln!("line mode runtime failed: {error}");
                        return;
                    }
                };
                for line in std::io::stdin().lock().lines() {
                    let Ok(content) = line else { break };
                    let trimmed = content.trim();
                    if matches!(trimmed, "/exit" | "/quit") {
                        let _ = shutdown.send(());
                        break;
                    }
                    if trimmed.is_empty() {
                        print!("> ");
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                        continue;
                    }
                    if trimmed.eq_ignore_ascii_case("/context") {
                        let session_key = crate::bus::clarification_session_key(
                            &channel_name_for_line,
                            &chat_id,
                            None,
                        );
                        let messages = rt.block_on(async {
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            let _ = memory_node
                                .send_packet(crate::memory::MemoryMessage::GetContext {
                                    thread_id: session_key.clone(),
                                    reply: crate::memory::SharedReply::new(tx),
                                })
                                .await;
                            rx.await.ok().and_then(|r| r.ok()).unwrap_or_default()
                        });
                        let user_turns = messages.iter().filter(|m| m.role == "user").count();
                        let approx_tokens: usize = messages
                            .iter()
                            .map(|m| m.content.as_ref().map_or(0, |c| c.text_content().len()) / 4)
                            .sum();
                        println!(
                            "[context] {} message(s) · {} user turn(s) · ~{} tokens (rough estimate). Use /compact to force compaction.",
                            messages.len(),
                            user_turns,
                            approx_tokens
                        );
                        print!("> ");
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                        continue;
                    }
                    if trimmed.eq_ignore_ascii_case("/compact")
                        || trimmed.to_ascii_lowercase().starts_with("/compact ")
                    {
                        let focus = trimmed
                            .strip_prefix("/compact")
                            .or_else(|| trimmed.strip_prefix("/COMPACT"))
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        let session_key = crate::bus::clarification_session_key(
                            &channel_name_for_line,
                            &chat_id,
                            None,
                        );
                        let msg = BusMessage::TriggerCompaction {
                            session_key,
                            focus_instructions: if focus.is_empty() {
                                None
                            } else {
                                Some(focus.clone())
                            },
                            trigger: Some(crate::bus::CompactionTrigger::Manual),
                        };
                        if bus_tx.blocking_send(msg).is_err() {
                            println!("[system] Bus closed; cannot trigger compaction.");
                            break;
                        }
                        if focus.is_empty() {
                            println!("[system] Compaction requested. It will run between turns.");
                        } else {
                            println!(
                                "[system] Compaction requested with focus: \"{focus}\"."
                            );
                        }
                        print!("> ");
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                        continue;
                    }

                    let (clean_text, mut attachments) =
                        crate::channels::terminal_ui::parse_terminal_attachments(
                            &content,
                            &sandbox_dir,
                        );
                    if !pending_host_files.is_empty() {
                        let (host_parts, warnings) =
                            crate::channels::terminal_ui::load_host_file_attachments(
                                &sandbox_dir,
                                &pending_host_files,
                            );
                        for warning in warnings {
                            eprintln!("Warning: {warning}");
                        }
                        attachments.extend(host_parts);
                        pending_host_files.clear();
                    }
                    if clean_text.trim().is_empty() && attachments.is_empty() {
                        print!("> ");
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                        continue;
                    }
                    if bus_tx
                        .blocking_send(BusMessage::Inbound(InboundMessage {
                            channel: "terminal".into(),
                            sender_id: "local_user".into(),
                            chat_id: chat_id.clone(),
                            thread_id: None,
                            content: if clean_text.is_empty() {
                                "(attached files)".into()
                            } else {
                                clean_text
                            },
                            attachments,
                            metadata: Default::default(),
                        }))
                        .is_err()
                    {
                        break;
                    }
                    print!("> ");
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
            });
            return Ok(());
        }
        let chat_id_clone = self.chat_id.clone();
        let status_model = self.status_model.clone();
        let logger_tx = self.logger_tx.clone();
        let shutdown_tx = self.shutdown_tx.clone();
        let sandbox_dir = self.sandbox_dir.clone();
        let workspace_dir = self.workspace_dir.clone();
        let providers_clone = self.providers.clone();

        let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
            "TerminalChannel",
            "Starting Terminal channel (Ratatui alternate screen)…",
        )));

        let (tx, rx) = std::sync::mpsc::channel::<OutboundMessage>();
        {
            let mut g = self
                .outbound_ui_tx
                .lock()
                .map_err(|_| "terminal outbound bridge poisoned".to_string())?;
            *g = Some(tx);
        }
        let bridge = self.outbound_ui_tx.clone();
        let bus_tx_clone = bus_tx.clone();
        let shutdown_clone = shutdown_tx.clone();
        let sandbox_clone = sandbox_dir.clone();
        let log_clone = logger_tx.clone();
        let memory_node_clone = self.memory_node.clone();
        let color_enabled = self.color_enabled;
        let theme = self.theme;
        let resume_session = self.resume_session;
        let initial_files = self.initial_files.clone();
        let status_permission = self.status_permission.clone();

        let opening_banner = format!(
            "ALTAI isanagent v{} — thread {}\n\
             Commands: /exit, /new, /context, /compact  ·  Attachments: @path (text/image/PDF) inside the workspace.",
            env!("CARGO_PKG_VERSION"),
            truncate_leading_ellipsis(&chat_id_clone, 13)
        );

        std::thread::Builder::new()
            .name("isanagent-terminal-tui".into())
            .spawn(move || {
                let res = crate::channels::terminal_ui::run_ratatui_main(
                    crate::channels::terminal_ui::RatatuiMainConfig {
                        bus_tx: bus_tx_clone,
                        outbound_rx: rx,
                        shutdown_tx: shutdown_clone,
                        workspace_dir,
                        sandbox_dir: sandbox_clone,
                        chat_id: chat_id_clone,
                        channel_name,
                        opening_banner,
                        status_model,
                        status_permission,
                        memory_node: memory_node_clone,
                        providers: providers_clone,
                        color_enabled,
                        theme,
                        resume_session,
                        initial_files,
                    },
                );
                if let Ok(mut g) = bridge.lock() {
                    *g = None;
                }
                if let Err(e) = res {
                    let _ = log_clone.send(BusMessage::Log(LogEvent::error(
                        "TerminalChannel",
                        &format!("Ratatui terminal ended: {e}"),
                    )));
                }
            })
            .map_err(|e| format!("failed to spawn terminal TUI thread: {e}"))?;

        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        let guard = self
            .outbound_ui_tx
            .lock()
            .map_err(|_| "terminal outbound bridge poisoned".to_string())?;
        if let Some(tx) = guard.as_ref() {
            if tx.send(msg).is_err() {
                error!("TerminalChannel: outbound UI disconnected; dropping message.");
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod preview_tests {
    use super::*;

    #[test]
    fn unknown_tool_falls_back_to_generic_description() {
        let args = r#"{"description":"do the thing","other":"x"}"#;
        let p = tool_call_preview_for_terminal("some_future_tool", args);
        assert!(
            p.contains("do the thing"),
            "expected description fallback: {p}"
        );
        assert!(!p.contains("\"other\""), "should not dump raw JSON: {p}");
    }

    #[test]
    fn unknown_tool_without_description_truncates_args() {
        let args = r#"{"a":1}"#;
        let p = tool_call_preview_for_terminal("some_future_tool", args);
        assert_eq!(p, args);
    }

    #[test]
    fn message_preview_unchanged() {
        let args = r#"{"content":"hi","channel":"terminal","chat_id":"x"}"#;
        let p = tool_call_preview_for_terminal("message", args);
        assert!(p.contains("hi"));
    }

    #[test]
    fn suppress_message_tool_call_to_terminal() {
        let args = r#"{"content":"hi","channel":"terminal","chat_id":"x"}"#;
        assert!(should_suppress_tool_notice_for_terminal("message", args));
    }

    #[test]
    fn keep_message_tool_call_to_other_channel() {
        let args = r#"{"content":"hi","channel":"slack","chat_id":"x"}"#;
        assert!(!should_suppress_tool_notice_for_terminal("message", args));
    }

    #[test]
    fn suppress_message_tool_result_to_terminal() {
        let result = "Message sent to terminal:abc-123";
        assert!(should_suppress_tool_notice_for_terminal("message", result));
    }

    #[test]
    fn keep_message_tool_result_to_other_channel() {
        let result = "Message sent to slack:abc-123";
        assert!(!should_suppress_tool_notice_for_terminal("message", result));
    }

    #[test]
    fn keep_message_tool_error_result() {
        let result = "Failed to send: connection refused";
        assert!(!should_suppress_tool_notice_for_terminal("message", result));
    }

    #[test]
    fn other_tools_never_suppressed() {
        let args = r#"{"channel":"terminal"}"#;
        assert!(!should_suppress_tool_notice_for_terminal(
            "execution_run",
            args
        ));
    }

    #[test]
    fn build_tool_call_terminal_notice_attaches_tool_call_id() {
        let notice = build_tool_call_terminal_notice(
            "chat-1",
            "execution_run",
            r#"{"description":"warm up"}"#,
            Some("call-abc"),
            None,
        );
        assert_eq!(
            notice
                .metadata
                .get(METADATA_TOOL_CALL_ID)
                .and_then(|v| v.as_str()),
            Some("call-abc"),
        );
    }

    #[test]
    fn build_tool_call_terminal_notice_omits_tool_call_id_when_none() {
        let notice = build_tool_call_terminal_notice(
            "chat-1",
            "execution_run",
            r#"{"description":"warm up"}"#,
            None,
            None,
        );
        assert!(!notice.metadata.contains_key(METADATA_TOOL_CALL_ID));
    }

    #[test]
    fn build_tool_result_terminal_notice_attaches_tool_call_id() {
        let notice = build_tool_result_terminal_notice(
            "chat-1",
            "execution_run",
            "exit 0 in 12ms",
            false,
            Some("call-abc"),
            None,
        );
        assert_eq!(
            notice
                .metadata
                .get(METADATA_TOOL_CALL_ID)
                .and_then(|v| v.as_str()),
            Some("call-abc"),
        );
    }

    #[test]
    fn build_tool_result_terminal_notice_keeps_text_preview_for_long_outputs() {
        let long = "x".repeat(200);
        let notice = build_tool_result_terminal_notice(
            "chat-1",
            "execution_run",
            long.as_str(),
            false,
            None,
            None,
        );
        let preview = notice
            .metadata
            .get(METADATA_TOOL_RESULT_PREVIEW)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(!preview.ends_with(" chars"), "{preview}");
        assert!(preview.ends_with('…'), "{preview}");
    }

    #[test]
    fn build_tool_result_terminal_notice_sets_char_count_metadata_for_long_outputs() {
        let long = "x".repeat(200);
        let notice = build_tool_result_terminal_notice(
            "chat-1",
            "execution_run",
            long.as_str(),
            false,
            None,
            None,
        );
        assert_eq!(
            notice
                .metadata
                .get(METADATA_TOOL_RESULT_CHAR_COUNT)
                .and_then(|v| v.as_u64()),
            Some(200)
        );
    }

    #[test]
    fn tool_result_terminal_phase_uses_typed_status_not_result_text() {
        let text_that_looks_like_error = build_tool_result_terminal_notice(
            "chat-1",
            "read_file",
            "Error: this is an expected literal from the file",
            false,
            None,
            None,
        );
        assert_eq!(
            text_that_looks_like_error
                .metadata
                .get(ISANAGENT_TOOL_PHASE)
                .and_then(|value| value.as_str()),
            Some("result"),
        );

        let generic_native_error = build_tool_result_terminal_notice(
            "chat-1",
            "write_file",
            "Permission denied by workspace policy",
            true,
            None,
            None,
        );
        assert_eq!(
            generic_native_error
                .metadata
                .get(ISANAGENT_TOOL_PHASE)
                .and_then(|value| value.as_str()),
            Some("fail"),
        );
    }
}
