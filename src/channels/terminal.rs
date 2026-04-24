use crate::bus::{BusMessage, LogEvent, OutboundMessage};
use crate::channels::Channel;
use crate::config::AppConfig;
use crate::logging::LoggerHandle;
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
    ISANAGENT_EXECUTION_JOB, ISANAGENT_EXECUTION_STREAM, METADATA_EXECUTION_DESCRIPTION,
    METADATA_EXECUTION_JOB_ID, METADATA_EXECUTION_JOB_STATUS, METADATA_EXECUTION_RUN_ID,
    METADATA_EXECUTION_SESSION_ID, METADATA_TOOL_CALL_PREVIEW, METADATA_TOOL_NAME,
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

fn tool_result_looks_like_failure(result: &str) -> bool {
    let t = result.trim_start();
    t.starts_with("Error:") || t.starts_with("error:")
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
    if t.chars().count() <= 120 {
        return t.to_string();
    }
    format!("{} chars", t.chars().count())
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
        _ => truncate_display(args, 220),
    }
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
        metadata.insert(
            METADATA_EXECUTION_DESCRIPTION.to_string(),
            json!(d.trim()),
        );
    }
    OutboundMessage {
        channel: channel.to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content: content.to_string(),
        metadata,
    }
}

/// Short user-visible line when a background execution job finishes (or fails).
pub fn build_execution_job_notice(
    chat_id: &str,
    channel: &str,
    job_id: &str,
    session_id: &str,
    status: &str,
    summary: &str,
    description: Option<&str>,
) -> OutboundMessage {
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_EXECUTION_JOB.to_string(), json!(true));
    metadata.insert(METADATA_EXECUTION_JOB_ID.to_string(), json!(job_id));
    metadata.insert(METADATA_EXECUTION_SESSION_ID.to_string(), json!(session_id));
    metadata.insert(
        METADATA_EXECUTION_JOB_STATUS.to_string(),
        json!(status),
    );
    if let Some(d) = description.filter(|s| !s.trim().is_empty()) {
        metadata.insert(
            METADATA_EXECUTION_DESCRIPTION.to_string(),
            json!(d.trim()),
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

/// Live terminal line when a tool is invoked (mirrors telemetry, user-visible).
pub fn build_tool_call_terminal_notice(
    chat_id: &str,
    tool_name: &str,
    args: &str,
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
    OutboundMessage {
        channel: "terminal".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content,
        metadata,
    }
}

/// Live terminal row for model reasoning / thought telemetry (Ratatui → `Cell::Thinking`).
pub fn build_agent_thought_terminal_notice(chat_id: &str, thought: &str) -> OutboundMessage {
    let mut metadata = HashMap::new();
    metadata.insert(
        crate::channels::terminal_ui::protocol::ISANAGENT_AGENT_THOUGHT.to_string(),
        json!(true),
    );
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
) -> OutboundMessage {
    let summary = summarize_tool_result_for_terminal(tool_name, result);
    let content = format!("{tool_name} → {summary}");
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_TOOL_NOTIFY.to_string(), json!(true));
    let phase = if tool_result_looks_like_failure(result) {
        "fail"
    } else {
        "result"
    };
    metadata.insert(ISANAGENT_TOOL_PHASE.to_string(), json!(phase));
    metadata.insert(METADATA_TOOL_NAME.to_string(), json!(tool_name));
    metadata.insert(METADATA_TOOL_RESULT_PREVIEW.to_string(), json!(summary));
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

/// Stdin/stdout terminal: always Ratatui (alternate screen). Requires an interactive TTY.
pub struct TerminalChannel {
    chat_id: String,
    logger_tx: LoggerHandle,
    shutdown_tx: tokio::sync::mpsc::UnboundedSender<()>,
    /// All user-supplied `@<filepath>` references are resolved relative to this
    /// directory.  Paths that escape the sandbox boundary are silently rejected.
    sandbox_dir: PathBuf,
    /// Provider model id for the status line (e.g. from config).
    status_model: String,
    /// Outbound messages for the Ratatui thread (set when `start` succeeds).
    outbound_ui_tx: Arc<Mutex<Option<std::sync::mpsc::Sender<OutboundMessage>>>>,
}

impl TerminalChannel {
    pub fn new(
        chat_id: &str,
        logger_tx: LoggerHandle,
        shutdown_tx: tokio::sync::mpsc::UnboundedSender<()>,
        sandbox_dir: PathBuf,
        status_model: String,
    ) -> Self {
        Self {
            chat_id: chat_id.to_string(),
            logger_tx,
            shutdown_tx,
            sandbox_dir,
            status_model,
            outbound_ui_tx: Arc::new(Mutex::new(None)),
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
        let chat_id_clone = self.chat_id.clone();
        let status_model = self.status_model.clone();
        let logger_tx = self.logger_tx.clone();
        let shutdown_tx = self.shutdown_tx.clone();
        let sandbox_dir = self.sandbox_dir.clone();

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

        let session_banner = format!(
            "isanagent v{} — session {}\n\
             Commands: /exit, /new  ·  Images: @path/to/file inside the workspace.",
            env!("CARGO_PKG_VERSION"),
            chat_id_clone
        );

        std::thread::Builder::new()
            .name("isanagent-terminal-tui".into())
            .spawn(move || {
                let res = crate::channels::terminal_ui::run_ratatui_main(
                    crate::channels::terminal_ui::RatatuiMainConfig {
                        bus_tx: bus_tx_clone,
                        outbound_rx: rx,
                        shutdown_tx: shutdown_clone,
                        sandbox_dir: sandbox_clone,
                        chat_id: chat_id_clone,
                        channel_name,
                        session_banner,
                        status_model,
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
