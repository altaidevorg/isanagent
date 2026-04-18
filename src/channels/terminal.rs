use crate::bus::{BusMessage, LogEvent, OutboundMessage};
use crate::channels::Channel;
use crate::config::AppConfig;
use crate::logging::LoggerHandle;
use async_trait::async_trait;
use log::error;
use serde_json::json;
use std::collections::HashMap;
use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::Sender;

const ISANAGENT_TOOL_NOTIFY: &str = "isanagent_tool_notify";
const ISANAGENT_TOOL_PHASE: &str = "isanagent_tool_phase";

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

fn summarize_tool_result_for_terminal(result: &str) -> String {
    let t = result.trim();
    if t.is_empty() {
        return "(empty output)".to_string();
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

/// Live terminal line when a tool is invoked (mirrors telemetry, user-visible).
pub fn build_tool_call_terminal_notice(
    chat_id: &str,
    tool_name: &str,
    args: &str,
) -> OutboundMessage {
    let detail = truncate_display(args, 220);
    let content = if detail.is_empty() {
        tool_name.to_string()
    } else {
        format!("{tool_name} {detail}")
    };
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_TOOL_NOTIFY.to_string(), json!(true));
    metadata.insert(ISANAGENT_TOOL_PHASE.to_string(), json!("call"));
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
    let summary = summarize_tool_result_for_terminal(result);
    let content = format!("{tool_name} → {summary}");
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_TOOL_NOTIFY.to_string(), json!(true));
    let phase = if tool_result_looks_like_failure(result) {
        "fail"
    } else {
        "result"
    };
    metadata.insert(ISANAGENT_TOOL_PHASE.to_string(), json!(phase));
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
For headless or piped runs, set [terminal] enable = false in config.toml (requires another inbound channel such as API, Slack, or Email)."
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
