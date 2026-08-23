use crate::bus::{BusMessage, InboundMessage, LogEvent, OutboundMessage};
use crate::channels::Channel;
use crate::config::AppConfig;
use crate::logging::LoggerHandle;
use crate::memory::MemoryMessage;
use crate::NodeHandle;
use async_trait::async_trait;
use log::error;
use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::Sender;

use crate::protocol::{truncate_leading_ellipsis, ISANAGENT_TOOL_NOTIFY};

/// When true, `main` skips the large colored stdout banner (Ratatui alternate screen owns the TTY).
pub fn terminal_startup_suppresses_plain_banner(cfg: &AppConfig) -> bool {
    use std::io::{self, IsTerminal};
    cfg.terminal_enabled() && io::stdin().is_terminal() && io::stdout().is_terminal()
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
            let workspace_dir = self.workspace_dir.clone();
            let mut providers = self.providers.clone();
            let memory_node = self.memory_node.clone();
            let channel_name_for_line = channel_name.clone();
            let mut pending_host_files = self.initial_files.clone();
            std::thread::spawn(move || {
                use std::io::BufRead;
                let mut active_provider_key =
                    crate::channels::terminal_ui::resolve_initial_active_provider_key(
                        &workspace_dir,
                        &providers,
                    );
                println!(
                    "ALTAI line mode · {sandbox_label} · {status_model} · {status_permission} · session {session_short}"
                );
                println!(
                    "Commands: /exit · /context · /compact [focus] · /key <api_key> · @file attachments. Color: {}",
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
                            println!("[system] Compaction requested with focus: \"{focus}\".");
                        }
                        print!("> ");
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                        continue;
                    }
                    if trimmed.eq_ignore_ascii_case("/key")
                        || trimmed.to_ascii_lowercase().starts_with("/key ")
                    {
                        let arg = trimmed.strip_prefix("/key").unwrap_or("").trim();
                        if arg.is_empty() {
                            println!(
                                "[system] Usage: /key <api_key>  (or /key <provider_config_key> <api_key>)"
                            );
                            print!("> ");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                            continue;
                        }
                        let mut parts = arg.splitn(2, char::is_whitespace);
                        let first = parts.next().unwrap_or("").trim();
                        let rest = parts.next().unwrap_or("").trim();
                        let (config_key_arg, secret): (Option<&str>, &str) =
                            if !rest.is_empty() && providers.contains_key(first) {
                                (Some(first), rest)
                            } else {
                                (None, arg)
                            };
                        let resolved_config_key = config_key_arg
                            .map(|s| s.to_string())
                            .or_else(|| active_provider_key.clone())
                            .or_else(|| {
                                if providers.len() == 1 {
                                    providers.keys().next().cloned()
                                } else {
                                    None
                                }
                            });
                        let Some(resolved_config_key) = resolved_config_key else {
                            let mut available: Vec<&str> =
                                providers.keys().map(|s| s.as_str()).collect();
                            available.sort_unstable();
                            println!(
                                "[system] Multiple providers configured; specify one: /key <provider_config_key> <api_key>. Available: {}. Or run /model first, then /key.",
                                available.join(", ")
                            );
                            print!("> ");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                            continue;
                        };
                        if crate::channels::terminal_ui::key_looks_like_placeholder(secret) {
                            println!(
                                "[system] That doesn't look like a real API key. Usage: /key <api_key>"
                            );
                            print!("> ");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                            continue;
                        }
                        match providers.get_mut(&resolved_config_key) {
                            None => {
                                let mut available: Vec<&str> =
                                    providers.keys().map(|s| s.as_str()).collect();
                                available.sort_unstable();
                                println!(
                                    "[system] Unknown provider '{resolved_config_key}'. Available: {}.",
                                    available.join(", ")
                                );
                            }
                            Some(cfg) => {
                                cfg.api_key = Some(secret.to_string());
                                let provider_name = cfg.provider_name.clone();
                                let model_name = cfg.model_name.clone();
                                let resolved_url = cfg.resolved_base_url().unwrap_or_default();
                                match cfg.resolve_api_key() {
                                    Ok(resolved_key) => {
                                        let masked =
                                            crate::channels::terminal_ui::mask_api_key_suffix(
                                                &resolved_key,
                                            );
                                        match crate::channels::terminal_ui::persist_provider_api_key(
                                            &workspace_dir,
                                            &resolved_config_key,
                                            secret,
                                        ) {
                                            Ok(()) => println!(
                                                "[system] API key updated for '{resolved_config_key}' (ends in {masked}) and saved to config.toml."
                                            ),
                                            Err(e) => println!(
                                                "[system] API key updated for '{resolved_config_key}' (ends in {masked}) for this session, but could not persist to config.toml: {e}"
                                            ),
                                        }
                                        if bus_tx
                                            .blocking_send(BusMessage::SwitchModel {
                                                provider_name,
                                                model_name,
                                                base_url: resolved_url,
                                                api_key: resolved_key,
                                            })
                                            .is_err()
                                        {
                                            println!("[system] Bus closed; exiting.");
                                            let _ = shutdown.send(());
                                            break;
                                        }
                                        active_provider_key = Some(resolved_config_key);
                                    }
                                    Err(e) => {
                                        println!(
                                            "[system] Key was set, but could not resolve it for switching: {e}"
                                        );
                                    }
                                }
                            }
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
