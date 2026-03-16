use std::time::Duration;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{mpsc, watch};

use agent_rs::{NodeHandle, ActorLogic, Supervisor, SupervisorPolicy};
use agent_rs::agent::AgentLogic;
use agent_rs::scheduler::CronActor;
use agent_rs::session::SessionManager;
use agent_rs::provider::OpenAIProvider;
use agent_rs::tools::ToolRegistry;
use agent_rs::tools::builtin::{ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, ShellExecTool, WebSearchTool, WebFetchTool, CronTool, MessageTool};
use agent_rs::onboarding::{onboard_workspace, BootstrapReport};
use agent_rs::workspace::{resolve_workspace_root, AltbotWorkspace};
use agent_rs::skills::SkillRegistry;
use agent_rs::bus::{BusMessage, LoggerControlMessage};
use agent_rs::channels::{Channel, terminal::TerminalChannel, slack::SlackChannel, api::ApiChannel, email::EmailChannel};
use agent_rs::logging::{create_logger_channel, init_runtime_logger, LoggingActor, LOGGER_QUEUE_CAPACITY};
use colored::Colorize;
use clap::{Args as ClapArgs, Parser, Subcommand};

/// Altbot: A terminal chat interface and autonomous agent engine
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Optional explicit path to the workspace directory. Defaults to ~/.altbot
    #[arg(short, long)]
    workspace: Option<String>,

    /// Optional path to a config.toml file. Defaults to <workspace>/config.toml
    #[arg(short, long)]
    config: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Onboard(OnboardArgs),
}

#[derive(ClapArgs, Debug)]
struct OnboardArgs {
    /// Optional explicit path to the workspace directory. Defaults to ~/.altbot
    #[arg(short, long)]
    workspace: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Onboard(args)) => run_onboard(args.workspace.or(cli.workspace)).await,
        None => run_altbot(cli.workspace, cli.config).await,
    }
}

async fn run_altbot(
    workspace_arg: Option<String>,
    config_arg: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace_dir = resolve_workspace_root(workspace_arg.as_deref());

    let (logger_bus_tx, logger_bus_rx) = create_logger_channel(LOGGER_QUEUE_CAPACITY);
    let (shutdown_tx, mut shutdown_rx) = mpsc::unbounded_channel::<()>();
    let (app_shutdown_tx, app_shutdown_rx) = watch::channel(false);
    init_runtime_logger(logger_bus_tx.clone()).map_err(|e| {
        std::io::Error::other(format!("failed to initialize runtime logger: {:?}", e))
    })?;

    let logger_factory = {
        let wd = workspace_dir.clone();
        move || {
            Box::new(LoggingActor::new(wd.clone()).expect("failed to initialize logging actor"))
                as Box<dyn ActorLogic<BusMessage>>
        }
    };
    let logger_sup = Supervisor::new(SupervisorPolicy::Restart, logger_factory);
    let logger_node = NodeHandle::<BusMessage>::new(logger_sup, 1000, 1, Duration::from_millis(10));
    let (logger_control_listener, mut logger_control_rx) =
        NodeHandle::<BusMessage>::create_listener("logger_control", 8);
    logger_node.wire("logger_control", &logger_control_listener).await;

    let logger_forward = logger_node.clone();
    let runtime_handle = tokio::runtime::Handle::current();
    std::thread::Builder::new()
        .name("logging-forwarder".to_string())
        .spawn(move || {
            while let Ok(msg) = logger_bus_rx.recv() {
                if runtime_handle.block_on(logger_forward.send_packet(msg)).is_err() {
                    break;
                }
            }
        })?;

    println!("Starting Advanced Agent-RS System...");
    log::info!("Starting Advanced Agent-RS System.");

    let workspace = AltbotWorkspace::new(workspace_arg.as_deref(), config_arg.as_deref())?;
    println!("Loading Altbot workspace at: {:?}", workspace.dir);
    log::info!("Loading Altbot workspace at {:?}", workspace.dir);

    // 1. Setup SqliteMemoryActor and SessionManager
    let db_path = workspace.dir.join(".system_generated").join("agent_memory.db");
    if let Some(parent) = db_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let memory_actor = agent_rs::memory::SqliteMemoryActor::new(db_path.to_str().unwrap())
        .expect("Failed to initialize SqliteMemoryActor");
    let memory_node = NodeHandle::<agent_rs::memory::MemoryMessage>::new(memory_actor, 100, 1, Duration::from_millis(5));
    
    let session_manager = SessionManager::new(memory_node.clone());

    // 4. Setup Tools
    let (global_outbound_tx, mut global_outbound_rx) = mpsc::channel(100);

    // 2. Setup Skills
    let skills = SkillRegistry::new(workspace.skills_path());
    let cron_logic = CronActor::new("DailyBriefingCron", db_path.to_str().unwrap(), logger_bus_tx.clone())
        .expect("Failed to initialize CronActor database");
    let cron_node = NodeHandle::new(cron_logic, 10, 3, Duration::from_millis(50));

    let mut tools = ToolRegistry::new();
    let restrict = workspace.config.restrict_to_workspace.unwrap_or(true);
    tools.register(Box::new(ReadFileTool {
        workspace_dir: workspace.sandbox_dir.clone(),
        restrict_to_workspace: restrict,
    }));
    tools.register(Box::new(WriteFileTool {
        workspace_dir: workspace.sandbox_dir.clone(),
        restrict_to_workspace: restrict,
    }));
    tools.register(Box::new(EditFileTool {
        workspace_dir: workspace.sandbox_dir.clone(),
        restrict_to_workspace: restrict,
    }));
    tools.register(Box::new(ListDirTool {
        workspace_dir: workspace.sandbox_dir.clone(),
        restrict_to_workspace: restrict,
    }));
    tools.register(Box::new(ShellExecTool {
        workspace_dir: workspace.sandbox_dir.clone(),
        restrict_to_workspace: restrict,
    }));
    tools.register(Box::new(WebSearchTool));
    tools.register(Box::new(WebFetchTool));
    tools.register(Box::new(CronTool {
        cron_node: cron_node.clone(),
    }));
    tools.register(Box::new(MessageTool {
        outbound_tx: global_outbound_tx.clone(),
    }));
    tools.register(Box::new(agent_rs::tools::builtin::SearchMemoryTool {
        memory_node: memory_node.clone(),
    }));
    tools.register(Box::new(agent_rs::tools::builtin::FetchMemoryByDateTool {
        memory_node: memory_node.clone(),
    }));

    // 5. Setup Provider (Dynamic from config)
    let (model_name, api_key_env, base_url) = if let Some(p) = workspace.config.provider.clone() {
        (p.model_name, p.api_key_env, p.base_url)
    } else {
        ("gemini-2.5-flash".to_string(), "GEMINI_API_KEY".to_string(), "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions".to_string())
    };

    let api_key = std::env::var(&api_key_env).unwrap_or_else(|_| panic!("{} must be set", api_key_env));

    let client = agent_rs::utils::LLMClient::new_openai_compatible(
        &base_url,
        &api_key,
        &model_name,
    ).with_temperature(0.3);
    let provider = Box::new(OpenAIProvider::new(client.clone()));

    // 5.5 Setup Reflection Engine
    let memory_config = workspace.config.memory.clone().unwrap_or_default();
    let reflection_engine = agent_rs::reflection::ReflectionEngine::new(
        memory_node.clone(),
        workspace.sandbox_dir.clone(),
        Box::new(OpenAIProvider::new(client.clone())),
        memory_config,
        logger_bus_tx.clone(),
        app_shutdown_rx.clone(),
    );
    let reflection_task = reflection_engine.start();

    // 6. Compile Agent System Prompt
    let system_prompt = workspace.compile_system_prompt();

    // Prepare startup visual references before we move the structs
    let skill_names = skills.get_skill_names().join(", ");
    let skill_count = skills.get_skill_names().len();
    
    let mut tool_names_list = tools.get_tool_names();
    tool_names_list.sort();
    let tool_names = tool_names_list.join(", ");
    let tool_count = tool_names_list.len();

    // 7. Create Agent Logic
    let max_iterations = workspace.config.max_iterations.unwrap_or(50);
    let max_tool_output_chars = workspace.config.max_tool_output_chars.unwrap_or(3000);
    let max_recent_summaries = workspace.config.memory.as_ref()
        .and_then(|m| m.max_recent_summaries)
        .unwrap_or(5);
    let short_term_threshold_turns = workspace.config.memory.as_ref()
        .and_then(|m| m.short_term_threshold_turns)
        .unwrap_or(20);
    let short_term_threshold_tokens = workspace.config.memory.as_ref()
        .and_then(|m| m.short_term_threshold_tokens)
        .unwrap_or(100000);

    let agent_logic = AgentLogic::new(
        "Altbot",
        provider,
        session_manager,
        tools,
        skills,
        &system_prompt,
        max_iterations,
        max_tool_output_chars,
        max_recent_summaries,
        short_term_threshold_turns,
        short_term_threshold_tokens,
        global_outbound_tx.clone(),
        logger_bus_tx.clone(),
    );

    // 8. Wrap Agent in NodeHandle
    let agent_node = NodeHandle::<BusMessage>::new(agent_logic, 100, 3, Duration::from_millis(50));

    // 10. Setup Terminal Channel
    let (inbound_tx, mut inbound_rx) = mpsc::channel(100);
    let mut out_channels: HashMap<String, Arc<dyn Channel>> = HashMap::new();

    let terminal_chat_id = uuid::Uuid::new_v4().to_string();
    let terminal = Arc::new(TerminalChannel::new(&terminal_chat_id, logger_bus_tx.clone(), shutdown_tx.clone()));
    terminal.start(inbound_tx.clone()).await?;
    out_channels.insert(terminal.name().to_string(), terminal);

    // 11. Setup Slack Channel
    if let Some(slack_cfg) = workspace.config.slack.clone() {
        if slack_cfg.enabled.unwrap_or(false) {
            let slack = Arc::new(SlackChannel::new(slack_cfg, logger_bus_tx.clone()));
            slack.start(inbound_tx.clone()).await?;
            out_channels.insert(slack.name().to_string(), slack);
        }
    }

    // 12. Setup API Channel
    if let Some(api_cfg) = workspace.config.api.clone() {
        if api_cfg.enabled.unwrap_or(false) {
            let api = Arc::new(ApiChannel::new(api_cfg.port, &db_path, logger_bus_tx.clone())?);
            api.start(inbound_tx.clone()).await?;
            out_channels.insert(api.name().to_string(), api);
        }
    }

    // 13. Setup Email Channel
    if let Some(email_cfg) = workspace.config.email.clone() {
        if email_cfg.enabled.unwrap_or(false) {
            let email_ch = Arc::new(EmailChannel::new(email_cfg, logger_bus_tx.clone()));
            email_ch.start(inbound_tx.clone()).await?;
            out_channels.insert(email_ch.name().to_string(), email_ch);
        }
    }

    // 14. Print clean startup banner
    println!("\n{}", "=============================================".blue());
    println!("Agent-RS Version: {}", env!("CARGO_PKG_VERSION").green());
    println!("Terminal Session ID: {}", terminal_chat_id.dimmed());
    println!("Loaded Skills ({}): {}", skill_count.to_string().cyan(), skill_names);
    println!("Loaded Tools ({}): {}", tool_count.to_string().yellow(), tool_names);
    println!("{}", "=============================================".blue());
    println!("\n{}", "Agent System is Running.".bold().green());
    println!("{}", "Available actions: type a message in the terminal or on active chat channels.".cyan());
    println!("{}", "Tip: Type '/exit' to securely shut down the engine.\n".dimmed());

    // Route inbound messages from all channels into the agent and logger
    let agent_tx = agent_node.clone();
    let logger_tx = logger_bus_tx.clone();
    tokio::spawn(async move {
        while let Some(msg) = inbound_rx.recv().await {
            let _ = logger_tx.send(BusMessage::Inbound(msg.clone()));
            let _ = agent_tx.send_packet(BusMessage::Inbound(msg)).await;
        }
    });

    // Listen for Outbound reasoning chunks and route back to the appropriate channel and logger
    let (listener_node, mut agent_rx) = NodeHandle::<BusMessage>::create_listener("completion", 100);
    let _ = &agent_node - "completion" >> &listener_node;

    // Listen to cron triggers
    let (cron_listener_node, mut cron_rx) = NodeHandle::<String>::create_listener("trigger", 100);
    let _ = &cron_node - "trigger" >> &cron_listener_node;

    let cron_agent_tx = agent_node.clone();
    let cron_logger_tx = logger_bus_tx.clone();
    tokio::spawn(async move {
        while let Some(msg) = cron_rx.recv().await {
            if let agent_rs::Message::Packet(payload) = msg {
                
                let mut channel_val = "cron".to_string();
                let mut chat_id_val = "cron_global".to_string();
                let mut content_val = payload.clone();

                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&payload) {
                    if let Some(msg) = parsed.get("message").and_then(|v| v.as_str()) {
                        content_val = msg.to_string();
                    }
                    if let Some(cid) = parsed.get("chat_id").and_then(|v| v.as_str()) {
                        chat_id_val = cid.to_string();
                    }
                    if let Some(ch) = parsed.get("channel").and_then(|v| v.as_str()) {
                        channel_val = ch.to_string();
                    }
                }

                let inbound = agent_rs::bus::InboundMessage {
                    channel: channel_val,
                    sender_id: "cron".to_string(),
                    chat_id: chat_id_val,
                    thread_id: None,
                    content: content_val,
                    metadata: HashMap::new(),
                };
                
                // Also emit a telemetry event so loggers see the trigger fired
                let tel = agent_rs::bus::TelemetryEvent::CronTrigger {
                    job_id: "cron_event".to_string(),
                    message: payload,
                };
                let _ = cron_logger_tx.send(BusMessage::Telemetry(tel));

                // Fire into the agent
                let _ = cron_agent_tx.send_packet(BusMessage::Inbound(inbound)).await;
            }
        }
    });

    let agent_outbound_tx = global_outbound_tx.clone();
    tokio::spawn(async move {
        while let Some(bus_msg) = agent_rx.recv().await {
            if let agent_rs::Message::Packet(packet) = bus_msg {
                match packet {
                    BusMessage::Outbound(out) => {
                        let _ = agent_outbound_tx.send(BusMessage::Outbound(out)).await;
                    }
                    BusMessage::Telemetry(tel) => {
                        let _ = agent_outbound_tx.send(BusMessage::Telemetry(tel)).await;
                    }
                    _ => {}
                }
            }
        }
    });

    let logger_tx_outbound = logger_bus_tx.clone();
    let delivery_channels = out_channels.clone();
    tokio::spawn(async move {
        while let Some(msg) = global_outbound_rx.recv().await {
            let _ = logger_tx_outbound.send(msg.clone());
            if let BusMessage::Outbound(out) = msg {
                if let Some(chan) = delivery_channels.get(&out.channel) {
                    if let Err(e) = chan.send(out).await {
                        log::error!("Failed to deliver message via channel [{}]: {}", chan.name(), e);
                    }
                }
            }
        }
    });

    tokio::select! {
        _ = shutdown_rx.recv() => {
            log::info!("Shutdown requested from terminal.");
        }
        _ = tokio::signal::ctrl_c() => {
            log::info!("Shutdown requested via Ctrl+C.");
        }
    }

    log::info!("Stopping channels and shutting down runtime.");
    for channel in out_channels.values() {
        let _ = channel.stop().await;
    }

    let _ = app_shutdown_tx.send(true);
    let _ = reflection_task.await;

    let _ = logger_bus_tx.send(BusMessage::LoggerControl(LoggerControlMessage::Flush));
    let flush_result = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(msg) = logger_control_rx.recv().await {
            if let agent_rs::Message::Packet(BusMessage::LoggerControl(LoggerControlMessage::Flushed)) = msg {
                return Ok::<(), ()>(());
            }
        }
        Err(())
    }).await;

    if flush_result.is_err() {
        log::warn!("Timed out waiting for LoggingActor flush acknowledgement.");
    }

    Ok(())
}

async fn run_onboard(workspace_arg: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let report = tokio::task::spawn_blocking(move || {
        let workspace_root = resolve_workspace_root(workspace_arg.as_deref());
        onboard_workspace(&workspace_root)
    })
    .await?
    .map_err(std::io::Error::other)?;
    print_onboarding_report(&report);
    Ok(())
}

fn print_onboarding_report(report: &BootstrapReport) {
    println!("Workspace onboarded at {}", report.root.display());
    println!();

    if !report.created.is_empty() {
        println!("Created:");
        for path in &report.created {
            println!("- {}", path.display());
        }
        println!();
    }

    if !report.skipped.is_empty() {
        println!("Skipped:");
        for path in &report.skipped {
            println!("- {}", path.display());
        }
        println!();
    }

    println!("Next steps:");
    println!("1. Set GEMINI_API_KEY");
    println!("2. Update <changethis> placeholders or disable unused channels in config.toml");
    println!("3. Run: altbot --workspace {}", report.root.display());
}
