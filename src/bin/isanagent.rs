use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};

use clap::{Args as ClapArgs, Parser, Subcommand};
use colored::Colorize;
use isanagent::agent::{AgentLogic, AgentLogicParams};
use isanagent::bus::{BusMessage, LoggerControlMessage, TelemetryEvent};
use isanagent::channels::terminal::{
    build_agent_thought_terminal_notice, build_tool_call_terminal_notice,
    build_tool_result_terminal_notice, terminal_startup_suppresses_plain_banner,
};
use isanagent::channels::{
    api::ApiChannel, email::EmailChannel, slack::SlackChannel, terminal::TerminalChannel, Channel,
};
use isanagent::clarification::ClarificationHub;
use isanagent::logging::{
    create_logger_channel, create_logging_actor_or_fallback, init_runtime_logger,
    LOGGER_QUEUE_CAPACITY,
};
use isanagent::onboarding::{onboard_workspace, BootstrapReport, OnboardOptions};
use isanagent::provider::OpenAIProvider;
use isanagent::scheduler::{
    validate_multi_tenant_edge_runtime, CronActor, CronSchedulingMode, CronTriggerPayload,
    MultiTenantEdgeCronScheduler,
};
use isanagent::session::SessionManager;
use isanagent::skills::SkillRegistry;
use isanagent::tools::builtin::{
    CronTool, EditFileTool, GitWorktreeTool, GlobFilesTool, ListDirTool, MessageTool, ReadFileTool,
    SearchTextTool, ShellExecTool, WebFetchTool, WebSearchTool, WriteFileTool,
};
use isanagent::tools::workflow::{AskUserTool, TodoWriteTool, ToolSearchTool};
use isanagent::tools::ToolRegistry;
use isanagent::workspace::{resolve_workspace_root, IsanagentWorkspace};
use isanagent::{NodeHandle, Supervisor, SupervisorPolicy};

const DEFAULT_PROVIDER_MODEL_NAME: &str = "gemini-2.5-flash";
const DEFAULT_PROVIDER_API_KEY_ENV: &str = "GEMINI_API_KEY";
const DEFAULT_PROVIDER_BASE_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";

/// isanagent: A terminal chat interface and autonomous agent engine
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Optional explicit path to the workspace directory. Defaults to ~/.isanagent
    #[arg(short, long)]
    workspace: Option<String>,

    /// Optional path to a config.toml file. Defaults to <workspace>/config.toml
    #[arg(short, long)]
    config: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create workspace layout and starter files; optional flags override generated config.toml
    Onboard(OnboardArgs),
}

#[derive(ClapArgs, Debug)]
struct OnboardArgs {
    /// Optional explicit path to the workspace directory. Defaults to ~/.isanagent
    #[arg(short, long)]
    workspace: Option<String>,
    /// Override embedded defaults for `config.toml` (see `isanagent onboard --help`)
    #[command(flatten)]
    options: OnboardOptions,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Onboard(args)) => run_onboard(cli.workspace, args).await,
        None => run_isanagent(cli.workspace, cli.config).await,
    }
}

async fn run_isanagent(
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
        move || create_logging_actor_or_fallback(wd.clone())
    };
    let logger_sup = Supervisor::new(SupervisorPolicy::Restart, logger_factory);
    let logger_node = NodeHandle::<BusMessage>::new(logger_sup, 1000, 1, Duration::from_millis(10));
    let (logger_control_listener, mut logger_control_rx) =
        NodeHandle::<BusMessage>::create_listener("logger_control", 8);
    logger_node
        .wire("logger_control", &logger_control_listener)
        .await;

    let logger_forward = logger_node.clone();
    let runtime_handle = tokio::runtime::Handle::current();
    std::thread::Builder::new()
        .name("logging-forwarder".to_string())
        .spawn(move || {
            while let Ok(msg) = logger_bus_rx.recv() {
                if runtime_handle
                    .block_on(logger_forward.send_packet(msg))
                    .is_err()
                {
                    break;
                }
            }
        })?;

    println!("Starting Advanced isanagent System...");
    log::info!("Starting Advanced isanagent System.");

    let workspace = IsanagentWorkspace::new(workspace_arg.as_deref(), config_arg.as_deref())?;
    println!("Loading Altbot workspace at: {:?}", workspace.dir);
    log::info!("Loading Altbot workspace at {:?}", workspace.dir);

    if !workspace.config.terminal_enabled() && !workspace.config.has_non_terminal_inbound_channel()
    {
        return Err(std::io::Error::other(
            "Invalid config: [terminal] enable = false requires at least one other inbound channel. \
Enable [api], [slack], or [email] (with enabled = true) so the agent can receive messages without stdin.",
        )
        .into());
    }

    // 1. Setup SqliteMemoryActor and SessionManager
    let db_path = workspace
        .dir
        .join(".system_generated")
        .join("agent_memory.db");
    if let Some(parent) = db_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let db_path_str = db_path
        .to_str()
        .ok_or_else(|| std::io::Error::other("workspace DB path is not valid UTF-8"))?;
    let memory_actor = isanagent::memory::SqliteMemoryActor::new(
        db_path_str,
        Some(workspace.dir.join("todos").as_path()),
    )
    .map_err(|e| std::io::Error::other(format!("Failed to initialize SqliteMemoryActor: {}", e)))?;
    let memory_node = NodeHandle::<isanagent::memory::MemoryMessage>::new(
        memory_actor,
        100,
        1,
        Duration::from_millis(5),
    );

    let session_manager = SessionManager::new(memory_node.clone());

    // 4. Setup Tools
    let (global_outbound_tx, mut global_outbound_rx) = mpsc::channel(100);
    let clarification_hub = ClarificationHub::shared();

    // 2. Setup Skills
    let skills = SkillRegistry::new(workspace.skills_path());
    let multi_tenant_edge_cfg = workspace
        .config
        .multi_tenant_edge
        .clone()
        .unwrap_or_default();
    let mte_cron_scheduler = if multi_tenant_edge_cfg
        .cron_scheduling_enabled
        .unwrap_or(false)
    {
        let api_enabled = workspace
            .config
            .api
            .as_ref()
            .and_then(|cfg| cfg.enabled)
            .unwrap_or(false);
        validate_multi_tenant_edge_runtime(api_enabled).map_err(std::io::Error::other)?;

        let client = isanagent::multi_tenant_edge::CronRegistrationClient::from_env()
            .map_err(std::io::Error::other)?;
        let scheduler = Arc::new(
            MultiTenantEdgeCronScheduler::new(db_path_str, client)
                .map_err(std::io::Error::other)?,
        );
        scheduler
            .sync_all(chrono::Utc::now())
            .await
            .map_err(|error| {
                std::io::Error::other(format!(
                    "Failed to sync cron jobs to multi-tenant-edge on startup: {}",
                    error
                ))
            })?;
        Some(scheduler)
    } else {
        None
    };
    let cron_mode = if mte_cron_scheduler.is_some() {
        CronSchedulingMode::MultiTenantEdge
    } else {
        CronSchedulingMode::Local
    };
    let cron_logic = CronActor::new(
        "DailyBriefingCron",
        db_path_str,
        logger_bus_tx.clone(),
        cron_mode,
    )
    .map_err(std::io::Error::other)?;
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
    tools.register(Box::new(GlobFilesTool {
        workspace_dir: workspace.sandbox_dir.clone(),
        restrict_to_workspace: restrict,
    }));
    tools.register(Box::new(SearchTextTool {
        workspace_dir: workspace.sandbox_dir.clone(),
        restrict_to_workspace: restrict,
        ripgrep_timeout_secs: workspace
            .config
            .effective_search_text_ripgrep_timeout_secs(),
    }));
    tools.register(Box::new(ShellExecTool {
        workspace_dir: workspace.sandbox_dir.clone(),
        restrict_to_workspace: restrict,
    }));
    if workspace.config.git_worktree_tool_enabled() {
        tools.register(Box::new(GitWorktreeTool {
            workspace_dir: workspace.sandbox_dir.clone(),
            restrict_to_workspace: restrict,
            allow_path_outside_sandbox: workspace.config.git_worktree_allow_path_outside_sandbox(),
        }));
    }
    let jina = workspace.config.jina_web_backend();
    let max_web_output_chars = workspace.config.effective_max_web_tool_output_chars();
    tools.register(Box::new(WebSearchTool {
        jina: jina.clone(),
        max_output_chars: max_web_output_chars,
    }));
    tools.register(Box::new(WebFetchTool {
        jina,
        max_output_chars: max_web_output_chars,
    }));
    tools.register(Box::new(CronTool {
        cron_node: cron_node.clone(),
        multi_tenant_edge_cron_enabled: mte_cron_scheduler.is_some(),
        mte_cron_scheduler: mte_cron_scheduler.clone(),
    }));
    tools.register(Box::new(MessageTool {
        outbound_tx: global_outbound_tx.clone(),
    }));
    tools.register(Box::new(AskUserTool {
        clarification_hub: clarification_hub.clone(),
        outbound_tx: global_outbound_tx.clone(),
    }));
    tools.register(Box::new(isanagent::tools::builtin::SearchMemoryTool {
        memory_node: memory_node.clone(),
    }));
    tools.register(Box::new(isanagent::tools::builtin::FetchMemoryByDateTool {
        memory_node: memory_node.clone(),
    }));

    tools.register(Box::new(TodoWriteTool {
        memory_node: memory_node.clone(),
    }));
    let tool_catalog = tools.catalog_handle();
    tools.register(Box::new(ToolSearchTool {
        catalog: tool_catalog,
    }));

    // 5. Setup Provider (Dynamic from config)
    let (model_name, api_key_env, base_url) = if let Some(p) = workspace.config.provider.clone() {
        (p.model_name, p.api_key_env, p.base_url)
    } else {
        (
            DEFAULT_PROVIDER_MODEL_NAME.to_string(),
            DEFAULT_PROVIDER_API_KEY_ENV.to_string(),
            DEFAULT_PROVIDER_BASE_URL.to_string(),
        )
    };

    let api_key = std::env::var(&api_key_env)
        .map_err(|_| std::io::Error::other(format!("{} must be set", api_key_env)))?;
    let client =
        isanagent::utils::LLMClient::new_openai_compatible(&base_url, &api_key, &model_name)
            .with_temperature(0.3);
    let provider = Box::new(OpenAIProvider::new(client.clone()));

    // 5.5 Setup Reflection Engine
    let memory_config = workspace.config.memory.clone().unwrap_or_default();
    let reflection_engine = isanagent::reflection::ReflectionEngine::new(
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
    let max_recent_summaries = workspace
        .config
        .memory
        .as_ref()
        .and_then(|m| m.max_recent_summaries)
        .unwrap_or(5);
    let short_term_threshold_turns = workspace
        .config
        .memory
        .as_ref()
        .and_then(|m| m.short_term_threshold_turns)
        .unwrap_or(20);
    let short_term_threshold_tokens = workspace
        .config
        .memory
        .as_ref()
        .and_then(|m| m.short_term_threshold_tokens)
        .unwrap_or(100000);
    let tool_execution_activity = if multi_tenant_edge_cfg
        .activity_heartbeat_enabled
        .unwrap_or(false)
    {
        match isanagent::multi_tenant_edge::ActivityHeartbeatClient::from_env(logger_bus_tx.clone())
        {
            Ok(client) => Some(std::sync::Arc::new(client)),
            Err(error) => {
                let _ = logger_bus_tx.send(BusMessage::Log(isanagent::bus::LogEvent::warn(
                    "Altbot", &error,
                )));
                None
            }
        }
    } else {
        None
    };

    let subagent = if workspace.config.subagent_harness_enabled() {
        Some(isanagent::agent::SubagentHarnessParams {
            cancel_children_on_parent_cancel: workspace
                .config
                .subagent_cancel_children_on_parent_cancel(),
            allowed_tools: workspace
                .config
                .subagent_allowed_tools_set()
                .map(std::sync::Arc::new),
            max_tasks: workspace.config.subagent_max_tasks(),
            max_wait_secs: workspace.config.subagent_max_wait_secs(),
        })
    } else {
        None
    };

    let agent_logic = AgentLogic::new(AgentLogicParams {
        name: "Altbot".to_string(),
        provider,
        session_manager,
        tools,
        skills,
        system_prompt,
        max_iterations,
        max_tool_output_chars,
        max_recent_summaries,
        short_term_threshold_turns,
        short_term_threshold_tokens,
        outbound_tx: global_outbound_tx.clone(),
        logger_tx: logger_bus_tx.clone(),
        clarification_hub,
        subagent,
    });
    let agent_logic = if let Some(tool_execution_activity) = tool_execution_activity {
        agent_logic.with_tool_execution_activity(tool_execution_activity)
    } else {
        agent_logic
    };

    // 8. Wrap Agent in NodeHandle
    let agent_node = NodeHandle::<BusMessage>::new(agent_logic, 100, 3, Duration::from_millis(50));

    // 10. Setup channels (terminal is optional for headless / Docker API-only runs)
    let (bus_tx, mut bus_rx) = mpsc::channel(100);
    let mut out_channels: HashMap<String, Arc<dyn Channel>> = HashMap::new();

    let terminal_chat_id = if workspace.config.terminal_enabled() {
        let id = uuid::Uuid::new_v4().to_string();
        let terminal = Arc::new(TerminalChannel::new(
            &id,
            logger_bus_tx.clone(),
            shutdown_tx.clone(),
            workspace.sandbox_dir.clone(),
            model_name.clone(),
        ));
        terminal.start(bus_tx.clone()).await?;
        out_channels.insert(terminal.name().to_string(), terminal);
        Some(id)
    } else {
        log::info!("Terminal channel disabled via config; stdin will not be read.");
        None
    };

    // 11. Setup Slack Channel
    if let Some(slack_cfg) = workspace.config.slack.clone() {
        if slack_cfg.enabled.unwrap_or(false) {
            let slack = Arc::new(SlackChannel::new(
                slack_cfg,
                &db_path,
                logger_bus_tx.clone(),
            )?);
            slack.start(bus_tx.clone()).await?;
            out_channels.insert(slack.name().to_string(), slack);
        }
    }

    // 12. Setup API Channel
    let api_local_url = if let Some(api_cfg) = workspace.config.api.clone() {
        if api_cfg.enabled.unwrap_or(false) {
            let api_port = api_cfg.port;
            let api = ApiChannel::new(
                api_cfg,
                &db_path,
                logger_bus_tx.clone(),
                memory_node.clone(),
            )?;
            let api = if let Some(mte_cron_scheduler) = mte_cron_scheduler.clone() {
                api.with_multi_tenant_edge_cron_scheduler(mte_cron_scheduler)
            } else {
                api
            };
            let api = Arc::new(api);
            api.start(bus_tx.clone()).await?;
            out_channels.insert(api.name().to_string(), api);
            Some(format!("http://127.0.0.1:{api_port}/"))
        } else {
            None
        }
    } else {
        None
    };

    // 13. Setup Email Channel
    if let Some(email_cfg) = workspace.config.email.clone() {
        if email_cfg.enabled.unwrap_or(false) {
            let email_ch = Arc::new(EmailChannel::new(email_cfg, logger_bus_tx.clone()));
            email_ch.start(bus_tx.clone()).await?;
            out_channels.insert(email_ch.name().to_string(), email_ch);
        }
    }

    // 14. Print clean startup banner (skipped when Ratatui owns the alternate screen)
    if !terminal_startup_suppresses_plain_banner(&workspace.config) {
        println!(
            "\n{}",
            "=============================================".blue()
        );
        println!("isanagent Version: {}", env!("CARGO_PKG_VERSION").green());
        if let Some(id) = terminal_chat_id.as_ref() {
            println!("Terminal Session ID: {}", id.dimmed());
        } else {
            println!("{}", "Terminal channel: disabled (headless mode)".dimmed());
        }
        if let Some(url) = &api_local_url {
            println!("HTTP API (Vite UI proxies here): {}", url.green());
        }
        println!(
            "Loaded Skills ({}): {}",
            skill_count.to_string().cyan(),
            skill_names
        );
        println!(
            "Loaded Tools ({}): {}",
            tool_count.to_string().yellow(),
            tool_names
        );
        println!("{}", "=============================================".blue());
        println!("\n{}", "Agent System is Running.".bold().green());
        if terminal_chat_id.is_some() {
            println!(
                "{}",
                "Available actions: type a message in the terminal or on active chat channels."
                    .cyan()
            );
            println!(
                "{}",
                "Tip: Type '/exit' to securely shut down the engine.\n".dimmed()
            );
        } else {
            println!(
                "{}",
                "Terminal input is disabled; use your enabled channel(s) (API, Slack, or Email)."
                    .cyan()
            );
            println!(
                "{}",
                "Tip: Press Ctrl+C to shut down the engine.\n".dimmed()
            );
        }
    }

    // Route inbound messages from all channels into the agent and logger
    let agent_tx = agent_node.clone();
    let logger_tx = logger_bus_tx.clone();
    tokio::spawn(async move {
        while let Some(msg) = bus_rx.recv().await {
            let _ = logger_tx.send(msg.clone());
            // Only route Inbound and Cancel messages to the agent logic.
            // This prevents the agent from being flooded with its own telemetry or other system messages.
            if matches!(msg, BusMessage::Inbound(_) | BusMessage::Cancel(_)) {
                let _ = agent_tx.send_packet(msg).await;
            }
        }
    });

    // Listen for Outbound reasoning chunks and route back to the appropriate channel and logger
    let (listener_node, mut agent_rx) =
        NodeHandle::<BusMessage>::create_listener("completion", 100);
    let _ = (&agent_node - "completion") >> &listener_node;

    // Listen to cron triggers
    let (cron_listener_node, mut cron_rx) = NodeHandle::<String>::create_listener("trigger", 100);
    let _ = (&cron_node - "trigger") >> &cron_listener_node;

    let cron_bus_tx = bus_tx.clone();
    let cron_logger_tx = logger_bus_tx.clone();
    tokio::spawn(async move {
        while let Some(msg) = cron_rx.recv().await {
            if let isanagent::Message::Packet(payload) = msg {
                let Ok(trigger) = serde_json::from_str::<CronTriggerPayload>(&payload) else {
                    let _ = cron_logger_tx.send(BusMessage::Log(isanagent::bus::LogEvent::warn(
                        "Altbot",
                        "Failed to parse cron trigger payload emitted by scheduler",
                    )));
                    continue;
                };

                let inbound = isanagent::bus::InboundMessage {
                    channel: trigger.channel.clone(),
                    sender_id: "cron".to_string(),
                    chat_id: trigger.chat_id.clone(),
                    thread_id: None,
                    content: trigger.message.clone(),
                    attachments: Vec::new(),
                    metadata: HashMap::from([
                        (
                            "cron_job_id".to_string(),
                            serde_json::Value::String(trigger.job_id.clone()),
                        ),
                        (
                            "trigger_source".to_string(),
                            serde_json::Value::String("local_scheduler".to_string()),
                        ),
                    ]),
                };

                // Also emit a telemetry event so loggers see the trigger fired
                let tel = isanagent::bus::TelemetryEvent::CronTrigger {
                    job_id: trigger.job_id.clone(),
                    message: trigger.message.clone(),
                };
                let _ = cron_logger_tx.send(BusMessage::Telemetry(tel));

                // Fire into the agent
                let _ = cron_bus_tx.send(BusMessage::Inbound(inbound)).await;
            }
        }
    });

    let agent_outbound_tx = global_outbound_tx.clone();
    tokio::spawn(async move {
        while let Some(bus_msg) = agent_rx.recv().await {
            if let isanagent::Message::Packet(packet) = bus_msg {
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
    let terminal_session_for_telemetry = terminal_chat_id.clone();
    tokio::spawn(async move {
        while let Some(msg) = global_outbound_rx.recv().await {
            // Deliver user-visible terminal traffic first. `LoggerHandle::send` uses a blocking
            // `sync_channel::send`; doing it before channel delivery can stall this task and make
            // tool-call lines and agent replies appear only after the run finishes.
            match &msg {
                BusMessage::Outbound(out) => {
                    if let Some(chan) = delivery_channels.get(&out.channel) {
                        if let Err(e) = chan.send(out.clone()).await {
                            log::error!(
                                "Failed to deliver message via channel [{}]: {}",
                                chan.name(),
                                e
                            );
                        }
                    }
                }
                BusMessage::Telemetry(TelemetryEvent::AgentThought { chat_id, thought }) => {
                    if terminal_session_for_telemetry.as_deref() == Some(chat_id.as_str()) {
                        let notice = build_agent_thought_terminal_notice(chat_id, thought);
                        if let Some(chan) = delivery_channels.get("terminal") {
                            if let Err(e) = chan.send(notice).await {
                                log::error!("Failed to deliver AgentThought to terminal: {}", e);
                            }
                        }
                    }
                    if let Some(api_chan) = delivery_channels.get("api") {
                        if let Some(api_chan) = api_chan.as_any().downcast_ref::<ApiChannel>() {
                            api_chan
                                .handle_telemetry(TelemetryEvent::AgentThought {
                                    chat_id: chat_id.clone(),
                                    thought: thought.clone(),
                                })
                                .await;
                        }
                    }
                }
                BusMessage::Telemetry(TelemetryEvent::ToolCall {
                    channel,
                    chat_id,
                    tool_name,
                    args,
                }) if channel == "terminal" => {
                    let notice = build_tool_call_terminal_notice(chat_id, tool_name, args);
                    if let Some(chan) = delivery_channels.get("terminal") {
                        if let Err(e) = chan.send(notice).await {
                            log::error!("Failed to deliver tool-call notice to terminal: {}", e);
                        }
                    }
                }
                BusMessage::Telemetry(TelemetryEvent::ToolResult {
                    channel,
                    chat_id,
                    tool_name,
                    result,
                }) if channel == "terminal" => {
                    let notice = build_tool_result_terminal_notice(chat_id, tool_name, result);
                    if let Some(chan) = delivery_channels.get("terminal") {
                        if let Err(e) = chan.send(notice).await {
                            log::error!("Failed to deliver tool-result notice to terminal: {}", e);
                        }
                    }
                }
                BusMessage::Telemetry(tel) => {
                    if let Some(api_chan) = delivery_channels.get("api") {
                        if let Some(api_chan) = api_chan.as_any().downcast_ref::<ApiChannel>() {
                            api_chan.handle_telemetry(tel.clone()).await;
                        }
                    }
                }
                _ => {}
            }

            let _ = logger_tx_outbound.send(msg);
        }
    });

    tokio::select! {
        _ = shutdown_rx.recv() => {
            log::info!("Shutdown requested (terminal /exit or internal signal).");
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
            if let isanagent::Message::Packet(BusMessage::LoggerControl(
                LoggerControlMessage::Flushed,
            )) = msg
            {
                return Ok::<(), ()>(());
            }
        }
        Err(())
    })
    .await;

    if flush_result.is_err() {
        log::warn!("Timed out waiting for LoggingActor flush acknowledgement.");
    }

    Ok(())
}

async fn run_onboard(
    global_workspace: Option<String>,
    args: OnboardArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace_arg = args.workspace.or(global_workspace);
    let options = args.options;
    let config_overrides_used = options.has_overrides();
    let report = tokio::task::spawn_blocking(move || {
        let workspace_root = resolve_workspace_root(workspace_arg.as_deref());
        onboard_workspace(&workspace_root, &options)
    })
    .await?
    .map_err(std::io::Error::other)?;
    print_onboarding_report(&report, config_overrides_used);
    Ok(())
}

fn print_onboarding_report(report: &BootstrapReport, config_overrides_used: bool) {
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

    if config_overrides_used {
        println!(
            "Note: config.toml was generated from merged settings (template comments were omitted)."
        );
        println!();
    }

    println!("Next steps:");
    println!("1. Set GEMINI_API_KEY (or the env named in provider.api_key_env)");
    println!("2. Update <changethis> placeholders or disable unused channels in config.toml");
    println!("3. Run: isanagent --workspace {}", report.root.display());
}
