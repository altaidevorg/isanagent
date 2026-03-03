use std::time::Duration;
use tokio::time::sleep;
use tokio::sync::mpsc;
use std::sync::Arc;
use std::collections::HashMap;

use agent_rs::{NodeHandle, ActorLogic, Supervisor, SupervisorPolicy};
use agent_rs::agent::AgentLogic;
use agent_rs::scheduler::CronActor;
use agent_rs::session::SessionManager;
use agent_rs::provider::OpenAIProvider;
use agent_rs::tools::ToolRegistry;
use agent_rs::tools::builtin::{ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, ShellExecTool, WebSearchTool, WebFetchTool, CronTool, MessageTool};
use agent_rs::workspace::AltbotWorkspace;
use agent_rs::skills::SkillRegistry;
use agent_rs::bus::BusMessage;
use agent_rs::channels::{Channel, terminal::TerminalChannel, slack::SlackChannel, api::ApiChannel, email::EmailChannel};
use agent_rs::logging::WorkspaceLoggingActor;
use colored::Colorize;
use clap::Parser;

/// Altbot: A terminal chat interface and autonomous agent engine
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Optional explicit path to the workspace directory. Defaults to ~/.altbot
    #[arg(short, long)]
    workspace: Option<String>,

    /// Optional path to a config.toml file. Defaults to <workspace>/config.toml
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    println!("Starting Advanced Agent-RS System...");

    // 0. Parse workspace CLI arguments
    let cli_args = Args::parse();

    let workspace = AltbotWorkspace::new(cli_args.workspace.as_deref(), cli_args.config.as_deref())?;
    println!("Loading Altbot workspace at: {:?}", workspace.dir);

    // 1. Setup SqliteMemoryActor and SessionManager
    let db_path = workspace.dir.join(".system_generated").join("agent_memory.db");
    if let Some(parent) = db_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let memory_actor = agent_rs::memory::SqliteMemoryActor::new(db_path.to_str().unwrap())
        .expect("Failed to initialize SqliteMemoryActor");
    let memory_node = NodeHandle::<agent_rs::memory::MemoryMessage>::new(memory_actor, 100, 1, Duration::from_millis(5));
    
    let session_manager = SessionManager::new(memory_node);

    // 2. Setup Skills
    let skills = SkillRegistry::new(workspace.skills_path());
    let cron_logic = CronActor::new("DailyBriefingCron");
    let cron_node = NodeHandle::new(cron_logic, 10, 3, Duration::from_millis(50));

    // 4. Setup Tools
    let (global_outbound_tx, mut global_outbound_rx) = mpsc::channel(100);

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
    let provider = Box::new(OpenAIProvider::new(client));

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

    let agent_logic = AgentLogic::new(
        "Altbot",
        provider,
        session_manager,
        tools,
        skills,
        &system_prompt,
        max_iterations,
        max_tool_output_chars,
        global_outbound_tx.clone(),
    );

    // 8. Wrap Agent in NodeHandle
    let agent_node = NodeHandle::<BusMessage>::new(agent_logic, 100, 3, Duration::from_millis(50));

    // 9. Setup Supervisor Logger Node
    let logger_factory = {
        let wd = workspace.dir.clone();
        move || {
            Box::new(WorkspaceLoggingActor::new(wd.clone())) as Box<dyn ActorLogic<BusMessage>>
        }
    };
    let logger_sup = Supervisor::new(SupervisorPolicy::Restart, logger_factory);
    let logger_node = NodeHandle::<BusMessage>::new(logger_sup, 100, 1, Duration::from_millis(10));

    // 10. Setup Terminal Channel
    let (inbound_tx, mut inbound_rx) = mpsc::channel(100);
    let mut out_channels: HashMap<String, Arc<dyn Channel>> = HashMap::new();

    let terminal_chat_id = uuid::Uuid::new_v4().to_string();
    let terminal = Arc::new(TerminalChannel::new(&terminal_chat_id));
    terminal.start(inbound_tx.clone()).await?;
    out_channels.insert(terminal.name().to_string(), terminal);

    // 11. Setup Slack Channel
    if let Some(slack_cfg) = workspace.config.slack.clone() {
        if slack_cfg.enabled.unwrap_or(false) {
            let slack = Arc::new(SlackChannel::new(slack_cfg));
            slack.start(inbound_tx.clone()).await?;
            out_channels.insert(slack.name().to_string(), slack);
        }
    }

    // 12. Setup API Channel
    if let Some(api_cfg) = workspace.config.api.clone() {
        if api_cfg.enabled.unwrap_or(false) {
            let api = Arc::new(ApiChannel::new(api_cfg.port));
            api.start(inbound_tx.clone()).await?;
            out_channels.insert(api.name().to_string(), api);
        }
    }

    // 13. Setup Email Channel
    if let Some(email_cfg) = workspace.config.email.clone() {
        if email_cfg.enabled.unwrap_or(false) {
            let email_ch = Arc::new(EmailChannel::new(email_cfg));
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
    let logger_tx = logger_node.clone();
    tokio::spawn(async move {
        while let Some(msg) = inbound_rx.recv().await {
            let _ = logger_tx.send_packet(BusMessage::Inbound(msg.clone())).await;
            let _ = agent_tx.send_packet(BusMessage::Inbound(msg)).await;
        }
    });

    // Listen for Outbound reasoning chunks and route back to the appropriate channel and logger
    let (listener_node, mut agent_rx) = NodeHandle::<BusMessage>::create_listener("completion", 100);
    let _ = &agent_node - "completion" >> &listener_node;

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

    let logger_tx_outbound = logger_node.clone();
    tokio::spawn(async move {
        while let Some(msg) = global_outbound_rx.recv().await {
            let _ = logger_tx_outbound.send_packet(msg.clone()).await;
            if let BusMessage::Outbound(out) = msg {
                if let Some(chan) = out_channels.get(&out.channel) {
                    let _ = chan.send(out).await;
                }
            }
        }
    });

    // Stay alive
    loop {
        sleep(Duration::from_secs(1)).await;
    }
}
