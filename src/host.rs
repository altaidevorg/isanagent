//! Public host entry point for embedding the full IsanAgent runtime.
//!
//! This module owns the same runtime construction that powers the isanagent
//! binary. Embedders can now start that runtime without spawning a second
//! process.

use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch, RwLock};

use crate::agent::{AgentLogic, AgentLogicParams};
use crate::bus::{BusMessage, InboundMessage, LoggerControlMessage, TelemetryEvent};
use crate::channels::terminal::{
    build_agent_thought_terminal_notice, build_tool_call_terminal_notice,
    build_tool_progress_terminal_notice, build_tool_result_terminal_notice,
    terminal_startup_suppresses_plain_banner, TerminalChannelConfig, TerminalMode,
};
use crate::channels::oneshot::OneshotChannel;
use crate::channels::{
    api::ApiChannel, email::EmailChannel, slack::SlackChannel, terminal::TerminalChannel, Channel,
};
use crate::clarification::ClarificationHub;
use crate::config::{AppConfig, HarnessConfig, ProviderConfig, ShellPolicyConfig};
use crate::execution::{ExecutionJobManager, InflightSyncRegistry};
use crate::logging::{
    create_logger_channel, create_logging_actor_or_fallback, init_runtime_logger,
    LOGGER_QUEUE_CAPACITY,
};
use crate::scheduler::{
    validate_multi_tenant_edge_runtime, CronActor, CronSchedulingMode, MultiTenantEdgeCronScheduler,
};
use crate::session::SessionManager;
use crate::skills::SkillRegistry;
use crate::tools::builtin::{
    CronTool, EditFileTool, GetEnvTool, GitWorktreeTool, GlobFilesTool, ListDirTool, MessageTool,
    PythonRunTool, ReadFileTool, SearchTextTool, ShellExecTool, WebFetchTool, WebSearchTool,
    WriteFileTool,
};
use crate::tools::execution::{
    ExecutionArtifactListTool, ExecutionCancelTool, ExecutionEnvInfoTool, ExecutionJobCancelTool,
    ExecutionJobListTool, ExecutionJobResultTool, ExecutionJobStatusTool,
    ExecutionRunBackgroundTool, ExecutionRunTool, ExecutionSessionCloseTool,
    ExecutionSessionCreateTool,
};
use crate::tools::ml_domain::{ArxivFetchTool, ArxivSearchTool, HfHubFileFetchTool};
use crate::tools::workflow::{AskUserTool, TodoWriteTool, ToolSearchTool};
use crate::tools::ToolRegistry;
use crate::workspace::{resolve_workspace_root, IsanagentWorkspace};
use crate::{NodeHandle, Supervisor, SupervisorPolicy};
use colored::Colorize;

const DEFAULT_PROVIDER_NAME: &str = "gemini";
const DEFAULT_PROVIDER_MODEL_NAME: &str = "gemini-2.5-flash";
const DEFAULT_PROVIDER_API_KEY_ENV: &str = "GEMINI_API_KEY";

const EXECUTION_HARNESS_SYSTEM_GUIDANCE: &str = r#"

--- Execution harness ---
- Call execution_env_info to read max_wall_secs and default_run_timeout_secs before long runs.
- Set timeout_secs explicitly for generation, training, or heavy I/O (up to max_wall_secs). Omit timeout_secs only for quick checks; use smaller values for tight polling loops.
- Prefer execution_run_background when work may block the reasoning loop for many minutes. When a job finishes, the harness may enqueue a synthetic follow-up message so you can call execution_job_status / execution_job_result without waiting for the user; if wake_on_job_terminal is off in config, poll manually.
- Pass description on execution_run and execution_run_background for runs that may exceed ~30 seconds or whenever you want a clear label in the terminal UI and audit logs.
- Pilot with a short execution_run, then scale; do not launch many parallel heavy jobs until one path succeeds.
- Prefer grep-friendly logging in training scripts (plain text lines) so stdout stays searchable in captured logs.
- Know where outputs live: sandbox-relative paths, execution_artifact_list, run journals under workspace_dir/.system_generated/execution_history/, and execution_runs.jsonl.
- For Jupyter/SSH: confirm interpreter and (if needed) GPU visibility with a tiny execution_run before a long job.
- For Google Colab: use the colab-cli skill (invoke colab commands via exec) instead of execution_run with a built-in provider.
"#;

#[derive(Debug, Clone, Default)]
pub struct HostConfig {
    /// Durable IsanAgent state directory.
    pub workspace: Option<PathBuf>,
    /// Path to the IsanAgent TOML configuration file.
    pub config: Option<PathBuf>,
    /// Project directory exposed to tools. Defaults to IsanAgent's own
    /// workspace sandbox for backward compatibility.
    pub sandbox: Option<PathBuf>,
    /// Optional `provider/model` or model-only override selected by the host.
    pub model: Option<String>,
    /// Optional provider/model used after the primary provider is exhausted.
    pub fallback_model: Option<String>,
    /// Optional interactive shell and file-edit policy override selected by
    /// the host application.
    pub permission: Option<HostPermissionMode>,
    /// Disable ANSI foreground colors while retaining terminal structure.
    pub no_color: bool,
    /// ALTAI terminal appearance. `no_color` / `NO_COLOR` still win.
    pub theme: HostThemeMode,
    /// Existing terminal chat identifier to load on startup.
    pub resume: Option<String>,
    /// Files preloaded into the terminal's next composed message.
    pub files: Vec<PathBuf>,
    pub line_mode: bool,
    /// Optional override for between-turn auto-compaction (`None` = config default).
    pub compact_auto: Option<bool>,
    /// Optional override for the auto-compaction token threshold.
    pub compact_threshold_tokens: Option<usize>,
    /// Optional override for recent-summary / tail retention.
    pub compact_tail_turns: Option<usize>,
    /// When set, disable the interactive terminal, inject this prompt once through
    /// the headless `altai-cli` channel, and shut down after the run terminates.
    pub oneshot_prompt: Option<String>,
    /// Optional observer for raw host bus messages during interactive or oneshot runs.
    pub observe_tx: Option<mpsc::UnboundedSender<BusMessage>>,
    /// Deterministic in-process assistant replies for tests and CI smokes.
    /// When set, the host skips network providers and returns these responses
    /// in order (the last response repeats).
    pub scripted_responses: Option<Vec<String>>,
}

pub use crate::channels::oneshot::{OneshotOutcome, OneshotResult};
pub use crate::channels::terminal_ui::HostThemeMode;

/// Interactive permission modes exposed to embedding hosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostPermissionMode {
    Ask,
    AutoEdit,
    Plan,
    Bypass,
}

pub type HostError = Box<dyn std::error::Error + Send + Sync>;
pub type HostResult<T> = Result<T, HostError>;

/// Lifecycle events emitted by a host started through [`spawn_host`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostEvent {
    /// The host task has been scheduled and is beginning runtime setup.
    Starting,
    /// The host task exited normally or with an error.
    Stopped { outcome: HostExit },
}

/// The terminal state of a spawned host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostExit {
    Completed,
    Failed(String),
}

/// A running IsanAgent host with explicit lifecycle observation and shutdown.
pub struct HostHandle {
    shutdown_tx: mpsc::UnboundedSender<()>,
    events: mpsc::UnboundedReceiver<HostEvent>,
    task: tokio::task::JoinHandle<HostResult<()>>,
}

impl HostHandle {
    /// Request graceful host shutdown. Returns false only when the host task
    /// has already stopped accepting control messages.
    pub fn shutdown(&self) -> bool {
        self.shutdown_tx.send(()).is_ok()
    }

    /// Receive the next lifecycle event, or `None` once the host task closes
    /// its event stream.
    pub async fn next_event(&mut self) -> Option<HostEvent> {
        self.events.recv().await
    }

    /// Wait for the complete host shutdown sequence and surface any startup or
    /// runtime error from the embedded host.
    pub async fn wait(self) -> HostResult<()> {
        self.task
            .await
            .map_err(|error| Box::new(error) as HostError)?
    }
}

/// Spawn the complete IsanAgent runtime and return explicit lifecycle control.
///
/// This is the embedding entry point for applications such as ALTAI. The
/// existing [`start_host`] function remains available for callers that simply
/// want to await the host until it exits.
pub fn spawn_host(config: HostConfig) -> HostHandle {
    let (shutdown_tx, shutdown_rx) = mpsc::unbounded_channel();
    let (events_tx, events) = mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        let _ = events_tx.send(HostEvent::Starting);
        let result = run_host(config, Some(shutdown_rx), None).await;
        let outcome = match &result {
            Ok(()) => HostExit::Completed,
            Err(error) => HostExit::Failed(error.to_string()),
        };
        let _ = events_tx.send(HostEvent::Stopped { outcome });
        result
    });

    HostHandle {
        shutdown_tx,
        events,
        task,
    }
}

/// Start the complete IsanAgent runtime, including its terminal channel when
/// enabled by the selected configuration.
pub async fn start_host(config: HostConfig) -> HostResult<()> {
    let _ = run_host(config, None, None).await?;
    Ok(())
}

/// Run a single headless prompt through the shared host runtime and shut down
/// after the matching reasoning run terminates.
pub async fn run_oneshot(mut config: HostConfig) -> HostResult<OneshotResult> {
    if config
        .oneshot_prompt
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .is_empty()
    {
        return Err(std::io::Error::other("oneshot prompt must be non-empty").into());
    }
    // One-shot runs never own an interactive TTY channel.
    config.line_mode = false;
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let host_result = run_host(config, None, Some(result_tx)).await;
    match result_rx.await {
        Ok(result) => Ok(result),
        Err(_) => Err(host_result.err().unwrap_or_else(|| {
            std::io::Error::other("oneshot host exited without a terminal result").into()
        })),
    }
}

async fn run_host(
    config: HostConfig,
    mut external_shutdown_rx: Option<mpsc::UnboundedReceiver<()>>,
    oneshot_result_tx: Option<tokio::sync::oneshot::Sender<OneshotResult>>,
) -> HostResult<()> {
    let workspace_arg = config
        .workspace
        .as_ref()
        .map(|path| path.to_string_lossy().to_string());
    let config_arg = config
        .config
        .as_ref()
        .map(|path| path.to_string_lossy().to_string());
    let sandbox = config.sandbox.as_deref();
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

    let oneshot_mode = config
        .oneshot_prompt
        .as_ref()
        .map(|prompt| prompt.trim())
        .is_some_and(|prompt| !prompt.is_empty());
    if oneshot_mode {
        eprintln!("Starting Advanced isanagent System...");
    } else {
        println!("Starting Advanced isanagent System...");
    }
    log::info!("Starting Advanced isanagent System.");

    let mut workspace = IsanagentWorkspace::new_with_sandbox(
        workspace_arg.as_deref(),
        config_arg.as_deref(),
        sandbox,
    )?;
    apply_host_overrides(&mut workspace.config, &config).map_err(std::io::Error::other)?;
    let oneshot_prompt = config
        .oneshot_prompt
        .as_ref()
        .map(|prompt| prompt.trim().to_string())
        .filter(|prompt| !prompt.is_empty());
    if oneshot_mode {
        // One-shot hosts never attach the interactive terminal channel.
        let terminal = workspace.config.terminal.get_or_insert_with(Default::default);
        terminal.enabled = Some(false);
        eprintln!("Loading isanagent workspace at: {:?}", workspace.dir);
    } else {
        println!("Loading isanagent workspace at: {:?}", workspace.dir);
    }
    log::info!("Loading isanagent workspace at {:?}", workspace.dir);

    if oneshot_prompt.is_none()
        && !workspace.config.terminal_enabled()
        && !workspace.config.has_non_terminal_inbound_channel()
    {
        return Err(std::io::Error::other(
            "Invalid config: [terminal] enabled = false requires at least one other inbound channel. \
Enable [api], [slack], or [email] (with enabled = true) so the agent can receive messages without stdin.",
        )
        .into());
    }

    if !oneshot_mode {
        maybe_prompt_uv_install_on_launch(&workspace).await;
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
    let memory_actor = crate::memory::SqliteMemoryActor::new(db_path_str).map_err(|e| {
        std::io::Error::other(format!("Failed to initialize SqliteMemoryActor: {}", e))
    })?;
    let memory_node = NodeHandle::<crate::memory::MemoryMessage>::new(
        memory_actor,
        100,
        1,
        Duration::from_millis(5),
    );

    let session_manager = SessionManager::new(memory_node.clone());

    // 4. Setup Tools
    let (global_outbound_tx, mut global_outbound_rx) = mpsc::channel(100);
    // Inbound bus: created early so execution job completion can enqueue synthetic follow-up messages
    // before channel setup; the forwarder to the agent is spawned after `agent_node` exists.
    let (bus_tx, mut bus_rx) = mpsc::channel(100);
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

        let client = crate::multi_tenant_edge::CronRegistrationClient::from_env()
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
        bus_tx.clone(),
    )
    .map_err(std::io::Error::other)?;
    let cron_node = NodeHandle::new(cron_logic, 10, 3, Duration::from_millis(50));

    let max_tool_output_chars = workspace
        .config
        .resolved_max_tool_output_chars()
        .unwrap_or(3000);

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
    tools.register(Box::new(GetEnvTool));
    tools.register(Box::new(PythonRunTool {
        workspace_dir: workspace.sandbox_dir.clone(),
    }));
    if workspace.config.git_worktree_tool_enabled() {
        tools.register(Box::new(GitWorktreeTool {
            workspace_dir: workspace.sandbox_dir.clone(),
            restrict_to_workspace: restrict,
            allow_path_outside_sandbox: workspace.config.git_worktree_allow_path_outside_sandbox(),
        }));
    }
    if workspace.config.checkpoint_enabled() {
        // Backups live in the outer rim (never inside the agent's editable sandbox); restores are
        // confined to the sandbox when the file tools are workspace-restricted.
        crate::checkpoint::init(
            workspace.dir.join(".system_generated").join("checkpoints"),
            restrict.then(|| workspace.sandbox_dir.clone()),
        );
        tools.register(Box::new(crate::checkpoint::CheckpointTool));
    }
    let mut inflight_sync_outer: Option<Arc<InflightSyncRegistry>> = None;
    let mut execution_harness_for_shutdown: Option<Arc<crate::execution::ExecutionHarness>> = None;
    if workspace.config.execution_harness_enabled() {
        let harness = crate::execution::build_execution_harness(
            workspace.dir.clone(),
            workspace.sandbox_dir.clone(),
            restrict,
            &workspace.config,
        )
        .map_err(|e| std::io::Error::other(format!("execution harness: {e}")))?;
        execution_harness_for_shutdown = Some(harness.clone());
        let execution_jobs = Arc::new(ExecutionJobManager::new(
            harness.clone(),
            global_outbound_tx.clone(),
            Some(bus_tx.clone()),
            workspace.config.execution_wake_on_job_terminal(),
        ));
        let inflight_sync = Arc::new(InflightSyncRegistry::new());
        inflight_sync_outer = Some(inflight_sync.clone());
        tools.register(Box::new(ExecutionSessionCreateTool {
            harness: harness.clone(),
        }));
        tools.register(Box::new(ExecutionRunTool {
            harness: harness.clone(),
            outbound_tx: global_outbound_tx.clone(),
            jobs: Some(execution_jobs.clone()),
            inflight: Some(inflight_sync.clone()),
        }));
        tools.register(Box::new(ExecutionRunBackgroundTool {
            harness: harness.clone(),
            jobs: execution_jobs.clone(),
        }));
        tools.register(Box::new(ExecutionJobStatusTool {
            jobs: execution_jobs.clone(),
        }));
        tools.register(Box::new(ExecutionJobResultTool {
            jobs: execution_jobs.clone(),
            max_tool_output_chars,
        }));
        tools.register(Box::new(crate::tools::execution::ExecutionReadLogTool {
            jobs: execution_jobs.clone(),
            harness: harness.clone(),
        }));
        tools.register(Box::new(ExecutionJobListTool {
            jobs: execution_jobs.clone(),
        }));
        tools.register(Box::new(ExecutionJobCancelTool {
            jobs: execution_jobs.clone(),
        }));
        tools.register(Box::new(ExecutionArtifactListTool {
            harness: harness.clone(),
        }));
        tools.register(Box::new(ExecutionCancelTool {
            harness: harness.clone(),
        }));
        tools.register(Box::new(ExecutionSessionCloseTool {
            harness: harness.clone(),
        }));
        tools.register(Box::new(ExecutionEnvInfoTool {
            harness: harness.clone(),
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
        workspace_dir: workspace.dir.clone(),
    }));
    tools.register(Box::new(ArxivSearchTool {
        max_output_chars: max_web_output_chars,
    }));
    tools.register(Box::new(ArxivFetchTool {
        workspace_dir: workspace.dir.clone(),
    }));
    tools.register(Box::new(HfHubFileFetchTool {
        max_output_chars: max_web_output_chars,
    }));
    tools.register(Box::new(CronTool {
        cron_node: cron_node.clone(),
        multi_tenant_edge_cron_enabled: mte_cron_scheduler.is_some(),
        mte_cron_scheduler: mte_cron_scheduler.clone(),
        db_path: db_path_str.to_string(),
    }));
    tools.register(Box::new(MessageTool {
        outbound_tx: global_outbound_tx.clone(),
    }));
    tools.register(Box::new(AskUserTool {
        clarification_hub: clarification_hub.clone(),
        outbound_tx: global_outbound_tx.clone(),
        memory_node: Some(memory_node.clone()),
    }));
    // PR-10: agent-triggered compaction. Tool posts a TriggerCompaction bus
    // message with `AgentSelf` reason; the agent processes it between turns
    // to respect the per-chat FIFO invariant (AGENTS.md).
    tools.register(Box::new(crate::tools::compact::CompactContextTool {
        outbound_tx: global_outbound_tx.clone(),
    }));
    // PR-7: re-materialize tool results that were compacted out of the active
    // conversation. Reads the cache populated by `do_compaction`'s swap step.
    tools.register(Box::new(crate::tools::recall::RecallToolResultTool {
        memory_node: memory_node.clone(),
        outbound_tx: global_outbound_tx.clone(),
    }));
    tools.register(Box::new(crate::tools::builtin::SearchMemoryTool {
        memory_node: memory_node.clone(),
    }));
    tools.register(Box::new(crate::tools::builtin::FetchMemoryByDateTool {
        memory_node: memory_node.clone(),
    }));

    tools.register(Box::new(TodoWriteTool {
        memory_node: memory_node.clone(),
    }));
    if workspace.config.kernel_porting_harness_enabled() {
        crate::tools::kernel_porting::register_kernel_porting_tools(
            &mut tools,
            workspace.sandbox_dir.clone(),
            std::sync::Arc::new(workspace.config.clone()),
        );
    }
    if workspace.config.autotrainess_harness_enabled() {
        crate::tools::autotrainess::register_autotrainess_tools(
            &mut tools,
            workspace.sandbox_dir.clone(),
            std::sync::Arc::new(workspace.config.clone()),
        );
    }
    let tool_catalog = tools.catalog_handle();
    tools.register(Box::new(ToolSearchTool {
        catalog: tool_catalog,
    }));

    // 5. Setup Provider (Dynamic from config)
    let default_provider_cfg =
        workspace
            .config
            .provider
            .clone()
            .unwrap_or_else(|| crate::config::ProviderConfig {
                provider_name: DEFAULT_PROVIDER_NAME.to_string(),
                model_name: DEFAULT_PROVIDER_MODEL_NAME.to_string(),
                models: None,
                api_key_env: DEFAULT_PROVIDER_API_KEY_ENV.to_string(),
                api_key: None,
                base_url: None,
            });

    // Expand family-format providers into flat per-model map (once, reused everywhere).
    let expanded_providers = workspace.config.expanded_providers();

    // Try to find any provider with a valid API key. No key = start with NoKeyProvider.
    // Priority: last_model file (remembers /model choice) → [provider] → first [providers.*] with key.
    let last_model_path = workspace.dir.join(".system_generated/last_model");
    let remembered_key = std::fs::read_to_string(&last_model_path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let (provider_cfg, api_key): (Option<crate::config::ProviderConfig>, Option<String>) = {
        // 0. Try remembered last model choice
        let mut found_remembered = None;
        if let Some(ref key_name) = remembered_key {
            if let Some(cfg) = expanded_providers.get(key_name) {
                if let Ok(key) = cfg.resolve_api_key() {
                    found_remembered = Some((cfg.clone(), key));
                }
            }
        }
        if let Some((cfg, key)) = found_remembered {
            (Some(cfg), Some(key))
        }
        // 1. Try default [provider]
        else if let Ok(key) = default_provider_cfg.resolve_api_key() {
            (Some(default_provider_cfg.clone()), Some(key))
        } else {
            // 2. Try any expanded [providers.*] entry
            let mut found: Option<(crate::config::ProviderConfig, String)> = None;
            for cfg in expanded_providers.values() {
                if let Ok(key) = cfg.resolve_api_key() {
                    found = Some((cfg.clone(), key));
                    break;
                }
            }
            if let Some((cfg, key)) = found {
                (Some(cfg), Some(key))
            } else {
                // No key anywhere — start without one (NoKeyProvider)
                (None, None)
            }
        }
    };

    let model_name = provider_cfg
        .as_ref()
        .map(|c| c.model_name.clone())
        .unwrap_or_else(|| "(no model)".to_string());

    let (provider, reflection_provider, fallback_providers): (
        Box<dyn crate::traits::Provider>,
        Box<dyn crate::traits::Provider>,
        Vec<crate::agent::FallbackProviderSpec>,
    ) = if let Some(responses) = config.scripted_responses.clone() {
        (
            Box::new(crate::provider::ScriptedProvider::new(responses.clone())),
            Box::new(crate::provider::ScriptedProvider::new(responses)),
            Vec::new(),
        )
    } else if let (Some(cfg), Some(key)) = (&provider_cfg, &api_key) {
        let base_url = cfg.resolved_base_url().map_err(std::io::Error::other)?;
        let p1 = crate::provider::create_provider(&cfg.provider_name, &base_url, key, &model_name);
        let p2 = crate::provider::create_provider(&cfg.provider_name, &base_url, key, &model_name);

        // Keep all configured providers as immutable candidates owned by this AgentLogic. Each run
        // filters its own primary by full (provider, base_url, model) identity while snapshotting,
        // so concurrent runs and `/model` switches cannot rewrite one another's fallback policy.
        let candidates: Vec<crate::agent::FallbackProviderSpec> = expanded_providers
            .values()
            .filter_map(|fb_cfg| {
                let fb_key = fb_cfg.resolve_api_key().ok()?;
                let fb_base = fb_cfg.resolved_base_url().ok()?;
                Some(crate::agent::FallbackProviderSpec {
                    provider_name: fb_cfg.provider_name.clone(),
                    base_url: fb_base,
                    api_key: fb_key,
                    model_name: fb_cfg.model_name.clone(),
                })
            })
            .collect();
        let initial_fallbacks = crate::agent::build_fallback_specs(
            &cfg.provider_name,
            &base_url,
            &model_name,
            candidates.clone(),
        );
        if !initial_fallbacks.is_empty() {
            log::info!(
                "Cross-provider failover enabled with {} fallback provider(s).",
                initial_fallbacks.len()
            );
        }

        (p1, p2, candidates)
    } else {
        // No API key found — list the env vars the user could set.
        let env_vars: Vec<String> = expanded_providers
            .values()
            .map(|c| c.resolved_api_key_env())
            .filter(|e| !e.is_empty())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        eprintln!("No API key configured for any provider.");
        if !env_vars.is_empty() {
            eprintln!("Set one of: {}", env_vars.join(", "));
        }
        eprintln!("Or run `isanagent onboard` to create a config.toml, then replace \"<changethis>\" with your key.");
        eprintln!("Use /model at runtime to configure one.");
        (
            Box::new(crate::provider::NoKeyProvider),
            Box::new(crate::provider::NoKeyProvider),
            Vec::new(),
        )
    };

    let provider_credentials = if let (Some(cfg), Some(key)) = (&provider_cfg, &api_key) {
        crate::provider::ProviderCredentials {
            provider_name: cfg.provider_name.clone(),
            base_url: cfg.resolved_base_url().unwrap_or_default(),
            api_key: key.clone(),
            model_name: model_name.clone(),
        }
    } else {
        crate::provider::ProviderCredentials::empty()
    };

    // 5.5 Setup Reflection Engine
    let memory_config = workspace.config.memory.clone().unwrap_or_default();
    let reflection_engine = crate::reflection::ReflectionEngine::new(
        memory_node.clone(),
        workspace.sandbox_dir.clone(),
        reflection_provider,
        memory_config,
        logger_bus_tx.clone(),
        app_shutdown_rx.clone(),
    );
    let reflection_task = reflection_engine.start();

    // 6. Compile Agent System Prompt
    let mut system_prompt = workspace.compile_system_prompt();
    if workspace.config.ml_engineer_harness_enabled() {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(crate::ml_engineer::HARNESS_OVERLAY);
    }
    if workspace.config.execution_harness_enabled() {
        system_prompt.push_str(EXECUTION_HARNESS_SYSTEM_GUIDANCE);
    }
    let subagent_system_prompt = if workspace.config.ml_engineer_subagent_research_overlay() {
        format!(
            "{}\n{}",
            system_prompt,
            crate::ml_engineer::SUBAGENT_RESEARCH_APPEND
        )
    } else {
        system_prompt.clone()
    };

    let harness_runtime_summary = workspace.config.runtime_harness_summary_lines().join("\n");
    let forbid_final_without_tools = workspace.config.ml_engineer_forbid_final_without_tools();
    let shell_policy = workspace.config.resolved_shell_policy();
    let default_harness = crate::config::HarnessConfig::default();
    let harness_ref = workspace
        .config
        .harness
        .as_ref()
        .unwrap_or(&default_harness);
    let hook_tool_ctx = crate::hooks::ToolCallHookContext::from_harness_config(
        &workspace.dir,
        &workspace.sandbox_dir,
        harness_ref,
    );

    // Prepare startup visual references before we move the structs
    let skill_names = skills.get_skill_names().join(", ");
    let skill_count = skills.get_skill_names().len();

    let mut tool_names_list = tools.get_tool_names();
    tool_names_list.sort();
    let tool_names = tool_names_list.join(", ");
    let tool_count = tool_names_list.len();

    // 7. Create Agent Logic
    let max_iterations = workspace.config.resolved_max_iterations().unwrap_or(50);
    let mut max_recent_summaries = workspace
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
    let mut short_term_threshold_tokens = workspace
        .config
        .memory
        .as_ref()
        .and_then(|m| m.short_term_threshold_tokens)
        .unwrap_or(100000);
    if config.compact_auto == Some(false) {
        short_term_threshold_tokens = usize::MAX;
    } else if let Some(tokens) = config.compact_threshold_tokens {
        short_term_threshold_tokens = tokens.max(8_000);
    }
    if let Some(tail) = config.compact_tail_turns {
        max_recent_summaries = tail.max(1);
    }
    let tool_execution_activity = if multi_tenant_edge_cfg
        .activity_heartbeat_enabled
        .unwrap_or(false)
    {
        match crate::multi_tenant_edge::ActivityHeartbeatClient::from_env(logger_bus_tx.clone()) {
            Ok(client) => Some(std::sync::Arc::new(client)),
            Err(error) => {
                let _ = logger_bus_tx.send(BusMessage::Log(crate::bus::LogEvent::warn(
                    "isanagent",
                    &error,
                )));
                None
            }
        }
    } else {
        None
    };

    // Load agent definitions from config; use built-in defaults when none configured.
    let agent_defs = workspace.config.agent_definitions();
    let agent_defs = if agent_defs.is_empty() {
        crate::agent::registry::default_agent_definitions()
    } else {
        agent_defs
    };
    let agent_registry = std::sync::Arc::new(crate::agent::AgentRegistry::from_definitions(
        &agent_defs,
        &workspace.sandbox_dir,
    ));

    // Inject agent descriptions into the system prompt
    let agent_prompt_section = agent_registry.compile_agent_prompt_section();
    if !agent_prompt_section.is_empty() {
        system_prompt.push_str(&agent_prompt_section);
    }

    let subagent = if workspace.config.subagent_harness_enabled() {
        Some(crate::agent::SubagentHarnessParams {
            cancel_children_on_parent_cancel: workspace
                .config
                .subagent_cancel_children_on_parent_cancel(),
            allowed_tools: workspace
                .config
                .subagent_allowed_tools_set()
                .map(std::sync::Arc::new),
            max_tasks: workspace.config.subagent_max_tasks(),
            max_wait_secs: workspace.config.subagent_max_wait_secs(),
            agent_registry: Some(agent_registry),
            wake_on_completion: workspace.config.subagent_wake_on_completion(),
            task_history_retention: workspace.config.subagent_task_history_retention(),
            bus_tx: Some(bus_tx.clone()),
            workspace_dir: workspace.sandbox_dir.clone(),
        })
    } else {
        None
    };

    let agent_logic = AgentLogic::new_with_fallback_providers(
        AgentLogicParams {
            name: "isanagent".to_string(),
            provider,
            provider_credentials,
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
            doom_loop_enabled: workspace.config.doom_loop_enabled(),
            harness_runtime_summary,
            subagent_system_prompt,
            forbid_final_without_tools,
            shell_policy,
            hook_tool_ctx,
        },
        fallback_providers,
    );
    let agent_logic = if let Some(tool_execution_activity) = tool_execution_activity {
        agent_logic.with_tool_execution_activity(tool_execution_activity)
    } else {
        agent_logic
    };

    // 8. Wrap Agent in NodeHandle
    let agent_node = NodeHandle::<BusMessage>::new(agent_logic, 100, 3, Duration::from_millis(50));

    // 10. Setup channels (terminal is optional for headless / Docker API-only runs)
    let mut out_channels: HashMap<String, Arc<dyn Channel>> = HashMap::new();

    let active_terminal_session_chat: Arc<RwLock<String>> = Arc::new(RwLock::new(String::new()));

    let terminal_chat_id = if workspace.config.terminal_enabled() {
        let id = config
            .resume
            .as_deref()
            .filter(|id| !id.trim().is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        *active_terminal_session_chat.write().await = id.clone();
        let terminal = Arc::new(TerminalChannel::new(TerminalChannelConfig {
            chat_id: id.clone(),
            logger_tx: logger_bus_tx.clone(),
            shutdown_tx: shutdown_tx.clone(),
            workspace_dir: workspace.dir.clone(),
            sandbox_dir: workspace.sandbox_dir.clone(),
            status_model: model_name.clone(),
            status_permission: host_permission_label(config.permission),
            memory_node: memory_node.clone(),
            providers: {
                // Merge default [provider] + expanded [providers.*] into one map for /model selector
                let mut all_providers = expanded_providers.clone();
                if let Some(def) = &workspace.config.provider {
                    let key = format!("{}/{}", def.provider_name, def.model_name);
                    all_providers.entry(key).or_insert_with(|| def.clone());
                }
                all_providers
            },
            color_enabled: !config.no_color && config.theme != HostThemeMode::NoColor,
            theme: config.theme,
            resume_session: config.resume.is_some(),
            initial_files: config.files.clone(),
            mode: if config.line_mode {
                TerminalMode::Line
            } else {
                TerminalMode::Tui
            },
        }));
        terminal.start(bus_tx.clone()).await?;
        out_channels.insert(terminal.name().to_string(), terminal);
        Some(id)
    } else {
        log::info!("Terminal channel disabled via config; stdin will not be read.");
        None
    };

    let oneshot_channel = if let Some(prompt) = oneshot_prompt.clone() {
        let result_tx = oneshot_result_tx.ok_or_else(|| {
            std::io::Error::other("oneshot prompt requires run_oneshot or an explicit result channel")
        })?;
        let id = config
            .resume
            .as_deref()
            .filter(|id| !id.trim().is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let (attachments, attach_warnings) =
            crate::channels::terminal_ui::load_host_file_attachments(
                &workspace.sandbox_dir,
                &config.files,
            );
        for warning in attach_warnings {
            log::warn!("oneshot attachment: {warning}");
            let _ = logger_bus_tx.send(BusMessage::Log(crate::bus::LogEvent::warn(
                "altai-cli",
                &warning,
            )));
        }
        let channel = Arc::new(OneshotChannel::new(
            id,
            prompt,
            config.files.clone(),
            attachments,
            result_tx,
            shutdown_tx.clone(),
            config.observe_tx.clone(),
        ));
        channel.start(bus_tx.clone()).await?;
        out_channels.insert(channel.name().to_string(), channel.clone());
        Some(channel)
    } else {
        if oneshot_result_tx.is_some() {
            return Err(std::io::Error::other(
                "oneshot result channel provided without oneshot_prompt",
            )
            .into());
        }
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
                workspace.sandbox_dir.clone(),
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
    if !oneshot_mode && !terminal_startup_suppresses_plain_banner(&workspace.config) {
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
    let inflight_promote = inflight_sync_outer.clone();
    let active_terminal_session_for_bus = active_terminal_session_chat.clone();
    let oneshot_for_bus = oneshot_channel.clone();
    let observe_tx = config.observe_tx.clone();
    tokio::spawn(async move {
        while let Some(msg) = bus_rx.recv().await {
            if let Some(tx) = observe_tx.as_ref() {
                let _ = tx.send(msg.clone());
            }
            if let Some(oneshot) = oneshot_for_bus.as_ref() {
                oneshot.observe_bus_message(&msg);
            }
            let _ = logger_tx.send(msg.clone());
            // Intercept /background slash commands and fire the in-flight oneshot.
            if let BusMessage::PromoteSyncToBackground(chat_id) = &msg {
                if let Some(reg) = inflight_promote.as_ref() {
                    let promoted = reg.promote(chat_id);
                    log::debug!("PromoteSyncToBackground chat={chat_id} promoted={promoted}");
                }
                continue;
            }
            if let BusMessage::SetTerminalSessionChat { chat_id } = &msg {
                *active_terminal_session_for_bus.write().await = chat_id.clone();
                continue;
            }
            // Only route Inbound, cancellation, and SwitchModel messages to the agent logic.
            // This prevents the agent from being flooded with its own telemetry or other system messages.
            if matches!(
                msg,
                BusMessage::Inbound(_)
                    | BusMessage::Cancel(_)
                    | BusMessage::CancelRun { .. }
                    | BusMessage::Steer { .. }
                    | BusMessage::SwitchModel { .. }
            ) {
                let _ = agent_tx.send_packet(msg).await;
            }
        }
    });

    // Listen for Outbound reasoning chunks and route back to the appropriate channel and logger
    let (listener_node, mut agent_rx) =
        NodeHandle::<BusMessage>::create_listener("completion", 100);
    let _ = (&agent_node - "completion") >> &listener_node;

    let agent_outbound_tx = global_outbound_tx.clone();
    tokio::spawn(async move {
        while let Some(bus_msg) = agent_rx.recv().await {
            if let crate::Message::Packet(packet) = bus_msg {
                match packet {
                    BusMessage::Outbound(out) => {
                        let _ = agent_outbound_tx.send(BusMessage::Outbound(out)).await;
                    }
                    BusMessage::Telemetry(tel) => {
                        let _ = agent_outbound_tx.send(BusMessage::Telemetry(tel)).await;
                    }
                    lifecycle @ BusMessage::RunLifecycle(_) => {
                        let _ = agent_outbound_tx.send(lifecycle).await;
                    }
                    _ => {}
                }
            }
        }
    });

    let logger_tx_outbound = logger_bus_tx.clone();
    let delivery_channels = out_channels.clone();
    let active_terminal_for_outbound = active_terminal_session_chat.clone();
    let oneshot_for_outbound = oneshot_channel.clone();
    let observe_tx_outbound = config.observe_tx.clone();
    tokio::spawn(async move {
        while let Some(msg) = global_outbound_rx.recv().await {
            if let Some(tx) = observe_tx_outbound.as_ref() {
                let _ = tx.send(msg.clone());
            }
            if let Some(oneshot) = oneshot_for_outbound.as_ref() {
                if !matches!(&msg, BusMessage::Outbound(_)) {
                    oneshot.observe_bus_message(&msg);
                }
            }
            // Deliver user-visible terminal traffic first. `LoggerHandle::send` uses a blocking
            // `sync_channel::send`; doing it before channel delivery can stall this task and make
            // tool-call lines and agent replies appear only after the run finishes.
            match &msg {
                BusMessage::Outbound(out) => {
                    if let Some(chan) = delivery_channels.get(&out.channel) {
                        if out.channel == "terminal" {
                            let active_chat = active_terminal_for_outbound.read().await.clone();
                            if out.chat_id != active_chat
                                && out
                                    .metadata
                                    .get("isanagent_notification")
                                    .and_then(|v| v.as_bool())
                                    != Some(true)
                            {
                                continue;
                            }
                        }
                        if let Err(e) = chan.send(out.clone()).await {
                            log::error!(
                                "Failed to deliver message via channel [{}]: {}",
                                chan.name(),
                                e
                            );
                        }
                    }
                }
                BusMessage::RunLifecycle(_) => {}
                BusMessage::Telemetry(TelemetryEvent::AgentThought {
                    chat_id,
                    thought,
                    background_job_id,
                }) => {
                    if active_terminal_for_outbound.read().await.as_str() == chat_id.as_str() {
                        let notice = build_agent_thought_terminal_notice(
                            chat_id,
                            thought,
                            background_job_id.as_deref(),
                        );
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
                                    background_job_id: background_job_id.clone(),
                                })
                                .await;
                        }
                    }
                }
                BusMessage::Telemetry(TelemetryEvent::ToolProgress {
                    chat_id,
                    channel,
                    tool_name,
                    tool_call_id,
                    message,
                    background_job_id,
                }) => {
                    if channel == "terminal"
                        && active_terminal_for_outbound.read().await.as_str() == chat_id.as_str()
                    {
                        let notice = build_tool_progress_terminal_notice(
                            chat_id,
                            tool_name,
                            message,
                            tool_call_id.as_deref(),
                            background_job_id.as_deref(),
                        );
                        if let Some(chan) = delivery_channels.get("terminal") {
                            if let Err(e) = chan.send(notice).await {
                                log::error!(
                                    "Failed to deliver tool-progress notice to terminal: {}",
                                    e
                                );
                            }
                        }
                    }
                    if let Some(api_chan) = delivery_channels.get("api") {
                        if let Some(api_chan) = api_chan.as_any().downcast_ref::<ApiChannel>() {
                            api_chan
                                .handle_telemetry(TelemetryEvent::ToolProgress {
                                    chat_id: chat_id.clone(),
                                    channel: channel.clone(),
                                    tool_name: tool_name.clone(),
                                    tool_call_id: tool_call_id.clone(),
                                    message: message.clone(),
                                    background_job_id: background_job_id.clone(),
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
                    tool_call_id,
                    background_job_id,
                }) if channel == "terminal" => {
                    if crate::channels::terminal::should_suppress_tool_notice_for_terminal(
                        tool_name, args,
                    ) {
                        // MessageTool already emits its own user-visible Outbound to the
                        // terminal; a synthetic tool-call notice would duplicate that line.
                    } else {
                        let notice = build_tool_call_terminal_notice(
                            chat_id,
                            tool_name,
                            args,
                            tool_call_id.as_deref(),
                            background_job_id.as_deref(),
                        );
                        if let Some(chan) = delivery_channels.get("terminal") {
                            if let Err(e) = chan.send(notice).await {
                                log::error!(
                                    "Failed to deliver tool-call notice to terminal: {}",
                                    e
                                );
                            }
                        }
                    }
                }
                BusMessage::Telemetry(TelemetryEvent::ToolResult {
                    channel,
                    chat_id,
                    tool_name,
                    result,
                    is_error,
                    tool_call_id,
                    background_job_id,
                }) if channel == "terminal" => {
                    if crate::channels::terminal::should_suppress_tool_notice_for_terminal(
                        tool_name, result,
                    ) {
                        // See ToolCall arm: avoid duplicating the user-visible MessageTool
                        // outbound with a redundant ack notice.
                    } else {
                        let notice = build_tool_result_terminal_notice(
                            chat_id,
                            tool_name,
                            result,
                            *is_error,
                            tool_call_id.as_deref(),
                            background_job_id.as_deref(),
                        );
                        if let Some(chan) = delivery_channels.get("terminal") {
                            if let Err(e) = chan.send(notice).await {
                                log::error!(
                                    "Failed to deliver tool-result notice to terminal: {}",
                                    e
                                );
                            }
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

    if workspace.config.background_jobs_enabled() && workspace.config.background_jobs_auto_resume()
    {
        recover_background_jobs_on_startup(&memory_node, &bus_tx, &global_outbound_tx).await;
    }

    tokio::select! {
        _ = shutdown_rx.recv() => {
            log::info!("Shutdown requested (terminal /exit or internal signal).");
        }
        _ = wait_for_external_shutdown(&mut external_shutdown_rx) => {
            log::info!("Shutdown requested by embedded host controller.");
        }
        _ = tokio::signal::ctrl_c() => {
            log::info!("Shutdown requested via Ctrl+C.");
        }
    }

    log::info!("Stopping channels and shutting down runtime.");

    if let Some(harness) = execution_harness_for_shutdown.as_ref() {
        let h = harness.clone();
        let shutdown_result =
            tokio::time::timeout(Duration::from_secs(5), async move { h.shutdown().await }).await;
        if shutdown_result.is_err() {
            log::warn!("Execution harness shutdown timed out after 5s; continuing exit anyway.");
        }
    }

    for channel in out_channels.values() {
        let _ = channel.stop().await;
    }

    let _ = app_shutdown_tx.send(true);
    let _ = reflection_task.await;

    let _ = logger_bus_tx.send(BusMessage::LoggerControl(LoggerControlMessage::Flush));
    let flush_result = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(msg) = logger_control_rx.recv().await {
            if let crate::Message::Packet(BusMessage::LoggerControl(
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

fn apply_host_overrides(config: &mut AppConfig, host: &HostConfig) -> Result<(), String> {
    if let Some(model) = host.model.as_deref() {
        let model = model.trim();
        if model.is_empty() {
            return Err("model override cannot be empty".to_string());
        }

        let provider = config.provider.get_or_insert_with(ProviderConfig::default);
        if let Some((provider_name, model_name)) = model.split_once('/') {
            if provider_name.is_empty() || model_name.is_empty() {
                return Err(format!("invalid provider/model override: {model}"));
            }
            provider.provider_name = provider_name.to_string();
            provider.model_name = model_name.to_string();
            // Let the provider's conventional key variable be inferred after
            // changing provider family (for example ANTHROPIC_API_KEY).
            provider.api_key_env.clear();
        } else {
            provider.model_name = model.to_string();
        }
    }

    if let Some(model) = host.fallback_model.as_deref() {
        let model = model.trim();
        if model.is_empty() {
            return Err("fallback model override cannot be empty".to_string());
        }
        let mut fallback = config.provider.clone().unwrap_or_default();
        if let Some((provider_name, model_name)) = model.split_once('/') {
            if provider_name.is_empty() || model_name.is_empty() {
                return Err(format!("invalid fallback provider/model override: {model}"));
            }
            fallback.provider_name = provider_name.to_string();
            fallback.model_name = model_name.to_string();
            fallback.api_key_env.clear();
        } else {
            fallback.model_name = model.to_string();
        }
        config
            .providers
            .get_or_insert_with(Default::default)
            .insert("__altai_cli_fallback".to_string(), fallback);
    }

    if let Some(permission) = host.permission {
        let shell_policy = config
            .harness
            .get_or_insert_with(HarnessConfig::default)
            .shell_policy
            .get_or_insert_with(ShellPolicyConfig::default);
        let (mode, edit_mode) = match permission {
            HostPermissionMode::Ask => ("ask", "ask"),
            HostPermissionMode::AutoEdit => ("ask", "allow"),
            // Match desktop: plan keeps shell at ask (read-only inspection with
            // approval for destructive commands) while denying file edits.
            HostPermissionMode::Plan => ("ask", "deny"),
            HostPermissionMode::Bypass => ("allow", "allow"),
        };
        shell_policy.mode = Some(mode.to_string());
        shell_policy.edit_mode = Some(edit_mode.to_string());
    }

    Ok(())
}

fn host_permission_label(permission: Option<HostPermissionMode>) -> String {
    match permission {
        Some(HostPermissionMode::Ask) => "ask".into(),
        Some(HostPermissionMode::AutoEdit) => "auto-edit".into(),
        Some(HostPermissionMode::Plan) => "plan".into(),
        Some(HostPermissionMode::Bypass) => "bypass".into(),
        None => "default".into(),
    }
}

async fn wait_for_external_shutdown(receiver: &mut Option<mpsc::UnboundedReceiver<()>>) {
    match receiver {
        Some(receiver) => {
            let _ = receiver.recv().await;
        }
        None => std::future::pending::<()>().await,
    }
}

async fn maybe_prompt_uv_install_on_launch(workspace: &IsanagentWorkspace) {
    if !workspace.config.execution_harness_enabled() {
        return;
    }
    let provider = workspace.config.execution_default_provider();
    let runtime = workspace.config.execution_local_python_runtime();
    let runtime_is_uv_managed = matches!(runtime.as_str(), "uv_managed" | "uvmanaged" | "uv");
    let requirements = workspace.config.execution_uv_requirements();

    if !(requirements.is_empty() || provider == "local" && runtime_is_uv_managed) {
        log::warn!(
            "[harness.execution].uv_requirements is set ({} entries) but the active provider/runtime \
does not consume it (provider={}, local_python_runtime={}). Move the dependencies into the active \
provider's environment, or set default_provider=\"local\" and local_python_runtime=\"uv_managed\".",
            requirements.len(),
            provider,
            runtime
        );
    }

    if provider != "local" || !runtime_is_uv_managed {
        return;
    }
    let uv_bin = workspace.config.execution_uv_binary();
    if !crate::execution::uv_binary_available(&uv_bin) {
        let interactive = workspace.config.terminal_enabled()
            && io::stdin().is_terminal()
            && io::stdout().is_terminal();
        if !interactive {
            log::warn!(
                "Execution local runtime is uv-managed but '{}' was not found on PATH. \
Install uv manually or run /install-python from terminal mode.",
                uv_bin
            );
            return;
        }

        let uv_bin_owned = uv_bin.to_string();
        let prompt_result = tokio::task::spawn_blocking(move || {
            println!(
                "\nExecution runtime is set to uv-managed, but '{}' was not found on PATH.",
                uv_bin_owned
            );
            println!("Install uv now? (yes/no)");
            let _ = io::stdout().flush();
            let mut line = String::new();
            loop {
                line.clear();
                if io::stdin().read_line(&mut line).is_err() {
                    println!("Unable to read input. Skipping uv installation prompt.");
                    break;
                }
                let ans = line.trim().to_ascii_lowercase();
                if matches!(ans.as_str(), "yes" | "y") {
                    match crate::execution::install_uv_best_effort() {
                        Ok(msg) => println!("{msg}"),
                        Err(err) => println!("Auto-install failed: {err}"),
                    }
                    break;
                }
                if matches!(ans.as_str(), "no" | "n") {
                    println!("Skipping uv installation. You can run /install-python anytime.");
                    break;
                }
                println!("Please answer yes or no:");
                let _ = io::stdout().flush();
            }
        })
        .await;

        if let Err(e) = prompt_result {
            log::warn!("uv installation prompt task failed: {e}");
        }
        return;
    }

    if requirements.is_empty() {
        return;
    }

    maybe_prompt_uv_requirements_install(workspace, &uv_bin, &requirements).await;
}

async fn recover_background_jobs_on_startup(
    memory_node: &NodeHandle<crate::memory::MemoryMessage>,
    bus_tx: &mpsc::Sender<BusMessage>,
    _outbound_tx: &mpsc::Sender<BusMessage>,
) {
    use crate::memory::{MemoryMessage, SharedReply};
    let (tx, rx) = tokio::sync::oneshot::channel();
    if memory_node
        .send_packet(MemoryMessage::ListBackgroundJobs {
            chat_id: None,
            channel: None,
            limit: 500,
            reply: SharedReply::new(tx),
        })
        .await
        .is_err()
    {
        return;
    }
    let rows = match rx.await {
        Ok(Ok(rows)) => rows,
        Ok(Err(e)) => {
            log::error!("Failed to list background jobs for recovery: {}", e);
            return;
        }
        Err(_) => {
            log::error!("Memory actor channel closed during background job recovery");
            return;
        }
    };
    let mut count = 0;
    for row in rows {
        if !row.resume_after_restart || row.state != "running" {
            continue;
        }
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            crate::bus::METADATA_SYNTHETIC_BACKGROUND_RESUME.to_string(),
            serde_json::Value::Bool(true),
        );
        metadata.insert(
            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
            serde_json::Value::String(row.job_id.clone()),
        );
        if let Err(e) = bus_tx
            .send(BusMessage::Inbound(InboundMessage {
                channel: row.channel.clone(),
                sender_id: "background_recovery".to_string(),
                chat_id: row.chat_id.clone(),
                thread_id: row.thread_id.clone(),
                content: format!("Resume background job {}", row.job_id),
                attachments: Vec::new(),
                metadata,
            }))
            .await
        {
            log::error!(
                "Failed to enqueue recovery message for job {}: {}",
                row.job_id,
                e
            );
        } else {
            log::info!("Recovered background job on startup: {}", row.job_id);
            count += 1;
        }
    }
    if count > 0 {
        log::info!(
            "Successfully resumed {} background job(s) on startup.",
            count
        );
    }
}

/// Inspect the uv-managed venv and prompt to install any missing `uv_requirements` entries.
/// No-op when the venv has not yet been created (first execution_run will populate it).
async fn maybe_prompt_uv_requirements_install(
    workspace: &IsanagentWorkspace,
    uv_bin: &str,
    requirements: &[String],
) {
    let local_cfg = crate::execution::LocalExecutionConfig {
        sandbox_dir: workspace.sandbox_dir.clone(),
        workspace_dir: workspace.dir.clone(),
        restrict_to_workspace: true,
        max_run_timeout_secs: workspace.config.execution_max_wall_secs(),
        max_output_bytes: workspace.config.execution_max_output_bytes(),
        max_sessions: workspace.config.execution_max_sessions(),
        python_executable: workspace.config.execution_python_executable(),
        python_repl: workspace.config.execution_local_python_repl_enabled(),
        python_runtime: crate::execution::LocalPythonRuntime::UvManaged,
        uv_binary: uv_bin.to_string(),
        uv_python: workspace.config.execution_uv_python(),
        uv_requirements: requirements.to_vec(),
        uv_env_root: workspace
            .dir
            .join(".system_generated")
            .join("uv")
            .join("envs"),
    };
    let Some(env_python) = crate::execution::uv_managed_env_python(&local_cfg) else {
        return;
    };
    if !env_python.exists() {
        // Venv not yet created; the first execution_run will create it and install requirements.
        return;
    }
    let uv_bin_owned = uv_bin.to_string();
    let env_python_owned = env_python.clone();
    let requirements_owned = requirements.to_vec();
    let status = tokio::task::spawn_blocking(move || {
        crate::execution::uv_requirements_status(
            &uv_bin_owned,
            &env_python_owned,
            &requirements_owned,
        )
    })
    .await;
    let missing = match status {
        Ok(Ok(Some(missing))) => missing,
        Ok(Ok(None)) => return,
        Ok(Err(e)) => {
            log::warn!("Could not verify uv_requirements (uv pip list failed): {e}");
            return;
        }
        Err(e) => {
            log::warn!("uv_requirements_status task join failed: {e}");
            return;
        }
    };
    if missing.is_empty() {
        return;
    }

    let interactive = workspace.config.terminal_enabled()
        && io::stdin().is_terminal()
        && io::stdout().is_terminal();
    if !interactive {
        log::warn!(
            "uv_requirements declares {} package(s) missing from the managed venv: {}. They will be \
installed on the next execution_run, or run /install-python (no-op when uv is present) and a \
quick `uv pip install` manually.",
            missing.len(),
            missing.join(", ")
        );
        return;
    }

    let uv_bin_owned = uv_bin.to_string();
    let env_python_owned = env_python.clone();
    let missing_owned = missing.clone();
    let prompt_result = tokio::task::spawn_blocking(move || {
        println!(
            "\n{} uv_requirements package(s) declared in config are not installed in the \
managed venv at {}:",
            missing_owned.len(),
            env_python_owned.display()
        );
        for r in &missing_owned {
            println!("  - {r}");
        }
        println!("Install them now? (yes/no)");
        let _ = io::stdout().flush();
        let mut line = String::new();
        loop {
            line.clear();
            if io::stdin().read_line(&mut line).is_err() {
                println!("Unable to read input. Skipping uv_requirements install prompt.");
                return;
            }
            let ans = line.trim().to_ascii_lowercase();
            if matches!(ans.as_str(), "yes" | "y") {
                let mut args = vec![
                    "pip".to_string(),
                    "install".to_string(),
                    "--python".to_string(),
                    env_python_owned.to_string_lossy().to_string(),
                ];
                args.extend(missing_owned.iter().cloned());
                let out = std::process::Command::new(&uv_bin_owned)
                    .args(&args)
                    .output();
                match out {
                    Ok(o) if o.status.success() => {
                        println!("uv_requirements installed.");
                    }
                    Ok(o) => {
                        let stderr = String::from_utf8_lossy(&o.stderr);
                        println!(
                            "uv pip install failed (status {:?}): {}",
                            o.status.code(),
                            stderr.trim()
                        );
                    }
                    Err(e) => println!("Could not invoke uv: {e}"),
                }
                return;
            }
            if matches!(ans.as_str(), "no" | "n") {
                println!(
                    "Skipping. The next execution_run will trigger automatic install when the \
venv is touched."
                );
                return;
            }
            println!("Please answer yes or no:");
            let _ = io::stdout().flush();
        }
    })
    .await;
    if let Err(e) = prompt_result {
        log::warn!("uv_requirements install prompt task failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn host_config_defaults_to_the_existing_workspace_resolution() {
        let config = HostConfig::default();
        assert!(config.workspace.is_none());
        assert!(config.config.is_none());
        assert!(config.sandbox.is_none());
        assert!(config.model.is_none());
        assert!(config.fallback_model.is_none());
        assert!(config.permission.is_none());
        assert!(!config.no_color);
        assert_eq!(config.theme, HostThemeMode::Auto);
        assert!(config.resume.is_none());
        assert!(config.files.is_empty());
        assert!(!config.line_mode);
        assert!(config.oneshot_prompt.is_none());
        assert!(config.observe_tx.is_none());
        assert!(config.scripted_responses.is_none());
    }

    #[test]
    fn host_model_override_selects_provider_and_model() {
        let mut config = AppConfig::default();
        apply_host_overrides(
            &mut config,
            &HostConfig {
                model: Some("anthropic/claude-test".to_string()),
                ..HostConfig::default()
            },
        )
        .expect("valid model override");

        let provider = config.provider.expect("provider override");
        assert_eq!(provider.provider_name, "anthropic");
        assert_eq!(provider.model_name, "claude-test");
        assert!(provider.api_key_env.is_empty());
    }

    #[test]
    fn host_permission_modes_map_to_shell_and_edit_policy() {
        for (permission, mode, edit_mode) in [
            (HostPermissionMode::Ask, "ask", "ask"),
            (HostPermissionMode::AutoEdit, "ask", "allow"),
            (HostPermissionMode::Plan, "ask", "deny"),
            (HostPermissionMode::Bypass, "allow", "allow"),
        ] {
            let mut config = AppConfig::default();
            apply_host_overrides(
                &mut config,
                &HostConfig {
                    permission: Some(permission),
                    ..HostConfig::default()
                },
            )
            .expect("valid permission override");

            let policy = config
                .harness
                .and_then(|harness| harness.shell_policy)
                .expect("shell policy override");
            assert_eq!(policy.mode.as_deref(), Some(mode));
            assert_eq!(policy.edit_mode.as_deref(), Some(edit_mode));
        }
    }

    #[test]
    fn host_fallback_override_registers_a_failover_provider() {
        let mut config = AppConfig::default();
        apply_host_overrides(
            &mut config,
            &HostConfig {
                fallback_model: Some("openai/gpt-test".to_string()),
                ..HostConfig::default()
            },
        )
        .expect("valid fallback override");
        let fallback = config
            .providers
            .and_then(|providers| providers.get("__altai_cli_fallback").cloned())
            .expect("fallback provider");
        assert_eq!(fallback.provider_name, "openai");
        assert_eq!(fallback.model_name, "gpt-test");
    }

    #[tokio::test]
    async fn spawned_host_accepts_an_external_graceful_shutdown() {
        let temp = tempfile::tempdir().expect("temporary workspace");
        let config_path = temp.path().join("config.toml");
        fs::write(
            &config_path,
            "[terminal]\nenabled = false\n\n[api]\nenabled = true\nport = 0\n",
        )
        .expect("headless config");

        let mut host = spawn_host(HostConfig {
            workspace: Some(temp.path().to_path_buf()),
            config: Some(config_path),
            sandbox: None,
            model: None,
            fallback_model: None,
            permission: None,
            no_color: false,
            theme: HostThemeMode::Auto,
            resume: None,
            files: Vec::new(),
            line_mode: false,
            compact_auto: None,
            compact_threshold_tokens: None,
            compact_tail_turns: None,
            oneshot_prompt: None,
            observe_tx: None,
            scripted_responses: None,
        });
        assert_eq!(host.next_event().await, Some(HostEvent::Starting));
        assert!(host.shutdown());

        let stopped = tokio::time::timeout(Duration::from_secs(10), host.next_event())
            .await
            .expect("host should stop promptly");
        assert_eq!(
            stopped,
            Some(HostEvent::Stopped {
                outcome: HostExit::Completed
            })
        );
        tokio::time::timeout(Duration::from_secs(10), host.wait())
            .await
            .expect("host task should join")
            .expect("host should complete cleanly");
    }

    #[tokio::test]
    async fn run_oneshot_completes_with_scripted_provider() {
        let temp = tempfile::tempdir().expect("temporary workspace");
        let config_path = temp.path().join("config.toml");
        fs::write(
            &config_path,
            "[terminal]\nenabled = false\n\n[logging]\nenabled = false\n",
        )
        .expect("write config");

        let result = tokio::time::timeout(
            Duration::from_secs(30),
            run_oneshot(HostConfig {
                workspace: Some(temp.path().to_path_buf()),
                config: Some(config_path),
                sandbox: Some(temp.path().to_path_buf()),
                model: None,
                fallback_model: None,
                permission: Some(HostPermissionMode::Plan),
                no_color: true,
                theme: HostThemeMode::NoColor,
                resume: None,
                files: Vec::new(),
                line_mode: false,
                compact_auto: None,
                compact_threshold_tokens: None,
                compact_tail_turns: None,
                oneshot_prompt: Some("reply with oneshot-ok".to_string()),
                observe_tx: None,
                scripted_responses: Some(vec!["oneshot-ok".to_string()]),
            }),
        )
        .await
        .expect("oneshot should finish promptly")
        .expect("oneshot should succeed");

        assert_eq!(result.outcome, OneshotOutcome::Completed);
        assert_eq!(result.final_text.as_deref(), Some("oneshot-ok"));
        assert!(!result.chat_id.is_empty());
    }
}
