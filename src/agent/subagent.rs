//! Sub-agent task harness (Phase 5): background reasoning loops keyed by parent chat.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

use super::{ActiveProviderConfig, ReasoningLoopCtx, RunProviderContext};
use crate::bus::{BusMessage, InboundMessage, OutboundMessage, TelemetryEvent};
use crate::channels::terminal_ui::protocol::{
    ISANAGENT_SUBAGENT_TASK_FINISHED, ISANAGENT_SUBAGENT_TASK_STARTED,
    METADATA_SUBAGENT_AGENT_NAME, METADATA_SUBAGENT_CHILD_CHAT_ID, METADATA_SUBAGENT_DISPLAY_NAME,
    METADATA_SUBAGENT_STATUS, METADATA_SUBAGENT_TASK_ID,
};
use crate::clarification::ClarificationHub;
use crate::config::{AgentMode, ResolvedShellPolicy};
use crate::logging::LoggerHandle;
use crate::memory::{MemoryMessage, SharedReply};
use crate::session::SessionManager;
use crate::skills::SharedSkillRegistry;
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tool_runtime::ToolExecCtx;
use crate::tools::ToolRegistry;
use crate::traits::Tool;
use crate::NodeHandle;
use tokio::sync::oneshot;

/// Arguments for [`SubagentHarness::spawn`].
pub struct SubagentSpawnSpec {
    pub parent_channel: String,
    pub parent_chat_id: String,
    pub parent_thread_id: Option<String>,
    pub parent_reasoning_cancel: Option<CancellationToken>,
    pub prompt: String,
    pub wait: bool,
    pub display_name: Option<String>,
    /// Named agent to invoke (e.g. "researcher", "coder"). Uses per-agent prompt/tools/model.
    pub agent_name: Option<String>,
    pub background_job_id: Option<String>,
}

/// Shared wiring for each spawned sub-agent run.
pub struct SubagentSpawnDeps {
    pub agent_name: String,
    /// Atomic provider + credentials pair, updated together by `/model`.
    pub(crate) provider_config: Arc<tokio::sync::RwLock<ActiveProviderConfig>>,
    /// Immutable candidate list; the selected primary is filtered from each run snapshot.
    pub(crate) fallback_candidates: Arc<Vec<super::FallbackProviderSpec>>,
    pub session_manager: Arc<SessionManager>,
    pub skills: SharedSkillRegistry,
    pub system_prompt: String,
    pub max_iterations: usize,
    pub max_tool_output_chars: usize,
    pub max_recent_summaries: usize,
    pub short_term_threshold_turns: usize,
    pub short_term_threshold_tokens: usize,
    pub tool_execution_activity: Option<SharedToolExecutionActivity>,
    pub outbound_tx: tokio::sync::mpsc::Sender<BusMessage>,
    pub logger_tx: LoggerHandle,
    pub clarification_hub: Arc<ClarificationHub>,
    pub cancel_children_on_parent_cancel: bool,
    pub default_allowlist: Option<Arc<HashSet<String>>>,
    pub max_tasks: usize,
    pub max_wait_secs: u64,
    pub doom_loop_enabled: bool,
    pub memory_node: NodeHandle<MemoryMessage>,
    /// Same harness snapshot as the parent agent (sub-agents do not use autonomy-forbid).
    pub harness_runtime_summary: String,
    /// Shell policy inherited from the parent agent.
    pub shell_policy: std::sync::Arc<ResolvedShellPolicy>,
    /// Optional hooks (observation + steering), same as parent.
    pub hook_tool_ctx: Option<std::sync::Arc<crate::hooks::ToolCallHookContext>>,
    /// Optional agent registry for named-agent spawns (Phase 5b).
    pub agent_registry: Option<Arc<super::registry::AgentRegistry>>,
    /// When true, auto-enqueue a synthetic inbound when a subagent finishes.
    pub wake_on_completion: bool,
    /// Max completed tasks retained in SQLite per parent chat.
    pub task_history_retention: usize,
    /// Optional extra bus sender for enqueuing synthetic follow-up messages.
    pub bus_tx: Option<tokio::sync::mpsc::Sender<BusMessage>>,
    /// Project root used by deterministic local worker agents. The worker
    /// never accepts a root from an LLM tool argument.
    pub workspace_dir: PathBuf,
}

struct TaskRecord {
    parent_chat_id: String,
    child_chat_id: String,
    prompt: String,
    agent_name: Option<String>,
    cancel: CancellationToken,
    status: std::sync::atomic::AtomicU8,
    result: tokio::sync::RwLock<Option<String>>,
    error: tokio::sync::RwLock<Option<String>>,
    done: tokio::sync::Notify,
}

const ST_PENDING: u8 = 0;
const ST_RUNNING: u8 = 1;
const ST_COMPLETED: u8 = 2;
const ST_FAILED: u8 = 3;
const ST_CANCELLED: u8 = 4;

fn truncate_sqlite_field(s: String, max: usize) -> String {
    let mut t = s;
    crate::utils::truncate_utf8_safe(&mut t, max, "\n… [truncated for sqlite]");
    t
}

fn truncate_for_tui(s: &str, max: usize) -> String {
    let t = s.trim();
    let n = t.chars().count();
    if n <= max {
        return t.to_string();
    }
    let shortened: String = t.chars().take(max.saturating_sub(1)).collect();
    format!("{shortened}…")
}

async fn persist_subagent_start(
    memory: &NodeHandle<MemoryMessage>,
    task_id: String,
    parent_chat_id: String,
    child_chat_id: String,
    display_name: Option<String>,
    agent_name: Option<String>,
    prompt: String,
) -> Result<(), String> {
    let prompt = truncate_sqlite_field(prompt, 150_000);
    let (tx, rx) = oneshot::channel();
    memory
        .send_packet(MemoryMessage::InsertSubagentTask {
            task_id,
            parent_chat_id,
            child_chat_id,
            display_name,
            agent_name,
            prompt,
            reply: SharedReply::new(tx),
        })
        .await
        .map_err(|e| format!("memory: {}", e))?;
    rx.await.map_err(|_| "memory actor closed".to_string())?
}

async fn persist_subagent_end(
    memory: &NodeHandle<MemoryMessage>,
    task_id: String,
    parent_chat_id: String,
    status: String,
    result: Option<String>,
    error: Option<String>,
    execution_job_id: Option<String>,
) {
    let result = result.map(|s| truncate_sqlite_field(s, 400_000));
    let error = error.map(|s| truncate_sqlite_field(s, 50_000));
    let (tx, rx) = oneshot::channel();
    let _ = memory
        .send_packet(MemoryMessage::FinalizeSubagentTask {
            task_id,
            parent_chat_id,
            status,
            result,
            error,
            execution_job_id,
            reply: SharedReply::new(tx),
        })
        .await;
    let _ = rx.await;
}

impl TaskRecord {
    fn is_terminal(&self) -> bool {
        let s = self.status.load(std::sync::atomic::Ordering::Acquire);
        s >= ST_COMPLETED
    }

    fn snapshot_json(&self) -> String {
        let status = match self.status.load(std::sync::atomic::Ordering::Acquire) {
            ST_PENDING => "pending",
            ST_RUNNING => "running",
            ST_COMPLETED => "completed",
            ST_FAILED => "failed",
            ST_CANCELLED => "cancelled",
            _ => "unknown",
        };
        let result = self
            .result
            .try_read()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_default();
        let error = self
            .error
            .try_read()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_default();
        serde_json::json!({
            "child_chat_id": self.child_chat_id,
            "parent_chat_id": self.parent_chat_id,
            "agent_name": self.agent_name,
            "status": status,
            "prompt": self.prompt,
            "result": result,
            "error": error,
        })
        .to_string()
    }
}

struct Inner {
    deps: SubagentSpawnDeps,
    tasks: DashMap<String, Arc<TaskRecord>>,
    tools: std::sync::OnceLock<Arc<ToolRegistry>>,
}

/// Task registry + spawn helper; tools call into this type.
#[derive(Clone)]
pub struct SubagentHarness {
    inner: Arc<Inner>,
}

impl SubagentHarness {
    pub fn new(deps: SubagentSpawnDeps) -> Self {
        Self {
            inner: Arc::new(Inner {
                deps,
                tasks: DashMap::new(),
                tools: std::sync::OnceLock::new(),
            }),
        }
    }

    pub fn bind_tools(&self, tools: Arc<ToolRegistry>) -> Result<(), String> {
        self.inner
            .tools
            .set(tools)
            .map_err(|_| "subagent bind_tools: tools already bound".to_string())
    }

    pub fn cancel_children_on_parent_cancel(&self) -> bool {
        self.inner.deps.cancel_children_on_parent_cancel
    }

    pub fn cancel_children_for_parent(&self, parent_chat_id: &str) {
        for e in self.inner.tasks.iter() {
            if e.value().parent_chat_id == parent_chat_id {
                e.value().cancel.cancel();
            }
        }
    }

    fn tools(&self) -> Result<Arc<ToolRegistry>, String> {
        self.inner
            .tools
            .get()
            .cloned()
            .ok_or_else(|| "Subagent harness is not wired to tools yet".to_string())
    }

    /// Run Semble as a constrained local worker. The LLM controls only the
    /// query text; the search root, executable shape, cache location, and
    /// content policy are owned by the host application.
    async fn run_semble_scout(workspace_dir: &Path, query: &str) -> Result<String, String> {
        let query = query.trim();
        if query.is_empty() {
            return Err("Semble Scout needs a non-empty code search query.".to_string());
        }
        if !workspace_dir.is_dir() {
            return Err(format!(
                "Semble Scout workspace is unavailable: {}",
                workspace_dir.display()
            ));
        }

        let cache_dir = workspace_dir
            .join(".isanagent")
            .join("cache")
            .join("semble");
        tokio::fs::create_dir_all(&cache_dir)
            .await
            .map_err(|e| format!("Could not prepare Semble cache: {e}"))?;

        let mut command = tokio::process::Command::new("uvx");
        command
            .args([
                "--from",
                "semble[mcp]==0.5.1",
                "semble",
                "search",
                "--content",
                "code",
                "--top-k",
                "8",
                "--max-snippet-lines",
                "16",
                "--",
                query,
            ])
            .arg(workspace_dir)
            .current_dir(workspace_dir)
            .env("SEMBLE_CACHE_LOCATION", &cache_dir)
            // A cancelled parent must not leave a package install or indexer
            // process running after its agent turn is gone.
            .kill_on_drop(true);

        // The first `uvx` run may need to download Semble and its model assets.
        // Do not impose an arbitrary timeout here: on a slow but healthy
        // network that would turn a one-time install into a false failure.
        // `kill_on_drop(true)` still cleans up the child if the runtime itself
        // drops this task during shutdown.
        let output = command.output().await.map_err(|e| {
            format!("Semble Scout could not start `uvx`: {e}. Install uv first, then retry.")
        })?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if !output.status.success() {
            let detail = if stderr.is_empty() { stdout } else { stderr };
            return Err(format!("Semble Scout search failed: {detail}"));
        }
        if stdout.is_empty() {
            return Ok("Semble Scout found no matching code snippets.".to_string());
        }
        Ok(format!(
            "Semble Scout results (local code search):\n\n{stdout}"
        ))
    }

    pub async fn spawn(&self, spec: SubagentSpawnSpec) -> Result<String, String> {
        let SubagentSpawnSpec {
            parent_channel,
            parent_chat_id,
            parent_thread_id,
            parent_reasoning_cancel,
            prompt,
            wait,
            display_name,
            agent_name,
            background_job_id,
        } = spec;

        if self.inner.tasks.len() >= self.inner.deps.max_tasks {
            return Err(format!(
                "Maximum number of sub-agent tasks ({}) reached",
                self.inner.deps.max_tasks
            ));
        }

        // Resolve named agent manifest if specified
        let manifest = match agent_name.as_deref() {
            Some(agent) => {
                let Some(registry) = self.inner.deps.agent_registry.as_ref() else {
                    return Err(
                        "Named agents are not configured. Use `agent_list` to inspect availability."
                            .to_string(),
                    );
                };
                let maybe = registry.get(agent).cloned();
                if maybe.is_none() {
                    return Err(format!(
                        "Named agent '{}' not found in registry. Use `agent_list` to see available agents.",
                        agent
                    ));
                }
                maybe
            }
            None => None,
        };

        // Semble Scout is deliberately a named agent rather than an MCP
        // integration: no LLM/provider is spawned for it. It receives the
        // coordinator's query, searches only the configured project root,
        // and returns the retrieved snippets directly to that coordinator.
        if manifest
            .as_ref()
            .is_some_and(|agent| agent.mode == AgentMode::SembleScout)
        {
            let result = Self::run_semble_scout(&self.inner.deps.workspace_dir, &prompt).await?;
            return Ok(serde_json::json!({
                "agent": "semble-scout",
                "status": "completed",
                "wait": true,
                "result": result,
            })
            .to_string());
        }

        let tools = self.tools()?;
        let task_id = uuid::Uuid::new_v4().simple().to_string();
        let child_chat_id = format!("subagent-{}", &task_id[..12.min(task_id.len())]);

        persist_subagent_start(
            &self.inner.deps.memory_node,
            task_id.clone(),
            parent_chat_id.clone(),
            child_chat_id.clone(),
            display_name.clone(),
            agent_name.clone(),
            prompt.clone(),
        )
        .await?;

        let task_cancel = match (
            &parent_reasoning_cancel,
            self.inner.deps.cancel_children_on_parent_cancel,
        ) {
            (Some(p), true) => p.child_token(),
            _ => CancellationToken::new(),
        };

        let record = Arc::new(TaskRecord {
            parent_chat_id: parent_chat_id.clone(),
            child_chat_id: child_chat_id.clone(),
            prompt: prompt.clone(),
            agent_name: agent_name.clone(),
            cancel: task_cancel.clone(),
            status: std::sync::atomic::AtomicU8::new(ST_PENDING),
            result: tokio::sync::RwLock::new(None),
            error: tokio::sync::RwLock::new(None),
            done: tokio::sync::Notify::new(),
        });

        self.inner.tasks.insert(task_id.clone(), record.clone());

        let _ = self
            .inner
            .deps
            .outbound_tx
            .send(BusMessage::Telemetry(TelemetryEvent::SubagentSpawned {
                parent_chat_id: parent_chat_id.clone(),
                child_chat_id: child_chat_id.clone(),
                task_id: task_id.clone(),
                display_name: display_name.clone(),
                agent_name: agent_name.clone(),
                background_job_id: background_job_id.clone(),
            }))
            .await;

        // Emit TUI notice for the agent-tasks strip
        {
            let mut meta = HashMap::new();
            meta.insert(
                ISANAGENT_SUBAGENT_TASK_STARTED.to_string(),
                serde_json::json!(true),
            );
            meta.insert(
                METADATA_SUBAGENT_TASK_ID.to_string(),
                serde_json::json!(task_id),
            );
            meta.insert(
                METADATA_SUBAGENT_CHILD_CHAT_ID.to_string(),
                serde_json::json!(&child_chat_id),
            );
            if let Some(ref a) = agent_name {
                meta.insert(
                    METADATA_SUBAGENT_AGENT_NAME.to_string(),
                    serde_json::json!(a),
                );
            }
            if let Some(ref d) = display_name {
                meta.insert(
                    METADATA_SUBAGENT_DISPLAY_NAME.to_string(),
                    serde_json::json!(d),
                );
            }
            let label = match (&agent_name, &display_name) {
                (Some(a), Some(d)) => format!("{a}: {d}"),
                (Some(a), None) => a.clone(),
                (None, Some(d)) => d.clone(),
                (None, None) => {
                    let short = &task_id[..8.min(task_id.len())];
                    format!("task-{short}")
                }
            };
            let _ = self
                .inner
                .deps
                .outbound_tx
                .send(BusMessage::Outbound(OutboundMessage {
                    channel: parent_channel.clone(),
                    chat_id: parent_chat_id.clone(),
                    thread_id: parent_thread_id.clone(),
                    content: format!("Sub-agent started: {label}"),
                    metadata: meta,
                }))
                .await;
        }

        // Build agent-specific system prompt and provider when manifest exists
        let agent_system_prompt = match &manifest {
            Some(m) => m.system_prompt.clone(),
            None => self.inner.deps.system_prompt.clone(),
        };
        let agent_allowlist: Option<Arc<HashSet<String>>> = match &manifest {
            Some(m) => match m.allowed_tools_set() {
                Some(v) if v.is_empty() => Some(Arc::new(HashSet::new())),
                Some(v) => Some(Arc::new(v.into_iter().collect())),
                None => self.inner.deps.default_allowlist.clone(),
            },
            None => self.inner.deps.default_allowlist.clone(),
        };
        let agent_max_iterations = manifest
            .as_ref()
            .and_then(|m| m.max_iterations)
            .unwrap_or(self.inner.deps.max_iterations);

        let active = self.inner.deps.provider_config.read().await;
        let mut run_credentials = active.credentials.clone();
        let provider = if let Some(ref m) = manifest {
            if m.model.is_some() || m.temperature.is_some() {
                if run_credentials.is_usable() {
                    if let Some(model) = &m.model {
                        run_credentials.model_name = model.clone();
                    }
                    crate::provider::provider_for_agent(
                        &run_credentials,
                        m.model.as_deref(),
                        m.temperature.map(|t| t as f32),
                    )
                } else {
                    dyn_clone::clone_box(&*active.provider)
                }
            } else {
                dyn_clone::clone_box(&*active.provider)
            }
        } else {
            dyn_clone::clone_box(&*active.provider)
        };
        drop(active);
        let run_active = ActiveProviderConfig {
            provider,
            credentials: run_credentials,
        };
        let run_provider =
            RunProviderContext::snapshot(&run_active, &self.inner.deps.fallback_candidates);

        let label = match (&agent_name, &display_name) {
            (Some(a), Some(d)) => format!("{a}: {d}"),
            (Some(a), None) => a.clone(),
            (None, Some(d)) => d.clone(),
            (None, None) => {
                let short = &task_id[..8.min(task_id.len())];
                format!("task-{short}")
            }
        };
        let inbound = InboundMessage {
            channel: parent_channel.clone(),
            sender_id: parent_chat_id.clone(),
            chat_id: child_chat_id.clone(),
            thread_id: parent_thread_id.clone(),
            content: format!("[Sub-agent: {}]\n\n{}", label, prompt),
            attachments: vec![],
            metadata: HashMap::new(),
        };
        let inbound_metadata = std::sync::Arc::new(inbound.metadata.clone());

        let tool_exec_ctx = ToolExecCtx::new(
            parent_channel.clone(),
            child_chat_id.clone(),
            parent_thread_id.clone(),
        )
        .with_reasoning_cancel(task_cancel.clone());

        let ctx = ReasoningLoopCtx {
            name: self.inner.deps.agent_name.clone(),
            run_provider,
            session_manager: self.inner.deps.session_manager.clone(),
            tools,
            skills: self.inner.deps.skills.clone(),
            system_prompt: agent_system_prompt,
            max_iterations: agent_max_iterations,
            max_tool_output_chars: self.inner.deps.max_tool_output_chars,
            max_recent_summaries: self.inner.deps.max_recent_summaries,
            short_term_threshold_turns: self.inner.deps.short_term_threshold_turns,
            short_term_threshold_tokens: self.inner.deps.short_term_threshold_tokens,
            tool_execution_activity: self.inner.deps.tool_execution_activity.clone(),
            outbound_tx: self.inner.deps.outbound_tx.clone(),
            logger_tx: self.inner.deps.logger_tx.clone(),
            inbound,
            run_id: format!("subagent-{task_id}"),
            steering: std::sync::Arc::new(std::sync::Mutex::new(super::SteeringInbox::open())),
            cancel_token: task_cancel.clone(),
            clarification_hub: self.inner.deps.clarification_hub.clone(),
            tool_exec_ctx,
            is_subagent: true,
            subagent_allowlist: agent_allowlist,
            doom_loop_enabled: self.inner.deps.doom_loop_enabled,
            harness_runtime_summary: self.inner.deps.harness_runtime_summary.clone(),
            forbid_final_without_tools: false,
            shell_policy: self.inner.deps.shell_policy.clone(),
            hook_tool_ctx: self.inner.deps.hook_tool_ctx.clone(),
            inbound_metadata,
        };

        record
            .status
            .store(ST_RUNNING, std::sync::atomic::Ordering::Release);

        let tasks = self.inner.tasks.clone();
        let tid = task_id.clone();
        let rec = record.clone();
        let memory = self.inner.deps.memory_node.clone();
        let outbound = self.inner.deps.outbound_tx.clone();
        let parent_for_db = parent_chat_id.clone();
        let wake_on_completion = self.inner.deps.wake_on_completion;
        let bus_tx = self.inner.deps.bus_tx.clone();
        let parent_channel_for_wake = parent_channel.clone();
        let parent_thread_for_wake = parent_thread_id.clone();
        let display_name_for_finish = display_name.clone();
        tokio::spawn(async move {
            let outcome = super::AgentLogic::run_reasoning_loop(ctx).await;
            let (status_str, result_opt, err_opt) = match outcome {
                Ok(super::ReasoningLoopExit::Cancelled { .. }) => {
                    rec.status
                        .store(ST_CANCELLED, std::sync::atomic::Ordering::Release);
                    ("cancelled".to_string(), None, Some("cancelled".to_string()))
                }
                Ok(super::ReasoningLoopExit::Failed { assistant_text, .. }) => {
                    *rec.error.write().await = Some(assistant_text.clone());
                    rec.status
                        .store(ST_FAILED, std::sync::atomic::Ordering::Release);
                    ("failed".to_string(), None, Some(assistant_text))
                }
                Ok(super::ReasoningLoopExit::WaitingForUser { ticket_id }) => {
                    // Keep the established parent-agent handoff contract. A subagent
                    // caller receives this result and uses the prefix to surface the
                    // outstanding clarification instead of treating it as an empty
                    // successful completion.
                    let text = format!("{}{}", super::WAITING_FOR_USER_RESULT_PREFIX, ticket_id);
                    {
                        let mut r = rec.result.write().await;
                        *r = Some(text.clone());
                    }
                    rec.status
                        .store(ST_COMPLETED, std::sync::atomic::Ordering::Release);
                    ("completed".to_string(), Some(text), None)
                }
                Ok(exit) => {
                    let text = exit.assistant_text().unwrap_or_default().to_string();
                    {
                        let mut r = rec.result.write().await;
                        *r = Some(text.clone());
                    }
                    rec.status
                        .store(ST_COMPLETED, std::sync::atomic::Ordering::Release);
                    ("completed".to_string(), Some(text), None)
                }
                Err(e) => {
                    *rec.error.write().await = Some(e.message.clone());
                    rec.status
                        .store(ST_FAILED, std::sync::atomic::Ordering::Release);
                    ("failed".to_string(), None, Some(e.message))
                }
            };

            let result_for_wake = result_opt.clone();
            let err_for_tui = err_opt.clone();
            persist_subagent_end(
                &memory,
                tid.clone(),
                parent_for_db.clone(),
                status_str.clone(),
                result_opt,
                err_opt,
                None,
            )
            .await;

            let _ = outbound
                .send(BusMessage::Telemetry(TelemetryEvent::SubagentFinished {
                    parent_chat_id: parent_for_db.clone(),
                    child_chat_id: rec.child_chat_id.clone(),
                    task_id: tid.clone(),
                    status: status_str.clone(),
                    agent_name: rec.agent_name.clone(),
                }))
                .await;

            // Emit TUI notice for the agent-tasks strip
            {
                let mut meta = HashMap::new();
                meta.insert(
                    ISANAGENT_SUBAGENT_TASK_FINISHED.to_string(),
                    serde_json::json!(true),
                );
                meta.insert(
                    METADATA_SUBAGENT_TASK_ID.to_string(),
                    serde_json::json!(tid),
                );
                meta.insert(
                    METADATA_SUBAGENT_CHILD_CHAT_ID.to_string(),
                    serde_json::json!(&rec.child_chat_id),
                );
                meta.insert(
                    METADATA_SUBAGENT_STATUS.to_string(),
                    serde_json::json!(status_str),
                );
                if let Some(ref a) = rec.agent_name {
                    meta.insert(
                        METADATA_SUBAGENT_AGENT_NAME.to_string(),
                        serde_json::json!(a),
                    );
                }
                let label = match (&rec.agent_name, &display_name_for_finish) {
                    (Some(a), Some(d)) => format!("{a}: {d}"),
                    (Some(a), None) => a.clone(),
                    (None, Some(d)) => d.clone(),
                    (None, None) => {
                        let short = &tid[..8.min(tid.len())];
                        format!("task-{short}")
                    }
                };
                let summary = result_for_wake
                    .as_deref()
                    .unwrap_or(err_for_tui.as_deref().unwrap_or(""))
                    .trim()
                    .to_string();
                let content = if summary.is_empty() {
                    format!("Sub-agent finished ({status_str}): {label}")
                } else {
                    format!(
                        "Sub-agent finished ({status_str}): {label} — {}",
                        truncate_for_tui(&summary, 120)
                    )
                };
                let _ = outbound
                    .send(BusMessage::Outbound(OutboundMessage {
                        channel: parent_channel_for_wake.clone(),
                        chat_id: parent_for_db.clone(),
                        thread_id: parent_thread_for_wake.clone(),
                        content,
                        metadata: meta,
                    }))
                    .await;
            }

            // Wake-on-completion: enqueue a synthetic inbound so the parent agent sees the result
            if wake_on_completion {
                if let Some(ref bus_tx) = bus_tx {
                    let synthetic_content = if let Some(ref name) = rec.agent_name {
                        format!(
                            "[Sub-agent {} task {} completed — {}]\n\n{}",
                            name,
                            &tid[..8.min(tid.len())],
                            status_str,
                            result_for_wake.as_deref().unwrap_or("(no output)")
                        )
                    } else {
                        format!(
                            "[Sub-agent task {} completed — {}]\n\n{}",
                            &tid[..8.min(tid.len())],
                            status_str,
                            result_for_wake.as_deref().unwrap_or("(no output)")
                        )
                    };
                    let mut meta = HashMap::new();
                    meta.insert(
                        crate::bus::METADATA_SYNTHETIC_SUBAGENT_COMPLETION.to_string(),
                        serde_json::Value::Bool(true),
                    );
                    let _ = bus_tx
                        .send(BusMessage::Inbound(crate::bus::InboundMessage {
                            channel: parent_channel_for_wake.clone(),
                            sender_id: "subagent-harness".to_string(),
                            chat_id: parent_for_db,
                            thread_id: parent_thread_for_wake.clone(),
                            content: synthetic_content,
                            attachments: vec![],
                            metadata: meta,
                        }))
                        .await;
                }
            }

            // Drop from the index before waking `wait=true` callers so `task_list` does not
            // briefly show a finished task after a blocking spawn returns.
            tasks.remove(&tid);
            rec.done.notify_waiters();
        });

        if wait {
            let max_wait = self.inner.deps.max_wait_secs;
            let wait_fut = async {
                loop {
                    let notified = record.done.notified();
                    if record.is_terminal() {
                        break;
                    }
                    notified.await;
                }
            };
            tokio::time::timeout(std::time::Duration::from_secs(max_wait), wait_fut)
                .await
                .map_err(|_| {
                    record.cancel.cancel();
                    format!("subagent_spawn timed out after {}s (wait=true)", max_wait)
                })?;
        }

        let mut body = serde_json::json!({
            "task_id": task_id,
            "child_chat_id": child_chat_id,
            "wait": wait,
        });
        if wait {
            let result_text = record.result.read().await.clone().unwrap_or_default();
            body["result"] = serde_json::Value::String(result_text);
        }
        Ok(body.to_string())
    }

    pub fn list_for_parent(&self, parent_chat_id: &str) -> String {
        let mut rows: Vec<String> = Vec::new();
        for e in self.inner.tasks.iter() {
            if e.value().parent_chat_id == parent_chat_id {
                rows.push(e.value().snapshot_json());
            }
        }
        if rows.is_empty() {
            return "No active sub-agent tasks for this chat.".to_string();
        }
        rows.join("\n")
    }

    fn find_task(&self, task_id: &str, parent_chat_id: &str) -> Result<Arc<TaskRecord>, String> {
        let t = self
            .inner
            .tasks
            .get(task_id)
            .ok_or_else(|| format!("Task '{}' not found", task_id))?;
        if t.parent_chat_id != parent_chat_id {
            return Err("Task is not owned by the current chat".to_string());
        }
        Ok(Arc::clone(t.value()))
    }

    pub fn get_task(&self, task_id: &str, parent_chat_id: &str) -> Result<String, String> {
        let t = self.find_task(task_id, parent_chat_id)?;
        Ok(t.snapshot_json())
    }

    pub fn cancel_task(&self, task_id: &str, parent_chat_id: &str) -> Result<String, String> {
        let t = self.find_task(task_id, parent_chat_id)?;
        t.cancel.cancel();
        Ok(format!("Cancellation requested for task {}", task_id))
    }
}

struct ParentContext {
    channel: String,
    chat_id: String,
    thread_id: Option<String>,
    token: Option<CancellationToken>,
    background_job_id: Option<String>,
}

fn current_parent_ids() -> Result<ParentContext, String> {
    let ctx = crate::tool_runtime::current_tool_exec_ctx()
        .ok_or_else(|| "subagent tools require an active agent tool scope".to_string())?;
    let bg_id = ctx
        .inbound_metadata
        .get(crate::bus::METADATA_BACKGROUND_JOB_ID)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Ok(ParentContext {
        channel: ctx.channel,
        chat_id: ctx.chat_id,
        thread_id: ctx.thread_id,
        token: ctx.reasoning_cancel,
        background_job_id: bg_id,
    })
}

pub struct SubagentSpawnTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for SubagentSpawnTool {
    fn name(&self) -> &str {
        "subagent_spawn"
    }

    fn description(&self) -> &str {
        "Spawn a background sub-agent that runs a separate reasoning loop with its own chat id (prefixed subagent-). Use wait=false for fire-and-forget; wait=true blocks until completion or timeout and includes the assistant-facing final text in the JSON field \"result\". Optionally specify `agent` to use a named agent (researcher, coder, evaluator) with its own system prompt and tool allowlist. Sub-agents cannot spawn nested sub-agents."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": { "type": "string", "description": "User task for the sub-agent" },
                "wait": { "type": "boolean", "description": "If true, block until the run finishes or times out (see harness max_wait_secs)." },
                "name": { "type": "string", "description": "Optional short label for logs / context." },
                "agent": { "type": "string", "description": "Optional named agent to invoke (e.g. researcher, coder). Use agent_list to see available agents." }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or("Missing prompt")?
            .to_string();
        let wait = args.get("wait").and_then(|v| v.as_bool()).unwrap_or(false);
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let agent_name = args
            .get("agent")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        let p = current_parent_ids()?;
        self.harness
            .spawn(SubagentSpawnSpec {
                parent_channel: p.channel,
                parent_chat_id: p.chat_id,
                parent_thread_id: p.thread_id,
                parent_reasoning_cancel: p.token,
                prompt,
                wait,
                display_name: name,
                agent_name,
                background_job_id: p.background_job_id,
            })
            .await
    }
}

pub struct TaskListTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for TaskListTool {
    fn name(&self) -> &str {
        "task_list"
    }

    fn description(&self) -> &str {
        "List active sub-agent tasks spawned from the current chat."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn execute(&self, _: Value) -> Result<String, String> {
        let parent_chat = current_parent_ids()?.chat_id;
        Ok(self.harness.list_for_parent(&parent_chat))
    }
}

pub struct TaskGetTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for TaskGetTool {
    fn name(&self) -> &str {
        "task_get"
    }

    fn description(&self) -> &str {
        "Get status snapshot JSON for a sub-agent task still listed for this chat (in-memory only while running). For completed runs use `task_history_list` (SQLite). Use subagent_spawn with wait=true for immediate `result` text in the spawn response."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_id": { "type": "string", "description": "Task id returned by subagent_spawn" }
            },
            "required": ["task_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let id = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing task_id")?;
        let parent_chat = current_parent_ids()?.chat_id;
        self.harness.get_task(id, &parent_chat)
    }
}

pub struct TaskCancelTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for TaskCancelTool {
    fn name(&self) -> &str {
        "task_cancel"
    }

    fn description(&self) -> &str {
        "Cancel a running sub-agent task owned by the current chat."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_id": { "type": "string" }
            },
            "required": ["task_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let id = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing task_id")?;
        let parent_chat = current_parent_ids()?.chat_id;
        self.harness.cancel_task(id, &parent_chat)
    }
}

#[derive(Deserialize)]
struct PlanStep {
    id: String,
    #[serde(default)]
    depends_on: Vec<String>,
    prompt: String,
    /// Optional named agent for this step (e.g. `gpu_to_jax`).
    #[serde(default)]
    agent: Option<String>,
}

#[derive(Deserialize)]
struct PlanInput {
    steps: Vec<PlanStep>,
}

pub struct SubagentPlanTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for SubagentPlanTool {
    fn name(&self) -> &str {
        "subagent_plan_execute"
    }

    fn description(&self) -> &str {
        "Run a multi-step plan: JSON object {\"steps\":[{\"id\":\"1\",\"depends_on\":[],\"prompt\":\"...\",\"agent\":\"optional_agent_name\"}, ...]}. Steps run in dependency order (sequential). Optional per-step `agent` selects a named sub-agent. Recommended deep-research pattern: discovery -> deep read -> contradiction check -> synthesis. Each step sees prior steps' assistant-facing final output (subagent_spawn result text) in its prompt prefix."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "JSON string: { \"steps\": [ { \"id\", \"depends_on\" (array of ids), \"prompt\" } ] }"
                }
            },
            "required": ["plan"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let plan_str = args
            .get("plan")
            .and_then(|v| v.as_str())
            .ok_or("Missing plan (JSON string)")?;
        let plan: PlanInput =
            serde_json::from_str(plan_str).map_err(|e| format!("Invalid plan JSON: {}", e))?;
        if plan.steps.is_empty() {
            return Err("plan.steps is empty".to_string());
        }

        let mut deps: HashMap<String, Vec<String>> = HashMap::new();
        let mut prompts: HashMap<String, String> = HashMap::new();
        let mut agents: HashMap<String, Option<String>> = HashMap::new();
        for s in &plan.steps {
            if s.id.is_empty() || s.prompt.is_empty() {
                return Err("Each step needs non-empty id and prompt".to_string());
            }
            deps.insert(s.id.clone(), s.depends_on.clone());
            prompts.insert(s.id.clone(), s.prompt.clone());
            agents.insert(s.id.clone(), s.agent.clone());
        }

        let mut done: HashSet<String> = HashSet::new();
        let mut results: HashMap<String, String> = HashMap::new();
        let mut out = String::from("# Plan execution\n\n");

        while done.len() < plan.steps.len() {
            let mut ready: Vec<String> = Vec::new();
            for s in &plan.steps {
                if done.contains(&s.id) {
                    continue;
                }
                let ok = deps[&s.id].iter().all(|d| done.contains(d));
                if ok {
                    ready.push(s.id.clone());
                }
            }
            if ready.is_empty() {
                return Err("Plan deadlock or missing dependency ids".to_string());
            }
            ready.sort();
            for step_id in ready {
                let mut body = String::new();
                for d in &deps[&step_id] {
                    if let Some(r) = results.get(d) {
                        body.push_str(&format!("## Prior step {} result:\n{}\n\n", d, r));
                    }
                }
                body.push_str(&prompts[&step_id]);
                let p = current_parent_ids()?;
                let label = format!("plan-{}", step_id);
                let spawn_json = self
                    .harness
                    .spawn(SubagentSpawnSpec {
                        parent_channel: p.channel,
                        parent_chat_id: p.chat_id,
                        parent_thread_id: p.thread_id,
                        parent_reasoning_cancel: p.token,
                        prompt: body,
                        wait: true,
                        display_name: Some(label),
                        agent_name: agents.get(&step_id).cloned().flatten(),
                        background_job_id: p.background_job_id,
                    })
                    .await?;
                let step_output = serde_json::from_str::<serde_json::Value>(&spawn_json)
                    .ok()
                    .and_then(|v| {
                        v.get("result")
                            .and_then(|r| r.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or(spawn_json.clone());
                results.insert(step_id.clone(), step_output.clone());
                out.push_str(&format!("## Step {}\n{}\n\n", step_id, step_output));
                done.insert(step_id);
            }
        }

        Ok(out)
    }
}

pub struct TaskHistoryListTool {
    pub memory_node: NodeHandle<MemoryMessage>,
}

#[async_trait]
impl Tool for TaskHistoryListTool {
    fn name(&self) -> &str {
        "task_history_list"
    }

    fn description(&self) -> &str {
        "List recent persisted sub-agent tasks for this chat (newest first, from SQLite). Use after parallel `subagent_spawn` runs complete to audit results. Optional `limit` (default 40, max 200)."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "limit": { "type": "integer", "description": "Max rows (default 40, max 200)" }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(40)
            .clamp(1, 200) as usize;
        let parent_chat = current_parent_ids()?.chat_id;
        let (tx, rx) = oneshot::channel();
        self.memory_node
            .send_packet(MemoryMessage::ListSubagentTasksForParent {
                parent_chat_id: parent_chat,
                limit,
                reply: SharedReply::new(tx),
            })
            .await
            .map_err(|e| format!("memory: {}", e))?;
        let rows = rx.await.map_err(|_| "memory actor closed".to_string())??;
        if rows.is_empty() {
            return Ok("No persisted sub-agent tasks for this chat yet.".to_string());
        }
        serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())
    }
}

pub struct AgentListTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for AgentListTool {
    fn name(&self) -> &str {
        "agent_list"
    }

    fn description(&self) -> &str {
        "List available named sub-agents and their capabilities (description, tools, model). Use to discover which agents are available for delegation via subagent_spawn or agent_spawn."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn execute(&self, _: Value) -> Result<String, String> {
        let registry = match &self.harness.inner.deps.agent_registry {
            Some(r) => r,
            None => return Ok("No named agents configured. Use default `subagent_spawn` without an `agent` parameter for a generic sub-agent, or configure `[agents.<name>]` in config.toml.".to_string()),
        };
        let agents = registry.list_visible();
        if agents.is_empty() {
            return Ok("No visible named agents configured. Use `subagent_spawn` without `agent` for a generic sub-agent.".to_string());
        }
        let mut lines = vec!["Available named agents:".to_string(), String::new()];
        for m in &agents {
            let tools = match &m.allowed_tools {
                None => "inherits harness allowlist".to_string(),
                Some(v) if v.is_empty() => "none (read-only)".to_string(),
                Some(v) => v.join(", "),
            };
            let model = m.model.as_deref().unwrap_or("(parent model)");
            let temp = m
                .temperature
                .map(|t| format!("{:.2}", t))
                .unwrap_or_else(|| "default".to_string());
            let iter = m
                .max_iterations
                .map(|n| n.to_string())
                .unwrap_or_else(|| "default".to_string());
            lines.push(format!(
                "- **{}**: {}\n  tools: [{}]\n  model: {}  temperature: {}  max_iterations: {}",
                m.name, m.description, tools, model, temp, iter
            ));
        }
        Ok(lines.join("\n"))
    }
}

pub struct TaskDashboardTool {
    pub harness: Arc<SubagentHarness>,
    pub memory_node: NodeHandle<MemoryMessage>,
}

#[async_trait]
impl Tool for TaskDashboardTool {
    fn name(&self) -> &str {
        "task_dashboard"
    }

    fn description(&self) -> &str {
        "Unified view of active and recently completed sub-agent tasks for this chat. Combines in-memory active tasks with SQLite history. Use after parallel subagent_spawn runs to see all statuses at once."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "limit": { "type": "integer", "description": "Max history rows (default 10, max 50)" }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(10)
            .clamp(1, 50) as usize;
        let parent_chat = current_parent_ids()?.chat_id;

        let mut out = String::from("# Task Dashboard\n\n## Active\n\n");
        let active = self.harness.list_for_parent(&parent_chat);
        if active == "No active sub-agent tasks for this chat." {
            out.push_str("(none)\n");
        } else {
            out.push_str(&active);
        }

        out.push_str("\n\n## Recent History\n\n");
        let (tx, rx) = oneshot::channel();
        self.memory_node
            .send_packet(MemoryMessage::ListSubagentTasksForParent {
                parent_chat_id: parent_chat,
                limit,
                reply: SharedReply::new(tx),
            })
            .await
            .map_err(|e| format!("memory: {}", e))?;
        let rows = rx.await.map_err(|_| "memory actor closed".to_string())??;
        if rows.is_empty() {
            out.push_str("(no history)\n");
        } else {
            out.push_str(&serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?);
        }
        Ok(out)
    }
}

pub fn register_subagent_tools(
    registry: &mut ToolRegistry,
    harness: Arc<SubagentHarness>,
    memory_node: NodeHandle<MemoryMessage>,
) {
    registry.register(Box::new(SubagentSpawnTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(TaskListTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(TaskGetTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(TaskCancelTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(TaskHistoryListTool {
        memory_node: memory_node.clone(),
    }));
    registry.register(Box::new(SubagentPlanTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(AgentListTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(TaskDashboardTool {
        harness: harness.clone(),
        memory_node: memory_node.clone(),
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::BusMessage;
    use crate::clarification::ClarificationHub;
    use crate::config::ResolvedShellPolicy;
    use crate::logging::create_logger_channel;
    use crate::memory::SqliteMemoryActor;
    use crate::session::SessionManager;
    use crate::skills::SkillRegistry;
    use crate::traits::Provider;
    use crate::utils::{ChatMessage, LLMError, LLMResponse};
    use crate::NodeHandle;
    use async_trait::async_trait;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;

    struct TempDir {
        path: std::path::PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let unique = format!(
                "isanagent-subagent-test-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos()
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[derive(Clone)]
    struct NeverUsedProvider;

    #[async_trait]
    impl Provider for NeverUsedProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            panic!("provider should not be called in this test")
        }
    }

    #[tokio::test]
    async fn spawn_with_named_agent_and_no_registry_returns_error_without_panic() {
        let memory_actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let memory_node = NodeHandle::new(memory_actor, 16, 1, Duration::from_millis(1));
        let session_manager = Arc::new(SessionManager::new(memory_node.clone()));
        let skills_dir = TempDir::new();
        let skills = Arc::new(tokio::sync::RwLock::new(SkillRegistry::new(
            skills_dir.path().to_path_buf(),
        )));
        let (outbound_tx, _outbound_rx) = mpsc::channel::<BusMessage>(8);
        let (logger_tx, _logger_rx) = create_logger_channel(16);
        let harness = Arc::new(SubagentHarness::new(SubagentSpawnDeps {
            agent_name: "SubagentTest".to_string(),
            provider_config: Arc::new(tokio::sync::RwLock::new(ActiveProviderConfig {
                provider: Box::new(NeverUsedProvider),
                credentials: crate::provider::ProviderCredentials::empty(),
            })),
            fallback_candidates: Arc::new(Vec::new()),
            session_manager,
            skills,
            system_prompt: "test system prompt".to_string(),
            max_iterations: 2,
            max_tool_output_chars: 4_000,
            max_recent_summaries: 0,
            short_term_threshold_turns: 10,
            short_term_threshold_tokens: 10_000,
            tool_execution_activity: None,
            outbound_tx,
            logger_tx,
            clarification_hub: Arc::new(ClarificationHub::new()),
            cancel_children_on_parent_cancel: true,
            default_allowlist: None,
            max_tasks: 5,
            max_wait_secs: 5,
            doom_loop_enabled: false,
            memory_node,
            harness_runtime_summary: String::new(),
            shell_policy: Arc::new(ResolvedShellPolicy {
                interactive_mode: crate::config::ShellPolicyMode::Ask,
                unattended_mode: crate::config::ShellPolicyMode::Deny,
                interactive_edit_mode: crate::config::ShellPolicyMode::Ask,
                unattended_edit_mode: crate::config::ShellPolicyMode::Deny,
                approval_patterns: Vec::new(),
            }),
            hook_tool_ctx: None,
            agent_registry: None,
            wake_on_completion: false,
            task_history_retention: 20,
            bus_tx: None,
            workspace_dir: skills_dir.path().to_path_buf(),
        }));

        let err = harness
            .spawn(SubagentSpawnSpec {
                parent_channel: "terminal".to_string(),
                parent_chat_id: "test-chat".to_string(),
                parent_thread_id: None,
                parent_reasoning_cancel: None,
                prompt: "do work".to_string(),
                wait: false,
                display_name: None,
                agent_name: Some("researcher".to_string()),
                background_job_id: None,
            })
            .await
            .expect_err("named agent without registry should fail");
        assert!(
            err.contains("Named agents are not configured"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn semble_scout_rejects_a_missing_workspace_before_spawning_processes() {
        let missing =
            std::env::temp_dir().join(format!("isanagent-absent-{}", uuid::Uuid::new_v4()));
        let err = SubagentHarness::run_semble_scout(&missing, "find agent startup")
            .await
            .expect_err("missing workspace must be rejected");
        assert!(err.contains("workspace is unavailable"));
    }
}
