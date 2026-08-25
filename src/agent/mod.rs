use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::mpsc;

mod budget;
pub mod compaction;
mod doom_loop;
pub mod registry;
mod subagent;
pub use registry::AgentRegistry;
pub use subagent::SubagentHarness;

// Audit X9: concerns extracted from this former god-file. Re-imports below
// keep every historical path under `crate::agent` resolving unchanged.
mod approval;
mod failover;
mod reasoning;
mod steering;
mod tool_dispatch;
pub use failover::{build_fallback_specs, FallbackProviderSpec};
pub(crate) use failover::{ActiveProviderConfig, RunProviderContext};
use reasoning::{
    ensure_run_id, estimate_context_tokens, metadata_truthy, spawn_main_chat_reasoning_turn,
    ReasoningLoopCtx, ReasoningLoopExit,
};
pub(crate) use steering::{steering_guard, SteeringInbox};

use crate::clarification::ClarificationHub;
use crate::hooks::ToolCallHookContext;

use crate::bus::{BusMessage, InboundMessage, LogEvent};
use crate::config::ResolvedShellPolicy;

use crate::logging::LoggerHandle;
use crate::memory::{MemoryMessage, SharedReply};
use crate::session::SessionManager;
use crate::skills::{SharedSkillRegistry, SkillRegistry};
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tools::ToolRegistry;
use crate::traits::{Memory, Provider, Tool, ToolPolicy};
use crate::{ActorError, ActorLogic};

static REDACTED_THINKING_STRIP_RE: OnceLock<Regex> = OnceLock::new();

pub const WAIT_SIGNAL_PREFIX: &str = "ISANAGENT_WAIT_FOR_USER:";
pub const WAITING_FOR_USER_RESULT_PREFIX: &str = "WAITING:";

/// Bundles everything needed to run one inbound reasoning task (spawned from `AgentLogic::process`).
/// Cloned into each spawned main-chat reasoning task (and used to chain queued inbounds).
#[derive(Clone)]
struct ReasoningSpawnArgs {
    name: String,
    session_manager: Arc<SessionManager>,
    tools: Arc<ToolRegistry>,
    skills: SharedSkillRegistry,
    system_prompt: String,
    max_iterations: usize,
    max_tool_output_chars: usize,
    max_recent_summaries: usize,
    short_term_threshold_turns: usize,
    short_term_threshold_tokens: usize,
    tool_execution_activity: Option<SharedToolExecutionActivity>,
    outbound_tx: mpsc::Sender<BusMessage>,
    logger_tx: LoggerHandle,
    clarification_hub: Arc<ClarificationHub>,
    doom_loop_enabled: bool,
    cancellation_tokens: Arc<dashmap::DashMap<String, ActiveRunHandle>>,
    pending_inbound: Arc<dashmap::DashMap<String, Mutex<VecDeque<QueuedInbound>>>>,
    harness_runtime_summary: String,
    forbid_final_without_tools: bool,
    shell_policy: Arc<ResolvedShellPolicy>,
    hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
}

#[derive(Clone)]
struct QueuedInbound {
    inbound: crate::bus::InboundMessage,
    run_provider: RunProviderContext,
}

#[derive(Clone)]
struct ActiveRunHandle {
    run_id: String,
    token: Arc<tokio_util::sync::CancellationToken>,
    steering: Arc<Mutex<SteeringInbox>>,
}

/// Constructor arguments for [`AgentLogic`], grouped to keep call sites readable.
//
// NOTE: `#[non_exhaustive]` is deliberately NOT applied here. Several fields (`provider`,
// `outbound_tx`, `logger_tx`, `clarification_hub`, …) cannot have a sensible `Default`,
// so the `..Default::default()` workaround that other Phase 0.0b structs use is not
// viable. Adding `#[non_exhaustive]` requires a builder API as a prerequisite — tracked
// as a follow-up to the Phase 0.0b sweep (see docs/public-api-surface.md §9.3).
pub struct AgentLogicParams {
    pub name: String,
    pub provider: Box<dyn Provider>,
    pub provider_credentials: crate::provider::ProviderCredentials,
    pub session_manager: SessionManager,
    pub tools: ToolRegistry,
    pub skills: SkillRegistry,
    pub system_prompt: String,
    pub max_iterations: usize,
    pub max_tool_output_chars: usize,
    pub max_recent_summaries: usize,
    pub short_term_threshold_turns: usize,
    pub short_term_threshold_tokens: usize,
    pub outbound_tx: mpsc::Sender<BusMessage>,
    pub logger_tx: LoggerHandle,
    pub clarification_hub: Arc<ClarificationHub>,
    /// When set, registers `subagent_*` / `task_*` tools and wires [`SubagentHarness`].
    pub subagent: Option<SubagentHarnessParams>,
    /// When true (default), inject corrective user text if repeated tool calls are detected.
    pub doom_loop_enabled: bool,
    /// Pre-formatted harness lines for system context (execution caps, subagent flags, etc.).
    pub harness_runtime_summary: String,
    /// System prompt used for `subagent_spawn` / plan runs (may include research appendix).
    pub subagent_system_prompt: String,
    /// Config default; inbound metadata `crate::bus::METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS` can override.
    pub forbid_final_without_tools: bool,
    /// Shell command safety policy (`exec`) resolved from config.
    pub shell_policy: ResolvedShellPolicy,
    /// Optional observation + steering hooks (`[harness.hooks]`).
    pub hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
}

/// Build-time options for the Phase 5 sub-agent harness (see `[harness.subagents]`).
//
// NOTE: `#[non_exhaustive]` is deferred — `src/main.rs` (a separate Cargo crate from the lib)
// constructs this struct directly when wiring the sub-agent harness. Adopting the marker
// requires either a builder or a constructor helper; tracked as a Phase 0.0b follow-up
// (see docs/public-api-surface.md §9.3).
#[derive(Clone, Debug)]
pub struct SubagentHarnessParams {
    pub cancel_children_on_parent_cancel: bool,
    pub allowed_tools: Option<Arc<HashSet<String>>>,
    pub max_tasks: usize,
    pub max_wait_secs: u64,
    pub agent_registry: Option<Arc<AgentRegistry>>,
    pub wake_on_completion: bool,
    pub task_history_retention: usize,
    pub bus_tx: Option<tokio::sync::mpsc::Sender<crate::bus::BusMessage>>,
    /// Canonical project root available to deterministic local worker agents.
    /// This is intentionally separate from IsanAgent's state directory.
    pub workspace_dir: std::path::PathBuf,
}

/// The central logic for an autonomous Agent running inside an ActorNode.
/// It holds a LLM Provider, a persistent Memory context, and available Tools.
pub struct AgentLogic {
    name: String,
    provider_config: Arc<tokio::sync::RwLock<ActiveProviderConfig>>,
    fallback_candidates: Arc<Vec<FallbackProviderSpec>>,
    session_manager: Arc<SessionManager>,
    tools: Arc<ToolRegistry>,
    skills: SharedSkillRegistry,
    system_prompt: String,
    max_iterations: usize,
    max_tool_output_chars: usize,
    max_recent_summaries: usize,
    short_term_threshold_turns: usize,
    short_term_threshold_tokens: usize,
    tool_execution_activity: Option<SharedToolExecutionActivity>,
    outbound_tx: mpsc::Sender<BusMessage>,
    logger_tx: LoggerHandle,
    cancellation_tokens: Arc<dashmap::DashMap<String, ActiveRunHandle>>,
    /// FIFO per `chat_id` when a new user inbound arrives while main reasoning is active.
    pending_inbound: Arc<dashmap::DashMap<String, Mutex<VecDeque<QueuedInbound>>>>,
    clarification_hub: Arc<ClarificationHub>,
    subagent_harness: Option<Arc<SubagentHarness>>,
    doom_loop_enabled: bool,
    harness_runtime_summary: String,
    forbid_final_without_tools: bool,
    shell_policy: Arc<ResolvedShellPolicy>,
    hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
}

impl AgentLogic {
    pub fn new(params: AgentLogicParams) -> Self {
        Self::new_with_fallback_providers(params, Vec::new())
    }

    /// Construct an agent with instance-owned failover candidates. The active primary is removed
    /// when each run snapshots its immutable provider context.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use isanagent::agent::{AgentLogic, AgentLogicParams, FallbackProviderSpec};
    /// use isanagent::provider::{create_provider, ProviderCredentials};
    ///
    /// async fn configure(
    ///     params: AgentLogicParams,
    ///     fallbacks: Vec<FallbackProviderSpec>,
    /// ) {
    ///     let agent = AgentLogic::new_with_fallback_providers(params, fallbacks);
    ///
    ///     let credentials = ProviderCredentials {
    ///         provider_name: "openai".to_string(),
    ///         base_url: "https://api.openai.com/v1".to_string(),
    ///         api_key: "replacement-key".to_string(),
    ///         model_name: "gpt-4o".to_string(),
    ///     };
    ///     let provider = create_provider(
    ///         &credentials.provider_name,
    ///         &credentials.base_url,
    ///         &credentials.api_key,
    ///         &credentials.model_name,
    ///     );
    ///     agent
    ///         .switch_provider_with_credentials(provider, credentials)
    ///         .await;
    /// }
    /// ```
    pub fn new_with_fallback_providers(
        params: AgentLogicParams,
        fallback_providers: Vec<FallbackProviderSpec>,
    ) -> Self {
        let AgentLogicParams {
            name,
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
            outbound_tx,
            logger_tx,
            clarification_hub,
            subagent,
            doom_loop_enabled,
            harness_runtime_summary,
            subagent_system_prompt,
            forbid_final_without_tools,
            shell_policy,
            hook_tool_ctx,
        } = params;

        let harness_for_subagent = harness_runtime_summary.clone();
        let session_manager = Arc::new(session_manager);
        let skills = Arc::new(tokio::sync::RwLock::new(skills));
        let tools = Arc::new(tools);
        let memory_node = session_manager.get_memory_node();
        let shell_policy = Arc::new(shell_policy);

        let provider_config = Arc::new(tokio::sync::RwLock::new(ActiveProviderConfig {
            provider,
            credentials: provider_credentials,
        }));
        let fallback_candidates = Arc::new(fallback_providers);

        let subagent_harness = subagent.map(|p| {
            Arc::new(SubagentHarness::new(subagent::SubagentSpawnDeps {
                agent_name: name.clone(),
                provider_config: provider_config.clone(),
                fallback_candidates: fallback_candidates.clone(),
                session_manager: session_manager.clone(),
                skills: skills.clone(),
                system_prompt: subagent_system_prompt,
                max_iterations,
                max_tool_output_chars,
                max_recent_summaries,
                short_term_threshold_turns,
                short_term_threshold_tokens,
                tool_execution_activity: None,
                outbound_tx: outbound_tx.clone(),
                logger_tx: logger_tx.clone(),
                clarification_hub: clarification_hub.clone(),
                cancel_children_on_parent_cancel: p.cancel_children_on_parent_cancel,
                default_allowlist: p.allowed_tools.clone(),
                max_tasks: p.max_tasks,
                max_wait_secs: p.max_wait_secs,
                doom_loop_enabled,
                memory_node: memory_node.clone(),
                harness_runtime_summary: harness_for_subagent.clone(),
                shell_policy: shell_policy.clone(),
                hook_tool_ctx: hook_tool_ctx.clone(),
                agent_registry: p.agent_registry.clone(),
                wake_on_completion: p.wake_on_completion,
                task_history_retention: p.task_history_retention,
                bus_tx: p.bus_tx.clone(),
                workspace_dir: p.workspace_dir.clone(),
            }))
        });

        let mut agent = Self {
            name,
            provider_config,
            fallback_candidates,
            session_manager,
            tools,
            skills,
            system_prompt,
            max_iterations,
            max_tool_output_chars,
            max_recent_summaries,
            short_term_threshold_turns,
            short_term_threshold_tokens,
            tool_execution_activity: None,
            outbound_tx,
            logger_tx,
            cancellation_tokens: Arc::new(dashmap::DashMap::new()),
            pending_inbound: Arc::new(dashmap::DashMap::new()),
            clarification_hub,
            subagent_harness: subagent_harness.clone(),
            doom_loop_enabled,
            harness_runtime_summary,
            forbid_final_without_tools,
            shell_policy: shell_policy.clone(),
            hook_tool_ctx,
        };

        let tools_mut = Arc::get_mut(&mut agent.tools)
            .expect("expected unique ownership of tools registry during initialization");
        if let Some(ref h) = subagent_harness {
            subagent::register_subagent_tools(tools_mut, h.clone(), memory_node);
        }
        let skill_reg = agent.skills.clone();
        let loader_tool = LoadSkillTool {
            registry: skill_reg,
        };
        tools_mut.register(Box::new(loader_tool));

        if let Some(ref h) = subagent_harness {
            h.bind_tools(agent.tools.clone())
                .expect("subagent bind_tools after unique registry init");
        }

        agent
    }

    /// Atomically replace the provider object and the credentials that created it. Active and
    /// already-queued runs keep their immutable snapshots; subsequent admissions see this pair.
    pub async fn switch_provider_with_credentials(
        &self,
        provider: Box<dyn Provider>,
        credentials: crate::provider::ProviderCredentials,
    ) {
        *self.provider_config.write().await = ActiveProviderConfig {
            provider,
            credentials,
        };
    }

    pub fn with_tool_execution_activity(
        mut self,
        tool_execution_activity: SharedToolExecutionActivity,
    ) -> Self {
        self.tool_execution_activity = Some(tool_execution_activity);
        self
    }

    fn reasoning_spawn_args(&self) -> ReasoningSpawnArgs {
        ReasoningSpawnArgs {
            name: self.name.clone(),
            session_manager: self.session_manager.clone(),
            tools: self.tools.clone(),
            skills: self.skills.clone(),
            system_prompt: self.system_prompt.clone(),
            max_iterations: self.max_iterations,
            max_tool_output_chars: self.max_tool_output_chars,
            max_recent_summaries: self.max_recent_summaries,
            short_term_threshold_turns: self.short_term_threshold_turns,
            short_term_threshold_tokens: self.short_term_threshold_tokens,
            tool_execution_activity: self.tool_execution_activity.clone(),
            outbound_tx: self.outbound_tx.clone(),
            logger_tx: self.logger_tx.clone(),
            clarification_hub: self.clarification_hub.clone(),
            doom_loop_enabled: self.doom_loop_enabled,
            cancellation_tokens: self.cancellation_tokens.clone(),
            pending_inbound: self.pending_inbound.clone(),
            harness_runtime_summary: self.harness_runtime_summary.clone(),
            forbid_final_without_tools: self.forbid_final_without_tools,
            shell_policy: self.shell_policy.clone(),
            hook_tool_ctx: self.hook_tool_ctx.clone(),
        }
    }

    async fn run_provider_context(&self) -> RunProviderContext {
        let active = self.provider_config.read().await;
        RunProviderContext::snapshot(&active, &self.fallback_candidates)
    }

    #[cfg(test)]
    async fn execute_tool_call(
        &self,
        chat_id: &str,
        tool_name: &str,
        args: Value,
    ) -> Result<String, String> {
        match crate::agent::tool_dispatch::execute_tool_call_with_activity(
            &self.tools,
            self.tool_execution_activity.clone(),
            chat_id,
            "test",
            &self.outbound_tx,
            tool_name,
            None,
            args,
            None,
            crate::agent::tool_dispatch::ToolCallRuntime {
                session: crate::tool_runtime::ToolExecCtx::new("test", chat_id, None),
                hub: self.clarification_hub.clone(),
                is_subagent: false,
                subagent_allowlist: None,
                shell_policy: self.shell_policy.clone(),
                unattended_session: false,
                hook_tool_ctx: None,
                inbound_metadata: Arc::new(std::collections::HashMap::new()),
            },
        )
        .await
        {
            crate::agent::tool_dispatch::ToolExecutionFinished::Completed(result) => {
                result.into_legacy_result()
            }
            crate::agent::tool_dispatch::ToolExecutionFinished::Waiting(ticket_id) => Err(format!(
                "tool call waiting for clarification ticket: {ticket_id}"
            )),
            crate::agent::tool_dispatch::ToolExecutionFinished::Cancelled => {
                Err("tool call cancelled without cancellation token".to_string())
            }
        }
    }

    fn cancel_active_run(&self, chat_id: &str, expected_run_id: Option<&str>) -> bool {
        let active = self
            .cancellation_tokens
            .get(chat_id)
            .map(|entry| entry.value().clone());
        let Some(active) = active else {
            if expected_run_id.is_none() {
                self.pending_inbound.remove(chat_id);
            }
            return false;
        };
        if expected_run_id.is_some_and(|run_id| run_id != active.run_id) {
            let _ = self.logger_tx.send(BusMessage::Log(
                LogEvent::warn(
                    &self.name,
                    &format!(
                        "Ignored cancellation for chat_id {chat_id} because run_id did not match the active run."
                    ),
                )
                .with_chat_id(chat_id),
            ));
            return false;
        }
        if let Some(harness) = &self.subagent_harness {
            if harness.cancel_children_on_parent_cancel() {
                harness.cancel_children_for_parent(chat_id);
            }
        }
        steering_guard(&active.steering).close();
        active.token.cancel();
        let _ = self.logger_tx.send(BusMessage::Log(
            LogEvent::info(
                &self.name,
                &format!("Cancelled reasoning loop for chat_id: {chat_id}"),
            )
            .with_chat_id(chat_id),
        ));
        self.pending_inbound.remove(chat_id);
        true
    }
}

/// The Agent processes incoming BusMessages, updates memory based on session key,
/// and outputs BusMessages (specifically Outbound) back to the channel.
#[async_trait]
impl ActorLogic<BusMessage> for AgentLogic {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn process(
        &mut self,
        packet: BusMessage,
    ) -> Result<Option<(String, BusMessage)>, ActorError> {
        match packet {
            BusMessage::Cancel(chat_id) => {
                // Keep ownership registered until the reasoning task emits its
                // terminal lifecycle event and finalizes. New inbound arriving
                // during cancellation must queue behind that acknowledgement.
                self.cancel_active_run(&chat_id, None);
                return Ok(None);
            }
            BusMessage::CancelRun { chat_id, run_id } => {
                self.cancel_active_run(&chat_id, Some(&run_id));
                return Ok(None);
            }
            BusMessage::Steer {
                chat_id,
                run_id,
                content,
            } => {
                if content.trim().is_empty() {
                    return Ok(None);
                }
                if let Some(active) = self.cancellation_tokens.get(&chat_id) {
                    if active.run_id == run_id {
                        steering_guard(&active.steering).push(content);
                    }
                }
                return Ok(None);
            }
            BusMessage::SwitchModel {
                provider_name,
                model_name,
                base_url,
                api_key,
            } => {
                let new_provider = crate::provider::create_provider(
                    &provider_name,
                    &base_url,
                    &api_key,
                    &model_name,
                );
                self.switch_provider_with_credentials(
                    new_provider,
                    crate::provider::ProviderCredentials {
                        provider_name: provider_name.clone(),
                        base_url: base_url.clone(),
                        api_key: api_key.clone(),
                        model_name: model_name.clone(),
                    },
                )
                .await;
                let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info(
                    &self.name,
                    &format!("Switched to provider={provider_name} model={model_name}"),
                )));
                return Ok(None);
            }
            BusMessage::InstallSkill {
                repo_url,
                skill_name,
            } => {
                let skills_arc = self.skills.clone();
                let logger_tx = self.logger_tx.clone();
                let name = self.name.clone();

                tokio::spawn(async move {
                    let mut skills_guard = skills_arc.write().await;
                    match skills_guard
                        .install_skills_from_repo(&repo_url, skill_name.as_deref())
                        .await
                    {
                        Ok(installed) => {
                            let msg = if installed.is_empty() {
                                "No skills found in the repository.".to_string()
                            } else {
                                format!("Successfully installed skills: {}", installed.join(", "))
                            };
                            let _ = logger_tx.send(BusMessage::Log(LogEvent::info(&name, &msg)));
                        }
                        Err(e) => {
                            let _ = logger_tx.send(BusMessage::Log(LogEvent::error(
                                &name,
                                &format!("Failed to install skills from {repo_url}: {e}"),
                            )));
                        }
                    }
                });
                return Ok(None);
            }
            BusMessage::Inbound(mut inbound) => {
                let run_id = match ensure_run_id(&mut inbound) {
                    Ok(run_id) => run_id,
                    Err(error) => {
                        let _ = self.logger_tx.send(BusMessage::Log(
                            LogEvent::error(
                                &self.name,
                                &format!("Rejecting inbound message: {error}"),
                            )
                            .with_chat_id(&inbound.chat_id),
                        ));
                        let notice = crate::protocol::build_channel_error_notice(
                            &inbound.channel,
                            &inbound.chat_id,
                            inbound.thread_id.as_deref(),
                            &error,
                        );
                        let _ = self.outbound_tx.send(BusMessage::Outbound(notice)).await;
                        return Ok(None);
                    }
                };
                let chat_id = inbound.chat_id.clone();
                let session_key = inbound.clarification_session_key();
                if self
                    .clarification_hub
                    .try_deliver_reply(&session_key, inbound.content.clone())
                {
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::debug(
                            &self.name,
                            "Inbound delivered as ask_user clarification reply (same session).",
                        )
                        .with_chat_id(&chat_id),
                    ));
                    return Ok(None);
                }

                // Check for background job resume via explicit clarification ticket UI interaction
                if let Some(res) = self
                    .try_resume_background_job_from_ticket(&inbound, &chat_id, &session_key)
                    .await
                {
                    return res;
                }

                let _ = self.logger_tx.send(BusMessage::Log(
                    LogEvent::info(
                        &self.name,
                        &format!(
                            "Received InboundMessage for chat_id [{}] ({} chars)",
                            chat_id,
                            inbound.content.len(),
                        ),
                    )
                    .with_chat_id(&chat_id),
                ));

                let run_provider = self.run_provider_context().await;
                if self.cancellation_tokens.contains_key(&chat_id) {
                    // Check if this is a synthetic cron trigger. If so, drop it if we're already busy.
                    if metadata_truthy(
                        &inbound.metadata,
                        crate::bus::METADATA_SYNTHETIC_CRON_TRIGGER,
                    ) {
                        let _ = self.logger_tx.send(BusMessage::Log(
                            LogEvent::info(
                                &self.name,
                                "Dropping synthetic cron trigger because chat is already active.",
                            )
                            .with_chat_id(&chat_id),
                        ));
                        return Ok(None);
                    }

                    let queue = self
                        .pending_inbound
                        .entry(chat_id.clone())
                        .or_insert_with(|| Mutex::new(VecDeque::new()));
                    let mut guard = match queue.lock() {
                        Ok(g) => g,
                        Err(poisoned) => {
                            let _ = self.logger_tx.send(BusMessage::Log(
                                LogEvent::warn(
                                    &self.name,
                                    "pending_inbound mutex poisoned; recovering queued inbound state.",
                                )
                                .with_chat_id(&chat_id),
                            ));
                            poisoned.into_inner()
                        }
                    };
                    guard.push_back(QueuedInbound {
                        inbound,
                        run_provider,
                    });
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::debug(
                            &self.name,
                            &format!(
                                "Queued inbound for chat_id {chat_id} (FIFO) — reasoning already active."
                            ),
                        )
                        .with_chat_id(&chat_id),
                    ));
                    return Ok(None);
                }

                // If not busy, check if there's a waiting background job for this chat
                // to automatically resume it (user replied to the thread instead of via ticket UI).
                if let Some(res) = self
                    .try_auto_resume_waiting_job(&inbound, &chat_id, &session_key)
                    .await
                {
                    return res;
                }

                spawn_main_chat_reasoning_turn(
                    self.reasoning_spawn_args(),
                    inbound,
                    run_id,
                    run_provider,
                );

                Ok(None)
            }
            BusMessage::TriggerCompaction {
                session_key,
                focus_instructions,
                trigger,
            } => {
                // PR-5 + PR-10: delegate to the internal `trigger_compaction_with_reason`
                // so the carried `trigger` (Manual vs AgentSelf) propagates into the
                // `CompactionTriggered` telemetry event. The per-chat FIFO guard
                // already lives inside that helper; failures are logged and dropped.
                let reason = trigger.unwrap_or(crate::bus::CompactionTrigger::Manual);
                if let Err(e) = self
                    .trigger_compaction_with_reason(session_key.clone(), focus_instructions, reason)
                    .await
                {
                    let _ = self.logger_tx.send(BusMessage::Log(LogEvent::warn(
                        &self.name,
                        &format!("TriggerCompaction dropped for session_key={session_key}: {e}"),
                    )));
                }
                Ok(None)
            }
            BusMessage::Outbound(_)
            | BusMessage::Telemetry(_)
            | BusMessage::RunLifecycle(_)
            | BusMessage::LoggerControl(_)
            | BusMessage::Log(_)
            | BusMessage::PromoteSyncToBackground(_)
            | BusMessage::SetTerminalSessionChat { .. }
            | BusMessage::StreamDelta { .. }
            | BusMessage::SessionProjection(_) => Ok(None),
        }
    }
}

impl AgentLogic {
    /// PR-5: manually trigger a compaction for `session_key` outside the normal
    /// threshold path. The compaction runs synchronously in the calling task and
    /// emits the full matched `CompactionTriggered { reason: Manual }` + (`Completed`
    /// or `Failed`) telemetry pair.
    ///
    /// Construct `session_key` via [`crate::bus::clarification_session_key`].
    /// `focus_instructions`, when present, is appended to the summarizer prompt
    /// as a `FOCUS:` block so the model can prioritize certain content.
    ///
    /// **Per-chat FIFO.** Returns `Err` if a reasoning turn is currently in flight
    /// for the same `chat_id` — the AGENTS.md invariant requires compaction to
    /// happen *between* turns, not during. Callers that arrive via the bus
    /// (`BusMessage::TriggerCompaction`) should expect drops in that case.
    pub async fn trigger_compaction(
        &self,
        session_key: String,
        focus_instructions: Option<String>,
    ) -> Result<crate::agent::compaction::CompactionOutcome, String> {
        self.trigger_compaction_with_reason(
            session_key,
            focus_instructions,
            crate::bus::CompactionTrigger::Manual,
        )
        .await
    }

    /// Internal entry point shared by the pub [`Self::trigger_compaction`] API
    /// and the PR-10 `compact_context` tool path (via `BusMessage::TriggerCompaction`
    /// with `trigger: Some(AgentSelf)`). Splits the public surface from the
    /// `CompactionTrigger` taxonomy so the eval pipeline can distinguish
    /// caller-driven (`Manual`) from agent-driven (`AgentSelf`) compactions.
    async fn trigger_compaction_with_reason(
        &self,
        session_key: String,
        focus_instructions: Option<String>,
        trigger_reason: crate::bus::CompactionTrigger,
    ) -> Result<crate::agent::compaction::CompactionOutcome, String> {
        // session_key format: `<channel>:<chat_id>:<thread_part>`. The chat_id
        // segment drives the in-flight guard and telemetry labelling.
        let chat_id = session_key
            .split(':')
            .nth(1)
            .ok_or_else(|| {
                format!("Malformed session_key (expected `channel:chat_id:thread`): {session_key}")
            })?
            .to_string();

        if self.cancellation_tokens.contains_key(&chat_id) {
            return Err(format!(
                "Refusing manual compaction: reasoning turn in flight for chat_id={chat_id}"
            ));
        }

        let mem = self
            .session_manager
            .get_session(&session_key)
            .await
            .map_err(|e| format!("get_session({session_key}): {e}"))?;
        let current_context = mem
            .get_context_since_reflection()
            .await
            .map_err(|e| format!("get_context_since_reflection({session_key}): {e}"))?;
        let user_turns = current_context.iter().filter(|m| m.role == "user").count();
        let approx_tokens: usize = estimate_context_tokens(&current_context);

        // Most recent summary keyed by the same channel:chat_id prefix
        // — same scheme used at the threshold-trigger site.
        let prefix = {
            let mut parts = session_key.splitn(3, ':');
            let channel = parts.next().unwrap_or("");
            format!("{channel}:{chat_id}")
        };
        let recent = self
            .session_manager
            .get_recent_summaries(&prefix, self.max_recent_summaries.max(1))
            .await
            .unwrap_or_default();

        let memory_node = self.session_manager.get_memory_node();
        let provider_guard = self.provider_config.read().await;
        // Manual triggers have no per-call cancellation; a token that never
        // fires keeps `do_compaction`'s `select!` valid without altering behavior.
        let cancel_token = tokio_util::sync::CancellationToken::new();

        let outcome =
            crate::agent::compaction::do_compaction(crate::agent::compaction::DoCompactionArgs {
                chat_id: &chat_id,
                session_key: &session_key,
                trigger_reason,
                tokens_before: approx_tokens.min(u32::MAX as usize) as u32,
                turns_before: user_turns.min(u32::MAX as usize) as u32,
                current_context: &current_context,
                existing_summary: recent.first().map(|s| s.as_str()),
                focus_instructions: focus_instructions.as_deref(),
                provider: provider_guard.provider.as_ref(),
                memory_node: &memory_node,
                outbound_tx: &self.outbound_tx,
                cancel_token: &cancel_token,
            })
            .await;
        Ok(outcome)
    }

    async fn try_resume_background_job_from_ticket(
        &mut self,
        inbound: &InboundMessage,
        chat_id: &str,
        session_key: &str,
    ) -> Option<Result<Option<(String, BusMessage)>, ActorError>> {
        if let Some(ticket_id) = inbound
            .metadata
            .get(crate::bus::METADATA_CLARIFICATION_TICKET_ID)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
        {
            let memory_node = self.session_manager.get_memory_node();
            let (tx, rx) = tokio::sync::oneshot::channel();
            let _ = memory_node
                .send_packet(MemoryMessage::GetClarificationTicket {
                    ticket_id: ticket_id.clone(),
                    reply: SharedReply::new(tx),
                })
                .await;

            if let Ok(Ok(Some(ticket))) = rx.await {
                let _ = self.logger_tx.send(BusMessage::Log(
                    LogEvent::info(
                        &self.name,
                        &format!(
                            "Resuming background job [{}] via clarification ticket [{}]",
                            ticket.job_id, ticket_id
                        ),
                    )
                    .with_chat_id(chat_id),
                ));

                if let Err(e) = self
                    .resolve_and_resume_job(
                        inbound,
                        &ticket.ticket_id,
                        &ticket.job_id,
                        ticket.tool_call_id.as_deref(),
                        session_key,
                    )
                    .await
                {
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::error(
                            &self.name,
                            &format!("Failed to resume job {}: {}", ticket.job_id, e),
                        )
                        .with_chat_id(chat_id),
                    ));
                    let notice = crate::protocol::build_channel_error_notice(
                        &inbound.channel,
                        chat_id,
                        inbound.thread_id.as_deref(),
                        &format!("Failed to resume background job [{}]: {}", ticket.job_id, e),
                    );
                    let _ = self.outbound_tx.try_send(BusMessage::Outbound(notice));
                    return Some(Ok(None));
                }
                return Some(Ok(None));
            }
        }
        None
    }

    async fn try_auto_resume_waiting_job(
        &mut self,
        inbound: &InboundMessage,
        chat_id: &str,
        session_key: &str,
    ) -> Option<Result<Option<(String, BusMessage)>, ActorError>> {
        if metadata_truthy(
            &inbound.metadata,
            crate::bus::METADATA_CLARIFICATION_TICKET_ID,
        ) {
            return None;
        }

        let memory_node = self.session_manager.get_memory_node();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = memory_node
            .send_packet(MemoryMessage::ListBackgroundJobs {
                chat_id: Some(chat_id.to_string()),
                // A background job may have been created through a different
                // channel for this same chat. Native recovery deliberately
                // remains chat-scoped; host embedders opt into channel scope
                // when their UI needs an isolated inbox.
                channel: None,
                limit: 10,
                reply: SharedReply::new(tx),
            })
            .await;

        if let Ok(Ok(jobs)) = rx.await {
            let waiting_jobs: Vec<_> = jobs.into_iter().filter(|j| j.state == "waiting").collect();
            for job in waiting_jobs {
                // Found a waiting job. Now find the latest ticket for it.
                let (tx2, rx2) = tokio::sync::oneshot::channel();
                let _ = memory_node
                    .send_packet(MemoryMessage::ListClarificationTickets {
                        job_id: Some(job.job_id.clone()),
                        chat_id: Some(chat_id.to_string()),
                        channel: None,
                        status: Some("waiting".to_string()),
                        limit: 1,
                        reply: SharedReply::new(tx2),
                    })
                    .await;

                if let Ok(Ok(tickets)) = rx2.await {
                    if let Some(ticket) = tickets.into_iter().next() {
                        let _ = self.logger_tx.send(BusMessage::Log(
                            LogEvent::info(
                                &self.name,
                                &format!(
                                    "Auto-resuming waiting background job [{}] via thread reply to ticket [{}]",
                                    job.job_id, ticket.ticket_id
                                ),
                            )
                            .with_chat_id(chat_id),
                        ));

                        if let Err(e) = self
                            .resolve_and_resume_job(
                                inbound,
                                &ticket.ticket_id,
                                &job.job_id,
                                ticket.tool_call_id.as_deref(),
                                session_key,
                            )
                            .await
                        {
                            let _ = self.logger_tx.send(BusMessage::Log(
                                LogEvent::error(
                                    &self.name,
                                    &format!("Failed to auto-resume job {}: {}", job.job_id, e),
                                )
                                .with_chat_id(chat_id),
                            ));
                            let notice = crate::protocol::build_channel_error_notice(
                                &inbound.channel,
                                chat_id,
                                inbound.thread_id.as_deref(),
                                &format!(
                                    "Failed to auto-resume background job [{}]: {}",
                                    job.job_id, e
                                ),
                            );
                            let _ = self.outbound_tx.try_send(BusMessage::Outbound(notice));
                            return Some(Ok(None));
                        }
                        return Some(Ok(None));
                    }
                }
            }
        }
        None
    }

    async fn resolve_and_resume_job(
        &mut self,
        inbound: &InboundMessage,
        ticket_id: &str,
        job_id: &str,
        tool_call_id: Option<&str>,
        session_key: &str,
    ) -> Result<(), String> {
        let memory_node = self.session_manager.get_memory_node();

        // 1. Resolve everything for this ticket in a single go
        let (tx, rx) = tokio::sync::oneshot::channel();
        memory_node
            .send_packet(MemoryMessage::ResolveClarificationTicketFull {
                ticket_id: ticket_id.to_string(),
                job_id: job_id.to_string(),
                response: inbound.content.clone(),
                reply: SharedReply::new(tx),
            })
            .await
            .map_err(|e| format!("Memory actor error: {e}"))?;

        rx.await
            .map_err(|_| "Memory actor channel closed".to_string())?
            .map_err(|e| format!("Memory node failed to resolve ticket fully: {e}"))?;

        // 2. Inject tool response into memory
        if let Some(id) = tool_call_id {
            if let Ok(mut mem) = self.session_manager.get_session(session_key).await {
                // Determine tool name from memory
                let mut tool_name_for_resume = None;
                if let Ok(context) = mem.get_context().await {
                    for msg in context.iter().rev() {
                        if msg.role == "assistant" {
                            if let Some(calls) = &msg.tool_calls {
                                if let Some(tc) = calls.iter().find(|c| c.id == id) {
                                    tool_name_for_resume = Some(tc.function.name.clone());
                                    break;
                                }
                            }
                        }
                    }
                }

                mem.add_message(crate::utils::ChatMessage::tool(
                    &inbound.content,
                    id,
                    tool_name_for_resume.as_deref(),
                ))
                .await
                .map_err(|e| format!("Failed to inject tool response into memory: {e}"))?;
            } else {
                return Err(format!("Failed to get session {session_key}"));
            }
        }

        // 3. Spawn turn with resume metadata
        let mut resumed_inbound = inbound.clone();
        resumed_inbound.metadata.insert(
            crate::bus::METADATA_SYNTHETIC_BACKGROUND_RESUME.to_string(),
            serde_json::json!(true),
        );
        resumed_inbound.metadata.insert(
            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
            serde_json::json!(job_id),
        );

        let run_id = ensure_run_id(&mut resumed_inbound)?;
        let run_provider = self.run_provider_context().await;
        spawn_main_chat_reasoning_turn(
            self.reasoning_spawn_args(),
            resumed_inbound,
            run_id,
            run_provider,
        );
        Ok(())
    }
}

/// A built-in tool that allows the agent to load the markdown instructions
/// for a skill dynamically from the SkillRegistry.
pub struct LoadSkillTool {
    registry: SharedSkillRegistry,
}

#[async_trait]
impl Tool for LoadSkillTool {
    fn name(&self) -> &str {
        "load_skill_instructions"
    }

    fn description(&self) -> &str {
        "Loads the full markdown instructions for a specific Agent Skill. Use this when you need to execute a skill."
    }

    fn policy(&self) -> ToolPolicy {
        // Read-only registry lookup.
        ToolPolicy::parallel()
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["load", "list"],
                    "description": "Use 'list' to enumerate discovered skills. Use 'load' (default) to fetch one skill by name."
                },
                "skill_name": {
                    "type": "string",
                    "description": "Exact skill name when action is load (e.g. 'code_review')."
                },
                "detail": {
                    "type": "string",
                    "enum": ["full", "metadata"],
                    "description": "When action is load: 'full' returns instruction body (default). 'metadata' returns name, availability, description, and body length without the full body."
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("load");

        if action == "list" {
            return Ok(self.registry.read().await.format_skill_directory());
        }

        let skill_name = args
            .get("skill_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'skill_name' when action is load (default).".to_string())?;

        let detail = args
            .get("detail")
            .and_then(|v| v.as_str())
            .unwrap_or("full");

        // First attempt against the registry as it was last scanned. The read guard is scoped to
        // this block and dropped before the rescan path below takes the write lock, so we never
        // hold a read guard across a write acquisition on the same RwLock (which would deadlock).
        let first = {
            let registry = self.registry.read().await;
            if detail == "metadata" {
                registry.get_skill_metadata(skill_name)
            } else {
                registry.get_skill_instructions(skill_name)
            }
        };
        if first.is_ok() {
            return first;
        }

        // Miss: a SKILL.md may have been dropped into the skills directory since the registry was
        // last scanned (it is scanned once at startup). Rescan once and re-resolve before reporting
        // the skill missing, so a freshly added skill is loadable without restarting the agent. The
        // rescan is paid only on an actual miss, so the common path (skill already present) is
        // unchanged. Hold a single write guard for both the rescan and the follow-up lookup (the
        // guard derefs to the registry for the immutable getters), avoiding a redundant drop-then-
        // reacquire and closing the gap between scan and read.
        let mut registry = self.registry.write().await;
        registry.scan_for_skills();
        if detail == "metadata" {
            registry.get_skill_metadata(skill_name)
        } else {
            registry.get_skill_instructions(skill_name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::failover::{build_llm_failed_banner, FailoverLogCtx};
    use super::reasoning::{effective_context_tokens, estimate_message_tokens, ReasoningLoopError};
    use super::{
        steering_guard, ActiveProviderConfig, AgentLogic, AgentLogicParams, QueuedInbound,
        ReasoningLoopCtx, ReasoningLoopExit, RunProviderContext, SteeringInbox,
    };
    use async_trait::async_trait;
    use axum::{
        body::Body,
        extract::State,
        http::{HeaderMap, StatusCode},
        routing::post,
        Router,
    };
    use serde_json::Value;
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;
    use tower::util::ServiceExt;

    use crate::bus::{
        clarification_session_key, BusMessage, InboundMessage, RunBudgetSnapshot, RunFailureKind,
        RunLifecycleEvent, RunOutcome, RunStuckReason, METADATA_RUN_ID,
    };
    use crate::clarification::ClarificationHub;
    use crate::logging::create_logger_channel;
    use crate::memory::SqliteMemoryActor;
    use crate::multi_tenant_edge::{ActivityHeartbeatClient, HeartbeatTransport};
    use crate::session::SessionManager;
    use crate::skills::SkillRegistry;
    use crate::tool_activity::SharedToolExecutionActivity;
    use crate::tool_runtime::ToolExecCtx;
    use crate::tools::ToolRegistry;
    use crate::traits::{Memory, Provider, Tool};
    use crate::utils::{ChatMessage, LLMError, LLMResponse};
    use crate::{ActorLogic, NodeHandle};

    struct LocalTempDir {
        path: PathBuf,
    }

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    impl LocalTempDir {
        fn new() -> Self {
            let unique = format!(
                "isanagent-heartbeat-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos(),
                NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).expect("tempdir");
            Self { path }
        }

        fn path(&self) -> &PathBuf {
            &self.path
        }
    }

    impl Drop for LocalTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn tauri_retryable_failure_does_not_advertise_terminal_commands() {
        let banner = build_llm_failed_banner("tauri", "chat-1", None, "provider unavailable", true);

        assert!(!banner.content.contains("/retry"));
        assert!(banner
            .content
            .contains("switch to another LLM from the client"));
        assert!(!banner
            .metadata
            .contains_key(crate::protocol::ISANAGENT_LLM_RETRY_AVAILABLE));
        assert!(!banner
            .metadata
            .contains_key(crate::protocol::ISANAGENT_TERMINAL_ERROR));
    }

    #[test]
    fn terminal_retryable_failure_advertises_retry_command() {
        let banner =
            build_llm_failed_banner("terminal", "chat-1", None, "provider unavailable", true);

        assert!(banner.content.contains("Press /retry"));
        assert_eq!(
            banner
                .metadata
                .get(crate::protocol::ISANAGENT_LLM_RETRY_AVAILABLE),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            banner
                .metadata
                .get(crate::protocol::ISANAGENT_TERMINAL_ERROR),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn terminal_non_retryable_failure_does_not_offer_retry() {
        let banner = build_llm_failed_banner("terminal", "chat-1", None, "context overflow", false);

        assert!(!banner.content.contains("/retry"));
        assert!(!banner
            .metadata
            .contains_key(crate::protocol::ISANAGENT_LLM_RETRY_AVAILABLE));
        assert_eq!(
            banner
                .metadata
                .get(crate::protocol::ISANAGENT_TERMINAL_ERROR),
            Some(&serde_json::json!(true))
        );
    }

    #[derive(Clone)]
    struct RecordedHeartbeat {
        received_at: Instant,
        authorization: Option<String>,
    }

    #[derive(Clone)]
    struct HeartbeatState {
        status: StatusCode,
        records: Arc<Mutex<Vec<RecordedHeartbeat>>>,
    }

    async fn heartbeat_handler(
        State(state): State<HeartbeatState>,
        headers: HeaderMap,
    ) -> StatusCode {
        state.records.lock().unwrap().push(RecordedHeartbeat {
            received_at: Instant::now(),
            authorization: headers
                .get("authorization")
                .and_then(|value| value.to_str().ok())
                .map(|value| value.to_string()),
        });
        state.status
    }

    #[derive(Clone)]
    struct RouterHeartbeatTransport {
        app: Router,
    }

    #[async_trait]
    impl HeartbeatTransport for RouterHeartbeatTransport {
        async fn post_activity(&self, url: &str, token: &str) -> Result<StatusCode, String> {
            let parsed_url = reqwest::Url::parse(url).map_err(|error| error.to_string())?;
            let request = axum::http::Request::builder()
                .method("POST")
                .uri(parsed_url.path())
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .map_err(|error| error.to_string())?;
            let response = self
                .app
                .clone()
                .oneshot(request)
                .await
                .map_err(|error| error.to_string())?;
            Ok(response.status())
        }
    }

    #[derive(Clone)]
    struct FailingHeartbeatTransport;

    #[async_trait]
    impl HeartbeatTransport for FailingHeartbeatTransport {
        async fn post_activity(&self, _url: &str, _token: &str) -> Result<StatusCode, String> {
            Err("connection refused".to_string())
        }
    }

    #[derive(Clone)]
    struct SequenceHeartbeatTransport {
        responses: Arc<Mutex<Vec<Result<StatusCode, String>>>>,
        call_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl HeartbeatTransport for SequenceHeartbeatTransport {
        async fn post_activity(&self, _url: &str, _token: &str) -> Result<StatusCode, String> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            let mut responses = self.responses.lock().unwrap();
            if responses.len() > 1 {
                responses.remove(0)
            } else {
                responses
                    .first()
                    .cloned()
                    .unwrap_or(Ok(StatusCode::NO_CONTENT))
            }
        }
    }

    #[derive(Clone)]
    struct SlowHeartbeatTransport {
        delay: Duration,
    }

    #[async_trait]
    impl HeartbeatTransport for SlowHeartbeatTransport {
        async fn post_activity(&self, _url: &str, _token: &str) -> Result<StatusCode, String> {
            tokio::time::sleep(self.delay).await;
            Ok(StatusCode::NO_CONTENT)
        }
    }

    fn build_heartbeat_transport(
        status: StatusCode,
    ) -> (
        Arc<dyn HeartbeatTransport>,
        Arc<Mutex<Vec<RecordedHeartbeat>>>,
    ) {
        let records = Arc::new(Mutex::new(Vec::new()));
        let state = HeartbeatState {
            status,
            records: records.clone(),
        };
        let app = Router::new()
            .route("/_internal/activity", post(heartbeat_handler))
            .with_state(state);

        (
            Arc::new(RouterHeartbeatTransport { app }) as Arc<dyn HeartbeatTransport>,
            records,
        )
    }

    #[derive(Clone)]
    struct DummyProvider;

    #[async_trait]
    impl Provider for DummyProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            unreachable!("DummyProvider is not used in heartbeat tests")
        }
    }

    /// First `chat` waits until the test releases `unblock_rx`; later calls return immediately.
    #[derive(Clone)]
    struct GateFirstChatProvider {
        calls: Arc<AtomicUsize>,
        first_unblock: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
    }

    #[async_trait]
    impl Provider for GateFirstChatProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                let mut slot = self.first_unblock.lock().await;
                if let Some(rx) = slot.take() {
                    let _ = rx.await;
                }
            }
            Ok(LLMResponse {
                content: format!("ok-{n}"),
                tool_calls: None,
                reasoning_content: None,
                usage: None,
            })
        }
    }

    #[derive(Clone)]
    struct LongSleepProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for LongSleepProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Err(LLMError::ApiError(
                "LongSleepProvider should have been cancelled".into(),
            ))
        }
    }

    #[derive(Clone)]
    struct NonTransientErrorProvider;

    #[async_trait]
    impl Provider for NonTransientErrorProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            Err(LLMError::ApiError("Status 400 bad request".into()))
        }
    }

    /// Returns `Ok` with `content` set to `tag` — a stand-in for a working fallback provider.
    #[derive(Clone)]
    struct RespondingProvider {
        tag: String,
    }

    #[async_trait]
    impl Provider for RespondingProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            Ok(LLMResponse {
                content: self.tag.clone(),
                tool_calls: None,
                reasoning_content: None,
                usage: None,
            })
        }
    }

    fn fb_spec(name: &str) -> super::FallbackProviderSpec {
        super::FallbackProviderSpec {
            provider_name: name.to_string(),
            base_url: String::new(),
            api_key: String::new(),
            model_name: format!("{name}-model"),
        }
    }

    fn fb_full(provider: &str, base: &str, model: &str) -> super::FallbackProviderSpec {
        super::FallbackProviderSpec {
            provider_name: provider.to_string(),
            base_url: base.to_string(),
            api_key: "k".to_string(),
            model_name: model.to_string(),
        }
    }

    #[test]
    fn build_fallback_specs_excludes_primary_by_full_identity() {
        let candidates = vec![
            // Same identity as the primary but with different casing and a trailing slash on the
            // base URL — must still be excluded after normalization.
            fb_full("Anthropic", "https://api.anthropic.com/", "Claude"),
            fb_full("openai", "https://api.openai.com", "gpt-4o"),
        ];
        let out = super::build_fallback_specs(
            "anthropic",
            "https://api.anthropic.com",
            "claude",
            candidates,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].provider_name, "openai");
    }

    #[test]
    fn build_fallback_specs_keeps_same_provider_different_model_or_url() {
        // Same provider but a different model — a legitimate fallback, must be kept.
        let candidates = vec![
            fb_full("openai", "u", "gpt-4o-mini"),
            fb_full("openai", "u2", "gpt-4o"), // same provider+model, different base_url -> kept
        ];
        let out = super::build_fallback_specs("openai", "u", "gpt-4o", candidates);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn fallback_spec_debug_redacts_api_key() {
        let dbg = format!("{:?}", fb_full("openai", "u", "m"));
        assert!(dbg.contains("[redacted]"), "{dbg}");
        assert!(!dbg.contains("\"k\""), "api key must not appear: {dbg}");
    }

    #[tokio::test]
    async fn concurrent_agents_snapshot_distinct_provider_credentials_and_fallbacks() {
        let (agent_a, _rx_a) = build_agent_with_provider_state(
            Box::new(RespondingProvider { tag: "a".into() }),
            provider_credentials("provider-a", "https://a.test/v1/", "secret-a", "model-a"),
            vec![
                fb_full("provider-a", "https://a.test/v1", "model-a"),
                fb_full("fallback-a", "https://fallback-a.test", "fallback-model-a"),
            ],
            ClarificationHub::shared(),
        );
        let (agent_b, _rx_b) = build_agent_with_provider_state(
            Box::new(RespondingProvider { tag: "b".into() }),
            provider_credentials("provider-b", "https://b.test/v1", "secret-b", "model-b"),
            vec![
                fb_full("provider-b", "https://b.test/v1/", "model-b"),
                fb_full("fallback-b", "https://fallback-b.test", "fallback-model-b"),
            ],
            ClarificationHub::shared(),
        );

        let (context_a, context_b) = tokio::join!(
            agent_a.run_provider_context(),
            agent_b.run_provider_context()
        );

        assert_eq!(context_a.identity.provider_name, "provider-a");
        assert_eq!(context_a.identity.model_name, "model-a");
        assert_eq!(context_a.fallback_providers.len(), 1);
        assert_eq!(context_a.fallback_providers[0].provider_name, "fallback-a");
        assert_eq!(context_b.identity.provider_name, "provider-b");
        assert_eq!(context_b.identity.model_name, "model-b");
        assert_eq!(context_b.fallback_providers.len(), 1);
        assert_eq!(context_b.fallback_providers[0].provider_name, "fallback-b");
        assert_ne!(
            context_a.identity.secret_identity,
            context_b.identity.secret_identity
        );
        assert!(!context_a.identity.secret_identity.contains("secret-a"));
        assert!(!context_b.identity.secret_identity.contains("secret-b"));
    }

    #[tokio::test]
    async fn atomic_provider_switch_replaces_the_complete_active_pair() {
        let (agent, _rx) = build_agent_with_provider_state(
            Box::new(RespondingProvider { tag: "old".into() }),
            provider_credentials("provider-a", "https://a.test", "secret-a", "model-a"),
            vec![fb_full(
                "fallback-b",
                "https://fallback-b.test",
                "fallback-model-b",
            )],
            ClarificationHub::shared(),
        );

        agent
            .switch_provider_with_credentials(
                Box::new(RespondingProvider { tag: "new".into() }),
                provider_credentials("provider-b", "https://b.test", "secret-b", "model-b"),
            )
            .await;

        let context = agent.run_provider_context().await;
        assert_eq!(context.identity.provider_name, "provider-b");
        assert_eq!(context.identity.model_name, "model-b");
        assert_ne!(context.identity.secret_identity, "none");
        assert_eq!(context.fallback_providers.len(), 1);
        assert_eq!(context.fallback_providers[0].provider_name, "fallback-b");
    }

    #[tokio::test]
    async fn try_fallbacks_returns_first_success() {
        let (logger, _rx) = crate::logging::create_logger_channel(64);
        let cancel = tokio_util::sync::CancellationToken::new();
        let specs = vec![fb_spec("a"), fb_spec("b")];
        // 'a' fails, 'b' succeeds -> first success wins, 'b' chosen.
        let out = super::failover::try_fallbacks(
            &specs,
            |s| -> Box<dyn Provider> {
                if s.provider_name == "b" {
                    Box::new(RespondingProvider { tag: "b-ok".into() })
                } else {
                    Box::new(NonTransientErrorProvider)
                }
            },
            &[],
            &None,
            &cancel,
            FailoverLogCtx {
                logger_tx: &logger,
                name: "agent",
                chat_id: "c1",
            },
        )
        .await;
        match out {
            super::failover::FallbackOutcome::Ok(r) => assert_eq!(r.content, "b-ok"),
            _ => panic!("expected Ok from fallback b"),
        }
    }

    #[tokio::test]
    async fn try_fallbacks_all_fail_is_exhausted() {
        let (logger, _rx) = crate::logging::create_logger_channel(64);
        let cancel = tokio_util::sync::CancellationToken::new();
        let specs = vec![fb_spec("a"), fb_spec("b")];
        let out = super::failover::try_fallbacks(
            &specs,
            |_| -> Box<dyn Provider> { Box::new(NonTransientErrorProvider) },
            &[],
            &None,
            &cancel,
            FailoverLogCtx {
                logger_tx: &logger,
                name: "agent",
                chat_id: "c1",
            },
        )
        .await;
        assert!(matches!(out, super::failover::FallbackOutcome::Exhausted));
    }

    #[tokio::test]
    async fn try_fallbacks_empty_is_exhausted() {
        let (logger, _rx) = crate::logging::create_logger_channel(64);
        let cancel = tokio_util::sync::CancellationToken::new();
        let out = super::failover::try_fallbacks(
            &[],
            |_| -> Box<dyn Provider> { Box::new(RespondingProvider { tag: "x".into() }) },
            &[],
            &None,
            &cancel,
            FailoverLogCtx {
                logger_tx: &logger,
                name: "agent",
                chat_id: "c1",
            },
        )
        .await;
        assert!(matches!(out, super::failover::FallbackOutcome::Exhausted));
    }

    #[tokio::test]
    async fn try_fallbacks_cancellation_short_circuits() {
        let (logger, _rx) = crate::logging::create_logger_channel(64);
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel(); // pre-cancelled; the slow provider's chat never wins the select
        let specs = vec![fb_spec("a")];
        let out = super::failover::try_fallbacks(
            &specs,
            |_| -> Box<dyn Provider> {
                Box::new(LongSleepProvider {
                    calls: Arc::new(AtomicUsize::new(0)),
                })
            },
            &[],
            &None,
            &cancel,
            FailoverLogCtx {
                logger_tx: &logger,
                name: "agent",
                chat_id: "c1",
            },
        )
        .await;
        assert!(matches!(out, super::failover::FallbackOutcome::Cancelled));
    }

    #[derive(Clone)]
    struct PanicProvider;

    #[async_trait]
    impl Provider for PanicProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            panic!("panic provider exploded")
        }
    }

    /// Always returns the SAME tool call — drives the doom-loop detector to fire repeatedly.
    #[derive(Clone)]
    struct IdenticalToolCallProvider;

    #[async_trait]
    impl Provider for IdenticalToolCallProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            Ok(LLMResponse {
                content: String::new(),
                tool_calls: Some(vec![crate::utils::ToolCallRequest {
                    id: "call_loop".to_string(),
                    tool_type: "function".to_string(),
                    extra_content: None,
                    function: crate::utils::ToolCallFunction {
                        name: "looping_tool".to_string(),
                        arguments: "{\"x\":1}".to_string(),
                    },
                }]),
                reasoning_content: None,
                usage: None,
            })
        }
    }

    /// Loops identically for the first 3 calls (triggers detection + a nudge), then emits
    /// distinct tool calls — simulating a model that corrects itself after the nudge.
    #[derive(Clone)]
    struct CorrectingProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for CorrectingProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            // First 3: identical args (X,X,X → detection). After: distinct args (loop no longer
            // active at the tail), so escalation must reset rather than hard-stop.
            let arguments = if n < 3 {
                "{\"x\":0}".to_string()
            } else {
                format!("{{\"x\":{n}}}")
            };
            Ok(LLMResponse {
                content: String::new(),
                tool_calls: Some(vec![crate::utils::ToolCallRequest {
                    id: format!("call_{n}"),
                    tool_type: "function".to_string(),
                    extra_content: None,
                    function: crate::utils::ToolCallFunction {
                        name: "looping_tool".to_string(),
                        arguments,
                    },
                }]),
                reasoning_content: None,
                usage: None,
            })
        }
    }

    #[derive(Clone)]
    struct MalformedToolArgumentsProvider {
        calls: Arc<AtomicUsize>,
        tool_names: Vec<String>,
    }

    #[async_trait]
    impl Provider for MalformedToolArgumentsProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<serde_json::Value>,
        ) -> Result<LLMResponse, LLMError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call > 0 {
                return Ok(LLMResponse {
                    content: "recovered after invalid arguments".to_string(),
                    tool_calls: None,
                    reasoning_content: None,
                    usage: None,
                });
            }
            let tool_calls = self
                .tool_names
                .iter()
                .enumerate()
                .map(|(index, name)| crate::utils::ToolCallRequest {
                    id: format!("malformed-{index}"),
                    tool_type: "function".to_string(),
                    extra_content: None,
                    function: crate::utils::ToolCallFunction {
                        name: name.clone(),
                        arguments: "{\"unterminated\":".to_string(),
                    },
                })
                .collect();
            Ok(LLMResponse {
                content: String::new(),
                tool_calls: Some(tool_calls),
                reasoning_content: None,
                usage: None,
            })
        }
    }

    async fn run_loop_once_for_test(
        provider: Box<dyn Provider>,
        max_iterations: usize,
        cancelled_before_start: bool,
        doom_loop_enabled: bool,
    ) -> (
        Result<ReasoningLoopExit, ReasoningLoopError>,
        Vec<ChatMessage>,
    ) {
        run_loop_once_for_test_with_autonomy(
            provider,
            max_iterations,
            cancelled_before_start,
            doom_loop_enabled,
            false,
        )
        .await
    }

    async fn run_loop_once_for_test_with_autonomy(
        provider: Box<dyn Provider>,
        max_iterations: usize,
        cancelled_before_start: bool,
        doom_loop_enabled: bool,
        forbid_final_without_tools: bool,
    ) -> (
        Result<ReasoningLoopExit, ReasoningLoopError>,
        Vec<ChatMessage>,
    ) {
        let memory_actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let memory_node = NodeHandle::new(memory_actor, 16, 1, Duration::from_millis(1));
        let session_manager = Arc::new(SessionManager::new(memory_node));
        let tools = Arc::new(ToolRegistry::new());
        let skills_temp = LocalTempDir::new();
        let skills = Arc::new(tokio::sync::RwLock::new(SkillRegistry::new(
            skills_temp.path().clone(),
        )));
        let (outbound_tx, mut outbound_rx) = mpsc::channel::<BusMessage>(8);
        // Drain outbound telemetry so the loop's `send().await` never blocks on a full buffer
        // (tool-call-returning providers emit several messages per iteration).
        tokio::spawn(async move { while outbound_rx.recv().await.is_some() {} });
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let cancel_token = tokio_util::sync::CancellationToken::new();
        if cancelled_before_start {
            cancel_token.cancel();
        }
        let inbound = test_inbound("loop-test-chat", "hello");
        let session_key = inbound.clarification_session_key();
        let inbound_metadata = Arc::new(inbound.metadata.clone());
        let run_provider = RunProviderContext::snapshot(
            &ActiveProviderConfig {
                provider,
                credentials: crate::provider::ProviderCredentials::empty(),
            },
            &[],
        );
        let result = AgentLogic::run_reasoning_loop(ReasoningLoopCtx {
            name: "LoopTestAgent".to_string(),
            run_provider,
            session_manager: session_manager.clone(),
            tools,
            skills,
            system_prompt: "test system prompt".to_string(),
            max_iterations,
            max_tool_output_chars: 4_000,
            max_recent_summaries: 0,
            short_term_threshold_turns: 10,
            short_term_threshold_tokens: 10_000,
            tool_execution_activity: None,
            outbound_tx,
            logger_tx,
            inbound,
            run_id: "test-run-id".to_string(),
            steering: Arc::new(Mutex::new(SteeringInbox::open())),
            cancel_token: cancel_token.clone(),
            clarification_hub: ClarificationHub::shared(),
            tool_exec_ctx: ToolExecCtx::new("terminal", "loop-test-chat", None)
                .with_reasoning_cancel(cancel_token),
            is_subagent: false,
            subagent_allowlist: None,
            doom_loop_enabled,
            harness_runtime_summary: String::new(),
            forbid_final_without_tools,
            shell_policy: Arc::new(crate::config::ResolvedShellPolicy {
                interactive_mode: crate::config::ShellPolicyMode::Ask,
                unattended_mode: crate::config::ShellPolicyMode::Deny,
                interactive_edit_mode: crate::config::ShellPolicyMode::Ask,
                unattended_edit_mode: crate::config::ShellPolicyMode::Deny,
                approval_patterns: Vec::new(),
                windows_runner: crate::config::WindowsShellRunner::default(),
            }),
            hook_tool_ctx: None,
            inbound_metadata,
        })
        .await;

        let session = session_manager
            .get_session(&session_key)
            .await
            .expect("session");
        let context = session.get_context().await.expect("context");
        (result, context)
    }

    struct SlowTool {
        delay: Duration,
        result: String,
    }

    #[async_trait]
    impl Tool for SlowTool {
        fn name(&self) -> &str {
            "slow_tool"
        }

        fn description(&self) -> &str {
            "Sleeps briefly and returns a static response."
        }

        fn parameters(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }

        async fn execute(&self, _args: Value) -> Result<String, String> {
            tokio::time::sleep(self.delay).await;
            Ok(self.result.clone())
        }
    }

    fn build_agent_with_provider_and_hub(
        provider: Box<dyn Provider>,
        clarification_hub: Arc<ClarificationHub>,
    ) -> (AgentLogic, mpsc::Receiver<BusMessage>) {
        build_agent_with_provider_state(
            provider,
            crate::provider::ProviderCredentials::empty(),
            Vec::new(),
            clarification_hub,
        )
    }

    fn build_agent_with_provider_state(
        provider: Box<dyn Provider>,
        provider_credentials: crate::provider::ProviderCredentials,
        fallback_providers: Vec<super::FallbackProviderSpec>,
        clarification_hub: Arc<ClarificationHub>,
    ) -> (AgentLogic, mpsc::Receiver<BusMessage>) {
        let memory_actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let memory_node = NodeHandle::new(memory_actor, 16, 1, Duration::from_millis(1));
        let session_manager = SessionManager::new(memory_node);

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(SlowTool {
            delay: Duration::from_millis(0),
            result: "tool complete".to_string(),
        }));

        let skills_temp = LocalTempDir::new();
        let skills = SkillRegistry::new(skills_temp.path().clone());

        let (outbound_tx, outbound_rx) = mpsc::channel::<BusMessage>(64);
        let (logger_tx, _logger_rx) = create_logger_channel(32);

        let agent = AgentLogic::new_with_fallback_providers(
            AgentLogicParams {
                name: "TestAgent".to_string(),
                provider,
                provider_credentials,
                session_manager,
                tools,
                skills,
                system_prompt: "test system prompt".to_string(),
                max_iterations: 4,
                max_tool_output_chars: 4_000,
                max_recent_summaries: 0,
                short_term_threshold_turns: 10,
                short_term_threshold_tokens: 10_000,
                outbound_tx,
                logger_tx,
                clarification_hub,
                subagent: None,
                doom_loop_enabled: false,
                harness_runtime_summary: String::new(),
                subagent_system_prompt: "test system prompt".to_string(),
                forbid_final_without_tools: false,
                shell_policy: crate::config::ResolvedShellPolicy {
                    interactive_mode: crate::config::ShellPolicyMode::Ask,
                    unattended_mode: crate::config::ShellPolicyMode::Deny,
                    interactive_edit_mode: crate::config::ShellPolicyMode::Ask,
                    unattended_edit_mode: crate::config::ShellPolicyMode::Deny,
                    approval_patterns: Vec::new(),
                    windows_runner: crate::config::WindowsShellRunner::default(),
                },
                hook_tool_ctx: None,
            },
            fallback_providers,
        );

        (agent, outbound_rx)
    }

    fn build_agent_with_provider(
        provider: Box<dyn Provider>,
    ) -> (AgentLogic, mpsc::Receiver<BusMessage>) {
        build_agent_with_provider_and_hub(provider, ClarificationHub::shared())
    }

    fn provider_credentials(
        provider_name: &str,
        base_url: &str,
        api_key: &str,
        model_name: &str,
    ) -> crate::provider::ProviderCredentials {
        crate::provider::ProviderCredentials {
            provider_name: provider_name.to_string(),
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            model_name: model_name.to_string(),
        }
    }

    fn build_test_agent(
        tool_execution_activity: Option<SharedToolExecutionActivity>,
        tool_delay: Duration,
    ) -> AgentLogic {
        let memory_actor = SqliteMemoryActor::new(":memory:").expect("memory actor");
        let memory_node = NodeHandle::new(memory_actor, 16, 1, Duration::from_millis(1));
        let session_manager = SessionManager::new(memory_node);

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(SlowTool {
            delay: tool_delay,
            result: "tool complete".to_string(),
        }));

        let skills_temp = LocalTempDir::new();
        let skills = SkillRegistry::new(skills_temp.path().clone());

        let (outbound_tx, _outbound_rx) = mpsc::channel::<BusMessage>(8);
        let (logger_tx, _logger_rx) = create_logger_channel(32);

        let agent = AgentLogic::new(AgentLogicParams {
            name: "TestAgent".to_string(),
            provider: Box::new(DummyProvider),
            provider_credentials: crate::provider::ProviderCredentials::empty(),
            session_manager,
            tools,
            skills,
            system_prompt: "test system prompt".to_string(),
            max_iterations: 4,
            max_tool_output_chars: 4_000,
            max_recent_summaries: 0,
            short_term_threshold_turns: 10,
            short_term_threshold_tokens: 10_000,
            outbound_tx,
            logger_tx,
            clarification_hub: ClarificationHub::shared(),
            subagent: None,
            doom_loop_enabled: false,
            harness_runtime_summary: String::new(),
            subagent_system_prompt: "test system prompt".to_string(),
            forbid_final_without_tools: false,
            shell_policy: crate::config::ResolvedShellPolicy {
                interactive_mode: crate::config::ShellPolicyMode::Ask,
                unattended_mode: crate::config::ShellPolicyMode::Deny,
                interactive_edit_mode: crate::config::ShellPolicyMode::Ask,
                unattended_edit_mode: crate::config::ShellPolicyMode::Deny,
                approval_patterns: Vec::new(),
                windows_runner: crate::config::WindowsShellRunner::default(),
            },
            hook_tool_ctx: None,
        });

        if let Some(tool_execution_activity) = tool_execution_activity {
            agent.with_tool_execution_activity(tool_execution_activity)
        } else {
            agent
        }
    }

    #[tokio::test]
    async fn execute_tool_call_sends_immediate_and_repeated_heartbeats() {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let (transport, records) = build_heartbeat_transport(StatusCode::NO_CONTENT);
        let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
            "http://edge.test/_internal/activity".to_string(),
            "edge-token".to_string(),
            Duration::from_millis(30),
            logger_tx,
            transport,
        ));
        let agent = build_test_agent(Some(heartbeat), Duration::from_millis(140));
        let started_at = Instant::now();

        let result = agent
            .execute_tool_call("chat-123", "slow_tool", serde_json::json!({}))
            .await
            .expect("tool result");

        assert_eq!(result, "tool complete");

        let records = records.lock().unwrap().clone();
        assert!(
            records.len() >= 3,
            "expected repeated heartbeats, got {}",
            records.len()
        );
        assert_eq!(
            records[0].authorization.as_deref(),
            Some("Bearer edge-token")
        );
        assert!(
            records[0].received_at.duration_since(started_at) < Duration::from_millis(100),
            "expected immediate heartbeat"
        );
    }

    #[tokio::test]
    async fn execute_tool_call_completes_when_heartbeat_endpoint_returns_statuses() {
        for status in [
            StatusCode::UNAUTHORIZED,
            StatusCode::NOT_FOUND,
            StatusCode::NOT_IMPLEMENTED,
        ] {
            let (logger_tx, _logger_rx) = create_logger_channel(32);
            let (transport, _records) = build_heartbeat_transport(status);
            let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
                "http://edge.test/_internal/activity".to_string(),
                "edge-token".to_string(),
                Duration::from_millis(30),
                logger_tx,
                transport,
            ));
            let agent = build_test_agent(Some(heartbeat), Duration::from_millis(40));

            let result = agent
                .execute_tool_call("chat-123", "slow_tool", serde_json::json!({}))
                .await
                .expect("tool result should succeed even when heartbeat fails");

            assert_eq!(result, "tool complete");
        }
    }

    #[tokio::test]
    async fn execute_tool_call_completes_when_heartbeat_endpoint_is_unavailable() {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
            "http://edge.test/_internal/activity".to_string(),
            "edge-token".to_string(),
            Duration::from_millis(30),
            logger_tx,
            Arc::new(FailingHeartbeatTransport),
        ));
        let agent = build_test_agent(Some(heartbeat), Duration::from_millis(40));

        let result = agent
            .execute_tool_call("chat-123", "slow_tool", serde_json::json!({}))
            .await
            .expect("tool result should succeed without heartbeat server");

        assert_eq!(result, "tool complete");
    }

    #[tokio::test]
    async fn execute_tool_call_retries_after_transient_heartbeat_failures() {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
            "http://edge.test/_internal/activity".to_string(),
            "edge-token".to_string(),
            Duration::from_millis(20),
            logger_tx,
            Arc::new(SequenceHeartbeatTransport {
                responses: Arc::new(Mutex::new(vec![
                    Ok(StatusCode::SERVICE_UNAVAILABLE),
                    Ok(StatusCode::NO_CONTENT),
                    Ok(StatusCode::NO_CONTENT),
                ])),
                call_count: call_count.clone(),
            }),
        ));
        let agent = build_test_agent(Some(heartbeat), Duration::from_millis(75));

        let result = agent
            .execute_tool_call("chat-123", "slow_tool", serde_json::json!({}))
            .await
            .expect("tool result should succeed after transient heartbeat failures");

        assert_eq!(result, "tool complete");
        assert!(
            call_count.load(Ordering::Relaxed) >= 2,
            "expected heartbeat retries after transient failure"
        );
    }

    #[tokio::test]
    async fn execute_tool_call_does_not_wait_for_hung_heartbeat_requests_on_stop() {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let heartbeat = Arc::new(ActivityHeartbeatClient::new_with_transport(
            "http://edge.test/_internal/activity".to_string(),
            "edge-token".to_string(),
            Duration::from_millis(20),
            logger_tx,
            Arc::new(SlowHeartbeatTransport {
                delay: Duration::from_millis(250),
            }),
        ));
        let agent = build_test_agent(Some(heartbeat), Duration::from_millis(10));

        let result = tokio::time::timeout(
            Duration::from_millis(100),
            agent.execute_tool_call("chat-123", "slow_tool", serde_json::json!({})),
        )
        .await
        .expect("tool should not wait for a hung heartbeat request")
        .expect("tool result");

        assert_eq!(result, "tool complete");
    }

    fn test_inbound(chat_id: &str, content: &str) -> InboundMessage {
        InboundMessage {
            channel: "terminal".to_string(),
            sender_id: "local_user".to_string(),
            chat_id: chat_id.to_string(),
            thread_id: None,
            content: content.to_string(),
            attachments: vec![],
            metadata: Default::default(),
        }
    }

    #[test]
    fn run_id_is_required_for_tauri_and_backfilled_for_legacy_channels() {
        let mut tauri = test_inbound("run-id-tauri", "hello");
        tauri.channel = "tauri".to_string();
        assert!(super::ensure_run_id(&mut tauri).is_err());

        let mut legacy = test_inbound("run-id-terminal", "hello");
        let generated = super::ensure_run_id(&mut legacy).expect("legacy run id");
        assert!(generated.starts_with("legacy-"));
        assert_eq!(
            legacy
                .metadata
                .get(METADATA_RUN_ID)
                .and_then(|value| value.as_str()),
            Some(generated.as_str())
        );
    }

    #[tokio::test]
    async fn invalid_tauri_inbound_is_rejected_without_stopping_the_actor() {
        let provider = RespondingProvider {
            tag: "done".to_string(),
        };
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(provider));
        let mut invalid = test_inbound("invalid-run-id", "hello");
        invalid.channel = "tauri".to_string();

        assert!(matches!(
            agent.process(BusMessage::Inbound(invalid)).await,
            Ok(None)
        ));
        assert!(matches!(
            outbound_rx.recv().await,
            Some(BusMessage::Outbound(_))
        ));

        let mut valid = test_inbound("valid-after-rejection", "hello");
        valid.channel = "tauri".to_string();
        valid.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("valid-run-id"),
        );
        agent
            .process(BusMessage::Inbound(valid))
            .await
            .expect("actor remains usable after rejecting malformed inbound");
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(2), outbound_rx.recv()).await,
            Ok(Some(BusMessage::RunLifecycle(
                RunLifecycleEvent::Started { .. }
            )))
        ));
    }

    #[tokio::test]
    async fn invalid_queued_inbound_does_not_strand_following_valid_message() {
        let (unblock_tx, unblock_rx) = tokio::sync::oneshot::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = GateFirstChatProvider {
            calls: calls.clone(),
            first_unblock: Arc::new(tokio::sync::Mutex::new(Some(unblock_rx))),
        };
        let (mut agent, _outbound_rx) = build_agent_with_provider(Box::new(provider));
        let chat_id = "skip-invalid-queued";
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start first turn");
        for _ in 0..200 {
            if calls.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let mut invalid = test_inbound(chat_id, "invalid queued");
        invalid.channel = "tauri".to_string();
        let mut valid = test_inbound(chat_id, "valid queued");
        valid.channel = "tauri".to_string();
        valid.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("queued-run-id"),
        );
        let run_provider = agent.run_provider_context().await;
        agent.pending_inbound.insert(
            chat_id.to_string(),
            Mutex::new(VecDeque::from([
                QueuedInbound {
                    inbound: invalid,
                    run_provider: run_provider.clone(),
                },
                QueuedInbound {
                    inbound: valid,
                    run_provider,
                },
            ])),
        );

        unblock_tx.send(()).expect("unblock first turn");
        for _ in 0..400 {
            if calls.load(Ordering::SeqCst) == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "the valid queued message must run after the invalid item is dropped"
        );
    }

    #[tokio::test]
    async fn model_switch_preserves_active_run_and_updates_later_queued_admission() {
        let (unblock_tx, unblock_rx) = tokio::sync::oneshot::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider_a = GateFirstChatProvider {
            calls: calls.clone(),
            first_unblock: Arc::new(tokio::sync::Mutex::new(Some(unblock_rx))),
        };
        let (mut agent, mut outbound_rx) = build_agent_with_provider_state(
            Box::new(provider_a),
            provider_credentials("provider-a", "https://a.test", "secret-a", "model-a"),
            Vec::new(),
            ClarificationHub::shared(),
        );
        let chat_id = "run-provider-admission";

        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start provider-a turn");
        for _ in 0..200 {
            if calls.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        agent
            .switch_provider_with_credentials(
                Box::new(RespondingProvider {
                    tag: "provider-b".to_string(),
                }),
                provider_credentials("provider-b", "https://b.test", "secret-b", "model-b"),
            )
            .await;
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "second")))
            .await
            .expect("queue provider-b turn");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            agent
                .pending_inbound
                .get(chat_id)
                .expect("queued second turn")
                .lock()
                .expect("pending queue lock")
                .len(),
            1
        );

        unblock_tx.send(()).expect("release provider-a turn");
        let (responses, terminal_count) = tokio::time::timeout(Duration::from_secs(5), async {
            let mut responses = Vec::new();
            let mut terminal_count = 0;
            while terminal_count < 2 {
                match outbound_rx.recv().await.expect("outbound channel open") {
                    BusMessage::Outbound(outbound) if outbound.chat_id == chat_id => {
                        responses.push(outbound.content);
                    }
                    BusMessage::RunLifecycle(RunLifecycleEvent::Terminated {
                        chat_id: event_chat,
                        ..
                    }) if event_chat == chat_id => terminal_count += 1,
                    _ => {}
                }
            }
            (responses, terminal_count)
        })
        .await
        .expect("both run snapshots complete");

        assert_eq!(terminal_count, 2);
        assert_eq!(responses, vec!["ok-0", "provider-b"]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn started_lifecycle_event_preserves_caller_run_id() {
        let provider = RespondingProvider {
            tag: "done".to_string(),
        };
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(provider));
        let mut inbound = test_inbound("run-id-chat", "hello");
        inbound.channel = "tauri".to_string();
        inbound.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("caller-run-123"),
        );

        agent
            .process(BusMessage::Inbound(inbound))
            .await
            .expect("process inbound");

        let event = tokio::time::timeout(Duration::from_secs(2), outbound_rx.recv())
            .await
            .expect("started lifecycle event before timeout")
            .expect("outbound event");
        assert!(matches!(
            event,
            BusMessage::RunLifecycle(RunLifecycleEvent::Started { run_id, chat_id })
                if run_id == "caller-run-123" && chat_id == "run-id-chat"
        ));

        let terminal = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(BusMessage::RunLifecycle(
                    event @ RunLifecycleEvent::Terminated { .. },
                )) = outbound_rx.recv().await
                {
                    return event;
                }
            }
        })
        .await
        .expect("terminal lifecycle event before timeout");
        assert!(matches!(
            terminal,
            RunLifecycleEvent::Terminated { run_id, chat_id, outcome: RunOutcome::Completed }
                if run_id == "caller-run-123" && chat_id == "run-id-chat"
        ));
    }

    #[tokio::test]
    async fn provider_retry_exhaustion_emits_one_typed_lifecycle_pair() {
        let (mut agent, mut outbound_rx) =
            build_agent_with_provider(Box::new(NonTransientErrorProvider));
        let mut inbound = test_inbound("provider-terminal-chat", "hello");
        inbound.channel = "tauri".to_string();
        inbound.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("provider-terminal-run"),
        );
        agent
            .process(BusMessage::Inbound(inbound))
            .await
            .expect("process inbound");

        let mut lifecycle_events = Vec::new();
        while lifecycle_events.len() < 2 {
            let event = tokio::time::timeout(Duration::from_secs(2), outbound_rx.recv())
                .await
                .expect("lifecycle event before timeout")
                .expect("outbound channel remains open");
            if let BusMessage::RunLifecycle(event) = event {
                lifecycle_events.push(event);
            }
        }

        assert!(matches!(
            lifecycle_events.as_slice(),
            [
                RunLifecycleEvent::Started { run_id, chat_id },
                RunLifecycleEvent::Terminated {
                    run_id: terminal_run_id,
                    chat_id: terminal_chat_id,
                    outcome: RunOutcome::Failed {
                        failure: RunFailureKind::ProviderRetriesExhausted,
                        retryable: true,
                    },
                },
            ] if run_id == "provider-terminal-run"
                && chat_id == "provider-terminal-chat"
                && terminal_run_id == run_id
                && terminal_chat_id == chat_id
        ));

        let extra_lifecycle = tokio::time::timeout(Duration::from_millis(100), async {
            loop {
                match outbound_rx.recv().await {
                    Some(BusMessage::RunLifecycle(event)) => return Some(event),
                    Some(_) => continue,
                    None => return None,
                }
            }
        })
        .await
        .ok()
        .flatten();
        assert!(
            extra_lifecycle.is_none(),
            "only one lifecycle pair is emitted"
        );
    }

    #[tokio::test]
    async fn repeated_root_cause_emits_warning_then_one_stuck_terminal() {
        let (mut agent, mut outbound_rx) =
            build_agent_with_provider(Box::new(IdenticalToolCallProvider));
        let mut inbound = test_inbound("budget-warning-chat", "hello");
        inbound.channel = "tauri".to_string();
        inbound.metadata.insert(
            METADATA_RUN_ID.to_string(),
            serde_json::json!("budget-warning-run"),
        );
        agent
            .process(BusMessage::Inbound(inbound))
            .await
            .expect("process inbound");

        let mut lifecycle_events = Vec::new();
        while lifecycle_events.len() < 3 {
            let event = tokio::time::timeout(Duration::from_secs(2), outbound_rx.recv())
                .await
                .expect("lifecycle event before timeout")
                .expect("outbound channel remains open");
            if let BusMessage::RunLifecycle(event) = event {
                lifecycle_events.push(event);
            }
        }

        assert!(matches!(
            lifecycle_events.as_slice(),
            [
                RunLifecycleEvent::Started { run_id, chat_id },
                RunLifecycleEvent::Warning {
                    run_id: warning_run_id,
                    chat_id: warning_chat_id,
                    warning: crate::bus::RunBudgetWarning {
                        reason: crate::bus::RunBudgetWarningReason::RepeatedRootCause {
                            failures: 2
                        },
                        ..
                    },
                },
                RunLifecycleEvent::Terminated {
                    run_id: terminal_run_id,
                    chat_id: terminal_chat_id,
                    outcome: RunOutcome::Stuck {
                        reason: RunStuckReason::RepeatedRootCause,
                    },
                },
            ] if run_id == "budget-warning-run"
                && chat_id == "budget-warning-chat"
                && warning_run_id == run_id
                && warning_chat_id == chat_id
                && terminal_run_id == run_id
                && terminal_chat_id == chat_id
        ));

        let extra_terminal = tokio::time::timeout(Duration::from_millis(100), async {
            loop {
                match outbound_rx.recv().await {
                    Some(BusMessage::RunLifecycle(
                        event @ RunLifecycleEvent::Terminated { .. },
                    )) => return Some(event),
                    Some(_) => continue,
                    None => return None,
                }
            }
        })
        .await
        .ok()
        .flatten();
        assert!(
            extra_terminal.is_none(),
            "only one terminal event is emitted"
        );
    }

    #[test]
    fn typed_terminal_exits_preserve_budget_and_doom_loop_outcomes() {
        let budget = ReasoningLoopExit::BudgetExhausted {
            assistant_text: "any localized assistant text".to_string(),
            budget: RunBudgetSnapshot {
                iterations_used: 7,
                iterations_limit: 7,
                ..RunBudgetSnapshot::default()
            },
        }
        .lifecycle_outcome();
        assert!(matches!(
            budget,
            RunOutcome::BudgetExhausted { budget }
                if budget.iterations_used == 7 && budget.iterations_limit == 7
        ));

        let stuck = ReasoningLoopExit::Stuck {
            assistant_text: "unrelated assistant text".to_string(),
            reason: RunStuckReason::DoomLoop,
        }
        .lifecycle_outcome();
        assert_eq!(
            stuck,
            RunOutcome::Stuck {
                reason: RunStuckReason::DoomLoop,
            }
        );
    }

    #[tokio::test]
    async fn inbound_queues_while_reasoning_active_second_chat_after_first() {
        let (unblock_tx, unblock_rx) = tokio::sync::oneshot::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let prov = GateFirstChatProvider {
            calls: calls.clone(),
            first_unblock: Arc::new(tokio::sync::Mutex::new(Some(unblock_rx))),
        };
        let (mut agent, _outbound_rx) = build_agent_with_provider(Box::new(prov));
        let cid = "queue-seq-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "first")))
            .await
            .expect("process");
        for _ in 0..200 {
            if calls.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "first reasoning should call provider.chat"
        );
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "second")))
            .await
            .expect("process");
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "second inbound must be queued, not start a concurrent provider.chat"
        );
        let _ = unblock_tx.send(());
        for _ in 0..400 {
            if calls.load(Ordering::SeqCst) >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "after first turn completes, queued inbound should run"
        );
    }

    #[tokio::test]
    async fn cancel_clears_pending_inbound_second_provider_chat_never_starts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let prov = LongSleepProvider {
            calls: calls.clone(),
        };
        let (mut agent, _outbound_rx) = build_agent_with_provider(Box::new(prov));
        let cid = "cancel-q-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "first")))
            .await
            .expect("process");
        for _ in 0..200 {
            if calls.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "second")))
            .await
            .expect("process");
        agent
            .process(BusMessage::Cancel(cid.to_string()))
            .await
            .expect("cancel");
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "queued follow-up must not run provider.chat after Cancel cleared the queue"
        );
    }

    #[tokio::test]
    async fn inbound_after_cancel_waits_for_old_terminal_before_new_start() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (mut agent, mut outbound_rx) =
            build_agent_with_provider(Box::new(LongSleepProvider { calls }));
        let chat_id = "cancel-serialization-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start first run");

        let first_run_id = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Started {
                    run_id,
                    chat_id: event_chat,
                })) if event_chat == chat_id => break run_id,
                Some(_) => continue,
                None => panic!("outbound channel closed before first start"),
            }
        };

        agent
            .process(BusMessage::Cancel(chat_id.to_string()))
            .await
            .expect("cancel accepted");
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "second")))
            .await
            .expect("queue second run while cancellation unwinds");

        let first_after_cancel = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(event)) => break event,
                Some(_) => continue,
                None => panic!("outbound channel closed during cancellation"),
            }
        };
        assert!(matches!(
            first_after_cancel,
            RunLifecycleEvent::Terminated {
                run_id,
                chat_id: event_chat,
                outcome: RunOutcome::Cancelled,
            } if run_id == first_run_id && event_chat == chat_id
        ));

        let second_start = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(event @ RunLifecycleEvent::Started { .. })) => {
                    break event;
                }
                Some(_) => continue,
                None => panic!("outbound channel closed before second start"),
            }
        };
        assert!(matches!(
            second_start,
            RunLifecycleEvent::Started { run_id, chat_id: event_chat }
                if run_id != first_run_id && event_chat == chat_id
        ));
    }

    #[tokio::test]
    async fn exact_cancel_does_not_interrupt_a_different_run() {
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(LongSleepProvider {
            calls: Arc::new(AtomicUsize::new(0)),
        }));
        let chat_id = "exact-cancel-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start run");
        let run_id = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Started {
                    run_id,
                    chat_id: event_chat,
                })) if event_chat == chat_id => break run_id,
                Some(_) => continue,
                None => panic!("outbound channel closed before start"),
            }
        };

        agent
            .process(BusMessage::CancelRun {
                chat_id: chat_id.to_string(),
                run_id: "wrong-run".to_string(),
            })
            .await
            .expect("wrong cancel is handled");
        assert!(
            tokio::time::timeout(Duration::from_millis(50), async {
                loop {
                    match outbound_rx.recv().await {
                        Some(BusMessage::RunLifecycle(RunLifecycleEvent::Terminated {
                            ..
                        })) => {
                            return;
                        }
                        Some(_) => continue,
                        None => return,
                    }
                }
            })
            .await
            .is_err(),
            "a mismatched cancel must not terminate the active run"
        );

        agent
            .process(BusMessage::CancelRun {
                chat_id: chat_id.to_string(),
                run_id: run_id.clone(),
            })
            .await
            .expect("exact cancel is accepted");
        let terminal = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(event @ RunLifecycleEvent::Terminated { .. })) => {
                    break event;
                }
                Some(_) => continue,
                None => panic!("outbound channel closed before terminal"),
            }
        };
        assert!(matches!(
            terminal,
            RunLifecycleEvent::Terminated {
                run_id: terminal_run_id,
                chat_id: event_chat,
                outcome: RunOutcome::Cancelled,
            } if terminal_run_id == run_id && event_chat == chat_id
        ));
    }

    #[tokio::test]
    async fn steer_is_accepted_only_for_the_exact_active_run() {
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(LongSleepProvider {
            calls: Arc::new(AtomicUsize::new(0)),
        }));
        let chat_id = "steer-run-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(chat_id, "first")))
            .await
            .expect("start run");
        let run_id = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Started { run_id, .. })) => {
                    break run_id
                }
                Some(_) => continue,
                None => panic!("outbound channel closed before start"),
            }
        };

        agent
            .process(BusMessage::Steer {
                chat_id: chat_id.to_string(),
                run_id: "stale-run".to_string(),
                content: "ignore this".to_string(),
            })
            .await
            .expect("stale steer is handled");
        {
            let active = agent.cancellation_tokens.get(chat_id).expect("active run");
            assert!(steering_guard(&active.steering).pending.is_empty());
        }

        agent
            .process(BusMessage::Steer {
                chat_id: chat_id.to_string(),
                run_id: run_id.clone(),
                content: "change direction".to_string(),
            })
            .await
            .expect("exact steer is handled");
        {
            let active = agent.cancellation_tokens.get(chat_id).expect("active run");
            let inbox = steering_guard(&active.steering);
            assert_eq!(
                inbox.pending.front().map(String::as_str),
                Some("change direction")
            );
        }

        agent
            .process(BusMessage::CancelRun {
                chat_id: chat_id.to_string(),
                run_id,
            })
            .await
            .expect("cancel test run");
    }

    #[test]
    fn steering_final_boundary_is_atomic_and_never_leaks_to_a_later_run() {
        let mut inbox = SteeringInbox::open();
        assert!(inbox.push("first revision".to_string()));
        assert_eq!(inbox.close_or_drain(), vec!["first revision"]);
        assert!(inbox.accepting, "draining a revision keeps this run open");

        assert!(inbox.close_or_drain().is_empty());
        assert!(!inbox.accepting, "empty final boundary closes acceptance");
        assert!(!inbox.push("too late".to_string()));
        assert!(inbox.pending.is_empty());
    }

    #[tokio::test]
    async fn steering_at_provider_boundary_is_persisted_before_the_next_response() {
        let (unblock_tx, unblock_rx) = tokio::sync::oneshot::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = GateFirstChatProvider {
            calls: calls.clone(),
            first_unblock: Arc::new(tokio::sync::Mutex::new(Some(unblock_rx))),
        };
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(provider));
        let chat_id = "steer-provider-boundary";
        agent
            .process(BusMessage::Inbound(test_inbound(
                chat_id,
                "original request",
            )))
            .await
            .expect("start run");
        let run_id = loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Started { run_id, .. })) => {
                    break run_id
                }
                Some(_) => continue,
                None => panic!("outbound channel closed before start"),
            }
        };
        agent
            .process(BusMessage::Steer {
                chat_id: chat_id.to_string(),
                run_id,
                content: "use the revised direction".to_string(),
            })
            .await
            .expect("queue steering");
        unblock_tx
            .send(())
            .expect("release first provider response");
        loop {
            match outbound_rx.recv().await {
                Some(BusMessage::RunLifecycle(RunLifecycleEvent::Terminated { .. })) => break,
                Some(_) => continue,
                None => panic!("outbound channel closed before terminal"),
            }
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let session_key = clarification_session_key("terminal", chat_id, None);
        let session = agent
            .session_manager
            .get_session(&session_key)
            .await
            .expect("session");
        let context = session.get_context().await.expect("context");
        let text: Vec<_> = context
            .iter()
            .map(|m| {
                m.content
                    .as_ref()
                    .map(|content| content.text_content())
                    .unwrap_or_default()
            })
            .collect();
        assert!(text
            .iter()
            .any(|value| value == "use the revised direction"));
        assert!(text.iter().any(|value| value == "ok-1"));
        assert!(!text.iter().any(|value| value == "ok-0"));
    }

    #[tokio::test]
    async fn clarification_inbound_routes_via_hub_before_reasoning_spawn() {
        let hub = Arc::new(ClarificationHub::new());
        let chat_id = "clar-chat";
        let sk = clarification_session_key("terminal", chat_id, None);
        let pending_rx = hub.begin_wait(&sk).expect("begin_wait");
        let (mut agent, _outbound_rx) =
            build_agent_with_provider_and_hub(Box::new(DummyProvider), hub);
        agent
            .process(BusMessage::Inbound(test_inbound(
                chat_id,
                "clarification reply text",
            )))
            .await
            .expect("process");
        let delivered = tokio::time::timeout(Duration::from_secs(2), pending_rx)
            .await
            .expect("timeout waiting clarification")
            .expect("clarification channel closed");
        assert_eq!(delivered, "clarification reply text");
    }

    #[tokio::test]
    async fn run_reasoning_loop_persists_terminal_message_on_llm_failure() {
        let (result, context) =
            run_loop_once_for_test(Box::new(NonTransientErrorProvider), 2, false, false).await;
        assert!(matches!(
            result,
            Ok(ReasoningLoopExit::Failed {
                failure: RunFailureKind::ProviderRetriesExhausted,
                retryable: true,
                ..
            })
        ));
        let last = context.last().expect("last message");
        assert_eq!(last.role, "assistant");
        let text = last
            .content
            .as_ref()
            .map(|c| c.text_content())
            .unwrap_or_default();
        assert!(
            text.contains("This LLM is failing after 3 retries"),
            "persisted terminal failure not found: {text}"
        );
    }

    #[tokio::test]
    async fn run_reasoning_loop_persists_terminal_message_on_max_iterations() {
        let (result, context) =
            run_loop_once_for_test(Box::new(DummyProvider), 0, false, false).await;
        assert!(matches!(
            result.expect("max iterations fallback"),
            ReasoningLoopExit::BudgetExhausted { .. }
        ));
        let last = context.last().expect("last message");
        assert_eq!(last.role, "assistant");
        let text = last
            .content
            .as_ref()
            .map(|c| c.text_content())
            .unwrap_or_default();
        assert!(text.contains("exhausted its LLM-turn budget"), "{text}");
    }

    #[tokio::test]
    async fn unresolved_no_progress_warning_cannot_complete_from_prose_alone() {
        let provider = Box::new(RespondingProvider {
            tag: "premature completion".to_string(),
        });
        let (result, context) =
            run_loop_once_for_test_with_autonomy(provider, 10, false, false, true).await;
        assert!(matches!(
            result.expect("typed stuck terminal"),
            ReasoningLoopExit::Stuck {
                reason: RunStuckReason::NoProgress,
                ..
            }
        ));
        let terminal_text = context
            .last()
            .and_then(|message| message.content.as_ref())
            .map(|content| content.text_content())
            .unwrap_or_default();
        assert!(terminal_text.contains("without observable progress"));
    }

    fn invalid_argument_codes(context: &[ChatMessage]) -> Vec<String> {
        context
            .iter()
            .filter(|message| message.role == "tool")
            .filter_map(|message| message.content.as_ref())
            .filter_map(|content| serde_json::from_str::<Value>(&content.text_content()).ok())
            .filter_map(|payload| {
                payload
                    .pointer("/error/code")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    #[tokio::test]
    async fn sequential_malformed_tool_arguments_return_typed_error_without_dispatch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = MalformedToolArgumentsProvider {
            calls: calls.clone(),
            tool_names: vec!["slow_tool".to_string()],
        };
        let (result, context) = run_loop_once_for_test(Box::new(provider), 2, false, false).await;

        assert!(matches!(result, Ok(ReasoningLoopExit::Completed { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(invalid_argument_codes(&context), ["invalid_tool_arguments"]);
    }

    #[tokio::test]
    async fn parallel_malformed_tool_arguments_return_typed_errors_without_dispatch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = MalformedToolArgumentsProvider {
            calls: calls.clone(),
            // Both names are classified parallel-safe, forcing the join_all path.
            tool_names: vec!["read_file".to_string(), "list_dir".to_string()],
        };
        let (result, context) = run_loop_once_for_test(Box::new(provider), 2, false, false).await;

        assert!(matches!(result, Ok(ReasoningLoopExit::Completed { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            invalid_argument_codes(&context),
            ["invalid_tool_arguments", "invalid_tool_arguments"]
        );
    }

    // Typed root-cause detection is earlier and more specific than the legacy context-pattern
    // detector for repeated failed calls, so it must stop this trace before the emergency ceiling.
    #[tokio::test]
    async fn repeated_typed_root_cause_preempts_legacy_doom_loop() {
        // max_iterations is high; the doom escalation should terminate the run much earlier.
        let (result, _context) =
            run_loop_once_for_test(Box::new(IdenticalToolCallProvider), 50, false, true).await;
        let exit = result.expect("terminal message");
        assert!(matches!(
            &exit,
            ReasoningLoopExit::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
                ..
            }
        ));
        let msg = exit.assistant_text().expect("stuck assistant text");
        assert!(
            msg.starts_with("Stopped:") && msg.contains("typed tool failure"),
            "expected typed-root-cause stuck message, got: {msg}"
        );
        // Must NOT have run to the iteration cap.
        assert_ne!(msg, "Agent reached max reasoning iterations.");
    }

    // The typed progress controller is independent of the optional legacy doom detector.
    #[tokio::test]
    async fn repeated_typed_root_cause_does_not_depend_on_doom_detection() {
        let (result, _context) =
            run_loop_once_for_test(Box::new(IdenticalToolCallProvider), 3, false, false).await;
        assert!(matches!(
            result.expect("terminal message"),
            ReasoningLoopExit::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
                ..
            }
        ));
    }

    // Merely varying arguments is not measurable progress when every call still reaches the same
    // typed root cause. This is the historical max-iteration failure shape T17 must stop.
    #[tokio::test]
    async fn varied_arguments_do_not_mask_the_same_typed_root_cause() {
        let provider = Box::new(CorrectingProvider {
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let (result, _context) = run_loop_once_for_test(provider, 8, false, true).await;
        assert!(matches!(
            result.expect("terminal message"),
            ReasoningLoopExit::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
                ..
            }
        ));
    }

    // P1.4: the shared token estimator counts tool_call argument bytes, not just content text
    // (the compaction sites previously under-counted tool-heavy turns).
    #[test]
    fn estimate_tokens_counts_tool_call_args() {
        let mut msg = ChatMessage::assistant("");
        msg.content = None;
        msg.tool_calls = Some(vec![crate::utils::ToolCallRequest {
            id: "1".to_string(),
            tool_type: "function".to_string(),
            extra_content: None,
            function: crate::utils::ToolCallFunction {
                name: "t".to_string(),
                arguments: "x".repeat(400), // 400 bytes / 4 = 100 tokens
            },
        }]);
        assert_eq!(estimate_message_tokens(&msg), 100);
        assert_eq!(
            super::estimate_context_tokens(std::slice::from_ref(&msg)),
            100
        );
    }

    #[test]
    fn effective_context_tokens_prefers_ground_truth() {
        // No usage yet -> fall back to the estimate.
        assert_eq!(effective_context_tokens(1000, None), 1000);
        // Provider's exact count exceeds the bytes/4 under-estimate -> use the ground truth so a
        // real overflow still triggers compaction.
        assert_eq!(effective_context_tokens(1000, Some(9000)), 9000);
        // Estimate larger (e.g. messages added since the last call) -> keep the estimate.
        assert_eq!(effective_context_tokens(9000, Some(1000)), 9000);
        // A zero ground-truth never lowers the estimate.
        assert_eq!(effective_context_tokens(1000, Some(0)), 1000);
    }

    #[tokio::test]
    async fn run_reasoning_loop_persists_terminal_message_on_cancel() {
        let (result, context) =
            run_loop_once_for_test(Box::new(DummyProvider), 2, true, false).await;
        assert!(matches!(
            result.expect("cancelled run"),
            ReasoningLoopExit::Cancelled { .. }
        ));
        let last = context.last().expect("last message");
        assert_eq!(last.role, "assistant");
        let text = last
            .content
            .as_ref()
            .map(|c| c.text_content())
            .unwrap_or_default();
        assert!(
            text.contains("Request cancelled while the agent was processing this turn."),
            "persisted cancel marker missing: {text}"
        );
    }

    #[tokio::test]
    async fn panic_in_provider_is_caught_and_surfaces_channel_notice() {
        let (mut agent, mut outbound_rx) = build_agent_with_provider(Box::new(PanicProvider));
        let cid = "panic-chat";
        agent
            .process(BusMessage::Inbound(test_inbound(cid, "first")))
            .await
            .expect("process");
        let notice = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match outbound_rx.recv().await {
                    Some(BusMessage::Outbound(msg)) if msg.chat_id == cid => break msg,
                    Some(_) => continue,
                    None => panic!("outbound channel closed"),
                }
            }
        })
        .await
        .expect("timeout waiting panic notice");
        assert!(
            notice
                .content
                .contains("Internal error: reasoning loop panicked and was stopped."),
            "unexpected panic notice: {}",
            notice.content
        );
    }

    #[tokio::test]
    async fn load_skill_tool_supports_list_and_metadata() {
        let root = LocalTempDir::new();
        let skills_root = root.path().join("skills");
        let skill_dir = skills_root.join("lint_skill");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: lint_skill\ndescription: Lint helper\n---\n\ndo things\n",
        )
        .unwrap();
        let reg = Arc::new(tokio::sync::RwLock::new(SkillRegistry::new(skills_root)));
        let tool = super::LoadSkillTool {
            registry: reg.clone(),
        };
        let listed = tool
            .execute(serde_json::json!({ "action": "list" }))
            .await
            .unwrap();
        assert!(listed.contains("lint_skill"), "{}", listed);

        let meta = tool
            .execute(serde_json::json!({
                "skill_name": "lint_skill",
                "detail": "metadata"
            }))
            .await
            .unwrap();
        assert!(meta.contains("Instruction length:"));
        assert!(meta.contains("Available: true"));

        let full = tool
            .execute(serde_json::json!({ "skill_name": "lint_skill", "detail": "full" }))
            .await
            .unwrap();
        assert!(full.contains("do things"));
    }

    #[tokio::test]
    async fn load_skill_rescans_on_miss_to_pick_up_a_new_skill() {
        let root = LocalTempDir::new();
        let skills_root = root.path().join("skills");

        // One skill present when the registry is first scanned (at startup).
        let first = skills_root.join("first_skill");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::write(
            first.join("SKILL.md"),
            "---\nname: first_skill\ndescription: present at startup\n---\n\nalpha body\n",
        )
        .unwrap();

        let reg = Arc::new(tokio::sync::RwLock::new(SkillRegistry::new(
            skills_root.clone(),
        )));
        let tool = super::LoadSkillTool {
            registry: reg.clone(),
        };

        // A skill dropped into the directory AFTER the startup scan: the in-memory registry has
        // never seen it.
        let late = skills_root.join("late_skill");
        std::fs::create_dir_all(&late).unwrap();
        std::fs::write(
            late.join("SKILL.md"),
            "---\nname: late_skill\ndescription: added after scan\n---\n\nomega instructions\n",
        )
        .unwrap();

        // Without rescan-on-miss this would error "skill not found"; the on-miss rescan must pick
        // the new skill up and return its instructions without an agent restart.
        let loaded = tool
            .execute(serde_json::json!({ "skill_name": "late_skill", "detail": "full" }))
            .await
            .expect("late skill should be loadable after the on-miss rescan");
        assert!(loaded.contains("omega instructions"), "{loaded}");

        // metadata path also benefits from the rescan (covers the detail == "metadata" branch).
        let late_meta = tool
            .execute(serde_json::json!({ "skill_name": "late_skill", "detail": "metadata" }))
            .await
            .expect("late skill metadata after rescan");
        assert!(late_meta.contains("Available: true"), "{late_meta}");

        // The originally-present skill still loads.
        let alpha = tool
            .execute(serde_json::json!({ "skill_name": "first_skill", "detail": "full" }))
            .await
            .expect("first skill still loads");
        assert!(alpha.contains("alpha body"), "{alpha}");

        // A genuinely non-existent skill still errors after a rescan turns up nothing new.
        let missing = tool
            .execute(serde_json::json!({ "skill_name": "no_such_skill" }))
            .await;
        assert!(
            missing.is_err(),
            "unknown skill should still error after rescan: {missing:?}"
        );
    }
}
