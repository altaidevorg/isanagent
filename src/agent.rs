use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;

use crate::clarification::ClarificationHub;
use crate::tool_runtime::ToolExecCtx;

use crate::bus::{BusMessage, LogEvent, OutboundMessage, TelemetryEvent};
use crate::logging::LoggerHandle;
use crate::session::SessionManager;
use crate::skills::SkillRegistry;
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tools::ToolRegistry;
use crate::traits::{Memory, Provider, Tool};
use crate::{ActorError, ActorLogic};

static REDACTED_THINKING_STRIP_RE: OnceLock<Regex> = OnceLock::new();

enum ToolExecutionFinished {
    Completed(Result<String, String>),
    Cancelled,
}

/// Session-scoped wiring for tools that need the active chat (e.g. `ask_user`).
#[derive(Clone)]
struct ToolCallRuntime {
    session: ToolExecCtx,
    hub: Arc<ClarificationHub>,
}

/// Runs a tool with optional per-call activity heartbeats and optional cooperative cancellation.
async fn execute_tool_call_with_activity(
    tools: &Arc<ToolRegistry>,
    tool_execution_activity: Option<SharedToolExecutionActivity>,
    chat_id: &str,
    tool_name: &str,
    args: Value,
    cancel_token: Option<&tokio_util::sync::CancellationToken>,
    runtime: ToolCallRuntime,
) -> ToolExecutionFinished {
    let session_key = runtime.session.session_key.clone();
    let hub = Arc::clone(&runtime.hub);
    let tool_exec_ctx = runtime.session;
    let chat_id = chat_id.to_string();
    let tool_name = tool_name.to_string();
    let tools = Arc::clone(tools);
    let cancel_owned = cancel_token.cloned();

    crate::tool_runtime::with_tool_exec_scope(tool_exec_ctx, async move {
        let activity_handle = tool_execution_activity
            .as_ref()
            .map(|a| a.start(chat_id.as_str(), tool_name.as_str()));

        let completed = match cancel_owned.as_ref() {
            None => Some(tools.execute_tool(&tool_name, args).await),
            Some(token) => {
                tokio::select! {
                    res = tools.execute_tool(&tool_name, args) => Some(res),
                    _ = token.cancelled() => None,
                }
            }
        };

        if let Some(handle) = activity_handle {
            handle.stop().await;
        }

        match completed {
            Some(res) => ToolExecutionFinished::Completed(res),
            None => {
                hub.cancel_wait(&session_key);
                ToolExecutionFinished::Cancelled
            }
        }
    })
    .await
}

/// Bundles everything needed to run one inbound reasoning task (spawned from `AgentLogic::process`).
struct ReasoningLoopCtx {
    name: String,
    provider: Box<dyn Provider>,
    session_manager: Arc<SessionManager>,
    tools: Arc<ToolRegistry>,
    skills: Arc<SkillRegistry>,
    system_prompt: String,
    max_iterations: usize,
    max_tool_output_chars: usize,
    max_recent_summaries: usize,
    short_term_threshold_turns: usize,
    short_term_threshold_tokens: usize,
    tool_execution_activity: Option<SharedToolExecutionActivity>,
    outbound_tx: mpsc::Sender<BusMessage>,
    logger_tx: LoggerHandle,
    inbound: crate::bus::InboundMessage,
    cancel_token: tokio_util::sync::CancellationToken,
    clarification_hub: Arc<ClarificationHub>,
    tool_exec_ctx: ToolExecCtx,
}

/// Constructor arguments for [`AgentLogic`], grouped to keep call sites readable.
pub struct AgentLogicParams {
    pub name: String,
    pub provider: Box<dyn Provider>,
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
}

/// The central logic for an autonomous Agent running inside an ActorNode.
/// It holds a LLM Provider, a persistent Memory context, and available Tools.
pub struct AgentLogic {
    name: String,
    provider: Box<dyn Provider>,
    session_manager: Arc<SessionManager>,
    tools: Arc<ToolRegistry>,
    skills: Arc<SkillRegistry>,
    system_prompt: String,
    max_iterations: usize,
    max_tool_output_chars: usize,
    max_recent_summaries: usize,
    short_term_threshold_turns: usize,
    short_term_threshold_tokens: usize,
    tool_execution_activity: Option<SharedToolExecutionActivity>,
    outbound_tx: mpsc::Sender<BusMessage>,
    logger_tx: LoggerHandle,
    cancellation_tokens: Arc<dashmap::DashMap<String, Arc<tokio_util::sync::CancellationToken>>>,
    clarification_hub: Arc<ClarificationHub>,
}

impl AgentLogic {
    pub fn new(params: AgentLogicParams) -> Self {
        let AgentLogicParams {
            name,
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
            outbound_tx,
            logger_tx,
            clarification_hub,
        } = params;

        let mut agent = Self {
            name,
            provider,
            session_manager: Arc::new(session_manager),
            tools: Arc::new(tools),
            skills: Arc::new(skills),
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
            clarification_hub,
        };

        // Inject the skill loader tool automatically
        let skill_reg = agent.skills.clone();
        let loader_tool = LoadSkillTool {
            registry: skill_reg,
        };
        let tools_mut = Arc::get_mut(&mut agent.tools)
            .expect("expected unique ownership of tools registry during initialization");
        tools_mut.register(Box::new(loader_tool));

        agent
    }

    pub fn with_tool_execution_activity(
        mut self,
        tool_execution_activity: SharedToolExecutionActivity,
    ) -> Self {
        self.tool_execution_activity = Some(tool_execution_activity);
        self
    }

    #[cfg(test)]
    async fn execute_tool_call(
        &self,
        chat_id: &str,
        tool_name: &str,
        args: Value,
    ) -> Result<String, String> {
        match execute_tool_call_with_activity(
            &self.tools,
            self.tool_execution_activity.clone(),
            chat_id,
            tool_name,
            args,
            None,
            ToolCallRuntime {
                session: ToolExecCtx::new("test", chat_id, None),
                hub: self.clarification_hub.clone(),
            },
        )
        .await
        {
            ToolExecutionFinished::Completed(res) => res,
            ToolExecutionFinished::Cancelled => {
                unreachable!("no cancellation token in execute_tool_call")
            }
        }
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
                if let Some((_, token)) = self.cancellation_tokens.remove(&chat_id) {
                    token.cancel();
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::info(
                            &self.name,
                            &format!("Cancelled reasoning loop for chat_id: {}", chat_id),
                        )
                        .with_chat_id(&chat_id),
                    ));
                }
                return Ok(None);
            }
            BusMessage::Inbound(inbound) => {
                let chat_id = inbound.chat_id.clone();
                let thread_part = inbound.thread_id.as_deref().unwrap_or("");
                let session_key =
                    format!("{}:{}:{}", inbound.channel, inbound.chat_id, thread_part);
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

                // 1. If there's an existing reasoning loop for this chat, cancel it first.
                // This ensures only one active reasoning task per conversation.
                if let Some((_, old_token)) = self.cancellation_tokens.remove(&chat_id) {
                    old_token.cancel();
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::debug(
                            &self.name,
                            &format!("Auto-cancelling previous task for chat_id: {}", chat_id),
                        )
                        .with_chat_id(&chat_id),
                    ));
                }

                let cancel_token = Arc::new(tokio_util::sync::CancellationToken::new());
                self.cancellation_tokens
                    .insert(chat_id.clone(), cancel_token.clone());

                // Clone necessary components for the task
                let cancellation_tokens = self.cancellation_tokens.clone();
                let name = self.name.clone();
                let provider = dyn_clone::clone_box(&*self.provider);
                let session_manager = self.session_manager.clone();
                let tools = self.tools.clone();
                let skills = self.skills.clone();
                let system_prompt = self.system_prompt.clone();
                let max_iterations = self.max_iterations;
                let max_tool_output_chars = self.max_tool_output_chars;
                let max_recent_summaries = self.max_recent_summaries;
                let short_term_threshold_turns = self.short_term_threshold_turns;
                let short_term_threshold_tokens = self.short_term_threshold_tokens;
                let tool_execution_activity = self.tool_execution_activity.clone();
                let outbound_tx = self.outbound_tx.clone();
                let logger_tx = self.logger_tx.clone();
                let clarification_hub = self.clarification_hub.clone();
                let tool_exec_ctx = ToolExecCtx::new(
                    inbound.channel.clone(),
                    inbound.chat_id.clone(),
                    inbound.thread_id.clone(),
                );
                let inbound_channel = inbound.channel.clone();

                tokio::spawn(async move {
                    let task_chat_id = chat_id.clone();
                    let task_token_arc = cancel_token.clone();

                    let agent_name = name.clone();
                    let _ = logger_tx.send(BusMessage::Log(
                        LogEvent::debug(
                            &agent_name,
                            &format!("Spawning reasoning task for chat_id: {}", task_chat_id),
                        )
                        .with_chat_id(&task_chat_id),
                    ));

                    let res = Self::run_reasoning_loop(ReasoningLoopCtx {
                        name,
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
                        tool_execution_activity,
                        outbound_tx: outbound_tx.clone(),
                        logger_tx: logger_tx.clone(),
                        inbound,
                        cancel_token: task_token_arc.as_ref().clone(),
                        clarification_hub,
                        tool_exec_ctx,
                    })
                    .await;

                    if let Err(e) = res {
                        let _ = logger_tx.send(BusMessage::Log(
                            LogEvent::error(
                                "AgentLogic",
                                &format!(
                                    "Reasoning loop failed for chat_id {}: {}",
                                    task_chat_id, e
                                ),
                            )
                            .with_chat_id(&task_chat_id),
                        ));
                        if inbound_channel == "terminal" {
                            let notice = crate::channels::terminal::build_terminal_error_notice(
                                &task_chat_id,
                                &e,
                            );
                            let _ = outbound_tx.send(BusMessage::Outbound(notice)).await;
                        }
                    } else if task_token_arc.is_cancelled() {
                        let _ = logger_tx.send(BusMessage::Log(
                            LogEvent::info(
                                &agent_name,
                                &format!(
                                    "Reasoning task for chat_id {} finished via cancellation.",
                                    task_chat_id
                                ),
                            )
                            .with_chat_id(&task_chat_id),
                        ));
                    } else {
                        let _ = logger_tx.send(BusMessage::Log(
                            LogEvent::debug(
                                &agent_name,
                                &format!(
                                    "Reasoning task for chat_id {} finished successfully.",
                                    task_chat_id
                                ),
                            )
                            .with_chat_id(&task_chat_id),
                        ));
                    }

                    // Drop our entry only if this task still owns the map slot (avoids races with a newer task).
                    let _ = cancellation_tokens.remove_if(&task_chat_id, |_key, stored| {
                        Arc::ptr_eq(stored, &task_token_arc)
                    });
                });

                Ok(None)
            }
            BusMessage::Outbound(_)
            | BusMessage::Telemetry(_)
            | BusMessage::LoggerControl(_)
            | BusMessage::Log(_) => Ok(None),
        }
    }
}

impl AgentLogic {
    async fn run_reasoning_loop(ctx: ReasoningLoopCtx) -> Result<(), String> {
        let ReasoningLoopCtx {
            name,
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
            tool_execution_activity,
            outbound_tx,
            logger_tx,
            inbound,
            cancel_token,
            clarification_hub,
            tool_exec_ctx,
        } = ctx;

        let session_key = tool_exec_ctx.session_key.clone();

        let mut mem = session_manager.get_session(&session_key).await?;

        // 1. Build runtime context and prepend to User message before adding to memory
        let thread_info = inbound
            .thread_id
            .as_deref()
            .map(|t| format!(", thread: '{}'", t))
            .unwrap_or_default();
        let now = chrono::Local::now().to_rfc3339();
        let runtime_context = format!(
            "[RUNTIME CONTEXT] Current time is {}. You are navigating and responding in channel: '{}', with chat ID: '{}'{}.",
            now,
            inbound.channel,
            inbound.chat_id,
            thread_info
        ) + crate::utils::RUNTIME_CONTEXT_END_SUFFIX;

        let contextualized_content = format!("{}{}", runtime_context, inbound.content);

        // Build the user message – multimodal when attachments are present
        let user_msg = if inbound.attachments.is_empty() {
            crate::utils::ChatMessage::user(&contextualized_content)
        } else {
            crate::utils::ChatMessage::user_multimodal(
                &contextualized_content,
                &inbound.attachments,
            )
        };
        mem.add_message(user_msg).await?;

        // Emit an initial thought so the user knows reasoning has started
        let _ = outbound_tx
            .send(BusMessage::Telemetry(TelemetryEvent::AgentThought {
                chat_id: inbound.chat_id.clone(),
                thought: "I am starting to process your request...".to_string(),
            }))
            .await;

        let thinking_strip_re = REDACTED_THINKING_STRIP_RE.get_or_init(|| {
            Regex::new(crate::utils::REDACTED_THINKING_STRIP_PATTERN)
                .expect("redacted thinking strip regex")
        });

        // 2. Loop until no more tool calls or max iterations reached
        let mut iterations = 0;

        while iterations < max_iterations {
            if cancel_token.is_cancelled() {
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::info(&name, "Reasoning loop cancelled before iteration start.")
                        .with_chat_id(&inbound.chat_id),
                ));
                return Ok(());
            }
            iterations += 1;

            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::debug(
                    &name,
                    &format!("Iteration {}/{}", iterations, max_iterations),
                )
                .with_chat_id(&inbound.chat_id),
            ));

            // Fetch context
            let mut context = mem.get_context_since_reflection().await?;

            // Strip any legacy static system prompts that SQLite may have persisted
            context.retain(|msg| msg.role != "system");

            // Fetch short term memory summaries
            let prefix = format!("{}:{}", inbound.channel, inbound.chat_id);
            let summaries = if max_recent_summaries > 0 {
                session_manager
                    .get_recent_summaries(&prefix, max_recent_summaries)
                    .await
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let summaries_text = if !summaries.is_empty() {
                format!(
                    "--- RECENT CONVERSATION SUMMARIES (SHORT-TERM MEMORY) ---\n{}",
                    summaries.join("\n\n")
                )
            } else {
                String::new()
            };

            // Inject the latest static system prompt to the beginning of the context
            let system_msg = crate::utils::ChatMessage::system(&format!(
                "{}\n\n{}\n\n{}",
                system_prompt,
                summaries_text,
                skills.get_capabilities_summary()
            ));
            context.insert(0, system_msg);

            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::debug(
                    &name,
                    &format!("Calling provider.chat (context size: {})", context.len()),
                )
                .with_chat_id(&inbound.chat_id),
            ));

            // Call Provider
            let tools_payload = Some(serde_json::json!(tools.list_tools()));

            // Call Provider with cancellation support
            let response = tokio::select! {
                res = provider.chat(&context, tools_payload) => {
                    res.map_err(|e| e.to_string())?
                }
                _ = cancel_token.cancelled() => {
                    let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                        &name,
                        "Reasoning loop cancelled during LLM call.",
                    ).with_chat_id(&inbound.chat_id)));
                    return Ok(());
                }
            };

            let _ = logger_tx.send(BusMessage::Log(
                LogEvent::debug(&name, "Provider responded.").with_chat_id(&inbound.chat_id),
            ));

            // Log USAGE telemetry
            if let Some(usage) = &response.usage {
                let _ = outbound_tx
                    .send(BusMessage::Telemetry(TelemetryEvent::AgentUsage {
                        chat_id: inbound.chat_id.clone(),
                        model: "llm_provider".to_string(),
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        total_tokens: usage.total_tokens,
                    }))
                    .await;
            }

            // Emit REASONING block as telemetry
            if let Some(reasoning) = &response.reasoning_content {
                let _ = outbound_tx
                    .send(BusMessage::Telemetry(TelemetryEvent::AgentThought {
                        chat_id: inbound.chat_id.clone(),
                        thought: reasoning.clone(),
                    }))
                    .await;
            }

            let response_text = response.content.clone();
            let mut tool_invoked = false;

            if let Some(tool_calls) = &response.tool_calls {
                // Record the assistant message that spawned the tool calls
                let assistant_msg = crate::utils::ChatMessage {
                    role: "assistant".to_string(),
                    content: if response_text.is_empty() {
                        None
                    } else {
                        Some(crate::utils::MessageContent::Text(response_text.clone()))
                    },
                    name: None,
                    tool_calls: Some(tool_calls.clone()),
                    tool_call_id: None,
                };
                mem.add_message(assistant_msg).await?;

                for tc in tool_calls {
                    if cancel_token.is_cancelled() {
                        return Ok(());
                    }

                    let tool_name = &tc.function.name;
                    let args_str = &tc.function.arguments;
                    let _ = logger_tx.send(BusMessage::Log(
                        LogEvent::info(&name, &format!("Invoking tool: {}", tool_name))
                            .with_chat_id(&inbound.chat_id),
                    ));

                    // Emit Telemetry Tool Call
                    let args = serde_json::from_str::<serde_json::Value>(args_str)
                        .unwrap_or_else(|_| serde_json::json!({}));
                    let _ = outbound_tx
                        .send(BusMessage::Telemetry(TelemetryEvent::ToolCall {
                            chat_id: inbound.chat_id.clone(),
                            channel: inbound.channel.clone(),
                            tool_name: tool_name.to_string(),
                            args: args_str.clone(),
                        }))
                        .await;
                    let _ = outbound_tx
                        .send(BusMessage::Telemetry(TelemetryEvent::ToolCallStarted {
                            chat_id: inbound.chat_id.clone(),
                            tool_name: tool_name.to_string(),
                            args: args_str.clone(),
                        }))
                        .await;

                    let tool_result = match execute_tool_call_with_activity(
                        &tools,
                        tool_execution_activity.clone(),
                        &inbound.chat_id,
                        tool_name,
                        args,
                        Some(&cancel_token),
                        ToolCallRuntime {
                            session: tool_exec_ctx.clone(),
                            hub: clarification_hub.clone(),
                        },
                    )
                    .await
                    {
                        ToolExecutionFinished::Completed(res) => res,
                        ToolExecutionFinished::Cancelled => return Ok(()),
                    };

                    let tool_result_text = match tool_result {
                        Ok(res) => {
                            let mut output = res;
                            if output.len() > max_tool_output_chars {
                                output.truncate(max_tool_output_chars);
                                output.push_str("\n... [TRUNCATED FOR LENGTH]");
                            }
                            output
                        }
                        Err(e) => format!("Error: {}", e),
                    };

                    // Emit Telemetry Tool Result
                    let _ = outbound_tx
                        .send(BusMessage::Telemetry(TelemetryEvent::ToolResult {
                            chat_id: inbound.chat_id.clone(),
                            channel: inbound.channel.clone(),
                            tool_name: tool_name.to_string(),
                            result: tool_result_text.clone(),
                        }))
                        .await;
                    let _ = outbound_tx
                        .send(BusMessage::Telemetry(TelemetryEvent::ToolCallFinished {
                            chat_id: inbound.chat_id.clone(),
                            tool_name: tool_name.to_string(),
                            result: tool_result_text.clone(),
                        }))
                        .await;

                    // Add the tool execution back as a tool role message natively
                    mem.add_message(crate::utils::ChatMessage::tool(&tool_result_text, &tc.id))
                        .await?;
                    tool_invoked = true;
                }
            } else {
                // Add vanilla assistant response to memory
                mem.add_message(crate::utils::ChatMessage::assistant(&response_text))
                    .await?;
            }

            if !tool_invoked {
                // Final outbound text
                let clean_response = thinking_strip_re
                    .replace_all(&response_text, "")
                    .to_string();

                // Emit outbound response payload.
                let outbound = OutboundMessage {
                    channel: inbound.channel.clone(),
                    chat_id: inbound.chat_id.clone(),
                    thread_id: inbound.thread_id.clone(),
                    content: clean_response,
                    metadata: HashMap::new(),
                };

                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::info(&name, "Sending final response.").with_chat_id(&inbound.chat_id),
                ));

                // Auto-compaction check
                let current_context = mem.get_context_since_reflection().await?;
                let user_turns = current_context.iter().filter(|m| m.role == "user").count();
                let approx_tokens: usize = current_context
                    .iter()
                    .map(|msg| msg.content.as_ref().map_or(0, |c| c.text_content().len()) / 4)
                    .sum();

                if user_turns >= short_term_threshold_turns
                    || approx_tokens >= short_term_threshold_tokens
                {
                    // ... (Summary generation logic - same as before but using local variables)
                    let mut transcript = String::new();
                    for msg in &current_context {
                        if msg.role != "system" {
                            if let Some(content) = &msg.content {
                                transcript.push_str(&format!("{}: {}\n\n", msg.role, content));
                            } else if let Some(_tc) = &msg.tool_calls {
                                transcript.push_str(&format!("{}: [Invoked Tools]\n\n", msg.role));
                            }
                        }
                    }

                    let existing_summary = if !summaries.is_empty() {
                        format!("\nEXISTING SUMMARY TO UPDATE:\n{}", summaries[0])
                    } else {
                        String::new()
                    };

                    let prompt = format!(
                        "Update the following conversation summary with new information from the transcript. \
                        If no existing summary is provided, create a new one. \
                        Extract key information, facts and any potential knowledge gaps.\n\
                        Format your response EXACTLY as a JSON object with these keys: \"summary\", \"key_info\", \"knowledge_gaps\".\n\n\
                        {}\n\n\
                        NEW TRANSCRIPT:\n{}", existing_summary, transcript
                    );

                    let summary_context = vec![
                        crate::utils::ChatMessage::system("You are a helpful assistant that summarizes conversations into structured JSON."),
                        crate::utils::ChatMessage::user(&prompt)
                    ];

                    let response = tokio::select! {
                        res = provider.chat(&summary_context, None) => res,
                        _ = cancel_token.cancelled() => {
                            return Ok(());
                        }
                    };

                    if let Ok(response) = response {
                        let text = response.content;
                        if let Some(val) = crate::utils::extract_json_from_llm_response(&text) {
                            let summary = val
                                .get("summary")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let key_info = val
                                .get("key_info")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let knowledge_gaps = val
                                .get("knowledge_gaps")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();

                            let memory_node = session_manager.get_memory_node();
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            let _ = memory_node
                                .send_packet(crate::memory::MemoryMessage::AddSummary {
                                    session_id: session_key.clone(),
                                    summary,
                                    key_info,
                                    knowledge_gaps,
                                    reply: crate::memory::SharedReply::new(tx),
                                })
                                .await;
                            let _ = rx.await;

                            // Update metadata
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            let msg = crate::memory::MemoryMessage::GetMessagesSinceReflection {
                                session_id: session_key.clone(),
                                reply: crate::memory::SharedReply::new(tx),
                            };
                            if let Ok(_) = memory_node.send_packet(msg).await {
                                if let Ok(Ok((rows, _))) = rx.await {
                                    if let Some((last_id, _)) = rows.last() {
                                        let (tx, rx) = tokio::sync::oneshot::channel();
                                        let _ = memory_node.send_packet(crate::memory::MemoryMessage::UpdateSessionMetadata {
                                            session_id: session_key.clone(),
                                            last_reflection_msg_id: Some(*last_id),
                                            reply: crate::memory::SharedReply::new(tx),
                                        }).await;
                                        let _ = rx.await;
                                    }
                                }
                            }
                        }
                    }
                }

                let _ = outbound_tx.send(BusMessage::Outbound(outbound)).await;
                return Ok(());
            }
        }

        let fallback = OutboundMessage {
            channel: inbound.channel,
            chat_id: inbound.chat_id,
            thread_id: inbound.thread_id,
            content: "Agent reached max reasoning iterations.".to_string(),
            metadata: HashMap::new(),
        };
        let _ = outbound_tx.send(BusMessage::Outbound(fallback)).await;
        Ok(())
    }
}

/// A built-in tool that allows the agent to load the markdown instructions
/// for a skill dynamically from the SkillRegistry.
pub struct LoadSkillTool {
    registry: Arc<SkillRegistry>,
}

#[async_trait]
impl Tool for LoadSkillTool {
    fn name(&self) -> &str {
        "load_skill_instructions"
    }

    fn description(&self) -> &str {
        "Loads the full markdown instructions for a specific Agent Skill. Use this when you need to execute a skill."
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
            return Ok(self.registry.format_skill_directory());
        }

        let skill_name = args
            .get("skill_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'skill_name' when action is load (default).".to_string())?;

        let detail = args
            .get("detail")
            .and_then(|v| v.as_str())
            .unwrap_or("full");

        if detail == "metadata" {
            return self.registry.get_skill_metadata(skill_name);
        }

        self.registry.get_skill_instructions(skill_name)
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentLogic, AgentLogicParams};
    use async_trait::async_trait;
    use axum::{
        body::Body,
        extract::State,
        http::{HeaderMap, StatusCode},
        routing::post,
        Router,
    };
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;
    use tower::util::ServiceExt;

    use crate::bus::BusMessage;
    use crate::clarification::ClarificationHub;
    use crate::logging::create_logger_channel;
    use crate::memory::SqliteMemoryActor;
    use crate::multi_tenant_edge::{ActivityHeartbeatClient, HeartbeatTransport};
    use crate::session::SessionManager;
    use crate::skills::SkillRegistry;
    use crate::tool_activity::SharedToolExecutionActivity;
    use crate::tools::ToolRegistry;
    use crate::traits::{Provider, Tool};
    use crate::utils::{ChatMessage, LLMResponse};
    use crate::NodeHandle;

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
                .header("authorization", format!("Bearer {}", token))
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
        ) -> Result<LLMResponse, crate::utils::LLMError> {
            unreachable!("DummyProvider is not used in heartbeat tests")
        }
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
        let reg = Arc::new(SkillRegistry::new(skills_root));
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
}
