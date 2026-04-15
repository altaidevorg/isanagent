use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::bus::{BusMessage, LogEvent, OutboundMessage, TelemetryEvent};
use crate::logging::LoggerHandle;
use crate::session::SessionManager;
use crate::skills::SkillRegistry;
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tools::ToolRegistry;
use crate::traits::{Memory, Provider, Tool};
use crate::{ActorError, ActorLogic};

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
    cancellation_tokens: Arc<dashmap::DashMap<String, tokio_util::sync::CancellationToken>>,
}

impl AgentLogic {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        provider: Box<dyn Provider>,
        session_manager: SessionManager,
        tools: ToolRegistry,
        skills: SkillRegistry,
        system_prompt: &str,
        max_iterations: usize,
        max_tool_output_chars: usize,
        max_recent_summaries: usize,
        short_term_threshold_turns: usize,
        short_term_threshold_tokens: usize,
        outbound_tx: mpsc::Sender<BusMessage>,
        logger_tx: LoggerHandle,
    ) -> Self {
        let mut agent = Self {
            name: name.to_string(),
            provider,
            session_manager: Arc::new(session_manager),
            tools: Arc::new(tools),
            skills: Arc::new(skills),
            system_prompt: system_prompt.to_string(),
            max_iterations,
            max_tool_output_chars,
            max_recent_summaries,
            short_term_threshold_turns,
            short_term_threshold_tokens,
            tool_execution_activity: None,
            outbound_tx,
            logger_tx,
            cancellation_tokens: Arc::new(dashmap::DashMap::new()),
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
                    let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info(
                        &self.name,
                        &format!("Cancelled reasoning loop for chat_id: {}", chat_id),
                    ).with_chat_id(&chat_id)));
                }
                return Ok(None);
            }
            BusMessage::Inbound(inbound) => {
                let chat_id = inbound.chat_id.clone();
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
                        ).with_chat_id(&chat_id)));
                }

                let cancel_token = tokio_util::sync::CancellationToken::new();
                self.cancellation_tokens.insert(chat_id.clone(), cancel_token.clone());

                // Clone necessary components for the task
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

                tokio::spawn(async move {
                    let task_chat_id = chat_id.clone();
                    let task_cancel_token = cancel_token.clone();
                    
                    let _ = logger_tx.send(BusMessage::Log(LogEvent::debug(
                        &name,
                        &format!("Spawning reasoning task for chat_id: {}", task_chat_id),
                    ).with_chat_id(&task_chat_id)));

                    let res = Self::run_reasoning_loop(
                        name.clone(),
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
                        outbound_tx.clone(),
                        logger_tx.clone(),
                        inbound,
                        task_cancel_token.clone(),
                    ).await;

                    if let Err(e) = res {
                        let _ = logger_tx.send(BusMessage::Log(LogEvent::error(
                            "AgentLogic",
                            &format!("Reasoning loop failed for chat_id {}: {}", task_chat_id, e),
                        ).with_chat_id(&task_chat_id)));
                    } else if task_cancel_token.is_cancelled() {
                        let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                            &name,
                            &format!("Reasoning task for chat_id {} finished via cancellation.", task_chat_id),
                        ).with_chat_id(&task_chat_id)));
                    } else {
                        let _ = logger_tx.send(BusMessage::Log(LogEvent::debug(
                            &name,
                            &format!("Reasoning task for chat_id {} finished successfully.", task_chat_id),
                        ).with_chat_id(&task_chat_id)));
                    }
                    
                    // Removed: cancellation_tokens.remove(&task_chat_id);
                    // We let the next Inbound or Cancel message handle removal.
                });

                Ok(None)
            }
            BusMessage::Outbound(_) | BusMessage::Telemetry(_) | BusMessage::LoggerControl(_) | BusMessage::Log(_) => {
                Ok(None)
            }
        }
    }
}

impl AgentLogic {
    #[allow(clippy::too_many_arguments)]
    async fn run_reasoning_loop(
        name: String,
        provider: Box<dyn Provider>,
        session_manager: Arc<SessionManager>,
        tools: Arc<ToolRegistry>,
        _skills: Arc<SkillRegistry>,
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
    ) -> Result<(), String> {
        let thread_part = inbound.thread_id.as_deref().unwrap_or("");
        let session_key = format!("{}:{}:{}", inbound.channel, inbound.chat_id, thread_part);

        let mut mem = session_manager
            .get_session(&session_key)
            .await?;

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
        let _ = outbound_tx.send(BusMessage::Telemetry(TelemetryEvent::AgentThought {
            chat_id: inbound.chat_id.clone(),
            thought: "I am starting to process your request...".to_string(),
        })).await;

        let thinking_strip_re = Regex::new(crate::utils::REDACTED_THINKING_STRIP_PATTERN)
            .expect("redacted thinking strip regex");

        // 2. Loop until no more tool calls or max iterations reached
        let mut iterations = 0;

        while iterations < max_iterations {
            if cancel_token.is_cancelled() {
                let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                    &name,
                    "Reasoning loop cancelled before iteration start.",
                ).with_chat_id(&inbound.chat_id)));
                return Ok(());
            }
            iterations += 1;

            let _ = logger_tx.send(BusMessage::Log(LogEvent::debug(
                &name,
                &format!("Iteration {}/{}", iterations, max_iterations),
            ).with_chat_id(&inbound.chat_id)));

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
            let system_msg =
                crate::utils::ChatMessage::system(&format!(
                    "{}\n\n{}\n\n{}",
                    system_prompt,
                    summaries_text,
                    _skills.get_capabilities_summary()
                ));
            context.insert(0, system_msg);

            let _ = logger_tx.send(BusMessage::Log(LogEvent::debug(
                &name,
                &format!("Calling provider.chat (context size: {})", context.len()),
            ).with_chat_id(&inbound.chat_id)));

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
                    if cancel_token.is_cancelled() { return Ok(()); }
                    
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

                    // Tool execution with cancellation support
                    let activity_handle = tool_execution_activity
                        .as_ref()
                        .map(|a| a.start(&inbound.chat_id, tool_name));
                    
                    let tool_result = tokio::select! {
                        res = tools.execute_tool(tool_name, args) => res,
                        _ = cancel_token.cancelled() => {
                            if let Some(handle) = activity_handle {
                                handle.stop().await;
                            }
                            return Ok(());
                        }
                    };
                    
                    if let Some(handle) = activity_handle {
                        handle.stop().await;
                    }

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
                    mem.add_message(crate::utils::ChatMessage::tool(&tool_result_text, &tc.id)).await?;
                    tool_invoked = true;
                }
            } else {
                // Add vanilla assistant response to memory
                mem.add_message(crate::utils::ChatMessage::assistant(&response_text)).await?;
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
                            let summary = val.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let key_info = val.get("key_info").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let knowledge_gaps = val.get("knowledge_gaps").and_then(|v| v.as_str()).unwrap_or("").to_string();

                            let memory_node = session_manager.get_memory_node();
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            let _ = memory_node.send_packet(crate::memory::MemoryMessage::AddSummary {
                                session_id: session_key.clone(),
                                summary,
                                key_info,
                                knowledge_gaps,
                                reply: crate::memory::SharedReply::new(tx),
                            }).await;
                            let _ = rx.await;

                            // Update metadata
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            let msg = crate::memory::MemoryMessage::GetMessagesSinceReflection {
                                session_id: session_key.clone(),
                                reply: crate::memory::SharedReply::new(tx),
                            };
                            if let Ok(_) = memory_node.send_packet(msg).await {
                                if let Ok(Ok((rows, _))) = rx.await {
                                    if let Some((last_id, _, _)) = rows.last() {
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
                "skill_name": {
                    "type": "string",
                    "description": "The exact name of the skill to load (e.g. 'tweet-author', 'code-reviewer')."
                }
            },
            "required": ["skill_name"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let skill_name = args
            .get("skill_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter 'skill_name'".to_string())?;

        self.registry.get_skill_instructions(skill_name)
    }
}
