use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value;
use log::{info, debug};
use std::collections::HashMap;
use regex::Regex;

use crate::{ActorLogic, ActorError};
use crate::traits::{Provider, Memory, Tool};
use crate::tools::ToolRegistry;
use crate::skills::SkillRegistry;
use crate::bus::{BusMessage, OutboundMessage, TelemetryEvent};
use crate::session::SessionManager;
use tokio::sync::mpsc;

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
    outbound_tx: mpsc::Sender<BusMessage>,
}

impl AgentLogic {
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
            outbound_tx,
        };

        // Inject the skill loader tool automatically
        let skill_reg = agent.skills.clone();
        let loader_tool = LoadSkillTool { registry: skill_reg };
        let tools_mut = Arc::get_mut(&mut agent.tools).unwrap();
        tools_mut.register(Box::new(loader_tool));

        agent
    }

    /// Helper to construct the dynamic system prompt containing the latest tool definitions
    fn build_system_prompt(&self, recent_summaries: &str) -> String {
        let content = format!(
            "{}\n\n{}\n\n{}\n\nYou have access to the following tools:\n{}\n\nIf you need to use a tool, output a JSON block exactly like this:\n```json\n{{\n  \"tool\": \"tool_name\",\n  \"args\": {{\"arg1\": \"value\"}}\n}}\n```\nDo not output anything else if you are calling a tool. If you are not calling a tool, respond conversationally.", 
            self.system_prompt,
            recent_summaries,
            self.skills.get_capabilities_summary(),
            serde_json::to_string_pretty(&self.tools.list_tools()).unwrap_or_default()
        );
        content
    }
}

/// The Agent processes incoming BusMessages, updates memory based on session key, 
/// and outputs BusMessages (specifically Outbound) back to the channel.
#[async_trait]
impl ActorLogic<BusMessage> for AgentLogic {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn process(&mut self, packet: BusMessage) -> Result<Option<(String, BusMessage)>, ActorError> {
        let inbound = match packet {
            BusMessage::Inbound(msg) => msg,
            BusMessage::Outbound(_) => {
                info!("[{}] Received OutboundMessage instead of Inbound, skipping.", self.name);
                return Ok(None);
            }
            BusMessage::Telemetry(_) => {
                info!("[{}] Received TelemetryEvent, skipping.", self.name);
                return Ok(None);
            }
        };

        info!("[{}] Received from [{}]: {}", self.name, inbound.channel, inbound.content);

        let thread_part = inbound.thread_id.as_deref().unwrap_or("");
        let session_key = format!("{}:{}:{}", inbound.channel, inbound.chat_id, thread_part);

        let mut mem = self.session_manager.get_session(&session_key).await.map_err(|e| ActorError::from(e))?;

        // 1. Build runtime context and prepend to User message before adding to memory
        let thread_info = inbound.thread_id.as_deref().map(|t| format!(", thread: '{}'", t)).unwrap_or_default();
        let now = chrono::Local::now().to_rfc3339();
        let runtime_context = format!("[RUNTIME CONTEXT] Current time is {}. You are navigating and responding in channel: '{}', with chat ID: '{}'{}.\n\n", now, inbound.channel, inbound.chat_id, thread_info);
        
        let contextualized_content = format!("{}{}", runtime_context, inbound.content);

        mem.add_user_message(&contextualized_content).await.map_err(|e| ActorError::from(e))?;

        // 2. Loop until no more tool calls or max iterations reached
        let mut iterations = 0;
        let max_iterations = self.max_iterations;

        while iterations < max_iterations {
            iterations += 1;

            // Fetch context
            let mut context = mem.get_context().await.map_err(|e| ActorError::from(e))?;

            // Strip any legacy static system prompts that SQLite may have persisted
            context.retain(|msg| msg.role != "system");

            // Fetch short term memory summaries
            let prefix = format!("{}:{}", inbound.channel, inbound.chat_id);
            let summaries = if self.max_recent_summaries > 0 {
                self.session_manager.get_recent_summaries(&prefix, self.max_recent_summaries).await.unwrap_or_default()
            } else {
                Vec::new()
            };
            
            let summaries_text = if !summaries.is_empty() {
                format!("--- RECENT CONVERSATION SUMMARIES (SHORT-TERM MEMORY) ---\n{}", summaries.join("\n\n"))
            } else {
                String::new()
            };

            // Inject the latest static system prompt to the beginning of the context
            let system_msg = crate::utils::ChatMessage::system(&self.build_system_prompt(&summaries_text));
            context.insert(0, system_msg);

            // Call Provider
            let response = self.provider.chat(&context).await.map_err(|e| ActorError::from(e.to_string()))?;
            debug!("[{}] Provider responded.", self.name);

            // Log USAGE telemetry
            if let Some(usage) = &response.usage {
                let _ = self.outbound_tx.send(BusMessage::Telemetry(TelemetryEvent::AgentUsage {
                    chat_id: inbound.chat_id.clone(),
                    model: "llm_provider".to_string(), // we don't have model name stored in AgentLogic nicely, defaulting it
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens,
                    total_tokens: usage.total_tokens,
                })).await;
            }

            // Emit REASONING block as telemetry
            if let Some(reasoning) = &response.reasoning_content {
                let _ = self.outbound_tx.send(BusMessage::Telemetry(TelemetryEvent::AgentThought {
                    chat_id: inbound.chat_id.clone(),
                    thought: reasoning.clone(),
                })).await;
            }

            // Add assistant response to memory
            let response_text = response.content.clone();
            mem.add_assistant_message(&response_text).await.map_err(|e| ActorError::from(e))?;

            // Check if response contains a tool call (naive JSON parsing for simplistic tool use)
            let mut tool_invoked = false;
            
            if let Some(json_start) = response_text.find("```json") {
                if let Some(json_end) = response_text[json_start+7..].find("```") {
                    let json_str = &response_text[json_start+7..json_start+7+json_end];
                    if let Ok(value) = serde_json::from_str::<Value>(json_str) {
                        if let (Some(tool_name), Some(args)) = (value.get("tool").and_then(|v| v.as_str()), value.get("args")) {
                            info!("[{}] Invoking tool: {}", self.name, tool_name);
                            
                            // Emit Telemetry Tool Call
                            let _ = self.outbound_tx.send(BusMessage::Telemetry(TelemetryEvent::ToolCall {
                                chat_id: inbound.chat_id.clone(),
                                tool_name: tool_name.to_string(),
                                args: serde_json::to_string(args).unwrap_or_default(),
                            })).await;

                            let tool_result = match self.tools.execute_tool(tool_name, args.clone()).await {
                                Ok(res) => {
                                    let mut output = res;
                                    if output.len() > self.max_tool_output_chars {
                                        output.truncate(self.max_tool_output_chars);
                                        output.push_str("\n... [TRUNCATED FOR LENGTH]");
                                    }
                                    format!("Tool '{}' execution result:\n{}", tool_name, output)
                                },
                                Err(e) => format!("Tool '{}' execution failed: {}", tool_name, e),
                            };

                            // Emit Telemetry Tool Result
                            let _ = self.outbound_tx.send(BusMessage::Telemetry(TelemetryEvent::ToolResult {
                                chat_id: inbound.chat_id.clone(),
                                tool_name: tool_name.to_string(),
                                result: tool_result.clone(),
                            })).await;

                            // Add the tool observation back as a user message
                            mem.add_user_message(&tool_result).await.map_err(|e| ActorError::from(e))?;
                            tool_invoked = true;
                        }
                    }
                }
            }

            if !tool_invoked {
                // Done reasoning. Strip <think>...</think> tags out of final output to user.
                let re = Regex::new(r"(?s)<think>.*?</think>\n*").unwrap();
                let clean_response = re.replace_all(&response_text, "").to_string();

                // Emit outbound response payload.
                let outbound = OutboundMessage {
                    channel: inbound.channel.clone(),
                    chat_id: inbound.chat_id.clone(),
                    thread_id: inbound.thread_id.clone(),
                    content: clean_response,
                    metadata: HashMap::new(),
                };

                // Auto-compaction check
                let current_context = mem.get_context().await.map_err(|e| ActorError::from(e))?;
                let turns = current_context.len();
                let approx_tokens: usize = current_context.iter().map(|msg| msg.content.len() / 4).sum();

                if turns >= self.short_term_threshold_turns || approx_tokens >= self.short_term_threshold_tokens {
                    info!("[{}] Session {} reached short-term auto-compaction threshold ({} turns, {} max; ~{} tokens, {} max)",
                          self.name, session_key, turns, self.short_term_threshold_turns, approx_tokens, self.short_term_threshold_tokens);
                    
                    let mut transcript = String::new();
                    for msg in &current_context {
                        if msg.role != "system" {
                            transcript.push_str(&format!("{}: {}\n\n", msg.role, msg.content));
                        }
                    }

                    let prompt = format!(
                        "Summarize the following conversation. Extract key information, facts and any potential knowledge gaps.\n\
                        Format your response EXACTLY as a JSON object with these keys: \"summary\", \"key_info\", \"knowledge_gaps\".\n\n\
                        Conversation:\n{}", transcript
                    );

                    let summary_context = vec![crate::utils::ChatMessage::user(&prompt)];
                    if let Ok(response) = self.provider.chat(&summary_context).await {
                        let text = response.content;
                        if let Some(val) = crate::utils::extract_json_from_llm_response(&text) {
                            let summary = val.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let key_info = val.get("key_info").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let knowledge_gaps = val.get("knowledge_gaps").and_then(|v| v.as_str()).unwrap_or("").to_string();

                                    let memory_node = self.session_manager.get_memory_node();

                                    let (tx, rx) = tokio::sync::oneshot::channel();
                                    if let Err(e) = memory_node.send_packet(crate::memory::MemoryMessage::AddSummary {
                                        session_id: session_key.clone(),
                                        summary,
                                        key_info,
                                        knowledge_gaps,
                                        reply: crate::memory::SharedReply::new(tx),
                                    }).await {
                                        log::error!("Failed to send AddSummary: {}", e);
                                    } else {
                                        let _ = rx.await;
                                    }

                                    let (tx, rx) = tokio::sync::oneshot::channel();
                                    if let Err(e) = memory_node.send_packet(crate::memory::MemoryMessage::UpdateSessionMetadata {
                                        session_id: session_key.clone(),
                                        last_reflection_msg_id: None, // Reset because we are clearing the messages array next
                                        reply: crate::memory::SharedReply::new(tx),
                                    }).await {
                                        log::error!("Failed to send UpdateSessionMetadata: {}", e);
                                    } else {
                                        let _ = rx.await;
                                    }

                                    // Clear raw chat history for this session, allowing it to start fresh while retaining short-term DB summaries
                                    if let Err(e) = mem.clear().await {
                                        log::error!("Failed to clear session after summary: {}", e);
                                    } else {
                                        info!("[{}] Session {} auto-compacted and cleared successfully.", self.name, session_key);
                                    }
                                }
                    }
                }

                return Ok(Some(("completion".to_string(), BusMessage::Outbound(outbound))));
            }
        }

        let fallback = OutboundMessage {
            channel: inbound.channel,
            chat_id: inbound.chat_id,
            thread_id: inbound.thread_id,
            content: "Agent reached max reasoning iterations.".to_string(),
            metadata: HashMap::new(),
        };
        Ok(Some(("completion".to_string(), BusMessage::Outbound(fallback))))
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
        let skill_name = args.get("skill_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter 'skill_name'".to_string())?;

        self.registry.get_skill_instructions(skill_name)
    }
}
