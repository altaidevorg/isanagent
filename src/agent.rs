use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;

use crate::bus::{BusMessage, LogEvent, OutboundMessage, TelemetryEvent};
use crate::logging::LoggerHandle;
use crate::session::SessionManager;
use crate::skills::SkillRegistry;
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tools::ToolRegistry;
use crate::traits::{Memory, Provider, Tool};
use crate::{ActorError, ActorLogic};

static REDACTED_THINKING_STRIP_RE: OnceLock<Regex> = OnceLock::new();

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

    /// Helper to construct the dynamic system prompt containing the latest tool definitions
    fn build_system_prompt(&self, recent_summaries: &str) -> String {
        format!(
            "{}\n\n{}\n\n{}",
            self.system_prompt,
            recent_summaries,
            self.skills.get_capabilities_summary()
        )
    }

    async fn execute_tool_call(
        &self,
        chat_id: &str,
        tool_name: &str,
        args: Value,
    ) -> Result<String, String> {
        let activity_handle = self
            .tool_execution_activity
            .as_ref()
            .map(|tool_execution_activity| tool_execution_activity.start(chat_id, tool_name));
        let output = self.tools.execute_tool(tool_name, args).await;
        if let Some(activity_handle) = activity_handle {
            activity_handle.stop().await;
        }
        output
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
        let inbound = match packet {
            BusMessage::Inbound(msg) => msg,
            BusMessage::Outbound(_) => {
                let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info(
                    &self.name,
                    "Received OutboundMessage instead of Inbound, skipping.",
                )));
                return Ok(None);
            }
            BusMessage::Telemetry(_) => {
                let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info(
                    &self.name,
                    "Received TelemetryEvent, skipping.",
                )));
                return Ok(None);
            }
            BusMessage::LoggerControl(_) => {
                return Ok(None);
            }
            BusMessage::Log(_) => {
                // AgentLogic ignores Log events sent by others
                return Ok(None);
            }
        };

        let _ = self.logger_tx.send(BusMessage::Log(
            LogEvent::info(
                &self.name,
                &format!(
                    "Received InboundMessage from [{}] ({} chars, {} attachments)",
                    inbound.channel,
                    inbound.content.len(),
                    inbound.attachments.len(),
                ),
            )
            .with_chat_id(&inbound.chat_id),
        ));

        let thread_part = inbound.thread_id.as_deref().unwrap_or("");
        let session_key = format!("{}:{}:{}", inbound.channel, inbound.chat_id, thread_part);

        let mut mem = self
            .session_manager
            .get_session(&session_key)
            .await
            .map_err(|e| ActorError::from(e))?;

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
        mem.add_message(user_msg)
            .await
            .map_err(|e| ActorError::from(e))?;

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
                self.session_manager
                    .get_recent_summaries(&prefix, self.max_recent_summaries)
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
                crate::utils::ChatMessage::system(&self.build_system_prompt(&summaries_text));
            context.insert(0, system_msg);

            // Call Provider
            let tools_payload = Some(serde_json::json!(self.tools.list_tools()));
            let response = self
                .provider
                .chat(&context, tools_payload)
                .await
                .map_err(|e| ActorError::from(e.to_string()))?;
            let _ = self.logger_tx.send(BusMessage::Log(
                LogEvent::debug(&self.name, "Provider responded.").with_chat_id(&inbound.chat_id),
            ));

            // Log USAGE telemetry
            if let Some(usage) = &response.usage {
                let _ = self
                    .outbound_tx
                    .send(BusMessage::Telemetry(TelemetryEvent::AgentUsage {
                        chat_id: inbound.chat_id.clone(),
                        model: "llm_provider".to_string(), // we don't have model name stored in AgentLogic nicely, defaulting it
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        total_tokens: usage.total_tokens,
                    }))
                    .await;
            }

            // Emit REASONING block as telemetry
            if let Some(reasoning) = &response.reasoning_content {
                let _ = self
                    .outbound_tx
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
                mem.add_message(assistant_msg)
                    .await
                    .map_err(|e| ActorError::from(e))?;

                for tc in tool_calls {
                    let tool_name = &tc.function.name;
                    let args_str = &tc.function.arguments;
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::info(&self.name, &format!("Invoking tool: {}", tool_name))
                            .with_chat_id(&inbound.chat_id),
                    ));

                    // Emit Telemetry Tool Call
                    let args = serde_json::from_str::<serde_json::Value>(args_str)
                        .unwrap_or_else(|_| serde_json::json!({}));
                    let _ = self
                        .outbound_tx
                        .send(BusMessage::Telemetry(TelemetryEvent::ToolCall {
                            chat_id: inbound.chat_id.clone(),
                            channel: inbound.channel.clone(),
                            tool_name: tool_name.to_string(),
                            args: args_str.clone(),
                        }))
                        .await;
                    let _ = self
                        .outbound_tx
                        .send(BusMessage::Telemetry(TelemetryEvent::ToolCallStarted {
                            chat_id: inbound.chat_id.clone(),
                            tool_name: tool_name.to_string(),
                            args: args_str.clone(),
                        }))
                        .await;

                    let tool_result = match self
                        .execute_tool_call(&inbound.chat_id, tool_name, args)
                        .await
                    {
                        Ok(res) => {
                            let mut output = res;
                            if output.len() > self.max_tool_output_chars {
                                output.truncate(self.max_tool_output_chars);
                                output.push_str("\n... [TRUNCATED FOR LENGTH]");
                            }
                            output
                        }
                        Err(e) => format!("Error: {}", e),
                    };

                    // Emit Telemetry Tool Result
                    let _ = self
                        .outbound_tx
                        .send(BusMessage::Telemetry(TelemetryEvent::ToolResult {
                            chat_id: inbound.chat_id.clone(),
                            channel: inbound.channel.clone(),
                            tool_name: tool_name.to_string(),
                            result: tool_result.clone(),
                        }))
                        .await;
                    let _ = self
                        .outbound_tx
                        .send(BusMessage::Telemetry(TelemetryEvent::ToolCallFinished {
                            chat_id: inbound.chat_id.clone(),
                            tool_name: tool_name.to_string(),
                            result: tool_result.clone(),
                        }))
                        .await;

                    // Add the tool execution back as a tool role message natively
                    mem.add_message(crate::utils::ChatMessage::tool(&tool_result, &tc.id))
                        .await
                        .map_err(|e| ActorError::from(e))?;
                    tool_invoked = true;
                }
            } else {
                // Add vanilla assistant response to memory
                mem.add_message(crate::utils::ChatMessage::assistant(&response_text))
                    .await
                    .map_err(|e| ActorError::from(e))?;
            }

            if !tool_invoked {
                // Final outbound text: strip blocks matched by `REDACTED_THINKING_STRIP_PATTERN`.
                let re = REDACTED_THINKING_STRIP_RE.get_or_init(|| {
                    Regex::new(crate::utils::REDACTED_THINKING_STRIP_PATTERN)
                        .expect("redacted thinking strip regex")
                });
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
                let approx_tokens: usize = current_context
                    .iter()
                    .map(|msg| msg.content.as_ref().map_or(0, |c| c.text_content().len()) / 4)
                    .sum();

                if turns >= self.short_term_threshold_turns
                    || approx_tokens >= self.short_term_threshold_tokens
                {
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::info(&self.name, &format!("Session {} reached short-term auto-compaction threshold ({} turns, {} max; ~{} tokens, {} max)",
                          session_key, turns, self.short_term_threshold_turns, approx_tokens, self.short_term_threshold_tokens))
                        .with_chat_id(&inbound.chat_id)
                    ));

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

                    let prompt = format!(
                        "Summarize the following conversation. Extract key information, facts and any potential knowledge gaps.\n\
                        Format your response EXACTLY as a JSON object with these keys: \"summary\", \"key_info\", \"knowledge_gaps\".\n\n\
                        Conversation:\n{}", transcript
                    );

                    let summary_context = vec![crate::utils::ChatMessage::user(&prompt)];
                    if let Ok(response) = self.provider.chat(&summary_context, None).await {
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

                            let memory_node = self.session_manager.get_memory_node();

                            let (tx, rx) = tokio::sync::oneshot::channel();
                            if let Err(e) = memory_node
                                .send_packet(crate::memory::MemoryMessage::AddSummary {
                                    session_id: session_key.clone(),
                                    summary,
                                    key_info,
                                    knowledge_gaps,
                                    reply: crate::memory::SharedReply::new(tx),
                                })
                                .await
                            {
                                log::error!("Failed to send AddSummary: {}", e);
                            } else {
                                let _ = rx.await;
                            }

                            let (tx, rx) = tokio::sync::oneshot::channel();
                            if let Err(e) = memory_node
                                .send_packet(crate::memory::MemoryMessage::UpdateSessionMetadata {
                                    session_id: session_key.clone(),
                                    last_reflection_msg_id: None, // Reset because we are clearing the messages array next
                                    reply: crate::memory::SharedReply::new(tx),
                                })
                                .await
                            {
                                log::error!("Failed to send UpdateSessionMetadata: {}", e);
                            } else {
                                let _ = rx.await;
                            }

                            // Clear raw chat history for this session, allowing it to start fresh while retaining short-term DB summaries
                            if let Err(e) = mem.clear().await {
                                log::error!("Failed to clear session after summary: {}", e);
                            } else {
                                let _ = self.logger_tx.send(BusMessage::Log(
                                    LogEvent::info(
                                        &self.name,
                                        &format!(
                                            "Session {} auto-compacted and cleared successfully.",
                                            session_key
                                        ),
                                    )
                                    .with_chat_id(&inbound.chat_id),
                                ));
                            }
                        }
                    }
                }

                return Ok(Some((
                    "completion".to_string(),
                    BusMessage::Outbound(outbound),
                )));
            }
        }

        let fallback = OutboundMessage {
            channel: inbound.channel,
            chat_id: inbound.chat_id,
            thread_id: inbound.thread_id,
            content: "Agent reached max reasoning iterations.".to_string(),
            metadata: HashMap::new(),
        };
        Ok(Some((
            "completion".to_string(),
            BusMessage::Outbound(fallback),
        )))
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

#[cfg(test)]
mod tests {
    use super::AgentLogic;
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

        let agent = AgentLogic::new(
            "TestAgent",
            Box::new(DummyProvider),
            session_manager,
            tools,
            skills,
            "test system prompt",
            4,
            4_000,
            0,
            10,
            10_000,
            outbound_tx,
            logger_tx,
        );

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
}
