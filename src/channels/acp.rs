use crate::acp::bridge::{classify_tool_kind, parse_acp_content_blocks};
use crate::acp::types::*;
use crate::bus::{
    BusMessage, InboundMessage, OutboundMessage, RunLifecycleEvent, RunOutcome, TelemetryEvent,
    METADATA_RUN_ID,
};
use crate::channels::Channel;
use async_trait::async_trait;
use dashmap::DashMap;
use serde_json::Value;
use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;

/// State tracked per active ACP session.
#[derive(Debug)]
pub struct AcpSessionState {
    pub session_id: String,
    pub cwd: String,
    pub additional_directories: Vec<String>,
    pub mcp_servers: Vec<AcpMcpServerConfig>,
    pub pending_prompt_reply: Arc<Mutex<Option<oneshot::Sender<SessionPromptResult>>>>,
    pub active_run_id: Arc<Mutex<Option<String>>>,
}

impl AcpSessionState {
    pub fn new(
        session_id: String,
        cwd: String,
        additional_directories: Vec<String>,
        mcp_servers: Vec<AcpMcpServerConfig>,
    ) -> Self {
        Self {
            session_id,
            cwd,
            additional_directories,
            mcp_servers,
            pending_prompt_reply: Arc::new(Mutex::new(None)),
            active_run_id: Arc::new(Mutex::new(None)),
        }
    }
}

/// ACP Channel implementing the Agent Client Protocol over stdio or custom streams.
pub struct AcpChannel {
    pub sessions: Arc<DashMap<String, Arc<AcpSessionState>>>,
    pub client_capabilities: Arc<Mutex<ClientCapabilities>>,
    pub is_running: Arc<AtomicBool>,
    pub outbound_tx: Arc<Mutex<Option<mpsc::Sender<String>>>>,
}

impl Default for AcpChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl AcpChannel {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            client_capabilities: Arc::new(Mutex::new(ClientCapabilities::default())),
            is_running: Arc::new(AtomicBool::new(false)),
            outbound_tx: Arc::new(Mutex::new(None)),
        }
    }

    /// Process a JSON-RPC line from the ACP client and generate response/notification string payloads.
    pub async fn handle_incoming_rpc(
        &self,
        line: &str,
        bus_tx: &mpsc::Sender<BusMessage>,
        raw_writer_tx: &mpsc::Sender<String>,
    ) -> Option<String> {
        let req: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                let err_resp = JsonRpcResponse::error(Value::Null, ERR_PARSE_ERROR, e.to_string());
                return serde_json::to_string(&err_resp).ok();
            }
        };

        match req.method.as_str() {
            "initialize" => {
                let params: InitializeParams = req
                    .params
                    .and_then(|p| serde_json::from_value(p).ok())
                    .unwrap_or_else(|| InitializeParams {
                        protocol_version: ACP_PROTOCOL_VERSION,
                        client_capabilities: ClientCapabilities::default(),
                        client_info: None,
                    });

                *self.client_capabilities.lock().await = params.client_capabilities;

                let result = InitializeResult {
                    protocol_version: ACP_PROTOCOL_VERSION,
                    agent_capabilities: AgentCapabilities::default(),
                    agent_info: ImplementationInfo {
                        name: "isanagent".to_string(),
                        title: Some("isanagent AI Agent".to_string()),
                        version: env!("CARGO_PKG_VERSION").to_string(),
                    },
                    auth_methods: vec![],
                };

                let resp = JsonRpcResponse::success(
                    req.id,
                    serde_json::to_value(result).unwrap_or(Value::Null),
                );
                serde_json::to_string(&resp).ok()
            }

            "session/new" => {
                let params: SessionNewParams = match req
                    .params
                    .and_then(|p| serde_json::from_value::<SessionNewParams>(p).ok())
                {
                    Some(p) => p,
                    None => {
                        let err_resp = JsonRpcResponse::error(
                            req.id,
                            ERR_INVALID_PARAMS,
                            "Invalid session/new params",
                        );
                        return serde_json::to_string(&err_resp).ok();
                    }
                };

                let session_id = format!("acp_sess_{}", Uuid::new_v4());
                let state = Arc::new(AcpSessionState::new(
                    session_id.clone(),
                    params.cwd,
                    params.additional_directories.unwrap_or_default(),
                    params.mcp_servers,
                ));

                self.sessions.insert(session_id.clone(), state);

                let result = SessionNewResult { session_id };
                let resp = JsonRpcResponse::success(
                    req.id,
                    serde_json::to_value(result).unwrap_or(Value::Null),
                );
                serde_json::to_string(&resp).ok()
            }

            "session/prompt" => {
                let params: SessionPromptParams = match req
                    .params
                    .and_then(|p| serde_json::from_value::<SessionPromptParams>(p).ok())
                {
                    Some(p) => p,
                    None => {
                        let err_resp = JsonRpcResponse::error(
                            req.id,
                            ERR_INVALID_PARAMS,
                            "Invalid session/prompt params",
                        );
                        return serde_json::to_string(&err_resp).ok();
                    }
                };

                let session = match self.sessions.get(&params.session_id) {
                    Some(s) => s.value().clone(),
                    None => {
                        let err_resp = JsonRpcResponse::error(
                            req.id,
                            ERR_INVALID_PARAMS,
                            format!("Session not found: {}", params.session_id),
                        );
                        return serde_json::to_string(&err_resp).ok();
                    }
                };

                let (text_content, attachments) = parse_acp_content_blocks(&params.prompt);
                let run_id = format!("run_acp_{}", Uuid::new_v4());

                *session.active_run_id.lock().await = Some(run_id.clone());

                let (reply_tx, reply_rx) = oneshot::channel::<SessionPromptResult>();
                *session.pending_prompt_reply.lock().await = Some(reply_tx);

                let mut metadata = std::collections::HashMap::new();
                metadata.insert(METADATA_RUN_ID.to_string(), Value::String(run_id.clone()));

                let inbound = InboundMessage {
                    channel: "acp".to_string(),
                    sender_id: "acp_client".to_string(),
                    chat_id: params.session_id.clone(),
                    thread_id: None,
                    content: text_content,
                    attachments,
                    metadata,
                };

                if let Err(e) = bus_tx.send(BusMessage::Inbound(inbound)).await {
                    let err_resp = JsonRpcResponse::error(
                        req.id,
                        ERR_INTERNAL_ERROR,
                        format!("Failed to dispatch to bus: {e}"),
                    );
                    return serde_json::to_string(&err_resp).ok();
                }

                let raw_writer_tx_clone = raw_writer_tx.clone();
                let req_id = req.id.clone();
                tokio::spawn(async move {
                    if let Ok(prompt_res) = reply_rx.await {
                        let resp = JsonRpcResponse::success(
                            req_id,
                            serde_json::to_value(prompt_res).unwrap_or(Value::Null),
                        );
                        if let Ok(line) = serde_json::to_string(&resp) {
                            let _ = raw_writer_tx_clone.send(line).await;
                        }
                    }
                });

                None
            }

            "session/cancel" => {
                let params: SessionCancelParams = match req
                    .params
                    .and_then(|p| serde_json::from_value::<SessionCancelParams>(p).ok())
                {
                    Some(p) => p,
                    None => {
                        let err_resp = JsonRpcResponse::error(
                            req.id,
                            ERR_INVALID_PARAMS,
                            "Invalid session/cancel params",
                        );
                        return serde_json::to_string(&err_resp).ok();
                    }
                };

                if let Some(session) = self.sessions.get(&params.session_id) {
                    let session = session.value();
                    let active_run = session.active_run_id.lock().await.take();
                    if let Some(run_id) = active_run {
                        let _ = bus_tx
                            .send(BusMessage::CancelRun {
                                chat_id: params.session_id.clone(),
                                run_id,
                            })
                            .await;
                    } else {
                        let _ = bus_tx
                            .send(BusMessage::Cancel(params.session_id.clone()))
                            .await;
                    }

                    if let Some(reply_tx) = session.pending_prompt_reply.lock().await.take() {
                        let _ = reply_tx.send(SessionPromptResult {
                            stop_reason: AcpStopReason::Cancelled,
                        });
                    }
                }

                let resp = JsonRpcResponse::success(req.id, serde_json::json!({}));
                serde_json::to_string(&resp).ok()
            }

            "session/close" => {
                let params: SessionCloseParams = match req
                    .params
                    .and_then(|p| serde_json::from_value::<SessionCloseParams>(p).ok())
                {
                    Some(p) => p,
                    None => {
                        let err_resp = JsonRpcResponse::error(
                            req.id,
                            ERR_INVALID_PARAMS,
                            "Invalid session/close params",
                        );
                        return serde_json::to_string(&err_resp).ok();
                    }
                };

                if let Some((_, session)) = self.sessions.remove(&params.session_id) {
                    let _ = bus_tx
                        .send(BusMessage::Cancel(params.session_id.clone()))
                        .await;

                    if let Some(reply_tx) = session.pending_prompt_reply.lock().await.take() {
                        let _ = reply_tx.send(SessionPromptResult {
                            stop_reason: AcpStopReason::Cancelled,
                        });
                    }
                }

                let resp = JsonRpcResponse::success(req.id, serde_json::json!({}));
                serde_json::to_string(&resp).ok()
            }

            "session/delete" => {
                let params: SessionDeleteParams = match req
                    .params
                    .and_then(|p| serde_json::from_value::<SessionDeleteParams>(p).ok())
                {
                    Some(p) => p,
                    None => {
                        let err_resp = JsonRpcResponse::error(
                            req.id,
                            ERR_INVALID_PARAMS,
                            "Invalid session/delete params",
                        );
                        return serde_json::to_string(&err_resp).ok();
                    }
                };

                self.sessions.remove(&params.session_id);
                let resp = JsonRpcResponse::success(req.id, serde_json::json!({}));
                serde_json::to_string(&resp).ok()
            }

            "session/load" | "session/resume" => {
                let params: SessionResumeParams = match req
                    .params
                    .and_then(|p| serde_json::from_value::<SessionResumeParams>(p).ok())
                {
                    Some(p) => p,
                    None => {
                        let err_resp = JsonRpcResponse::error(
                            req.id,
                            ERR_INVALID_PARAMS,
                            "Invalid session params",
                        );
                        return serde_json::to_string(&err_resp).ok();
                    }
                };

                if !self.sessions.contains_key(&params.session_id) {
                    let state = Arc::new(AcpSessionState::new(
                        params.session_id.clone(),
                        params.cwd,
                        params.additional_directories.unwrap_or_default(),
                        params.mcp_servers,
                    ));
                    self.sessions.insert(params.session_id, state);
                }

                let resp = JsonRpcResponse::success(req.id, serde_json::json!({}));
                serde_json::to_string(&resp).ok()
            }

            _ => {
                let err_resp = JsonRpcResponse::error(
                    req.id,
                    ERR_METHOD_NOT_FOUND,
                    format!("Method not found: {}", req.method),
                );
                serde_json::to_string(&err_resp).ok()
            }
        }
    }

    /// Forward outbound message text from AgentLogic to the active ACP prompt stream.
    pub async fn handle_outbound_msg(&self, msg: &OutboundMessage) {
        if let Some(_session) = self.sessions.get(&msg.chat_id) {
            let update_notification = JsonRpcNotification::new(
                "session/update",
                serde_json::to_value(SessionUpdateParams {
                    session_id: msg.chat_id.clone(),
                    update: AcpSessionUpdate::AgentMessageChunk {
                        message_id: None,
                        content: AcpTextContentPart::Text {
                            text: msg.content.clone(),
                        },
                    },
                })
                .unwrap_or(Value::Null),
            );

            if let Ok(line) = serde_json::to_string(&update_notification) {
                if let Some(tx) = self.outbound_tx.lock().await.as_ref() {
                    let _ = tx.send(line).await;
                }
            }
        }
    }

    /// Forward telemetry events (tool calls, usage, thoughts) to ACP session/update notifications.
    pub async fn handle_telemetry_event(&self, event: &TelemetryEvent) {
        match event {
            TelemetryEvent::AgentThought {
                chat_id, thought, ..
            } => {
                if let Some(session) = self.sessions.get(chat_id) {
                    let notification = JsonRpcNotification::new(
                        "session/update",
                        serde_json::to_value(SessionUpdateParams {
                            session_id: session.session_id.clone(),
                            update: AcpSessionUpdate::AgentMessageChunk {
                                message_id: None,
                                content: AcpTextContentPart::Text {
                                    text: format!("*Thought*: {thought}\n"),
                                },
                            },
                        })
                        .unwrap_or(Value::Null),
                    );

                    if let Ok(line) = serde_json::to_string(&notification) {
                        if let Some(tx) = self.outbound_tx.lock().await.as_ref() {
                            let _ = tx.send(line).await;
                        }
                    }
                }
            }
            TelemetryEvent::ToolCallStarted {
                chat_id,
                tool_name,
                args,
                tool_call_id,
                ..
            } => {
                if let Some(session) = self.sessions.get(chat_id) {
                    let call_id = tool_call_id
                        .clone()
                        .unwrap_or_else(|| format!("call_{}", Uuid::new_v4()));
                    let raw_input_val: Option<Value> = serde_json::from_str(args).ok();
                    let kind = classify_tool_kind(tool_name);

                    let notification = JsonRpcNotification::new(
                        "session/update",
                        serde_json::to_value(SessionUpdateParams {
                            session_id: session.session_id.clone(),
                            update: AcpSessionUpdate::ToolCall {
                                tool_call_id: call_id,
                                title: tool_name.clone(),
                                kind,
                                status: AcpToolCallStatus::InProgress,
                                raw_input: raw_input_val,
                            },
                        })
                        .unwrap_or(Value::Null),
                    );

                    if let Ok(line) = serde_json::to_string(&notification) {
                        if let Some(tx) = self.outbound_tx.lock().await.as_ref() {
                            let _ = tx.send(line).await;
                        }
                    }
                }
            }
            TelemetryEvent::ToolCallFinished {
                chat_id,
                result,
                is_error,
                tool_call_id,
                ..
            } => {
                if let Some(session) = self.sessions.get(chat_id) {
                    let call_id = tool_call_id
                        .clone()
                        .unwrap_or_else(|| format!("call_{}", Uuid::new_v4()));
                    let status = if *is_error {
                        AcpToolCallStatus::Failed
                    } else {
                        AcpToolCallStatus::Completed
                    };

                    let content_block = AcpToolCallContent::Content {
                        content: AcpContentBlock::Text {
                            text: result.clone(),
                        },
                    };

                    let notification = JsonRpcNotification::new(
                        "session/update",
                        serde_json::to_value(SessionUpdateParams {
                            session_id: session.session_id.clone(),
                            update: AcpSessionUpdate::ToolCallUpdate {
                                tool_call_id: call_id,
                                status: Some(status),
                                content: Some(vec![content_block]),
                                locations: None,
                            },
                        })
                        .unwrap_or(Value::Null),
                    );

                    if let Ok(line) = serde_json::to_string(&notification) {
                        if let Some(tx) = self.outbound_tx.lock().await.as_ref() {
                            let _ = tx.send(line).await;
                        }
                    }
                }
            }
            TelemetryEvent::AgentUsage {
                chat_id,
                total_tokens,
                prompt_tokens,
                ..
            } => {
                if let Some(session) = self.sessions.get(chat_id) {
                    let notification = JsonRpcNotification::new(
                        "session/update",
                        serde_json::to_value(SessionUpdateParams {
                            session_id: session.session_id.clone(),
                            update: AcpSessionUpdate::UsageUpdate {
                                used: *prompt_tokens as u64,
                                size: *total_tokens as u64,
                                cost: None,
                            },
                        })
                        .unwrap_or(Value::Null),
                    );

                    if let Ok(line) = serde_json::to_string(&notification) {
                        if let Some(tx) = self.outbound_tx.lock().await.as_ref() {
                            let _ = tx.send(line).await;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Handle run lifecycle termination signals to finish the pending session/prompt request.
    pub async fn handle_run_lifecycle(&self, event: &RunLifecycleEvent) {
        if let RunLifecycleEvent::Terminated {
            chat_id, outcome, ..
        } = event
        {
            if let Some(session) = self.sessions.get(chat_id) {
                let stop_reason = match outcome {
                    RunOutcome::Completed => AcpStopReason::EndTurn,
                    RunOutcome::Cancelled => AcpStopReason::Cancelled,
                    RunOutcome::BudgetExhausted { .. } => AcpStopReason::MaxTokens,
                    RunOutcome::Failed { .. } | RunOutcome::Stuck { .. } => AcpStopReason::Refusal,
                };

                if let Some(reply_tx) = session.pending_prompt_reply.lock().await.take() {
                    let _ = reply_tx.send(SessionPromptResult { stop_reason });
                }

                *session.active_run_id.lock().await = None;
            }
        }
    }
}

#[async_trait]
impl Channel for AcpChannel {
    fn name(&self) -> &str {
        "acp"
    }

    async fn start(&self, bus_tx: mpsc::Sender<BusMessage>) -> Result<(), String> {
        self.is_running.store(true, Ordering::SeqCst);
        let (raw_writer_tx, mut raw_writer_rx) = mpsc::channel::<String>(100);
        *self.outbound_tx.lock().await = Some(raw_writer_tx.clone());

        tokio::spawn(async move {
            let mut stdout = tokio::io::stdout();
            while let Some(msg) = raw_writer_rx.recv().await {
                let mut line = msg;
                line.push('\n');
                let _ = stdout.write_all(line.as_bytes()).await;
                let _ = stdout.flush().await;
            }
        });

        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin).lines();

        while self.is_running.load(Ordering::SeqCst) {
            match reader.next_line().await {
                Ok(Some(line)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    if let Some(resp_str) = self
                        .handle_incoming_rpc(trimmed, &bus_tx, &raw_writer_tx)
                        .await
                    {
                        let _ = raw_writer_tx.send(resp_str).await;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("ACP Channel stdin error: {e}");
                    break;
                }
            }
        }

        self.is_running.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        self.is_running.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        self.handle_outbound_msg(&msg).await;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
