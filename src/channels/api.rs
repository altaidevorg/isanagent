use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use dashmap::{mapref::entry::Entry, DashMap};
use log::{error, info};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;

#[path = "api_store.rs"]
mod api_store;

use crate::bus::{BusMessage, InboundMessage, LogEvent, OutboundMessage};
use crate::channels::Channel;
use crate::logging::LoggerHandle;
use api_store::{ResponseStore, StoredResponse};

const AGENT_TIMEOUT_SECS: u64 = 60;
const DEFAULT_API_USER: &str = "api_user";
const DEFAULT_RESPONSE_MODEL: &str = "agent-rs";

#[derive(Clone)]
struct ApiState {
    inbound_tx: Sender<InboundMessage>,
    pending_requests: Arc<DashMap<String, oneshot::Sender<OutboundMessage>>>,
    responses_cache: Arc<DashMap<String, StoredResponse>>,
    response_store: Arc<ResponseStore>,
    channel_name: String,
    logger_tx: LoggerHandle,
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    user: Option<String>,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesRequest {
    input: Value,
    model: Option<String>,
    previous_response_id: Option<String>,
    store: Option<bool>,
    user: Option<String>,
}

#[derive(Serialize)]
struct ResponsesResponse {
    id: String,
    object: &'static str,
    created_at: i64,
    model: String,
    status: &'static str,
    previous_response_id: Option<String>,
    output: Vec<ResponsesOutputItem>,
}

#[derive(Serialize)]
struct ResponsesOutputItem {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    content: Vec<ResponsesOutputText>,
}

#[derive(Serialize)]
struct ResponsesOutputText {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
    annotations: Vec<Value>,
}

#[derive(Serialize)]
struct ApiErrorEnvelope {
    error: ApiErrorBody,
}

#[derive(Serialize)]
struct ApiErrorBody {
    code: &'static str,
    message: String,
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiErrorEnvelope {
                error: ApiErrorBody {
                    code: self.code,
                    message: self.message,
                },
            }),
        )
            .into_response()
    }
}

pub struct ApiChannel {
    port: u16,
    pending_requests: Arc<DashMap<String, oneshot::Sender<OutboundMessage>>>,
    responses_cache: Arc<DashMap<String, StoredResponse>>,
    response_store: Arc<ResponseStore>,
    logger_tx: LoggerHandle,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    task_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl ApiChannel {
    pub fn new(
        port: u16,
        db_path: impl AsRef<Path>,
        logger_tx: LoggerHandle,
    ) -> Result<Self, String> {
        let (shutdown_tx, _) = tokio::sync::watch::channel(false);
        Ok(Self {
            port,
            pending_requests: Arc::new(DashMap::new()),
            responses_cache: Arc::new(DashMap::new()),
            response_store: Arc::new(ResponseStore::new(db_path)?),
            logger_tx,
            shutdown_tx,
            task_handle: Mutex::new(None),
        })
    }
}

#[async_trait]
impl Channel for ApiChannel {
    fn name(&self) -> &str {
        "api"
    }

    async fn start(&self, inbound_tx: Sender<InboundMessage>) -> Result<(), String> {
        let port = self.port;
        let state = ApiState {
            inbound_tx,
            pending_requests: self.pending_requests.clone(),
            responses_cache: self.responses_cache.clone(),
            response_store: self.response_store.clone(),
            channel_name: self.name().to_string(),
            logger_tx: self.logger_tx.clone(),
        };

        let logger_tx = self.logger_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
            "ApiChannel",
            "Starting API channel...",
        )));

        let handle = tokio::spawn(async move {
            let app = Router::new()
                .route("/v1/chat/completions", post(handle_chat))
                .route("/v1/responses", post(handle_responses))
                .with_state(state);

            let addr = format!("0.0.0.0:{}", port);
            let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                "ApiChannel",
                &format!("API channel listening on http://{}", addr),
            )));

            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .expect("Failed to bind API port");
            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                while shutdown_rx.changed().await.is_ok() {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
            });
            if let Err(e) = server.await {
                error!("API server crashed: {}", e);
            }
        });
        *self.task_handle.lock().unwrap() = Some(handle);

        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        info!("Stopping API channel...");
        let _ = self.shutdown_tx.send(true);
        let handle = self.task_handle.lock().unwrap().take();
        if let Some(handle) = handle {
            let _ = handle.await;
        }
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        let chat_id = &msg.chat_id;
        if let Some((_, sender)) = self.pending_requests.remove(chat_id) {
            let _ = sender.send(msg);
        } else {
            error!(
                "ApiChannel: No pending request found for chat_id: {}",
                chat_id
            );
        }
        Ok(())
    }
}

async fn handle_chat(State(state): State<ApiState>, Json(payload): Json<ChatRequest>) -> Response {
    let chat_id = uuid::Uuid::new_v4().to_string();
    let sender_id = payload.user.unwrap_or_else(|| DEFAULT_API_USER.to_string());

    match dispatch_agent_turn(&state, sender_id, chat_id, payload.message).await {
        Ok(outbound) => Json(ChatResponse {
            response: outbound.content,
        })
        .into_response(),
        Err(err) => err.into_response(),
    }
}

async fn handle_responses(
    State(state): State<ApiState>,
    Json(payload): Json<ResponsesRequest>,
) -> Response {
    let input = match normalize_responses_input(&payload.input) {
        Ok(input) => input,
        Err(message) => {
            return ApiError::new(StatusCode::BAD_REQUEST, "invalid_input", message).into_response()
        }
    };

    let store_response = payload.store.unwrap_or(true);
    let now = chrono::Utc::now().timestamp();

    let (internal_chat_id, sender_id, model, previous_response_id) =
        match payload.previous_response_id.as_deref() {
            Some(previous_response_id) => {
                let stored = if let Some(stored) = state
                    .responses_cache
                    .get(previous_response_id)
                    .map(|entry| entry.clone())
                {
                    stored
                } else {
                    match state.response_store.get(previous_response_id).await {
                        Ok(Some(stored)) => {
                            state
                                .responses_cache
                                .insert(previous_response_id.to_string(), stored.clone());
                            log_api(
                                &state.logger_tx,
                                LogEvent::debug(
                                    "ApiChannel",
                                    &format!(
                                        "Loaded response state from DB for previous_response_id {}",
                                        previous_response_id
                                    ),
                                ),
                            );
                            stored
                        }
                        Ok(None) => {
                            log_api(
                                &state.logger_tx,
                                LogEvent::warn(
                                    "ApiChannel",
                                    &format!(
                                    "Responses request referenced unknown previous_response_id {}",
                                    previous_response_id
                                ),
                                ),
                            );
                            return ApiError::new(
                                StatusCode::NOT_FOUND,
                                "previous_response_not_found",
                                format!("Unknown previous_response_id: {}", previous_response_id),
                            )
                            .into_response();
                        }
                        Err(e) => {
                            log_api(
                                &state.logger_tx,
                                LogEvent::error(
                                    "ApiChannel",
                                    &format!(
                                        "Failed to load response state for {}: {}",
                                        previous_response_id, e
                                    ),
                                ),
                            );
                            return ApiError::new(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "response_store_unavailable",
                                "Failed to load response state.",
                            )
                            .into_response();
                        }
                    }
                };

                (
                    stored.internal_chat_id,
                    payload.user.unwrap_or(stored.sender_id),
                    payload.model.unwrap_or(stored.model),
                    Some(previous_response_id.to_string()),
                )
            }
            None => (
                uuid::Uuid::new_v4().to_string(),
                payload.user.unwrap_or_else(|| DEFAULT_API_USER.to_string()),
                payload
                    .model
                    .unwrap_or_else(|| DEFAULT_RESPONSE_MODEL.to_string()),
                None,
            ),
        };

    let outbound =
        match dispatch_agent_turn(&state, sender_id.clone(), internal_chat_id.clone(), input).await
        {
            Ok(outbound) => outbound,
            Err(err) => return err.into_response(),
        };

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    if store_response {
        let stored = StoredResponse {
            internal_chat_id: internal_chat_id.clone(),
            sender_id,
            model: model.clone(),
        };
        if let Err(e) = state
            .response_store
            .insert(&response_id, previous_response_id.as_deref(), &stored, now)
            .await
        {
            log_api(
                &state.logger_tx,
                LogEvent::error(
                    "ApiChannel",
                    &format!("Failed to persist response {}: {}", response_id, e),
                )
                .with_chat_id(&internal_chat_id),
            );
            return ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "response_store_unavailable",
                "Failed to persist response state.",
            )
            .into_response();
        }
        state.responses_cache.insert(response_id.clone(), stored);
    }

    log_api(
        &state.logger_tx,
        LogEvent::info(
            "ApiChannel",
            &format!(
                "Responses request completed with response_id {}",
                response_id
            ),
        )
        .with_chat_id(&internal_chat_id),
    );

    Json(ResponsesResponse {
        id: response_id,
        object: "response",
        created_at: now,
        model,
        status: "completed",
        previous_response_id,
        output: vec![ResponsesOutputItem {
            id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            kind: "message",
            role: "assistant",
            content: vec![ResponsesOutputText {
                kind: "output_text",
                text: outbound.content,
                annotations: Vec::new(),
            }],
        }],
    })
    .into_response()
}

async fn dispatch_agent_turn(
    state: &ApiState,
    sender_id: String,
    chat_id: String,
    content: String,
) -> Result<OutboundMessage, ApiError> {
    let (tx, rx) = oneshot::channel();
    match state.pending_requests.entry(chat_id.clone()) {
        Entry::Occupied(_) => {
            return Err(ApiError::new(
                StatusCode::CONFLICT,
                "conversation_busy",
                "A request is already in-flight for this conversation.",
            ));
        }
        Entry::Vacant(entry) => {
            entry.insert(tx);
        }
    }

    let msg = InboundMessage {
        channel: state.channel_name.clone(),
        sender_id,
        chat_id: chat_id.clone(),
        thread_id: None,
        content,
        metadata: Default::default(),
    };

    if let Err(e) = state.inbound_tx.send(msg).await {
        state.pending_requests.remove(&chat_id);
        log_api(
            &state.logger_tx,
            LogEvent::error(
                "ApiChannel",
                &format!("API handler failed to send inbound message: {}", e),
            )
            .with_chat_id(&chat_id),
        );
        return Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "agent_queue_unavailable",
            "Failed to enqueue request for the agent.",
        ));
    }

    match tokio::time::timeout(Duration::from_secs(AGENT_TIMEOUT_SECS), rx).await {
        Ok(Ok(outbound)) => Ok(outbound),
        Ok(Err(_)) => {
            state.pending_requests.remove(&chat_id);
            Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "agent_channel_closed",
                "Agent channel closed unexpectedly.",
            ))
        }
        Err(_) => {
            state.pending_requests.remove(&chat_id);
            log_api(
                &state.logger_tx,
                LogEvent::warn("ApiChannel", "Request timed out waiting for Agent.")
                    .with_chat_id(&chat_id),
            );
            Err(ApiError::new(
                StatusCode::GATEWAY_TIMEOUT,
                "agent_timeout",
                "Request timed out waiting for Agent.",
            ))
        }
    }
}

fn normalize_responses_input(input: &Value) -> Result<String, String> {
    let mut segments = Vec::new();
    collect_text_segments(input, &mut segments)?;

    let normalized = segments
        .into_iter()
        .map(|segment| segment.trim().to_string())
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    if normalized.is_empty() {
        return Err("Responses input did not contain any supported text content.".to_string());
    }

    Ok(normalized)
}

fn collect_text_segments(value: &Value, segments: &mut Vec<String>) -> Result<(), String> {
    match value {
        Value::String(text) => {
            segments.push(text.clone());
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                collect_text_segments(item, segments)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                segments.push(text.to_string());
                return Ok(());
            }

            if let Some(content) = map.get("content") {
                return collect_text_segments(content, segments);
            }

            Err("Unsupported responses input object. Expected text or content.".to_string())
        }
        _ => {
            Err("Unsupported responses input type. Expected string, array, or object.".to_string())
        }
    }
}

fn log_api(logger_tx: &LoggerHandle, event: LogEvent) {
    let _ = logger_tx.send(BusMessage::Log(event));
}
