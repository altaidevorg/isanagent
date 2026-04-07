use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use axum::{
    body::{Body, Bytes},
    extract::{OriginalUri, Path as AxumPath, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use dashmap::{mapref::entry::Entry, DashMap};
use log::{error, info};
use moka::policy::EvictionPolicy;
use moka::sync::Cache;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;

use crate::bus::{BusMessage, InboundMessage, LogEvent, OutboundMessage, TelemetryEvent};
use crate::channels::api_store::{ResponseStore, StoredResponse};
use crate::channels::Channel;
use crate::config::ApiConfig;
use crate::logging::LoggerHandle;
use crate::memory::{MemoryMessage, SharedReply};
use crate::utils::ChatMessage;
use crate::NodeHandle;
use crate::scheduler::{
    CronWebhookError, MultiTenantEdgeCronScheduler, PendingCronTriggerFinalize,
};
use crate::utils::{ContentPart, ImageUrl};

const AGENT_TIMEOUT_SECS: u64 = 60;
const DEFAULT_API_USER: &str = "api_user";
const DEFAULT_RESPONSE_MODEL: &str = "isanagent";
const MAX_RESPONSE_CACHE_ENTRIES: u64 = 1024;

include!(concat!(env!("OUT_DIR"), "/ui_assets.rs"));

#[derive(Clone)]
struct ApiState {
    inbound_tx: Sender<InboundMessage>,
    pending_requests: std::sync::Arc<DashMap<String, oneshot::Sender<OutboundMessage>>>,
    responses_cache: Cache<String, StoredResponse>,
    response_store: std::sync::Arc<ResponseStore>,
    mte_cron_scheduler: Option<std::sync::Arc<MultiTenantEdgeCronScheduler>>,
    channel_name: String,
    logger_tx: LoggerHandle,
    memory_node: NodeHandle<MemoryMessage>,
}

/// A parsed and normalised chat request: plain text plus optional image attachments.
struct ParsedChatInput {
    content: String,
    attachments: Vec<ContentPart>,
}

/// Request body for `POST /v1/chat/completions`.
///
/// `message` may be a plain JSON string **or** an OpenAI-compatible content-part
/// array (e.g. `[{"type":"text","text":"..."},{"type":"image_url","image_url":{...}}]`).
#[derive(Deserialize)]
struct ChatRequest {
    message: Value,
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
    /// Same key as `messages.session_id` in workspace SQLite (terminal-style session).
    internal_chat_id: String,
    output: Vec<ResponsesOutputItem>,
}

#[derive(Serialize)]
struct SessionHistoryMessage {
    role: String,
    content: String,
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
    bind_address: Option<String>,
    serve_ui: bool,
    pending_requests: std::sync::Arc<DashMap<String, oneshot::Sender<OutboundMessage>>>,
    responses_cache: Cache<String, StoredResponse>,
    response_store: std::sync::Arc<ResponseStore>,
    mte_cron_scheduler: Option<std::sync::Arc<MultiTenantEdgeCronScheduler>>,
    logger_tx: LoggerHandle,
    memory_node: NodeHandle<MemoryMessage>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    task_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl ApiChannel {
    pub fn new(
        config: ApiConfig,
        db_path: impl AsRef<Path>,
        logger_tx: LoggerHandle,
        memory_node: NodeHandle<MemoryMessage>,
    ) -> Result<Self, String> {
        let (shutdown_tx, _) = tokio::sync::watch::channel(false);
        let bind_address = config
            .bind_address
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        Ok(Self {
            port: config.port,
            bind_address,
            serve_ui: config.serve_ui.unwrap_or(false),
            pending_requests: std::sync::Arc::new(DashMap::new()),
            responses_cache: Cache::builder()
                .max_capacity(MAX_RESPONSE_CACHE_ENTRIES)
                .eviction_policy(EvictionPolicy::lru())
                .build(),
            response_store: std::sync::Arc::new(ResponseStore::new(db_path)?),
            mte_cron_scheduler: None,
            logger_tx,
            memory_node,
            shutdown_tx,
            task_handle: Mutex::new(None),
        })
    }

    pub fn with_multi_tenant_edge_cron_scheduler(
        mut self,
        mte_cron_scheduler: std::sync::Arc<MultiTenantEdgeCronScheduler>,
    ) -> Self {
        self.mte_cron_scheduler = Some(mte_cron_scheduler);
        self
    }

    fn resolved_bind_address(&self) -> String {
        if let Some(bind_address) = &self.bind_address {
            bind_address.clone()
        } else if self.serve_ui {
            "127.0.0.1".to_string()
        } else {
            "0.0.0.0".to_string()
        }
    }
}

#[async_trait]
impl Channel for ApiChannel {
    fn name(&self) -> &str {
        "api"
    }

    async fn start(&self, inbound_tx: Sender<InboundMessage>) -> Result<(), String> {
        let port = self.port;
        let serve_ui = self.serve_ui;
        let state = ApiState {
            inbound_tx,
            pending_requests: self.pending_requests.clone(),
            responses_cache: self.responses_cache.clone(),
            response_store: self.response_store.clone(),
            mte_cron_scheduler: self.mte_cron_scheduler.clone(),
            channel_name: self.name().to_string(),
            logger_tx: self.logger_tx.clone(),
            memory_node: self.memory_node.clone(),
        };

        let logger_tx = self.logger_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
            "ApiChannel",
            "Starting API channel...",
        )));
        let addr = format!("{}:{}", self.resolved_bind_address(), port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| format!("Failed to bind API port on {}: {}", addr, e))?;
        let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
            "ApiChannel",
            &format!("API channel listening on http://{}", addr),
        )));

        let handle = tokio::spawn(async move {
            let app = build_router(state, serve_ui);
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
        *self
            .task_handle
            .lock()
            .map_err(|_| "Failed to lock API channel task handle.".to_string())? = Some(handle);

        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        info!("Stopping API channel...");
        let _ = self.shutdown_tx.send(true);
        let handle = self
            .task_handle
            .lock()
            .map_err(|_| "Failed to lock API channel task handle.".to_string())?
            .take();
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

fn build_router(state: ApiState, serve_ui: bool) -> Router {
    let mut app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/v1/responses", post(handle_responses))
        .route(
            "/v1/sessions/{session_id}/messages",
            get(handle_session_messages),
        );

    if state.mte_cron_scheduler.is_some() {
        app = app.route("/_mte/cron/{job_id}/{token}", get(handle_mte_cron_webhook));
    }

    if serve_ui {
        app = app
            .route("/", get(handle_ui_index))
            .route("/assets/{*asset_path}", get(handle_ui_asset))
            .fallback(get(handle_ui_fallback));
    }

    app.with_state(state)
}

fn find_ui_asset(path: &str) -> Option<&'static EmbeddedUiAsset> {
    EMBEDDED_UI_ASSETS.iter().find(|asset| asset.path == path)
}

fn index_asset() -> Option<&'static EmbeddedUiAsset> {
    find_ui_asset("index.html")
}

fn ui_cache_control(asset: &'static EmbeddedUiAsset) -> &'static str {
    if asset.path == "index.html" {
        "no-cache"
    } else if asset.path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "public, max-age=3600"
    }
}

fn ui_asset_response(asset: &'static EmbeddedUiAsset) -> Response {
    let mut response = Response::new(Body::from(Bytes::from_static(asset.bytes)));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(asset.content_type),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(ui_cache_control(asset)),
    );
    response
}

fn is_reserved_api_path(path: &str) -> bool {
    path == "v1" || path.starts_with("v1/") || path == "_mte" || path.starts_with("_mte/")
}

async fn handle_ui_index() -> Response {
    match index_asset() {
        Some(asset) => ui_asset_response(asset),
        None => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn handle_ui_asset(AxumPath(asset_path): AxumPath<String>) -> Response {
    let asset_key = format!("assets/{}", asset_path);
    match find_ui_asset(&asset_key) {
        Some(asset) => ui_asset_response(asset),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn handle_ui_fallback(OriginalUri(uri): OriginalUri) -> Response {
    let path = uri.path().trim_start_matches('/');

    if path.is_empty() {
        return handle_ui_index().await;
    }

    if is_reserved_api_path(path) {
        return StatusCode::NOT_FOUND.into_response();
    }

    if let Some(asset) = find_ui_asset(path) {
        return ui_asset_response(asset);
    }

    if path == "assets" || path.starts_with("assets/") || path.contains('.') {
        return StatusCode::NOT_FOUND.into_response();
    }

    handle_ui_index().await
}

async fn handle_chat(State(state): State<ApiState>, Json(payload): Json<ChatRequest>) -> Response {
    let chat_id = uuid::Uuid::new_v4().to_string();
    let sender_id = payload.user.unwrap_or_else(|| DEFAULT_API_USER.to_string());

    let parsed = match parse_chat_message_value(&payload.message) {
        Ok(parsed) => parsed,
        Err(message) => {
            return ApiError::new(StatusCode::BAD_REQUEST, "invalid_input", message).into_response()
        }
    };

    match dispatch_agent_turn(&state, sender_id, chat_id, parsed).await {
        Ok(outbound) => Json(ChatResponse {
            response: outbound.content,
        })
        .into_response(),
        Err(err) => err.into_response(),
    }
}

async fn handle_mte_cron_webhook(
    AxumPath((job_id, token)): AxumPath<(String, String)>,
    State(state): State<ApiState>,
) -> StatusCode {
    let now = chrono::Utc::now();
    let Some(mte_cron_scheduler) = state.mte_cron_scheduler.as_ref() else {
        return StatusCode::NOT_FOUND;
    };

    let pending_trigger = match mte_cron_scheduler.begin_trigger(&job_id, &token, now).await {
        Ok(trigger) => trigger,
        Err(CronWebhookError::NotFound) => return StatusCode::NOT_FOUND,
        Err(CronWebhookError::Internal(error)) => {
            log_api(
                &state.logger_tx,
                LogEvent::error(
                    "ApiChannel",
                    &format!(
                        "Failed to process multi-tenant-edge cron webhook for job {}: {}",
                        job_id, error
                    ),
                ),
            );
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    };
    let trigger = pending_trigger.payload().clone();

    let metadata = HashMap::from([
        (
            "cron_job_id".to_string(),
            Value::String(trigger.job_id.clone()),
        ),
        (
            "trigger_source".to_string(),
            Value::String("multi_tenant_edge".to_string()),
        ),
    ]);
    let inbound = InboundMessage {
        channel: trigger.channel.clone(),
        sender_id: "cron".to_string(),
        chat_id: trigger.chat_id.clone(),
        thread_id: None,
        content: trigger.message.clone(),
        attachments: Vec::new(),
        metadata,
    };

    if let Err(error) = state.inbound_tx.send(inbound).await {
        if let Err(rollback_error) = pending_trigger.rollback().await {
            log_api(
                &state.logger_tx,
                LogEvent::error(
                    "ApiChannel",
                    &format!(
                        "Failed to enqueue multi-tenant-edge cron job {}: {}. Rollback/reschedule also failed: {}",
                        trigger.job_id, error, rollback_error
                    ),
                )
                .with_chat_id(&trigger.chat_id),
            );
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
        log_api(
            &state.logger_tx,
            LogEvent::error(
                "ApiChannel",
                &format!(
                    "Failed to enqueue multi-tenant-edge cron job {}: {}",
                    trigger.job_id, error
                ),
            )
            .with_chat_id(&trigger.chat_id),
        );
        return StatusCode::INTERNAL_SERVER_ERROR;
    }

    if let Err(error) = pending_trigger.mark_delivered(now.timestamp_millis()) {
        log_api(
            &state.logger_tx,
            LogEvent::error(
                "ApiChannel",
                &format!(
                    "Accepted multi-tenant-edge cron job {} into the inbound queue, but failed to mark the one-shot delivery as durable: {}",
                    trigger.job_id, error
                ),
            )
            .with_chat_id(&trigger.chat_id),
        );
        return StatusCode::INTERNAL_SERVER_ERROR;
    }

    let _ = state
        .logger_tx
        .send(BusMessage::Telemetry(TelemetryEvent::CronTrigger {
            job_id: trigger.job_id.clone(),
            message: trigger.message.clone(),
        }));

    match pending_trigger.complete().await {
        Ok(PendingCronTriggerFinalize::Completed) => {}
        Ok(PendingCronTriggerFinalize::CompletedWithWarning(error)) => {
            log_api(
                &state.logger_tx,
                LogEvent::warn(
                    "ApiChannel",
                    &format!(
                        "Accepted multi-tenant-edge cron webhook for job {}, but completion cleanup emitted a warning: {}",
                        trigger.job_id, error
                    ),
                )
                .with_chat_id(&trigger.chat_id),
            );
        }
        Err(error) => unreachable!(
            "complete() is not expected to fail after delivery is marked: {}",
            error
        ),
    }

    log_api(
        &state.logger_tx,
        LogEvent::info(
            "ApiChannel",
            &format!(
                "Accepted multi-tenant-edge cron webhook for job {}",
                trigger.job_id
            ),
        )
        .with_chat_id(&trigger.chat_id),
    );

    StatusCode::NO_CONTENT
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
                let stored = if let Some(stored) = state.responses_cache.get(previous_response_id) {
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
        internal_chat_id: internal_chat_id.clone(),
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
    input: ParsedChatInput,
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
        content: input.content,
        attachments: input.attachments,
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

/// Parses an OpenAI-compatible `message` value from a chat request.
///
/// Accepts either:
/// - A plain JSON string: `"hello"`
/// - An array of content parts: `[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"..."}}]`
fn parse_chat_message_value(value: &Value) -> Result<ParsedChatInput, String> {
    match value {
        Value::String(text) => Ok(ParsedChatInput {
            content: text.clone(),
            attachments: Vec::new(),
        }),
        Value::Array(parts) => {
            let mut text_segments: Vec<String> = Vec::new();
            let mut attachments: Vec<ContentPart> = Vec::new();
            for part in parts {
                collect_content_parts(part, &mut text_segments, &mut attachments)?;
            }
            let content = text_segments
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            if content.is_empty() && attachments.is_empty() {
                return Err("Chat message did not contain any text or image content.".to_string());
            }
            Ok(ParsedChatInput { content, attachments })
        }
        _ => Err("Chat message must be a string or an array of content parts.".to_string()),
    }
}

fn normalize_responses_input(input: &Value) -> Result<ParsedChatInput, String> {
    let mut text_segments: Vec<String> = Vec::new();
    let mut attachments: Vec<ContentPart> = Vec::new();
    collect_input_parts(input, &mut text_segments, &mut attachments)?;

    let content = text_segments
        .into_iter()
        .map(|segment| segment.trim().to_string())
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    if content.is_empty() && attachments.is_empty() {
        return Err("Responses input did not contain any text or image content.".to_string());
    }

    Ok(ParsedChatInput { content, attachments })
}

/// Collects text segments and image attachments from an OpenAI content part object or array.
fn collect_content_parts(
    value: &Value,
    text_segments: &mut Vec<String>,
    attachments: &mut Vec<ContentPart>,
) -> Result<(), String> {
    match value {
        Value::String(text) => {
            text_segments.push(text.clone());
            Ok(())
        }
        Value::Object(map) => {
            match map.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = map.get("text").and_then(Value::as_str) {
                        text_segments.push(text.to_string());
                    }
                    Ok(())
                }
                Some("image_url") => {
                    if let Some(img_obj) = map.get("image_url").and_then(Value::as_object) {
                        if let Some(url) = img_obj.get("url").and_then(Value::as_str) {
                            let detail = img_obj
                                .get("detail")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned);
                            attachments.push(ContentPart::ImageUrl {
                                image_url: ImageUrl {
                                    url: url.to_string(),
                                    detail,
                                },
                            });
                        }
                    }
                    Ok(())
                }
                _ => {
                    // Gracefully ignore unknown content part types
                    Ok(())
                }
            }
        }
        _ => Err("Unsupported content part type. Expected string or object.".to_string()),
    }
}

/// Recursively collects text and image parts from a Responses API `input` value.
fn collect_input_parts(
    value: &Value,
    text_segments: &mut Vec<String>,
    attachments: &mut Vec<ContentPart>,
) -> Result<(), String> {
    match value {
        Value::String(text) => {
            text_segments.push(text.clone());
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                collect_input_parts(item, text_segments, attachments)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            // Handle explicit content part objects (type: "text" / "image_url")
            if let Some(kind) = map.get("type").and_then(Value::as_str) {
                return collect_content_parts(value, text_segments, attachments)
                    .map_err(|_| format!("Unsupported content part type: {}", kind));
            }

            // Convenience: objects with a top-level `text` field
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                text_segments.push(text.to_string());
                return Ok(());
            }

            // Convenience: objects with a nested `content` field
            if let Some(content) = map.get("content") {
                return collect_input_parts(content, text_segments, attachments);
            }

            Err("Unsupported responses input object. Expected text or content.".to_string())
        }
        _ => {
            Err("Unsupported responses input type. Expected string, array, or object.".to_string())
        }
    }
}

fn session_history_row(message: &ChatMessage) -> SessionHistoryMessage {
    let mut content = message
        .content
        .as_ref()
        .map(|c| c.text_content())
        .unwrap_or_default();
    if content.is_empty() {
        if let Some(tool_calls) = &message.tool_calls {
            if let Ok(s) = serde_json::to_string_pretty(tool_calls) {
                content = s;
            }
        }
    }
    if content.is_empty() {
        if let Some(name) = &message.name {
            content = format!("({})", name);
        }
    }
    SessionHistoryMessage {
        role: message.role.clone(),
        content,
    }
}

/// Maps a path segment to the SQLite `messages.session_id` key.
///
/// [`AgentLogic`](crate::agent::AgentLogic) uses `format!("{}:{}:{}", channel, chat_id, thread_id)`;
/// for API messages with no thread that is `api:<uuid>:` (note trailing colon).
/// Clients pass the bare `internal_chat_id` (uuid only); we qualify it with this channel name.
fn resolve_memory_session_id<'a>(state: &ApiState, raw: &'a str) -> Cow<'a, str> {
    let s = raw.trim();
    if s.contains(':') {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(format!("{}:{}:", state.channel_name, s))
    }
}

async fn memory_get_context(
    memory_node: &NodeHandle<MemoryMessage>,
    session_id: &str,
) -> Result<Vec<ChatMessage>, String> {
    let (tx, rx) = oneshot::channel();
    let msg = MemoryMessage::GetContext {
        session_id: session_id.to_string(),
        reply: SharedReply::new(tx),
    };
    memory_node.send_packet(msg).await.map_err(|e| e.to_string())?;
    rx.await
        .map_err(|_| "Memory actor channel closed".to_string())?
}

async fn handle_session_messages(
    State(state): State<ApiState>,
    AxumPath(session_id): AxumPath<String>,
) -> Response {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_session",
            "Empty session id.",
        )
        .into_response();
    }
    let memory_session_id = resolve_memory_session_id(&state, session_id);
    match memory_get_context(&state.memory_node, memory_session_id.as_ref()).await {
        Ok(rows) => {
            let body: Vec<SessionHistoryMessage> = rows.iter().map(session_history_row).collect();
            Json(body).into_response()
        }
        Err(message) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "memory_unavailable",
            message,
        )
        .into_response(),
    }
}

fn log_api(logger_tx: &LoggerHandle, event: LogEvent) {
    let _ = logger_tx.send(BusMessage::Log(event));
}

#[cfg(test)]
mod tests {
    use super::{build_router, ApiState, EMBEDDED_UI_ASSETS};
    use crate::bus::OutboundMessage;
    use crate::channels::api_store::ResponseStore;
    use crate::config::ApiConfig;
    use crate::logging::create_logger_channel;
    use crate::memory::{MemoryMessage, SharedReply, SqliteMemoryActor};
    use crate::utils::ChatMessage;
    use crate::multi_tenant_edge::{CronRegistrationClient, CronRule, CronTransport};
    use crate::scheduler::{ActiveJob, CronStore, MultiTenantEdgeCronScheduler, ScheduleKind};
    use crate::NodeHandle;
    use async_trait::async_trait;
    use axum::body::{to_bytes, Body};
    use reqwest::StatusCode;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use tokio::sync::{mpsc, oneshot};
    use tower::ServiceExt;

    struct LocalTempDir {
        path: std::path::PathBuf,
    }

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    impl LocalTempDir {
        fn new() -> Self {
            let unique = format!(
                "isanagent-api-cron-{}-{}-{}",
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

        fn db_path(&self) -> std::path::PathBuf {
            self.path.join("agent.db")
        }
    }

    impl Drop for LocalTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    struct RecordingCronTransport {
        statuses: Arc<Mutex<Vec<StatusCode>>>,
        records: Arc<Mutex<Vec<Vec<CronRule>>>>,
    }

    #[async_trait]
    impl CronTransport for RecordingCronTransport {
        async fn put_crons(
            &self,
            _url: &str,
            _token: &str,
            cron_rules: &[CronRule],
        ) -> Result<StatusCode, String> {
            self.records.lock().unwrap().push(cron_rules.to_vec());
            Ok(self
                .statuses
                .lock()
                .unwrap()
                .pop()
                .unwrap_or(StatusCode::NO_CONTENT))
        }
    }

    fn cron_job(schedule: ScheduleKind, token: &str) -> ActiveJob {
        ActiveJob {
            id: "job-1".to_string(),
            schedule,
            message: "wake up".to_string(),
            last_run_at_ms: None,
            chat_id: "chat-123".to_string(),
            channel: "terminal".to_string(),
            webhook_token: token.to_string(),
        }
    }

    fn build_state(
        db_path: &std::path::Path,
        inbound_tx: mpsc::Sender<crate::bus::InboundMessage>,
        scheduler: Option<Arc<MultiTenantEdgeCronScheduler>>,
    ) -> ApiState {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let memory_actor =
            SqliteMemoryActor::new(db_path.to_str().expect("utf8 db path")).expect("memory actor");
        let memory_node = NodeHandle::<MemoryMessage>::new(
            memory_actor,
            100,
            1,
            Duration::from_millis(5),
        );
        ApiState {
            inbound_tx,
            pending_requests: Arc::new(dashmap::DashMap::<
                String,
                oneshot::Sender<crate::bus::OutboundMessage>,
            >::new()),
            responses_cache: moka::sync::Cache::builder().max_capacity(16).build(),
            response_store: Arc::new(ResponseStore::new(db_path).expect("response store")),
            mte_cron_scheduler: scheduler,
            channel_name: "api".to_string(),
            logger_tx,
            memory_node,
        }
    }

    #[test]
    fn api_config_supports_ui_flags() {
        let config: ApiConfig = toml::from_str(
            r#"
enabled = true
port = 8080
serve_ui = true
bind_address = "127.0.0.1"
"#,
        )
        .expect("api config parses");

        assert_eq!(config.enabled, Some(true));
        assert_eq!(config.port, 8080);
        assert_eq!(config.serve_ui, Some(true));
        assert_eq!(config.bind_address.as_deref(), Some("127.0.0.1"));
    }

    #[tokio::test]
    async fn ui_root_is_not_served_when_disabled() {
        let temp = LocalTempDir::new();
        let (inbound_tx, _inbound_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), inbound_tx, None), false);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn ui_root_serves_embedded_index_when_enabled() {
        let temp = LocalTempDir::new();
        let (inbound_tx, _inbound_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), inbound_tx, None), true);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        let status = response.status();
        let content_type = response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let cache_control = response
            .headers()
            .get(axum::http::header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let body_text = String::from_utf8(body.to_vec()).expect("utf8 html");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/html; charset=utf-8");
        assert_eq!(cache_control, "no-cache");
        assert!(body_text.contains("<div id=\"root\"></div>"));
    }

    #[tokio::test]
    async fn ui_asset_route_serves_embedded_asset_with_content_type() {
        let asset = EMBEDDED_UI_ASSETS
            .iter()
            .find(|candidate| candidate.path.starts_with("assets/"))
            .expect("built ui asset");
        let temp = LocalTempDir::new();
        let (inbound_tx, _inbound_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), inbound_tx, None), true);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri(format!("/{}", asset.path))
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        let status = response.status();
        let content_type = response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let cache_control = response
            .headers()
            .get(axum::http::header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, asset.content_type);
        assert_eq!(cache_control, "public, max-age=31536000, immutable");
        assert!(!body.is_empty());
    }

    #[tokio::test]
    async fn ui_fallback_serves_index_for_client_routes() {
        let temp = LocalTempDir::new();
        let (inbound_tx, _inbound_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), inbound_tx, None), true);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/conversations/demo")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        let content_type = response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(content_type, "text/html; charset=utf-8");
    }

    #[tokio::test]
    async fn session_messages_qualifies_bare_chat_id_with_api_channel_prefix() {
        let temp = LocalTempDir::new();
        let db_path = temp.db_path();
        let (inbound_tx, _inbound_rx) = mpsc::channel(4);
        let state = build_state(&db_path, inbound_tx, None);
        let chat_suffix = "list-me-123e4567-e89b-12d3-a456-426614174000";
        let memory_key = format!("api:{}:", chat_suffix);
        let (tx, rx) = oneshot::channel();
        state
            .memory_node
            .send_packet(MemoryMessage::AddMessage {
                session_id: memory_key,
                message: ChatMessage::user("hello from test"),
                reply: SharedReply::new(tx),
            })
            .await
            .expect("send add message");
        rx.await.expect("oneshot").expect("add ok");

        let app = build_router(state, false);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri(format!("/v1/sessions/{chat_suffix}/messages"))
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let rows: Vec<Value> = serde_json::from_slice(&body).expect("json array");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["role"], Value::String("user".to_string()));
        assert!(
            rows[0]["content"]
                .as_str()
                .expect("content string")
                .contains("hello from test")
        );
    }

    #[tokio::test]
    async fn ui_fallback_does_not_intercept_unknown_api_paths() {
        let temp = LocalTempDir::new();
        let (inbound_tx, _inbound_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), inbound_tx, None), true);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/unknown")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn responses_route_still_works_when_ui_is_enabled() {
        let temp = LocalTempDir::new();
        let pending_requests = Arc::new(dashmap::DashMap::<
            String,
            oneshot::Sender<crate::bus::OutboundMessage>,
        >::new());
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let response_store = Arc::new(ResponseStore::new(temp.db_path()).expect("response store"));
        let (inbound_tx, mut inbound_rx) = mpsc::channel(4);
        let memory_actor = SqliteMemoryActor::new(temp.db_path().to_str().expect("utf8"))
            .expect("memory actor");
        let memory_node = NodeHandle::<MemoryMessage>::new(
            memory_actor,
            100,
            1,
            Duration::from_millis(5),
        );
        let state = ApiState {
            inbound_tx,
            pending_requests: pending_requests.clone(),
            responses_cache: moka::sync::Cache::builder().max_capacity(16).build(),
            response_store,
            mte_cron_scheduler: None,
            channel_name: "api".to_string(),
            logger_tx,
            memory_node,
        };
        let app = build_router(state, true);

        tokio::spawn(async move {
            let inbound = inbound_rx.recv().await.expect("inbound message");
            let outbound = OutboundMessage {
                channel: "api".to_string(),
                chat_id: inbound.chat_id.clone(),
                thread_id: None,
                content: "UI path still reaches responses.".to_string(),
                metadata: HashMap::new(),
            };
            let (_, sender) = pending_requests
                .remove(&inbound.chat_id)
                .expect("pending request sender");
            let _ = sender.send(outbound);
        });

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/responses")
                    .method("POST")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"input":"Hello from UI","store":true}"#))
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let payload: Value = serde_json::from_slice(&body).expect("json body");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            payload.get("object"),
            Some(&Value::String("response".to_string()))
        );
        assert_eq!(
            payload["output"][0]["content"][0]["text"],
            Value::String("UI path still reaches responses.".to_string())
        );
    }

    #[tokio::test]
    async fn mte_cron_webhook_route_enqueues_saved_message_and_returns_no_content() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("cron store");
        let token = "secret-token";
        store
            .insert_job(&cron_job(
                ScheduleKind::Cron {
                    cron_expr: "0 15 9 * * *".to_string(),
                },
                token,
            ))
            .expect("insert job");

        let scheduler = Arc::new(
            MultiTenantEdgeCronScheduler::new(
                &temp.db_path().to_string_lossy(),
                CronRegistrationClient::new_with_transport(
                    "https://edge.example.com/_internal/crons".to_string(),
                    "cron-token".to_string(),
                    Arc::new(RecordingCronTransport {
                        statuses: Arc::new(Mutex::new(Vec::new())),
                        records: Arc::new(Mutex::new(Vec::new())),
                    }),
                ),
            )
            .expect("scheduler"),
        );
        let (inbound_tx, mut inbound_rx) = mpsc::channel(4);
        let app = build_router(
            build_state(&temp.db_path(), inbound_tx, Some(scheduler)),
            false,
        );

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/_mte/cron/job-1/secret-token")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        let inbound = inbound_rx.recv().await.expect("cron inbound");
        assert_eq!(inbound.channel, "terminal");
        assert_eq!(inbound.chat_id, "chat-123");
        assert_eq!(inbound.content, "wake up");
        assert_eq!(inbound.sender_id, "cron");
        assert_eq!(
            inbound.metadata.get("cron_job_id"),
            Some(&Value::String("job-1".to_string()))
        );
    }

    #[tokio::test]
    async fn mte_cron_webhook_route_returns_not_found_for_unknown_job_or_token() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("cron store");
        store
            .insert_job(&cron_job(
                ScheduleKind::Cron {
                    cron_expr: "0 15 9 * * *".to_string(),
                },
                "secret-token",
            ))
            .expect("insert job");

        let scheduler = Arc::new(
            MultiTenantEdgeCronScheduler::new(
                &temp.db_path().to_string_lossy(),
                CronRegistrationClient::new_with_transport(
                    "https://edge.example.com/_internal/crons".to_string(),
                    "cron-token".to_string(),
                    Arc::new(RecordingCronTransport {
                        statuses: Arc::new(Mutex::new(Vec::new())),
                        records: Arc::new(Mutex::new(Vec::new())),
                    }),
                ),
            )
            .expect("scheduler"),
        );
        let (inbound_tx, mut inbound_rx) = mpsc::channel(4);
        let app = build_router(
            build_state(&temp.db_path(), inbound_tx, Some(scheduler)),
            false,
        );

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/_mte/cron/job-1/wrong-token")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(inbound_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn mte_cron_webhook_route_removes_one_shot_jobs_after_first_trigger() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("cron store");
        let token = "one-shot-token";
        store
            .insert_job(&cron_job(
                ScheduleKind::At {
                    at_ms: chrono::Utc::now().timestamp_millis() + 60_000,
                },
                token,
            ))
            .expect("insert job");

        let sync_records = Arc::new(Mutex::new(Vec::new()));
        let scheduler = Arc::new(
            MultiTenantEdgeCronScheduler::new(
                &temp.db_path().to_string_lossy(),
                CronRegistrationClient::new_with_transport(
                    "https://edge.example.com/_internal/crons".to_string(),
                    "cron-token".to_string(),
                    Arc::new(RecordingCronTransport {
                        statuses: Arc::new(Mutex::new(Vec::new())),
                        records: sync_records.clone(),
                    }),
                ),
            )
            .expect("scheduler"),
        );
        let (inbound_tx, _inbound_rx) = mpsc::channel(4);
        let app = build_router(
            build_state(&temp.db_path(), inbound_tx, Some(scheduler)),
            false,
        );

        let first = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/_mte/cron/job-1/one-shot-token")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("first request succeeds");
        let second = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/_mte/cron/job-1/one-shot-token")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("second request succeeds");

        assert_eq!(first.status(), StatusCode::NO_CONTENT);
        assert_eq!(second.status(), StatusCode::NOT_FOUND);
        assert!(store.load_jobs().expect("remaining jobs").is_empty());
        assert_eq!(sync_records.lock().unwrap().len(), 1);
        assert!(sync_records.lock().unwrap()[0].is_empty());
    }

    #[tokio::test]
    async fn mte_cron_webhook_route_keeps_one_shot_job_when_enqueue_fails() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("cron store");
        let token = "one-shot-token";
        let before_request = chrono::Utc::now();
        let original_at_ms = (before_request - chrono::Duration::seconds(1)).timestamp_millis();
        store
            .insert_job(&cron_job(
                ScheduleKind::At {
                    at_ms: original_at_ms,
                },
                token,
            ))
            .expect("insert job");

        let sync_records = Arc::new(Mutex::new(Vec::new()));
        let scheduler = Arc::new(
            MultiTenantEdgeCronScheduler::new(
                &temp.db_path().to_string_lossy(),
                CronRegistrationClient::new_with_transport(
                    "https://edge.example.com/_internal/crons".to_string(),
                    "cron-token".to_string(),
                    Arc::new(RecordingCronTransport {
                        statuses: Arc::new(Mutex::new(Vec::new())),
                        records: sync_records.clone(),
                    }),
                ),
            )
            .expect("scheduler"),
        );
        let (inbound_tx, inbound_rx) = mpsc::channel(1);
        drop(inbound_rx);
        let app = build_router(
            build_state(&temp.db_path(), inbound_tx, Some(scheduler)),
            false,
        );

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/_mte/cron/job-1/one-shot-token")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let job = store
            .find_job("job-1")
            .expect("job lookup")
            .expect("job should remain");
        match job.schedule {
            ScheduleKind::At { at_ms } => {
                assert!(at_ms > before_request.timestamp_millis());
                assert!(at_ms > original_at_ms);
            }
            other => panic!("expected rescheduled one-shot job, got {:?}", other),
        }
        assert_eq!(sync_records.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn mte_cron_webhook_route_keeps_one_shot_job_when_enqueue_and_rollback_sync_fail() {
        let temp = LocalTempDir::new();
        let store = CronStore::new(&temp.db_path().to_string_lossy()).expect("cron store");
        let token = "one-shot-token";
        store
            .insert_job(&cron_job(
                ScheduleKind::At {
                    at_ms: (chrono::Utc::now() - chrono::Duration::seconds(1)).timestamp_millis(),
                },
                token,
            ))
            .expect("insert job");

        let scheduler = Arc::new(
            MultiTenantEdgeCronScheduler::new(
                &temp.db_path().to_string_lossy(),
                CronRegistrationClient::new_with_transport(
                    "https://edge.example.com/_internal/crons".to_string(),
                    "cron-token".to_string(),
                    Arc::new(RecordingCronTransport {
                        statuses: Arc::new(Mutex::new(vec![StatusCode::INTERNAL_SERVER_ERROR])),
                        records: Arc::new(Mutex::new(Vec::new())),
                    }),
                ),
            )
            .expect("scheduler"),
        );
        let (inbound_tx, inbound_rx) = mpsc::channel(1);
        drop(inbound_rx);
        let app = build_router(
            build_state(&temp.db_path(), inbound_tx, Some(scheduler)),
            false,
        );

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/_mte/cron/job-1/one-shot-token")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(store.find_job("job-1").expect("job lookup").is_some());
    }
}
