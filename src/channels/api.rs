use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use axum::{
    body::{Body, Bytes},
    extract::{OriginalUri, Path as AxumPath, Query, Request, State},
    http::{header, HeaderName, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{sse::Event, sse::Sse, IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use dashmap::{mapref::entry::Entry, DashMap};
use log::{error, info};
use moka::policy::EvictionPolicy;
use moka::sync::Cache;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::{self, Sender};
use tokio::sync::oneshot;

use crate::bus::{BusMessage, InboundMessage, LogEvent, OutboundMessage, TelemetryEvent};
use crate::channels::api_store::{ResponseStore, StoredResponse};
use crate::channels::Channel;
use crate::config::ApiConfig;
use crate::logging::LoggerHandle;
use crate::memory::{MemoryMessage, SharedReply};
use crate::scheduler::{
    CronWebhookError, MultiTenantEdgeCronScheduler, PendingCronTriggerFinalize,
};
use crate::tools::builtin::resolve_path;
use crate::utils::ChatMessage;
use crate::utils::{
    ContentPart, ImageUrl, MessageContent, REDACTED_THINKING_STRIP_PATTERN,
    RUNTIME_CONTEXT_END_SUFFIX,
};
use crate::NodeHandle;

const AGENT_TIMEOUT_SECS: u64 = 60;
const DEFAULT_API_USER: &str = "api_user";
const DEFAULT_RESPONSE_MODEL: &str = "isanagent";
const MAX_RESPONSE_CACHE_ENTRIES: u64 = 1024;

include!(concat!(env!("OUT_DIR"), "/ui_assets.rs"));

#[derive(Clone)]
struct ApiState {
    bus_tx: Sender<BusMessage>,
    pending_requests: std::sync::Arc<DashMap<String, PendingRequest>>,
    responses_cache: Cache<String, StoredResponse>,
    response_store: std::sync::Arc<ResponseStore>,
    mte_cron_scheduler: Option<std::sync::Arc<MultiTenantEdgeCronScheduler>>,
    channel_name: String,
    logger_tx: LoggerHandle,
    memory_node: NodeHandle<MemoryMessage>,
    /// Agent sandbox (`<workspace>/workspace`); same root as filesystem tools.
    workspace_sandbox: std::path::PathBuf,
    /// When `Some`, a `Authorization: Bearer <token>` is required on all `/v1` routes.
    auth_token: Option<std::sync::Arc<String>>,
}

enum PendingRequest {
    Sync(oneshot::Sender<OutboundMessage>),
    Stream(StreamingResponsePending),
}

/// Context for a streaming `/v1/responses` turn: SSE sender plus fields needed to persist like the sync path.
struct StreamingResponsePending {
    stream_tx: mpsc::Sender<StreamEvent>,
    sender_id: String,
    model: String,
    previous_response_id: Option<String>,
    store_response: bool,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    ToolCallStarted {
        tool_name: String,
        args: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
    },
    ToolProgress {
        tool_name: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
    },
    ToolCallFinished {
        tool_name: String,
        result: String,
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
    },
    AgentThought {
        thought: String,
    },
    Completion {
        content: String,
        thread_id: String,
        response_id: String,
    },
    Error {
        message: String,
    },
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
    stream: Option<bool>,
    /// When starting a new chain (no `previous_response_id`), the UI may supply a UUID so cancel works before any SSE is read.
    thread_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpdateSummaryRequest {
    summary: String,
    key_info: String,
    knowledge_gaps: String,
}

#[derive(Serialize)]
struct ResponsesResponse {
    id: String,
    object: &'static str,
    created_at: i64,
    model: String,
    status: &'static str,
    previous_response_id: Option<String>,
    /// Same key as `messages.thread_id` in workspace SQLite (API thread / chat scope).
    thread_id: String,
    output: Vec<ResponsesOutputItem>,
}

#[derive(Serialize)]
struct SessionHistoryMessage {
    role: String,
    content: String,
    /// Image URLs or `data:` URIs from multimodal user/assistant turns (OpenAI-style parts).
    #[serde(skip_serializing_if = "Option::is_none")]
    image_urls: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct ThreadsQueryParams {
    user: String,
    /// Max threads to return (default 100, clamped 1–500).
    #[serde(default)]
    limit: Option<u32>,
}

fn clamp_thread_list_limit(raw: Option<u32>) -> u32 {
    const DEFAULT: u32 = 100;
    const MAX: u32 = 500;
    raw.unwrap_or(DEFAULT).clamp(1, MAX)
}

#[derive(Serialize)]
struct ThreadListEntry {
    thread_id: String,
    updated_at: i64,
    latest_response_id: String,
    /// First user line (truncated), ChatGPT-style sidebar label.
    preview: String,
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

/// Helper to send a request to the memory node and map common errors to ApiError.
async fn memory_request<T>(
    memory_node: &NodeHandle<MemoryMessage>,
    msg_ctor: impl FnOnce(SharedReply<Result<T, String>>) -> MemoryMessage,
) -> Result<T, ApiError> {
    let (tx, rx) = oneshot::channel();
    let msg = msg_ctor(SharedReply::new(tx));
    memory_node.send_packet(msg).await.map_err(|e| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "memory_unavailable",
            e.to_string(),
        )
    })?;
    match rx.await {
        Ok(Ok(val)) => Ok(val),
        Ok(Err(e)) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "memory_error",
            e,
        )),
        Err(_) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "memory_unavailable",
            "Memory actor channel closed",
        )),
    }
}

pub struct ApiChannel {
    port: u16,
    bind_address: Option<String>,
    serve_ui: bool,
    auth_token: Option<std::sync::Arc<String>>,
    workspace_sandbox: std::path::PathBuf,
    pending_requests: std::sync::Arc<DashMap<String, PendingRequest>>,
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
        workspace_sandbox: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let (shutdown_tx, _) = tokio::sync::watch::channel(false);
        let bind_address = config
            .bind_address
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        // Config value wins; fall back to the ISANAGENT_API_TOKEN env var so the secret need
        // not live in the workspace TOML.
        let auth_token = config
            .auth_token
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .or_else(|| {
                std::env::var("ISANAGENT_API_TOKEN")
                    .ok()
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
            })
            .map(std::sync::Arc::new);
        let workspace_sandbox = std::fs::canonicalize(workspace_sandbox.as_ref()).map_err(|e| {
            format!(
                "Failed to canonicalize workspace sandbox {:?}: {}",
                workspace_sandbox.as_ref(),
                e
            )
        })?;
        Ok(Self {
            port: config.port,
            bind_address,
            serve_ui: config.serve_ui.unwrap_or(false),
            auth_token,
            workspace_sandbox,
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
        } else {
            // Safe by default: loopback. Exposing the control plane to the network now requires
            // an explicit `bind_address` AND an auth token (enforced in `start`). Previously the
            // headless default was `0.0.0.0`, which silently network-exposed chat control and
            // workspace file read/write.
            "127.0.0.1".to_string()
        }
    }
}

/// Loopback / local-only binds that are safe to serve without authentication.
///
/// Parses the host as an IP and uses `is_loopback()` (covers all of `127.0.0.0/8` and `::1`,
/// including the bracketed `[::1]` form), so a non-loopback bind (`0.0.0.0`, `::`, a public IP) —
/// or anything unparseable, including a hostname like `127.example.com` — is treated as NOT
/// loopback and therefore requires a token. Fails safe (unknown ⇒ not loopback).
fn is_loopback_bind(addr: &str) -> bool {
    let host = addr.trim();
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    // Accept the bracketed IPv6 literal `[::1]` as well as bare `::1`.
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// Refuse to expose the control plane to a non-loopback address without a token. The `/v1`
/// surface includes chat control and workspace file read/write, so an unauthenticated public
/// bind is an unauthenticated remote-code-execution surface.
fn validate_bind_security(addr: &str, has_token: bool) -> Result<(), String> {
    if is_loopback_bind(addr) || has_token {
        Ok(())
    } else {
        Err(format!(
            "Refusing to bind the API control plane to non-loopback address {addr:?} without authentication. \
             Set [api].auth_token (or the ISANAGENT_API_TOKEN env var), or bind to 127.0.0.1."
        ))
    }
}

/// Constant-time bearer-token check for `Authorization: Bearer <token>`.
fn bearer_token_matches(auth_header: Option<&str>, expected: &str) -> bool {
    if expected.is_empty() {
        return false;
    }
    let Some(header) = auth_header else {
        return false;
    };
    let Some(token) = header
        .strip_prefix("Bearer ")
        .or_else(|| header.strip_prefix("bearer "))
    else {
        return false;
    };
    let token = token.trim();
    if token.len() != expected.len() {
        return false;
    }
    // Constant-time compare so a near-miss token cannot be discovered byte-by-byte via timing.
    let mut diff = 0u8;
    for (a, b) in token.bytes().zip(expected.bytes()) {
        diff |= a ^ b;
    }
    diff == 0
}

/// Middleware gating the `/v1` routes when an auth token is configured.
async fn require_bearer_auth(
    State(expected): State<std::sync::Arc<String>>,
    req: Request,
    next: Next,
) -> Response {
    let provided = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    if bearer_token_matches(provided, expected.as_str()) {
        next.run(req).await
    } else {
        ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "Missing or invalid `Authorization: Bearer` token.",
        )
        .into_response()
    }
}

#[async_trait]
impl Channel for ApiChannel {
    fn name(&self) -> &str {
        "api"
    }

    async fn start(&self, bus_tx: Sender<BusMessage>) -> Result<(), String> {
        let port = self.port;
        let serve_ui = self.serve_ui;
        let state = ApiState {
            bus_tx,
            pending_requests: self.pending_requests.clone(),
            responses_cache: self.responses_cache.clone(),
            response_store: self.response_store.clone(),
            mte_cron_scheduler: self.mte_cron_scheduler.clone(),
            channel_name: self.name().to_string(),
            logger_tx: self.logger_tx.clone(),
            memory_node: self.memory_node.clone(),
            workspace_sandbox: self.workspace_sandbox.clone(),
            auth_token: self.auth_token.clone(),
        };

        let resolved_bind = self.resolved_bind_address();
        validate_bind_security(&resolved_bind, self.auth_token.is_some())?;

        let logger_tx = self.logger_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
            "ApiChannel",
            "Starting API channel...",
        )));
        let addr = format!("{}:{}", resolved_bind, port);
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
        if let Some((_, pending)) = self.pending_requests.remove(chat_id) {
            match pending {
                PendingRequest::Sync(sender) => {
                    let _ = sender.send(msg);
                }
                PendingRequest::Stream(pending) => {
                    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
                    let thread_id = msg.chat_id.clone();
                    if pending.store_response {
                        let now = chrono::Utc::now().timestamp();
                        let stored = StoredResponse {
                            thread_id: thread_id.clone(),
                            sender_id: pending.sender_id.clone(),
                            model: pending.model.clone(),
                        };
                        if let Err(e) = self
                            .response_store
                            .insert(
                                &response_id,
                                pending.previous_response_id.as_deref(),
                                &stored,
                                now,
                            )
                            .await
                        {
                            log_api(
                                &self.logger_tx,
                                LogEvent::error(
                                    "ApiChannel",
                                    &format!(
                                        "Failed to persist streaming response {}: {}",
                                        response_id, e
                                    ),
                                )
                                .with_chat_id(&thread_id),
                            );
                            if let Err(send_err) = pending.stream_tx.try_send(StreamEvent::Error {
                                message: "Failed to persist response state.".to_string(),
                            }) {
                                error!(
                                    "Failed to send stream error after persist failure: {}",
                                    send_err
                                );
                            }
                            return Ok(());
                        }
                        self.responses_cache.insert(response_id.clone(), stored);
                    }
                    log_api(
                        &self.logger_tx,
                        LogEvent::info(
                            "ApiChannel",
                            &format!(
                                "Streaming responses request completed with response_id {}",
                                response_id
                            ),
                        )
                        .with_chat_id(&thread_id),
                    );
                    if let Err(e) = pending.stream_tx.try_send(StreamEvent::Completion {
                        content: msg.content,
                        thread_id,
                        response_id,
                    }) {
                        error!("Failed to send completion to stream: {}", e);
                    }
                }
            }
        } else {
            error!(
                "ApiChannel: No pending request found for chat_id: {}",
                chat_id
            );
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ApiChannel {
    pub async fn handle_telemetry(&self, event: TelemetryEvent) {
        match event {
            TelemetryEvent::ToolCallStarted {
                chat_id,
                tool_name,
                args,
                tool_call_id,
                ..
            } => {
                if let Some(pending) = self.pending_requests.get(&chat_id) {
                    if let PendingRequest::Stream(pending) = pending.value() {
                        if let Err(e) = pending.stream_tx.try_send(StreamEvent::ToolCallStarted {
                            tool_name,
                            args,
                            tool_call_id,
                        }) {
                            error!("Failed to send tool_call_started to stream: {}", e);
                        }
                    }
                }
            }
            TelemetryEvent::ToolProgress {
                chat_id,
                tool_name,
                message,
                tool_call_id,
                ..
            } => {
                if let Some(pending) = self.pending_requests.get(&chat_id) {
                    if let PendingRequest::Stream(pending) = pending.value() {
                        if let Err(e) = pending.stream_tx.try_send(StreamEvent::ToolProgress {
                            tool_name,
                            message,
                            tool_call_id,
                        }) {
                            error!("Failed to send tool_progress to stream: {}", e);
                        }
                    }
                }
            }
            TelemetryEvent::ToolCallFinished {
                chat_id,
                tool_name,
                result,
                is_error,
                tool_call_id,
                ..
            } => {
                if let Some(pending) = self.pending_requests.get(&chat_id) {
                    if let PendingRequest::Stream(pending) = pending.value() {
                        if let Err(e) = pending.stream_tx.try_send(StreamEvent::ToolCallFinished {
                            tool_name,
                            result,
                            is_error,
                            tool_call_id,
                        }) {
                            error!("Failed to send tool_call_finished to stream: {}", e);
                        }
                    }
                }
            }
            TelemetryEvent::AgentThought {
                chat_id, thought, ..
            } => {
                if let Some(pending) = self.pending_requests.get(&chat_id) {
                    if let PendingRequest::Stream(pending) = pending.value() {
                        if let Err(e) = pending
                            .stream_tx
                            .try_send(StreamEvent::AgentThought { thought })
                        {
                            error!("Failed to send agent_thought to stream: {}", e);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn build_router(state: ApiState, serve_ui: bool) -> Router {
    let auth_token = state.auth_token.clone();
    let mut v1 = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/v1/responses", post(handle_responses))
        .route("/v1/threads", get(handle_list_threads))
        .route(
            "/v1/threads/{thread_id}/messages",
            get(handle_thread_messages),
        )
        .route("/v1/threads/{thread_id}", delete(handle_delete_thread))
        .route(
            "/v1/threads/{thread_id}/summaries",
            get(handle_get_thread_summaries),
        )
        .route("/v1/summaries", get(handle_get_all_summaries))
        .route("/v1/summaries/{id}", post(handle_update_summary))
        .route("/v1/summaries/{id}", delete(handle_delete_summary))
        .route("/v1/chat/cancel/{chat_id}", post(handle_cancel_chat))
        .route("/v1/background-jobs", get(handle_list_background_jobs))
        .route(
            "/v1/background-jobs/{job_id}/dismiss",
            post(handle_background_job_dismiss),
        )
        .route("/v1/notifications", get(handle_list_notifications))
        .route(
            "/v1/notifications/{notification_id}/seen",
            post(handle_notification_seen),
        )
        .route(
            "/v1/notifications/{notification_id}/resolve",
            post(handle_notification_resolve),
        )
        .route(
            "/v1/clarification-tickets/{ticket_id}/reply",
            post(handle_clarification_ticket_reply),
        )
        .route(
            "/v1/clarification-tickets/{ticket_id}/dismiss",
            post(handle_clarification_ticket_dismiss),
        )
        .route("/v1/workspace/list", get(handle_workspace_list))
        .route("/v1/workspace/file", get(handle_workspace_file))
        // POST on a distinct path so clients are not blocked by proxies or older builds that only registered GET on `/v1/workspace/file`.
        .route("/v1/workspace/file/save", post(handle_workspace_file_put))
        .route("/v1/workspace/rename", post(handle_workspace_rename));

    // Gate the `/v1` control surface with bearer auth when a token is configured. The cron
    // webhook (its own per-job URL token) and the embedded UI assets are intentionally left
    // open: external cron callers and browser asset GETs cannot carry the bearer header.
    if let Some(token) = auth_token {
        v1 = v1.route_layer(middleware::from_fn_with_state(token, require_bearer_auth));
    }

    let mut app = Router::new().merge(v1);

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

    let mut metadata = HashMap::from([
        (
            crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
            Value::String(format!("cron:{}", trigger.job_id)),
        ),
        (
            "cron_job_id".to_string(),
            Value::String(trigger.job_id.clone()),
        ),
        (
            "trigger_source".to_string(),
            Value::String("multi_tenant_edge".to_string()),
        ),
    ]);
    metadata.insert(
        crate::bus::METADATA_SYNTHETIC_CRON_TRIGGER.to_string(),
        serde_json::Value::Bool(true),
    );
    metadata.insert(
        crate::bus::METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS.to_string(),
        serde_json::Value::Bool(true),
    );
    let inbound = InboundMessage {
        channel: trigger.channel.clone(),
        sender_id: "cron".to_string(),
        chat_id: trigger.chat_id.clone(),
        thread_id: None,
        content: trigger.message.clone(),
        attachments: Vec::new(),
        metadata,
    };

    if let Err(error) = state.bus_tx.send(BusMessage::Inbound(inbound)).await {
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

    let stream_requested = payload.stream.unwrap_or(false);

    let (conv_thread_id, sender_id, model, previous_response_id) =
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
                    stored.thread_id,
                    payload.user.unwrap_or(stored.sender_id),
                    payload.model.unwrap_or(stored.model),
                    Some(previous_response_id.to_string()),
                )
            }
            None => {
                let conv_thread_id = if let Some(ref raw) = payload.thread_id {
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        return ApiError::new(
                            StatusCode::BAD_REQUEST,
                            "invalid_thread_id",
                            "thread_id must be a non-empty UUID when provided.",
                        )
                        .into_response();
                    }
                    match uuid::Uuid::parse_str(trimmed) {
                        Ok(u) => u.to_string(),
                        Err(_) => {
                            return ApiError::new(
                                StatusCode::BAD_REQUEST,
                                "invalid_thread_id",
                                "thread_id must be a valid UUID.",
                            )
                            .into_response();
                        }
                    }
                } else {
                    uuid::Uuid::new_v4().to_string()
                };
                (
                    conv_thread_id,
                    payload.user.unwrap_or_else(|| DEFAULT_API_USER.to_string()),
                    payload
                        .model
                        .unwrap_or_else(|| DEFAULT_RESPONSE_MODEL.to_string()),
                    None,
                )
            }
        };

    if stream_requested {
        let (stream_tx, mut stream_rx) = mpsc::channel(100);
        match state.pending_requests.entry(conv_thread_id.clone()) {
            Entry::Occupied(_) => {
                return ApiError::new(
                    StatusCode::CONFLICT,
                    "conversation_busy",
                    "A request is already in-flight for this conversation.",
                )
                .into_response();
            }
            Entry::Vacant(entry) => {
                entry.insert(PendingRequest::Stream(StreamingResponsePending {
                    stream_tx: stream_tx.clone(),
                    sender_id: sender_id.clone(),
                    model: model.clone(),
                    previous_response_id: previous_response_id.clone(),
                    store_response,
                }));
            }
        }

        // Send an initial event with `thread_id` so the client can cancel even before completion
        let _ = stream_tx.try_send(StreamEvent::Completion {
            content: String::new(),
            thread_id: conv_thread_id.clone(),
            response_id: String::new(),
        });

        let inbound = InboundMessage {
            channel: state.channel_name.clone(),
            sender_id: sender_id.clone(),
            chat_id: conv_thread_id.clone(),
            thread_id: None,
            content: input.content,
            attachments: input.attachments,
            metadata: Default::default(),
        };

        if let Err(e) = state.bus_tx.send(BusMessage::Inbound(inbound)).await {
            state.pending_requests.remove(&conv_thread_id);
            return ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "agent_queue_unavailable",
                format!("Failed to enqueue request: {}", e),
            )
            .into_response();
        }

        let stream = async_stream::stream! {
            while let Some(event) = stream_rx.recv().await {
                match serde_json::to_string(&event) {
                    Ok(json) => yield Ok::<Event, std::convert::Infallible>(Event::default().data(json)),
                    Err(e) => {
                        error!("Failed to serialize stream event: {}", e);
                    }
                }
            }
        };

        let mut sse_response = Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
        if let Ok(hv) = HeaderValue::from_str(&conv_thread_id) {
            sse_response
                .headers_mut()
                .insert(HeaderName::from_static("x-thread-id"), hv);
        }
        return sse_response;
    }

    let outbound =
        match dispatch_agent_turn(&state, sender_id.clone(), conv_thread_id.clone(), input).await {
            Ok(outbound) => outbound,
            Err(err) => return err.into_response(),
        };

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    if store_response {
        let stored = StoredResponse {
            thread_id: conv_thread_id.clone(),
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
                .with_chat_id(&conv_thread_id),
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
        .with_chat_id(&conv_thread_id),
    );

    Json(ResponsesResponse {
        id: response_id,
        object: "response",
        created_at: now,
        model,
        status: "completed",
        previous_response_id,
        thread_id: conv_thread_id.clone(),
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
            entry.insert(PendingRequest::Sync(tx));
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

    if let Err(e) = state.bus_tx.send(BusMessage::Inbound(msg)).await {
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
            Ok(ParsedChatInput {
                content,
                attachments,
            })
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

    Ok(ParsedChatInput {
        content,
        attachments,
    })
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

fn runtime_context_prefix_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\[RUNTIME CONTEXT\] Current time is .+?\. You are navigating and responding in channel: '[^']*', with chat ID: '[^']*'(?:, thread: '[^']*')?\.\s*\n\s*",
        )
        .expect("runtime context strip regex")
    })
}

fn redacted_thinking_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(REDACTED_THINKING_STRIP_PATTERN).expect("redacted thinking strip regex")
    })
}

fn strip_runtime_context_prefix(text: &str) -> String {
    if let Some(idx) = text.find(RUNTIME_CONTEXT_END_SUFFIX) {
        return text[idx + RUNTIME_CONTEXT_END_SUFFIX.len()..]
            .trim_start()
            .to_string();
    }
    // Legacy rows persisted before the stable end marker existed.
    runtime_context_prefix_re()
        .replace(text, "")
        .trim_start()
        .to_string()
}

fn strip_model_thinking_markup(text: &str) -> String {
    redacted_thinking_re()
        .replace_all(text, "")
        .trim()
        .to_string()
}

fn truncate_chat_preview(text: &str) -> String {
    let line = text.split('\n').next().unwrap_or(text).trim();
    let mut iter = line.chars();
    let chunk: String = iter.by_ref().take(56).collect();
    if iter.next().is_some() {
        format!("{}…", chunk)
    } else {
        chunk
    }
}

fn text_and_images_from_message(message: &ChatMessage) -> (String, Vec<String>) {
    match &message.content {
        Some(MessageContent::Parts(parts)) => {
            let mut texts = Vec::new();
            let mut urls = Vec::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => texts.push(text.as_str()),
                    ContentPart::ImageUrl { image_url } => urls.push(image_url.url.clone()),
                    ContentPart::Document { .. } => texts.push("[document attachment]"),
                }
            }
            (texts.join("\n\n"), urls)
        }
        Some(MessageContent::Text(s)) => (s.clone(), Vec::new()),
        None => (String::new(), Vec::new()),
    }
}

/// Transcript rows suitable for the web UI: hides tool traces, runtime injection prefixes, and
/// tool-only assistant turns (no user-visible assistant text).
fn chat_messages_to_ui_transcript(messages: &[ChatMessage]) -> Vec<SessionHistoryMessage> {
    let mut out = Vec::new();
    for message in messages {
        if message.role == "tool" || message.role == "system" {
            continue;
        }

        let (mut text, image_urls) = text_and_images_from_message(message);

        if message.role == "user" {
            text = strip_runtime_context_prefix(&text);
        } else if message.role == "assistant" {
            text = strip_model_thinking_markup(&text);
            let visible = text.trim();
            if visible.is_empty() && message.tool_calls.is_some() && image_urls.is_empty() {
                continue;
            }
        }

        let text = text.trim().to_string();
        if text.is_empty() && image_urls.is_empty() {
            continue;
        }

        out.push(SessionHistoryMessage {
            role: message.role.clone(),
            content: text,
            image_urls: if image_urls.is_empty() {
                None
            } else {
                Some(image_urls)
            },
        });
    }
    out
}

/// Maps a path segment to the SQLite `messages.thread_id` key.
///
/// [`AgentLogic`](crate::agent::AgentLogic) uses `format!("{}:{}:{}", channel, chat_id, thread_id)`;
/// for API messages with no agent sub-thread that is `api:<uuid>:` (note trailing colon).
/// Clients pass the bare thread id (uuid only); we qualify it with this channel name.
fn resolve_memory_thread_id<'a>(state: &ApiState, raw: &'a str) -> Cow<'a, str> {
    let s = raw.trim();
    let prefix = format!("{}:", state.channel_name);
    if s.starts_with(&prefix) {
        // Correct channel prefix; ensure canonical trailing colon.
        if s.ends_with(':') {
            Cow::Borrowed(s)
        } else {
            Cow::Owned(format!("{}:", s))
        }
    } else {
        // Extract the bare ID by taking the last non-empty colon-delimited segment.
        // This strips any caller-supplied channel prefix (e.g. "terminal:<uuid>:")
        // and re-qualifies it with the current channel, preventing cross-channel
        // history access.
        let id = s.rsplit(':').find(|seg| !seg.is_empty()).unwrap_or(s);
        Cow::Owned(format!("{}{}:", prefix, id))
    }
}

async fn memory_get_context(
    memory_node: &NodeHandle<MemoryMessage>,
    memory_thread_id: &str,
) -> Result<Vec<ChatMessage>, String> {
    let (tx, rx) = oneshot::channel();
    let msg = MemoryMessage::GetContext {
        thread_id: memory_thread_id.to_string(),
        reply: SharedReply::new(tx),
    };
    memory_node
        .send_packet(msg)
        .await
        .map_err(|e| e.to_string())?;
    rx.await
        .map_err(|_| "Memory actor channel closed".to_string())?
}

async fn memory_first_user_previews_batch(
    memory_node: &NodeHandle<MemoryMessage>,
    thread_ids: Vec<String>,
) -> Result<Vec<Option<String>>, String> {
    if thread_ids.is_empty() {
        return Ok(Vec::new());
    }
    let (tx, rx) = oneshot::channel();
    let msg = MemoryMessage::FirstUserMessagePreviewsBatch {
        thread_ids,
        reply: SharedReply::new(tx),
    };
    memory_node
        .send_packet(msg)
        .await
        .map_err(|e| e.to_string())?;
    rx.await
        .map_err(|_| "Memory actor channel closed".to_string())?
}

async fn memory_clear_thread(
    memory_node: &NodeHandle<MemoryMessage>,
    memory_thread_id: &str,
) -> Result<(), String> {
    let (tx, rx) = oneshot::channel();
    let msg = MemoryMessage::Clear {
        thread_id: memory_thread_id.to_string(),
        keep_last: 0,
        reply: SharedReply::new(tx),
    };
    memory_node
        .send_packet(msg)
        .await
        .map_err(|e| e.to_string())?;
    rx.await
        .map_err(|_| "Memory actor channel closed".to_string())?
}

async fn handle_list_threads(
    State(state): State<ApiState>,
    Query(params): Query<ThreadsQueryParams>,
) -> Response {
    let user = params.user.trim();
    if user.is_empty() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_user",
            "Query parameter `user` is required.",
        )
        .into_response();
    }
    let limit = clamp_thread_list_limit(params.limit);
    match state
        .response_store
        .list_threads_by_sender(user, limit)
        .await
    {
        Ok(rows) => {
            let memory_thread_ids: Vec<String> = rows
                .iter()
                .map(|row| resolve_memory_thread_id(&state, &row.thread_id).into_owned())
                .collect();
            let preview_opts =
                match memory_first_user_previews_batch(&state.memory_node, memory_thread_ids).await
                {
                    Ok(v) if v.len() == rows.len() => v,
                    Ok(v) => {
                        error!(
                            "thread list preview batch length mismatch: got {} want {}",
                            v.len(),
                            rows.len()
                        );
                        vec![None; rows.len()]
                    }
                    Err(e) => {
                        error!("thread list preview batch failed: {}", e);
                        vec![None; rows.len()]
                    }
                };
            let body: Vec<ThreadListEntry> = rows
                .into_iter()
                .zip(preview_opts)
                .map(|(row, preview_raw)| {
                    let preview = preview_raw
                        .as_deref()
                        .map(strip_runtime_context_prefix)
                        .map(|s| truncate_chat_preview(&s))
                        .filter(|s| !s.is_empty())
                        .unwrap_or_default();
                    ThreadListEntry {
                        thread_id: row.thread_id,
                        updated_at: row.updated_at,
                        latest_response_id: row.latest_response_id,
                        preview,
                    }
                })
                .collect();
            Json(body).into_response()
        }
        Err(message) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "thread_list_unavailable",
            message,
        )
        .into_response(),
    }
}

async fn handle_delete_thread(
    State(state): State<ApiState>,
    AxumPath(thread_id): AxumPath<String>,
    Query(params): Query<ThreadsQueryParams>,
) -> Response {
    let thread_id = thread_id.trim();
    if thread_id.is_empty() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_thread",
            "Empty thread id.",
        )
        .into_response();
    }
    let user = params.user.trim();
    if user.is_empty() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_user",
            "Query parameter `user` is required.",
        )
        .into_response();
    }

    let bare_id = thread_id
        .rsplit(':')
        .find(|s| !s.is_empty())
        .unwrap_or(thread_id);

    // Verify ownership via the API store before touching memory: `delete` only removes rows
    // where `thread_id` and `sender_id` match, so `removed == 0` means no access.
    match state
        .response_store
        .delete_thread_responses(bare_id, user)
        .await
    {
        Ok(removed) if removed > 0 => {
            let memory_thread_id = resolve_memory_thread_id(&state, thread_id);
            if let Err(message) =
                memory_clear_thread(&state.memory_node, memory_thread_id.as_ref()).await
            {
                return ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "memory_unavailable",
                    message,
                )
                .into_response();
            }
            Json(serde_json::json!({
                "deleted": true,
                "responses_removed": removed,
                "thread_id": bare_id,
            }))
            .into_response()
        }
        Ok(removed) => Json(serde_json::json!({
            "deleted": false,
            "responses_removed": removed,
            "thread_id": bare_id,
        }))
        .into_response(),
        Err(message) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "thread_delete_unavailable",
            message,
        )
        .into_response(),
    }
}

async fn handle_thread_messages(
    State(state): State<ApiState>,
    AxumPath(thread_id): AxumPath<String>,
) -> Response {
    let thread_id = thread_id.trim();
    if thread_id.is_empty() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_thread",
            "Empty thread id.",
        )
        .into_response();
    }
    let memory_thread_id = resolve_memory_thread_id(&state, thread_id);
    match memory_get_context(&state.memory_node, memory_thread_id.as_ref()).await {
        Ok(rows) => {
            let body = chat_messages_to_ui_transcript(&rows);
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

async fn handle_get_thread_summaries(
    State(state): State<ApiState>,
    AxumPath(thread_id): AxumPath<String>,
) -> Response {
    let thread_id = thread_id.trim();
    if thread_id.is_empty() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_thread",
            "Empty thread id.",
        )
        .into_response();
    }
    let memory_thread_id = resolve_memory_thread_id(&state, thread_id);
    let res = memory_request(&state.memory_node, |reply| MemoryMessage::GetSummaries {
        thread_id: memory_thread_id.to_string(),
        limit: 50,
        reply,
    })
    .await;

    match res {
        Ok(summaries) => Json(summaries).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn handle_update_summary(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<i64>,
    Json(payload): Json<UpdateSummaryRequest>,
) -> Response {
    let res = memory_request(&state.memory_node, |reply| MemoryMessage::UpdateSummary {
        id,
        summary: payload.summary,
        key_info: payload.key_info,
        knowledge_gaps: payload.knowledge_gaps,
        reply,
    })
    .await;

    match res {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => e.into_response(),
    }
}

async fn handle_get_all_summaries(State(state): State<ApiState>) -> Response {
    let res = memory_request(&state.memory_node, |reply| MemoryMessage::GetSummaries {
        thread_id: String::new(), // Empty string means get all
        limit: 100,
        reply,
    })
    .await;

    match res {
        Ok(summaries) => Json(summaries).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn handle_delete_summary(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<i64>,
) -> Response {
    let res = memory_request(&state.memory_node, |reply| MemoryMessage::DeleteSummary {
        id,
        reply,
    })
    .await;

    match res {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => e.into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct WorkspaceListQuery {
    #[serde(default)]
    path: String,
}

#[derive(Serialize)]
struct WorkspaceEntryDto {
    name: String,
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
}

#[derive(Serialize)]
struct WorkspaceListBody {
    path: String,
    entries: Vec<WorkspaceEntryDto>,
}

#[derive(Debug, Deserialize)]
struct WorkspaceFileQuery {
    path: String,
}

#[derive(Serialize)]
struct WorkspaceFileBody {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct WorkspaceFilePutBody {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct WorkspaceRenameBody {
    from: String,
    to: String,
}

#[derive(Serialize)]
struct WorkspaceRenameResponse {
    path: String,
}

#[derive(Debug, Deserialize)]
struct JobsQuery {
    #[serde(default)]
    chat_id: Option<String>,
    #[serde(default)]
    channel: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct NotificationsQuery {
    #[serde(default)]
    chat_id: Option<String>,
    #[serde(default)]
    channel: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    unseen_only: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ClarificationReplyBody {
    response: String,
}

/// Matches a generous UI preview cap (binary files are rejected earlier).
const WORKSPACE_FILE_MAX_BYTES: usize = 2_000_000;

/// Caps workspace directory listing size (avoids huge allocations / slow UI).
const WORKSPACE_LIST_MAX_ENTRIES: usize = 1000;

/// Offloads `resolve_path` (blocking `std::fs::canonicalize`) from the async runtime.
async fn spawn_workspace_resolve(
    rel: String,
    sandbox: std::path::PathBuf,
) -> Result<std::path::PathBuf, Response> {
    match tokio::task::spawn_blocking(move || resolve_path(&rel, &sandbox, true)).await {
        Ok(Ok(path)) => Ok(path),
        Ok(Err(message)) => {
            Err(ApiError::new(StatusCode::BAD_REQUEST, "invalid_path", message).into_response())
        }
        Err(join_err) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "path_resolution_failed",
            join_err.to_string(),
        )
        .into_response()),
    }
}

async fn handle_workspace_list(
    State(state): State<ApiState>,
    Query(q): Query<WorkspaceListQuery>,
) -> Response {
    let trimmed = q.path.trim();
    let rel_for_resolve = if trimmed.is_empty() {
        ".".to_string()
    } else {
        trimmed.to_string()
    };
    let dir = match spawn_workspace_resolve(rel_for_resolve, state.workspace_sandbox.clone()).await
    {
        Ok(p) => p,
        Err(resp) => return resp,
    };
    let dir_meta = match tokio::fs::metadata(&dir).await {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return ApiError::new(
                StatusCode::NOT_FOUND,
                "path_not_found",
                "The specified path does not exist",
            )
            .into_response();
        }
        Err(e) => {
            return ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "metadata_failed",
                e.to_string(),
            )
            .into_response()
        }
    };
    if !dir_meta.is_dir() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "not_a_directory",
            "Path is not a directory",
        )
        .into_response();
    }

    let mut entries: Vec<WorkspaceEntryDto> = Vec::new();
    let mut read_dir = match tokio::fs::read_dir(&dir).await {
        Ok(rd) => rd,
        Err(e) => {
            return ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "read_dir_failed",
                e.to_string(),
            )
            .into_response()
        }
    };

    let mut scanned: usize = 0;
    loop {
        let next = match read_dir.next_entry().await {
            Ok(v) => v,
            Err(e) => {
                return ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "read_dir_failed",
                    e.to_string(),
                )
                .into_response()
            }
        };

        let Some(entry) = next else {
            break;
        };
        scanned += 1;
        if scanned > WORKSPACE_LIST_MAX_ENTRIES {
            break;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let meta = match entry.metadata().await {
            Ok(m) => m,
            Err(_) => continue,
        };
        let kind = if meta.is_dir() {
            "dir".to_string()
        } else if meta.is_file() {
            "file".to_string()
        } else {
            continue;
        };
        let size = if meta.is_file() {
            Some(meta.len())
        } else {
            None
        };
        entries.push(WorkspaceEntryDto { name, kind, size });
    }

    entries.sort_by(|a, b| match (a.kind.as_str(), b.kind.as_str()) {
        ("dir", "file") => std::cmp::Ordering::Less,
        ("file", "dir") => std::cmp::Ordering::Greater,
        _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
    });

    let display_path = if trimmed.is_empty() {
        String::new()
    } else {
        trimmed.to_string()
    };

    Json(WorkspaceListBody {
        path: display_path,
        entries,
    })
    .into_response()
}

async fn handle_workspace_file(
    State(state): State<ApiState>,
    Query(q): Query<WorkspaceFileQuery>,
) -> Response {
    let trimmed = q.path.trim();
    if trimmed.is_empty() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "missing_path",
            "Query parameter `path` is required",
        )
        .into_response();
    }
    let path =
        match spawn_workspace_resolve(trimmed.to_string(), state.workspace_sandbox.clone()).await {
            Ok(p) => p,
            Err(resp) => return resp,
        };
    let metadata = match tokio::fs::metadata(&path).await {
        Ok(m) => m,
        Err(e) => {
            return ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "metadata_failed",
                e.to_string(),
            )
            .into_response()
        }
    };
    if !metadata.is_file() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "not_a_file",
            "Path is not a regular file",
        )
        .into_response();
    }
    let len = metadata.len();
    if len > WORKSPACE_FILE_MAX_BYTES as u64 {
        return ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "file_too_large",
            format!("File is {len} bytes (max {}).", WORKSPACE_FILE_MAX_BYTES),
        )
        .into_response();
    }
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) => {
            return ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "read_failed",
                e.to_string(),
            )
            .into_response()
        }
    };
    let content = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => {
            return ApiError::new(
                StatusCode::BAD_REQUEST,
                "not_utf8",
                "File is not valid UTF-8 text.",
            )
            .into_response()
        }
    };
    Json(WorkspaceFileBody {
        path: trimmed.to_string(),
        content,
    })
    .into_response()
}

async fn handle_workspace_file_put(
    State(state): State<ApiState>,
    Json(body): Json<WorkspaceFilePutBody>,
) -> Response {
    let trimmed = body.path.trim();
    if trimmed.is_empty() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "missing_path",
            "Field `path` is required",
        )
        .into_response();
    }
    if body.content.len() > WORKSPACE_FILE_MAX_BYTES {
        return ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "content_too_large",
            format!(
                "Content is {} bytes (max {}).",
                body.content.len(),
                WORKSPACE_FILE_MAX_BYTES
            ),
        )
        .into_response();
    }
    let path =
        match spawn_workspace_resolve(trimmed.to_string(), state.workspace_sandbox.clone()).await {
            Ok(p) => p,
            Err(resp) => return resp,
        };
    if let Ok(meta) = tokio::fs::metadata(&path).await {
        if meta.is_dir() {
            return ApiError::new(
                StatusCode::BAD_REQUEST,
                "is_directory",
                "Path is a directory, not a file",
            )
            .into_response();
        }
    }
    if let Some(parent) = path.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            return ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "create_dir_failed",
                e.to_string(),
            )
            .into_response();
        }
    }
    if let Err(e) = tokio::fs::write(&path, body.content.as_bytes()).await {
        return ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "write_failed",
            e.to_string(),
        )
        .into_response();
    }
    Json(WorkspaceFileBody {
        path: trimmed.to_string(),
        content: body.content,
    })
    .into_response()
}

async fn handle_workspace_rename(
    State(state): State<ApiState>,
    Json(body): Json<WorkspaceRenameBody>,
) -> Response {
    let from_trim = body.from.trim();
    let to_trim = body.to.trim();
    if from_trim.is_empty() || to_trim.is_empty() {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "missing_path",
            "Fields `from` and `to` are required",
        )
        .into_response();
    }

    let sandbox = state.workspace_sandbox.clone();
    let from_path = match spawn_workspace_resolve(from_trim.to_string(), sandbox.clone()).await {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // Prevent renaming the sandbox root itself. `from_path` and `workspace_sandbox` are canonical.
    if from_path == state.workspace_sandbox {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_operation",
            "Cannot rename or move the workspace root directory",
        )
        .into_response();
    }

    if !tokio::fs::try_exists(&from_path).await.unwrap_or(false) {
        return ApiError::new(
            StatusCode::NOT_FOUND,
            "source_not_found",
            "Source path does not exist",
        )
        .into_response();
    }

    let to_path = match spawn_workspace_resolve(to_trim.to_string(), sandbox).await {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    if from_path == to_path {
        return ApiError::new(
            StatusCode::BAD_REQUEST,
            "same_path",
            "Source and destination are the same",
        )
        .into_response();
    }

    if tokio::fs::try_exists(&to_path).await.unwrap_or(false) {
        return ApiError::new(
            StatusCode::CONFLICT,
            "destination_exists",
            "A file or folder already exists at the destination path",
        )
        .into_response();
    }

    if let Some(parent) = to_path.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            return ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "create_dir_failed",
                e.to_string(),
            )
            .into_response();
        }
    }

    if let Err(e) = tokio::fs::rename(&from_path, &to_path).await {
        return ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "rename_failed",
            e.to_string(),
        )
        .into_response();
    }

    Json(WorkspaceRenameResponse {
        path: to_trim.to_string(),
    })
    .into_response()
}

async fn handle_cancel_chat(
    State(state): State<ApiState>,
    AxumPath(chat_id): AxumPath<String>,
) -> Response {
    // 1. Signal the agent to stop reasoning
    let msg = BusMessage::Cancel(chat_id.clone());
    if let Err(e) = state.bus_tx.send(msg).await {
        return ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "agent_unavailable",
            e.to_string(),
        )
        .into_response();
    }

    // 2. Clear the pending request lock so the user can send a new message immediately
    state.pending_requests.remove(&chat_id);

    StatusCode::OK.into_response()
}

async fn handle_list_background_jobs(
    State(state): State<ApiState>,
    Query(params): Query<JobsQuery>,
) -> Response {
    let res = memory_request(&state.memory_node, |reply| {
        MemoryMessage::ListBackgroundJobs {
            chat_id: params.chat_id,
            channel: params.channel,
            limit: params.limit.unwrap_or(100),
            reply,
        }
    })
    .await;

    match res {
        Ok(rows) => Json(rows).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn handle_list_notifications(
    State(state): State<ApiState>,
    Query(params): Query<NotificationsQuery>,
) -> Response {
    let res = memory_request(&state.memory_node, |reply| {
        MemoryMessage::ListNotifications {
            chat_id: params.chat_id,
            channel: params.channel,
            limit: params.limit.unwrap_or(100),
            unseen_only: params.unseen_only.unwrap_or(false),
            reply,
        }
    })
    .await;

    match res {
        Ok(rows) => Json(rows).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn handle_notification_seen(
    State(state): State<ApiState>,
    AxumPath(notification_id): AxumPath<String>,
) -> Response {
    let res = memory_request(&state.memory_node, |reply| {
        MemoryMessage::MarkNotificationSeen {
            notification_id,
            reply,
        }
    })
    .await;

    match res {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => e.into_response(),
    }
}

async fn handle_notification_resolve(
    State(state): State<ApiState>,
    AxumPath(notification_id): AxumPath<String>,
) -> Response {
    let res = memory_request(&state.memory_node, |reply| {
        MemoryMessage::ResolveNotification {
            notification_id,
            reply,
        }
    })
    .await;

    match res {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => e.into_response(),
    }
}

async fn handle_background_job_dismiss(
    State(state): State<ApiState>,
    AxumPath(job_id): AxumPath<String>,
) -> Response {
    let res = memory_request(&state.memory_node, |reply| {
        MemoryMessage::DismissBackgroundJob {
            job_id: Some(job_id),
            ticket_id: None,
            reply,
        }
    })
    .await;

    match res {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => e.into_response(),
    }
}

async fn handle_clarification_ticket_reply(
    State(state): State<ApiState>,
    AxumPath(ticket_id): AxumPath<String>,
    Json(body): Json<ClarificationReplyBody>,
) -> Response {
    let ticket = match memory_request(&state.memory_node, |reply| {
        MemoryMessage::GetClarificationTicket {
            ticket_id: ticket_id.clone(),
            reply,
        }
    })
    .await
    {
        Ok(Some(t)) => t,
        Ok(None) => {
            return ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found",
                "Unknown clarification ticket",
            )
            .into_response();
        }
        Err(e) => return e.into_response(),
    };

    if let Err(e) = memory_request(&state.memory_node, |reply| {
        MemoryMessage::ResolveClarificationTicket {
            ticket_id: ticket_id.clone(),
            response: body.response.clone(),
            reply,
        }
    })
    .await
    {
        return e.into_response();
    }

    let mut metadata = HashMap::new();
    metadata.insert(
        crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
        Value::String(ticket.job_id),
    );
    metadata.insert(
        crate::bus::METADATA_SYNTHETIC_BACKGROUND_RESUME.to_string(),
        Value::Bool(true),
    );
    metadata.insert(
        crate::bus::METADATA_CLARIFICATION_TICKET_ID.to_string(),
        Value::String(ticket_id),
    );
    if let Err(e) = state
        .bus_tx
        .send(BusMessage::Inbound(InboundMessage {
            channel: ticket.channel,
            sender_id: "notification_reply".to_string(),
            chat_id: ticket.chat_id,
            thread_id: ticket.thread_id,
            content: body.response,
            attachments: Vec::new(),
            metadata,
        }))
        .await
    {
        return ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "agent_queue_unavailable",
            format!("Failed to enqueue ticket response: {}", e),
        )
        .into_response();
    }
    StatusCode::OK.into_response()
}

async fn handle_clarification_ticket_dismiss(
    State(state): State<ApiState>,
    AxumPath(ticket_id): AxumPath<String>,
) -> Response {
    let res = memory_request(&state.memory_node, |reply| {
        MemoryMessage::DismissBackgroundJob {
            job_id: None,
            ticket_id: Some(ticket_id),
            reply,
        }
    })
    .await;

    match res {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => e.into_response(),
    }
}

fn log_api(logger_tx: &LoggerHandle, event: LogEvent) {
    let _ = logger_tx.send(BusMessage::Log(event));
}

#[cfg(test)]
mod tests {
    use super::{
        bearer_token_matches, build_router, is_loopback_bind, validate_bind_security, ApiState,
        PendingRequest, StreamEvent, EMBEDDED_UI_ASSETS,
    };
    use crate::bus::{BusMessage, OutboundMessage};
    use crate::channels::api_store::ResponseStore;
    use crate::config::ApiConfig;
    use crate::logging::create_logger_channel;
    use crate::memory::{MemoryMessage, SharedReply, SqliteMemoryActor};
    use crate::multi_tenant_edge::{CronRegistrationClient, CronRule, CronTransport};
    use crate::scheduler::{ActiveJob, CronStore, MultiTenantEdgeCronScheduler, ScheduleKind};
    use crate::utils::ChatMessage;
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

    #[test]
    fn streamed_tool_completion_preserves_typed_error_status() {
        for (result, is_error) in [
            ("Error: this is file content, not a failure", false),
            ("native failure without a text prefix", true),
        ] {
            let event = StreamEvent::ToolCallFinished {
                tool_name: "read_file".to_string(),
                result: result.to_string(),
                is_error,
                tool_call_id: Some("call-42".to_string()),
            };
            let encoded = serde_json::to_value(event).expect("serialize stream event");

            assert_eq!(encoded["type"], "tool_call_finished");
            assert_eq!(encoded["is_error"], is_error);
            assert_eq!(encoded["tool_call_id"], "call-42");
        }
    }

    #[test]
    fn streamed_tool_lifecycle_uses_one_correlation_id() {
        let events = [
            StreamEvent::ToolCallStarted {
                tool_name: "exec".to_string(),
                args: r#"{"cmd":"cargo test"}"#.to_string(),
                tool_call_id: Some("call-7".to_string()),
            },
            StreamEvent::ToolProgress {
                tool_name: "exec".to_string(),
                message: "still running".to_string(),
                tool_call_id: Some("call-7".to_string()),
            },
            StreamEvent::ToolCallFinished {
                tool_name: "exec".to_string(),
                result: "done".to_string(),
                is_error: false,
                tool_call_id: Some("call-7".to_string()),
            },
        ];

        for event in events {
            let encoded = serde_json::to_value(event).expect("serialize stream event");
            assert_eq!(encoded["tool_call_id"], "call-7");
        }
    }

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
        bus_tx: mpsc::Sender<BusMessage>,
        scheduler: Option<Arc<MultiTenantEdgeCronScheduler>>,
    ) -> ApiState {
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let memory_actor =
            SqliteMemoryActor::new(db_path.to_str().expect("utf8 db path")).expect("memory actor");
        let memory_node =
            NodeHandle::<MemoryMessage>::new(memory_actor, 100, 1, Duration::from_millis(5));
        let workspace_sandbox = db_path.parent().expect("db path parent").join("workspace");
        std::fs::create_dir_all(&workspace_sandbox).expect("workspace sandbox");
        let workspace_sandbox =
            std::fs::canonicalize(&workspace_sandbox).expect("canonicalize workspace sandbox");
        ApiState {
            bus_tx,
            pending_requests: Arc::new(dashmap::DashMap::<String, PendingRequest>::new()),
            responses_cache: moka::sync::Cache::builder().max_capacity(16).build(),
            response_store: Arc::new(ResponseStore::new(db_path).expect("response store")),
            mte_cron_scheduler: scheduler,
            channel_name: "api".to_string(),
            logger_tx,
            memory_node,
            workspace_sandbox,
            auth_token: None,
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

    // ---- 0.2: HTTP control-plane auth + safe bind ----

    #[test]
    fn loopback_binds_recognized() {
        assert!(is_loopback_bind("127.0.0.1"));
        assert!(is_loopback_bind("::1"));
        assert!(is_loopback_bind("[::1]")); // bracketed IPv6 literal
        assert!(is_loopback_bind("localhost"));
        assert!(is_loopback_bind("127.0.0.5")); // all of 127.0.0.0/8
        assert!(!is_loopback_bind("0.0.0.0"));
        assert!(!is_loopback_bind("::")); // unspecified IPv6 is NOT loopback
        assert!(!is_loopback_bind("192.168.1.10"));
        // A hostname that merely starts with "127." must NOT be treated as loopback
        // (previously fail-open via a `starts_with("127.")` shortcut).
        assert!(!is_loopback_bind("127.example.com"));
    }

    #[test]
    fn refuses_public_bind_without_token() {
        assert!(validate_bind_security("127.0.0.1", false).is_ok());
        let err =
            validate_bind_security("0.0.0.0", false).expect_err("must refuse public+no-token");
        assert!(
            err.contains("auth_token") || err.contains("127.0.0.1"),
            "err={err}"
        );
        assert!(validate_bind_security("0.0.0.0", true).is_ok());
    }

    #[test]
    fn bearer_match_rules() {
        assert!(bearer_token_matches(Some("Bearer s3cret"), "s3cret"));
        assert!(bearer_token_matches(Some("bearer s3cret"), "s3cret"));
        assert!(!bearer_token_matches(Some("Bearer wrong0"), "s3cret"));
        assert!(!bearer_token_matches(Some("s3cret"), "s3cret")); // missing scheme
        assert!(!bearer_token_matches(None, "s3cret"));
        assert!(!bearer_token_matches(Some("Bearer s3cret"), "")); // empty expected never matches
    }

    #[tokio::test]
    async fn v1_requires_bearer_when_token_configured() {
        let temp = LocalTempDir::new();
        let (bus_tx, _bus_rx) = mpsc::channel(4);
        let mut state = build_state(&temp.db_path(), bus_tx, None);
        state.auth_token = Some(std::sync::Arc::new("topsecret".to_string()));
        let app = build_router(state, false);

        let unauthorized = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/threads")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let authorized = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/threads")
                    .method("GET")
                    .header("authorization", "Bearer topsecret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");
        assert_ne!(authorized.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn v1_open_when_no_token_configured() {
        let temp = LocalTempDir::new();
        let (bus_tx, _bus_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), bus_tx, None), false);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/threads")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("request succeeds");
        assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn ui_root_is_not_served_when_disabled() {
        let temp = LocalTempDir::new();
        let (bus_tx, _bus_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), bus_tx, None), false);

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
        let (bus_tx, _bus_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), bus_tx, None), true);

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
        let (bus_tx, _bus_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), bus_tx, None), true);

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
        let (bus_tx, _bus_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), bus_tx, None), true);

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
    async fn thread_messages_qualifies_bare_chat_id_with_api_channel_prefix() {
        let temp = LocalTempDir::new();
        let db_path = temp.db_path();
        let (bus_tx, _bus_rx) = mpsc::channel(4);
        let state = build_state(&db_path, bus_tx, None);
        let chat_suffix = "list-me-123e4567-e89b-12d3-a456-426614174000";
        let memory_key = format!("api:{}:", chat_suffix);
        let (tx, rx) = oneshot::channel();
        state
            .memory_node
            .send_packet(MemoryMessage::AddMessage {
                thread_id: memory_key,
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
                    .uri(format!("/v1/threads/{chat_suffix}/messages"))
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
        assert!(rows[0]["content"]
            .as_str()
            .expect("content string")
            .contains("hello from test"));
    }

    #[tokio::test]
    async fn ui_fallback_does_not_intercept_unknown_api_paths() {
        let temp = LocalTempDir::new();
        let (bus_tx, _bus_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), bus_tx, None), true);

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
        let pending_requests = Arc::new(dashmap::DashMap::<String, PendingRequest>::new());
        let (logger_tx, _logger_rx) = create_logger_channel(32);
        let response_store = Arc::new(ResponseStore::new(temp.db_path()).expect("response store"));
        let (bus_tx, mut bus_rx) = mpsc::channel(4);
        let memory_actor =
            SqliteMemoryActor::new(temp.db_path().to_str().expect("utf8")).expect("memory actor");
        let memory_node =
            NodeHandle::<MemoryMessage>::new(memory_actor, 100, 1, Duration::from_millis(5));
        let workspace_sandbox = temp.path.join("workspace");
        std::fs::create_dir_all(&workspace_sandbox).expect("workspace sandbox");
        let workspace_sandbox =
            std::fs::canonicalize(&workspace_sandbox).expect("canonicalize workspace sandbox");
        let state = ApiState {
            bus_tx,
            pending_requests: pending_requests.clone(),
            responses_cache: moka::sync::Cache::builder().max_capacity(16).build(),
            response_store,
            mte_cron_scheduler: None,
            channel_name: "api".to_string(),
            logger_tx,
            memory_node,
            workspace_sandbox,
            auth_token: None,
        };
        let app = build_router(state, true);

        tokio::spawn(async move {
            let msg = bus_rx.recv().await.expect("bus message");
            let BusMessage::Inbound(inbound) = msg else {
                panic!("expected BusMessage::Inbound");
            };
            let outbound = OutboundMessage {
                channel: "api".to_string(),
                chat_id: inbound.chat_id.clone(),
                thread_id: None,
                content: "UI path still reaches responses.".to_string(),
                metadata: HashMap::new(),
            };
            let (_, pending) = pending_requests
                .remove(&inbound.chat_id)
                .expect("pending request sender");
            match pending {
                PendingRequest::Sync(sender) => {
                    let _ = sender.send(outbound);
                }
                PendingRequest::Stream(_) => panic!("unexpected stream pending"),
            }
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
        let (bus_tx, mut bus_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), bus_tx, Some(scheduler)), false);

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
        let msg = bus_rx.recv().await.expect("cron bus message");
        let BusMessage::Inbound(inbound) = msg else {
            panic!("expected BusMessage::Inbound");
        };
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
        let (bus_tx, mut bus_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), bus_tx, Some(scheduler)), false);

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
        assert!(bus_rx.try_recv().is_err());
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
        let (bus_tx, _bus_rx) = mpsc::channel(4);
        let app = build_router(build_state(&temp.db_path(), bus_tx, Some(scheduler)), false);

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
        let (bus_tx, bus_rx) = mpsc::channel(1);
        drop(bus_rx);
        let app = build_router(build_state(&temp.db_path(), bus_tx, Some(scheduler)), false);

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
        let (bus_tx, bus_rx) = mpsc::channel(1);
        drop(bus_rx);
        let app = build_router(build_state(&temp.db_path(), bus_tx, Some(scheduler)), false);

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
