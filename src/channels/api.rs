use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
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
use crate::logging::LoggerHandle;
use crate::scheduler::{
    CronWebhookError, MultiTenantEdgeCronScheduler, PendingCronTriggerFinalize,
};

const AGENT_TIMEOUT_SECS: u64 = 60;
const DEFAULT_API_USER: &str = "api_user";
const DEFAULT_RESPONSE_MODEL: &str = "agent-rs";
const MAX_RESPONSE_CACHE_ENTRIES: u64 = 1024;

#[derive(Clone)]
struct ApiState {
    inbound_tx: Sender<InboundMessage>,
    pending_requests: std::sync::Arc<DashMap<String, oneshot::Sender<OutboundMessage>>>,
    responses_cache: Cache<String, StoredResponse>,
    response_store: std::sync::Arc<ResponseStore>,
    mte_cron_scheduler: Option<std::sync::Arc<MultiTenantEdgeCronScheduler>>,
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
    pending_requests: std::sync::Arc<DashMap<String, oneshot::Sender<OutboundMessage>>>,
    responses_cache: Cache<String, StoredResponse>,
    response_store: std::sync::Arc<ResponseStore>,
    mte_cron_scheduler: Option<std::sync::Arc<MultiTenantEdgeCronScheduler>>,
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
            pending_requests: std::sync::Arc::new(DashMap::new()),
            responses_cache: Cache::builder()
                .max_capacity(MAX_RESPONSE_CACHE_ENTRIES)
                .eviction_policy(EvictionPolicy::lru())
                .build(),
            response_store: std::sync::Arc::new(ResponseStore::new(db_path)?),
            mte_cron_scheduler: None,
            logger_tx,
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
            mte_cron_scheduler: self.mte_cron_scheduler.clone(),
            channel_name: self.name().to_string(),
            logger_tx: self.logger_tx.clone(),
        };

        let logger_tx = self.logger_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
            "ApiChannel",
            "Starting API channel...",
        )));
        let addr = format!("0.0.0.0:{}", port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| format!("Failed to bind API port on {}: {}", addr, e))?;
        let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
            "ApiChannel",
            &format!("API channel listening on http://{}", addr),
        )));

        let handle = tokio::spawn(async move {
            let app = build_router(state);
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

fn build_router(state: ApiState) -> Router {
    let mut app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/v1/responses", post(handle_responses));

    if state.mte_cron_scheduler.is_some() {
        app = app.route("/_mte/cron/{job_id}/{token}", get(handle_mte_cron_webhook));
    }

    app.with_state(state)
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
        Err(error) => unreachable!("complete() is not expected to fail after delivery is marked: {}", error),
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

#[cfg(test)]
mod tests {
    use super::{build_router, ApiState};
    use crate::channels::api_store::ResponseStore;
    use crate::logging::create_logger_channel;
    use crate::multi_tenant_edge::{CronRegistrationClient, CronRule, CronTransport};
    use crate::scheduler::{ActiveJob, CronStore, MultiTenantEdgeCronScheduler, ScheduleKind};
    use async_trait::async_trait;
    use axum::body::Body;
    use reqwest::StatusCode;
    use serde_json::Value;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::sync::{mpsc, oneshot};
    use tower::ServiceExt;

    struct LocalTempDir {
        path: std::path::PathBuf,
    }

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    impl LocalTempDir {
        fn new() -> Self {
            let unique = format!(
                "agent-rs-api-cron-{}-{}-{}",
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
        }
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
        let app = build_router(build_state(&temp.db_path(), inbound_tx, Some(scheduler)));

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
        let app = build_router(build_state(&temp.db_path(), inbound_tx, Some(scheduler)));

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
        let app = build_router(build_state(&temp.db_path(), inbound_tx, Some(scheduler)));

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
        let app = build_router(build_state(&temp.db_path(), inbound_tx, Some(scheduler)));

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
        let app = build_router(build_state(&temp.db_path(), inbound_tx, Some(scheduler)));

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
