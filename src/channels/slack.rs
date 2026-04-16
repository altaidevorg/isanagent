use async_trait::async_trait;
use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use log::{error, info, warn};
use moka::sync::Cache;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc::Sender, watch, Mutex, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use crate::bus::{BusMessage, InboundMessage, LogEvent, OutboundMessage};
use crate::channels::slack_store::{SlackUserProfileStore, StoredSlackUserProfile};
use crate::channels::Channel;
use crate::config::{SlackConfig, SlackMode};
use crate::logging::LoggerHandle;
use crate::utils::{ContentPart, ImageUrl};
use std::path::Path;

const SLACK_CHANNEL_NAME: &str = "slack";
const DEFAULT_REACTION_EMOJI: &str = "eyes";
const DEFAULT_WEBHOOK_PATH: &str = "/slack/events";
const DEFAULT_TIMESTAMP_TOLERANCE_SECS: i64 = 300;
const MAX_WEBHOOK_DEDUPE_ENTRIES: u64 = 10_000;
const WEBHOOK_DEDUPE_TTL_SECS: u64 = 600;
const MAX_USER_NAME_CACHE_ENTRIES: u64 = 10_000;
const USER_NAME_CACHE_TTL_SECS: u64 = 604_800;
const BOT_USER_ID_RETRY_COOLDOWN_SECS: u64 = 30;
const DEFAULT_MAX_RETRIES: usize = 3;
const DEFAULT_INITIAL_BACKOFF_SECS: u64 = 2;

type HmacSha256 = Hmac<Sha256>;

pub struct SlackChannel {
    config: SlackConfig,
    shared: Arc<SlackRuntimeState>,
    shutdown_tx: watch::Sender<bool>,
    task_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
    max_retries: usize,
    initial_backoff: Duration,
    timestamp_tolerance_secs: i64,
}

struct SlackRuntimeState {
    client: Client,
    logger_tx: LoggerHandle,
    bot_user_id: RwLock<Option<String>>,
    bot_user_id_refresh_lock: tokio::sync::Mutex<()>,
    last_bot_user_id_refresh_attempt: Mutex<Option<SystemTime>>,
    user_names_cache: Cache<String, String>,
    user_profile_store: Arc<SlackUserProfileStore>,
    webhook_dedupe: Cache<String, bool>,
    api_base_url: String,
}

#[derive(Clone)]
struct SlackWebhookState {
    shared: Arc<SlackRuntimeState>,
    bus_tx: Sender<BusMessage>,
    config: SlackConfig,
    signing_secret: String,
    timestamp_tolerance_secs: i64,
}

enum SlackStartMode {
    Socket {
        app_token: String,
    },
    Webhook {
        signing_secret: String,
        port: u16,
        path: String,
    },
}

#[derive(Clone, Copy)]
enum SlackIngressMode {
    Socket,
    Webhook,
}

struct SlackDispatch {
    inbound: InboundMessage,
    reaction: Option<SlackReaction>,
}

#[derive(Clone)]
struct SlackReaction {
    channel: String,
    timestamp: String,
    emoji: String,
}

struct SlackHttpResponse {
    status: StatusCode,
    body: String,
    retry_after: Option<Duration>,
}

enum SlackSendDecision {
    Success,
    Retry { delay_override: Option<Duration> },
    Fatal(String),
}

#[derive(Deserialize)]
struct SlackApiResponse {
    ok: bool,
    error: Option<String>,
}

impl SlackRuntimeState {
    fn new(
        logger_tx: LoggerHandle,
        api_base_url: String,
        user_profile_store: Arc<SlackUserProfileStore>,
    ) -> Self {
        Self {
            client: crate::utils::build_reqwest_client(),
            logger_tx,
            bot_user_id: RwLock::new(None),
            bot_user_id_refresh_lock: tokio::sync::Mutex::new(()),
            last_bot_user_id_refresh_attempt: Mutex::new(None),
            user_names_cache: Cache::builder()
                .max_capacity(MAX_USER_NAME_CACHE_ENTRIES)
                .time_to_live(Duration::from_secs(USER_NAME_CACHE_TTL_SECS))
                .build(),
            user_profile_store,
            webhook_dedupe: Cache::builder()
                .max_capacity(MAX_WEBHOOK_DEDUPE_ENTRIES)
                .time_to_live(Duration::from_secs(WEBHOOK_DEDUPE_TTL_SECS))
                .build(),
            api_base_url,
        }
    }

    fn api_url(&self, method: &str) -> String {
        format!("{}/{}", self.api_base_url.trim_end_matches('/'), method)
    }

    async fn refresh_bot_user_id(&self, bot_token: &str) -> Option<String> {
        info!("Requesting Slack bot user ID...");
        let auth_res = self
            .client
            .post(self.api_url("auth.test"))
            .header("Authorization", format!("Bearer {}", bot_token))
            .send()
            .await;

        match auth_res {
            Ok(response) => match response.json::<Value>().await {
                Ok(json) if json["ok"].as_bool() == Some(true) => {
                    if let Some(uid) = json["user_id"].as_str() {
                        let uid = uid.to_string();
                        *self.bot_user_id.write().await = Some(uid.clone());
                        log_slack(
                            &self.logger_tx,
                            LogEvent::info(
                                "SlackChannel",
                                &format!("Slack bot connected as {}", uid),
                            ),
                        );
                        return Some(uid);
                    }
                }
                Ok(json) => {
                    log_slack(
                        &self.logger_tx,
                        LogEvent::warn(
                            "SlackChannel",
                            &format!("Slack auth.test failed: {:?}", json),
                        ),
                    );
                }
                Err(e) => {
                    log_slack(
                        &self.logger_tx,
                        LogEvent::warn(
                            "SlackChannel",
                            &format!("Failed to decode Slack auth.test response: {}", e),
                        ),
                    );
                }
            },
            Err(e) => {
                log_slack(
                    &self.logger_tx,
                    LogEvent::warn(
                        "SlackChannel",
                        &format!("Failed to request Slack auth.test: {}", e),
                    ),
                );
            }
        }

        None
    }

    async fn ensure_bot_user_id(&self, bot_token: &str) -> Option<String> {
        if let Some(uid) = self.bot_user_id.read().await.clone() {
            return Some(uid);
        }

        let _refresh_guard = self.bot_user_id_refresh_lock.lock().await;
        if let Some(uid) = self.bot_user_id.read().await.clone() {
            return Some(uid);
        }

        {
            let mut last_attempt = self.last_bot_user_id_refresh_attempt.lock().await;
            let now = SystemTime::now();
            if let Some(previous) = last_attempt.as_ref().cloned() {
                if let Ok(elapsed) = now.duration_since(previous) {
                    if elapsed < Duration::from_secs(BOT_USER_ID_RETRY_COOLDOWN_SECS) {
                        return None;
                    }
                }
            }
            *last_attempt = Some(now);
        }

        self.refresh_bot_user_id(bot_token).await
    }

    async fn resolve_display_name(&self, user: &str, bot_token: &str) -> String {
        self.resolve_display_name_with_fetcher(user, || {
            self.fetch_display_name_from_slack(user, bot_token)
        })
        .await
    }

    async fn resolve_display_name_with_fetcher<F, Fut>(&self, user: &str, fetcher: F) -> String
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Option<String>>,
    {
        if user.is_empty() {
            return String::new();
        }

        if let Some(name) = self.user_names_cache.get(user) {
            return name;
        }

        let now_unix_secs = current_unix_timestamp();
        let mut stale_persisted_name = None;
        match self.user_profile_store.get(user).await {
            Ok(Some(stored)) if slack_user_profile_is_fresh(&stored, now_unix_secs) => {
                self.user_names_cache
                    .insert(user.to_string(), stored.display_name.clone());
                return stored.display_name;
            }
            Ok(Some(stored)) => {
                stale_persisted_name = Some(stored.display_name);
            }
            Ok(None) => {}
            Err(e) => warn!(
                "Failed to load cached Slack user profile for {}: {}",
                user, e
            ),
        }

        if let Some(name) = fetcher().await {
            self.cache_display_name(user, &name, now_unix_secs).await;
            return name;
        }

        if let Some(name) = stale_persisted_name {
            self.user_names_cache.insert(user.to_string(), name.clone());
            return name;
        }

        user.to_string()
    }

    async fn fetch_display_name_from_slack(&self, user: &str, bot_token: &str) -> Option<String> {
        let info_url = self.api_url(&format!("users.info?user={}", user));
        let info_res = self
            .client
            .get(info_url)
            .header("Authorization", format!("Bearer {}", bot_token))
            .send()
            .await;

        match info_res {
            Ok(response) => match response.json::<Value>().await {
                Ok(json) if json["ok"].as_bool() == Some(true) => {
                    return slack_profile_display_name(&json);
                }
                Ok(_) => {}
                Err(e) => warn!("Failed to decode Slack users.info response: {}", e),
            },
            Err(e) => warn!("Failed to fetch Slack user profile for {}: {}", user, e),
        }

        None
    }

    async fn cache_display_name(&self, user: &str, display_name: &str, fetched_at_unix_secs: i64) {
        self.user_names_cache
            .insert(user.to_string(), display_name.to_string());

        if let Err(e) = self
            .user_profile_store
            .upsert(
                user,
                &StoredSlackUserProfile {
                    display_name: display_name.to_string(),
                    fetched_at_unix_secs,
                },
            )
            .await
        {
            warn!("Failed to persist Slack user profile for {}: {}", user, e);
        }
    }

    /// Downloads a Slack file (accessible via `url_private`) using the bot token
    /// and returns it as an OpenAI-compatible `ContentPart::ImageUrl` with a
    /// base64 data URI.  Returns `None` when the file is not a supported image
    /// type or when the download fails.
    async fn download_slack_file(&self, file: &Value, bot_token: &str) -> Option<ContentPart> {
        let mime = file["mimetype"].as_str()?;
        if !mime.starts_with("image/") {
            return None;
        }
        // Only process MIME types that OpenAI vision supports
        match mime {
            "image/jpeg" | "image/png" | "image/gif" | "image/webp" => {}
            _ => return None,
        }

        let url = file["url_private"].as_str()?;
        match self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", bot_token))
            .send()
            .await
        {
            Err(e) => {
                warn!("Failed to download Slack file {}: {}", url, e);
                None
            }
            Ok(response) => {
                if !response.status().is_success() {
                    warn!(
                        "Slack file download failed with status {}: {}",
                        response.status(),
                        url
                    );
                    return None;
                }
                match response.bytes().await {
                    Err(e) => {
                        warn!("Failed to read Slack file bytes from {}: {}", url, e);
                        None
                    }
                    Ok(bytes) => {
                        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        let data_uri = format!("data:{};base64,{}", mime, encoded);
                        Some(ContentPart::ImageUrl {
                            image_url: ImageUrl {
                                url: data_uri,
                                detail: None,
                            },
                        })
                    }
                }
            }
        }
    }

    async fn process_event_callback(
        self: &Arc<Self>,
        envelope: Value,
        ingress: SlackIngressMode,
        bus_tx: Sender<BusMessage>,
        config: &SlackConfig,
    ) {
        let Some(event) = envelope.get("event") else {
            return;
        };

        let bot_user_id = self.ensure_bot_user_id(&config.bot_token).await;
        if !should_process_event(event, bot_user_id.as_deref()) {
            return;
        }

        let user = event["user"].as_str().unwrap_or_default().to_string();
        let display_name = self.resolve_display_name(&user, &config.bot_token).await;
        let event_id = envelope
            .get("event_id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        let Some(mut dispatch) = normalize_slack_event(
            config,
            event,
            bot_user_id.as_deref(),
            &display_name,
            event_id.as_deref(),
            ingress,
        ) else {
            return;
        };

        // Download any image files attached to the Slack message
        if let Some(files) = event.get("files").and_then(Value::as_array) {
            for file in files {
                if let Some(attachment) = self.download_slack_file(file, &config.bot_token).await {
                    dispatch.inbound.attachments.push(attachment);
                }
            }
        }

        let reaction = dispatch.reaction.clone();
        let chat_id = dispatch.inbound.chat_id.clone();
        if let Err(e) = bus_tx.send(BusMessage::Inbound(dispatch.inbound)).await {
            warn!("Failed to route InboundMessage from Slack: {}", e);
            return;
        }

        log_slack(
            &self.logger_tx,
            LogEvent::info("SlackChannel", "Slack event routed to inbound bus.")
                .with_chat_id(&chat_id),
        );

        if let Some(reaction) = reaction {
            self.spawn_reaction(reaction, config.bot_token.clone());
        }
    }

    fn spawn_reaction(self: &Arc<Self>, reaction: SlackReaction, bot_token: String) {
        let client = self.client.clone();
        let api_url = self.api_url("reactions.add");
        tokio::spawn(async move {
            let body = json!({
                "channel": reaction.channel,
                "timestamp": reaction.timestamp,
                "name": reaction.emoji,
            });

            match client
                .post(api_url)
                .header("Authorization", format!("Bearer {}", bot_token))
                .json(&body)
                .send()
                .await
            {
                Ok(response) => {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    if let Err(err) =
                        validate_simple_slack_api_response("Slack reaction", status, &body)
                    {
                        warn!("{}", err);
                    }
                }
                Err(e) => warn!("Network error adding Slack emoji reaction: {}", e),
            }
        });
    }
}

impl SlackChannel {
    pub fn new(
        config: SlackConfig,
        db_path: impl AsRef<Path>,
        logger_tx: LoggerHandle,
    ) -> Result<Self, String> {
        let user_profile_store = Arc::new(SlackUserProfileStore::new(db_path)?);
        Ok(Self::new_internal(
            config,
            logger_tx,
            "https://slack.com/api".to_string(),
            DEFAULT_MAX_RETRIES,
            Duration::from_secs(DEFAULT_INITIAL_BACKOFF_SECS),
            DEFAULT_TIMESTAMP_TOLERANCE_SECS,
            user_profile_store,
        ))
    }

    fn new_internal(
        config: SlackConfig,
        logger_tx: LoggerHandle,
        api_base_url: String,
        max_retries: usize,
        initial_backoff: Duration,
        timestamp_tolerance_secs: i64,
        user_profile_store: Arc<SlackUserProfileStore>,
    ) -> Self {
        let (shutdown_tx, _) = watch::channel(false);
        Self {
            config,
            shared: Arc::new(SlackRuntimeState::new(
                logger_tx,
                api_base_url,
                user_profile_store,
            )),
            shutdown_tx,
            task_handle: Mutex::new(None),
            max_retries,
            initial_backoff,
            timestamp_tolerance_secs,
        }
    }

    fn start_mode(&self) -> Result<SlackStartMode, String> {
        match self.config.mode() {
            SlackMode::Socket => {
                let app_token = self
                    .config
                    .app_token
                    .clone()
                    .filter(|token| !token.trim().is_empty())
                    .ok_or_else(|| {
                        "Slack socket mode requires a non-empty slack.app_token".to_string()
                    })?;
                Ok(SlackStartMode::Socket { app_token })
            }
            SlackMode::Webhook => {
                let signing_secret = self
                    .config
                    .signing_secret
                    .clone()
                    .filter(|secret| !secret.trim().is_empty())
                    .ok_or_else(|| {
                        "Slack webhook mode requires a non-empty slack.signing_secret".to_string()
                    })?;
                let port = self
                    .config
                    .webhook_port
                    .ok_or_else(|| "Slack webhook mode requires slack.webhook_port".to_string())?;
                Ok(SlackStartMode::Webhook {
                    signing_secret,
                    port,
                    path: normalize_webhook_path(self.config.webhook_path.as_deref()),
                })
            }
        }
    }

    async fn store_task_handle(&self, handle: tokio::task::JoinHandle<()>) -> Result<(), String> {
        *self.task_handle.lock().await = Some(handle);
        Ok(())
    }
}

#[async_trait]
impl Channel for SlackChannel {
    fn name(&self) -> &str {
        SLACK_CHANNEL_NAME
    }

    async fn start(&self, bus_tx: Sender<BusMessage>) -> Result<(), String> {
        log_slack(
            &self.shared.logger_tx,
            LogEvent::info("SlackChannel", "Starting Slack channel..."),
        );

        let start_mode = self.start_mode()?;
        self.shared
            .refresh_bot_user_id(&self.config.bot_token)
            .await;

        match start_mode {
            SlackStartMode::Socket { app_token } => {
                let shared = self.shared.clone();
                let config = self.config.clone();
                let mut shutdown_rx = self.shutdown_tx.subscribe();
                let handle = tokio::spawn(async move {
                    run_socket_mode(shared, config, app_token, bus_tx, &mut shutdown_rx).await;
                });
                self.store_task_handle(handle).await?;
            }
            SlackStartMode::Webhook {
                signing_secret,
                port,
                path,
            } => {
                let addr = format!("0.0.0.0:{}", port);
                let listener = tokio::net::TcpListener::bind(&addr)
                    .await
                    .map_err(|e| format!("Failed to bind Slack webhook port on {}: {}", addr, e))?;
                log_slack(
                    &self.shared.logger_tx,
                    LogEvent::info(
                        "SlackChannel",
                        &format!("Slack webhook listening on http://{}{}", addr, path),
                    ),
                );

                let state = SlackWebhookState {
                    shared: self.shared.clone(),
                    bus_tx,
                    config: self.config.clone(),
                    signing_secret,
                    timestamp_tolerance_secs: self.timestamp_tolerance_secs,
                };
                let app = build_webhook_router(state, &path);
                let mut shutdown_rx = self.shutdown_tx.subscribe();
                let handle = tokio::spawn(async move {
                    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                        while shutdown_rx.changed().await.is_ok() {
                            if *shutdown_rx.borrow() {
                                break;
                            }
                        }
                    });
                    if let Err(e) = server.await {
                        error!("Slack webhook server crashed: {}", e);
                    }
                });
                self.store_task_handle(handle).await?;
            }
        }

        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        info!("Stopping Slack channel...");
        let _ = self.shutdown_tx.send(true);
        let handle = {
            let mut task_handle = self.task_handle.lock().await;
            task_handle.take()
        };
        if let Some(handle) = handle {
            let _ = handle.await;
        }
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        let body = build_post_message_body(&msg);
        let client = self.shared.client.clone();
        let api_url = self.shared.api_url("chat.postMessage");
        let bot_token = self.config.bot_token.clone();

        execute_post_message_with_retries(self.max_retries, self.initial_backoff, move |_| {
            let client = client.clone();
            let api_url = api_url.clone();
            let bot_token = bot_token.clone();
            let body = body.clone();

            async move {
                let response = client
                    .post(api_url)
                    .header("Authorization", format!("Bearer {}", bot_token))
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| e.to_string())?;

                let status = response.status();
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(Duration::from_secs);
                let body = response.text().await.map_err(|e| e.to_string())?;

                Ok(SlackHttpResponse {
                    status,
                    body,
                    retry_after,
                })
            }
        })
        .await
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn build_post_message_body(msg: &OutboundMessage) -> Value {
    let mut body = json!({
        "channel": msg.chat_id,
        "text": msg.content,
    });

    if let Some(ts) = msg.thread_id.as_deref() {
        if !msg.chat_id.starts_with('D') {
            body.as_object_mut()
                .unwrap()
                .insert("thread_ts".to_string(), Value::String(ts.to_string()));
        }
    }

    body
}

async fn execute_post_message_with_retries<F, Fut>(
    max_retries: usize,
    initial_backoff: Duration,
    mut execute: F,
) -> Result<(), String>
where
    F: FnMut(usize) -> Fut,
    Fut: Future<Output = Result<SlackHttpResponse, String>>,
{
    let mut backoff = initial_backoff;

    for attempt in 1..=max_retries {
        match execute(attempt).await {
            Ok(response) => match classify_post_message_response(response) {
                SlackSendDecision::Success => return Ok(()),
                SlackSendDecision::Fatal(err) => return Err(err),
                SlackSendDecision::Retry { delay_override } => {
                    if attempt < max_retries {
                        tokio::time::sleep(delay_override.unwrap_or(backoff)).await;
                        backoff *= 2;
                        continue;
                    }
                }
            },
            Err(e) => {
                error!(
                    "Slack postMessage network error (attempt {}): {}",
                    attempt, e
                );
                if attempt < max_retries {
                    tokio::time::sleep(backoff).await;
                    backoff *= 2;
                    continue;
                }
            }
        }
    }

    error!("Slack send failed after {} attempts.", max_retries);
    Err("Slack send max retries exceeded".to_string())
}

fn classify_post_message_response(response: SlackHttpResponse) -> SlackSendDecision {
    if response.status == StatusCode::TOO_MANY_REQUESTS {
        error!("Slack postMessage rate limited: {}", response.body);
        return SlackSendDecision::Retry {
            delay_override: response.retry_after,
        };
    }

    if response.status.is_server_error() {
        error!(
            "Slack postMessage 5xx error: {} - {}",
            response.status, response.body
        );
        return SlackSendDecision::Retry {
            delay_override: None,
        };
    }

    if response.status.is_client_error() {
        error!(
            "Slack postMessage fatal 4xx error: {} - {}",
            response.status, response.body
        );
        return SlackSendDecision::Fatal("Slack API returned fatal 4xx error".to_string());
    }

    match serde_json::from_str::<SlackApiResponse>(&response.body) {
        Ok(api_response) if api_response.ok => SlackSendDecision::Success,
        Ok(api_response) => {
            let err = api_response
                .error
                .unwrap_or_else(|| "unknown_error".to_string());
            error!("Slack postMessage failed with ok=false: {}", err);
            SlackSendDecision::Fatal(format!("Slack API returned ok=false: {}", err))
        }
        Err(e) => SlackSendDecision::Fatal(format!(
            "Failed to decode Slack postMessage response: {}",
            e
        )),
    }
}

fn validate_simple_slack_api_response(
    action: &str,
    status: StatusCode,
    body: &str,
) -> Result<(), String> {
    if !status.is_success() {
        return Err(format!(
            "{} failed with status {}: {}",
            action, status, body
        ));
    }

    let api_response = serde_json::from_str::<SlackApiResponse>(body)
        .map_err(|e| format!("Failed to decode {} response: {}", action, e))?;
    if api_response.ok {
        return Ok(());
    }

    Err(format!(
        "{} failed with ok=false: {}",
        action,
        api_response
            .error
            .unwrap_or_else(|| "unknown_error".to_string())
    ))
}

fn build_webhook_router(state: SlackWebhookState, path: &str) -> Router {
    Router::new()
        .route(path, post(handle_slack_webhook))
        .with_state(state)
}

async fn handle_slack_webhook(
    State(state): State<SlackWebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let now = current_unix_timestamp();
    if let Err(err) = verify_slack_signature(
        &state.signing_secret,
        &headers,
        body.as_ref(),
        now,
        state.timestamp_tolerance_secs,
    ) {
        warn!("Slack webhook signature verification failed: {}", err);
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let payload: Value = match serde_json::from_slice(body.as_ref()) {
        Ok(payload) => payload,
        Err(e) => {
            warn!("Failed to parse Slack webhook payload: {}", e);
            return StatusCode::BAD_REQUEST.into_response();
        }
    };

    match payload["type"].as_str().unwrap_or_default() {
        "url_verification" => payload["challenge"]
            .as_str()
            .unwrap_or_default()
            .to_string()
            .into_response(),
        "event_callback" => {
            let Some(event_id) = payload
                .get("event_id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
            else {
                return StatusCode::OK.into_response();
            };

            if state.shared.webhook_dedupe.get(&event_id).is_some() {
                return StatusCode::OK.into_response();
            }
            state.shared.webhook_dedupe.insert(event_id, true);

            let shared = state.shared.clone();
            let bus_tx = state.bus_tx.clone();
            let config = state.config.clone();
            tokio::spawn(async move {
                shared
                    .process_event_callback(payload, SlackIngressMode::Webhook, bus_tx, &config)
                    .await;
            });
            StatusCode::OK.into_response()
        }
        _ => StatusCode::OK.into_response(),
    }
}

async fn run_socket_mode(
    shared: Arc<SlackRuntimeState>,
    config: SlackConfig,
    app_token: String,
    bus_tx: Sender<BusMessage>,
    shutdown_rx: &mut watch::Receiver<bool>,
) {
    let mut backoff_secs = DEFAULT_INITIAL_BACKOFF_SECS;

    'outer: loop {
        if *shutdown_rx.borrow() {
            break;
        }

        let _ = shared.ensure_bot_user_id(&config.bot_token).await;

        log_slack(
            &shared.logger_tx,
            LogEvent::info("SlackChannel", "Requesting Slack Socket Mode URL..."),
        );
        let res = shared
            .client
            .post(shared.api_url("apps.connections.open"))
            .header("Authorization", format!("Bearer {}", app_token))
            .header("Content-type", "application/x-www-form-urlencoded")
            .send()
            .await;

        let ws_url = match res {
            Ok(r) => match r.json::<Value>().await {
                Ok(json) if json["ok"].as_bool() == Some(true) => {
                    json["url"].as_str().unwrap_or_default().to_string()
                }
                Ok(json) => {
                    log_slack(
                        &shared.logger_tx,
                        LogEvent::error(
                            "SlackChannel",
                            &format!("Slack apps.connections.open failed: {:?}", json),
                        ),
                    );
                    if wait_for_shutdown_or_timeout(shutdown_rx, Duration::from_secs(backoff_secs))
                        .await
                    {
                        break;
                    }
                    backoff_secs = std::cmp::min(backoff_secs * 2, 60);
                    continue;
                }
                Err(e) => {
                    log_slack(
                        &shared.logger_tx,
                        LogEvent::error(
                            "SlackChannel",
                            &format!("Failed to parse Slack response: {}", e),
                        ),
                    );
                    if wait_for_shutdown_or_timeout(shutdown_rx, Duration::from_secs(backoff_secs))
                        .await
                    {
                        break;
                    }
                    backoff_secs = std::cmp::min(backoff_secs * 2, 60);
                    continue;
                }
            },
            Err(e) => {
                log_slack(
                    &shared.logger_tx,
                    LogEvent::error(
                        "SlackChannel",
                        &format!("Failed to request Slack websockets URL: {}", e),
                    ),
                );
                if wait_for_shutdown_or_timeout(shutdown_rx, Duration::from_secs(backoff_secs))
                    .await
                {
                    break;
                }
                backoff_secs = std::cmp::min(backoff_secs * 2, 60);
                continue;
            }
        };

        log_slack(
            &shared.logger_tx,
            LogEvent::info("SlackChannel", "Connecting to Slack Socket Mode..."),
        );
        let (ws_stream, _) = match connect_async(&ws_url).await {
            Ok(stream) => {
                backoff_secs = DEFAULT_INITIAL_BACKOFF_SECS;
                stream
            }
            Err(e) => {
                log_slack(
                    &shared.logger_tx,
                    LogEvent::error(
                        "SlackChannel",
                        &format!("WebSocket connection failed: {}", e),
                    ),
                );
                if wait_for_shutdown_or_timeout(shutdown_rx, Duration::from_secs(backoff_secs))
                    .await
                {
                    break;
                }
                backoff_secs = std::cmp::min(backoff_secs * 2, 60);
                continue;
            }
        };

        log_slack(
            &shared.logger_tx,
            LogEvent::info("SlackChannel", "Slack Socket Mode connected successfully."),
        );

        let (mut write, mut read) = ws_stream.split();
        loop {
            let maybe_msg = tokio::select! {
                msg = read.next() => msg,
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        break 'outer;
                    }
                    continue;
                }
            };

            let Some(msg) = maybe_msg else {
                break;
            };

            let msg = match msg {
                Ok(m) => m,
                Err(e) => {
                    error!("Slack websocket read error: {}", e);
                    break;
                }
            };

            if let Message::Text(text) = msg {
                let Ok(payload) = serde_json::from_str::<Value>(&text) else {
                    continue;
                };

                if let Some(envelope_id) = payload.get("envelope_id").and_then(Value::as_str) {
                    let ack = json!({ "envelope_id": envelope_id });
                    if let Err(e) = write.send(Message::Text(ack.to_string().into())).await {
                        error!("Failed to ack Slack envelope: {}", e);
                    }
                }

                if payload["type"].as_str() != Some("events_api") {
                    continue;
                }

                let event_payload = payload.get("payload").cloned().unwrap_or(Value::Null);
                shared
                    .process_event_callback(
                        event_payload,
                        SlackIngressMode::Socket,
                        bus_tx.clone(),
                        &config,
                    )
                    .await;
            }
        }

        warn!(
            "Slack Socket Mode disconnected. Reconnecting in {} seconds...",
            backoff_secs
        );
        if wait_for_shutdown_or_timeout(shutdown_rx, Duration::from_secs(backoff_secs)).await {
            break;
        }
        backoff_secs = std::cmp::min(backoff_secs * 2, 60);
    }

    log_slack(
        &shared.logger_tx,
        LogEvent::info("SlackChannel", "Slack channel stopped."),
    );
}

fn normalize_slack_event(
    config: &SlackConfig,
    event: &Value,
    bot_user_id: Option<&str>,
    display_name: &str,
    event_id: Option<&str>,
    ingress: SlackIngressMode,
) -> Option<SlackDispatch> {
    let ev_type = event["type"].as_str().unwrap_or_default();
    if ev_type != "message" && ev_type != "app_mention" {
        return None;
    }

    if event.get("bot_id").is_some() || event.get("subtype").is_some() {
        return None;
    }

    let text = event["text"].as_str().unwrap_or_default().to_string();
    if ev_type == "message" {
        if let Some(uid) = bot_user_id {
            if text.contains(&format!("<@{uid}>")) {
                return None;
            }
        }
    }

    let user = event["user"].as_str().unwrap_or_default().to_string();
    let chat_id = event["channel"].as_str().unwrap_or_default().to_string();
    if user.is_empty() || chat_id.is_empty() {
        return None;
    }

    let ts = event
        .get("ts")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let mut thread_ts = event
        .get("thread_ts")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    if thread_ts.is_none()
        && config.reply_in_thread.unwrap_or(false)
        && !chat_id.starts_with('D')
        && !ts.is_empty()
    {
        thread_ts = Some(ts.clone());
    }

    let stripped_text = strip_bot_mention(&text, bot_user_id);
    let payload_text = format!("(Slack User: {}) {}", display_name, stripped_text)
        .trim()
        .to_string();

    let mut metadata = HashMap::new();
    metadata.insert(
        "slack_event_id".to_string(),
        Value::String(event_id.unwrap_or_default().to_string()),
    );
    metadata.insert(
        "slack_ingress_mode".to_string(),
        Value::String(ingress.as_str().to_string()),
    );

    let reaction = config
        .reaction_emoji
        .clone()
        .unwrap_or_else(|| DEFAULT_REACTION_EMOJI.to_string());
    let reaction = if reaction.is_empty() || ts.is_empty() {
        None
    } else {
        Some(SlackReaction {
            channel: chat_id.clone(),
            timestamp: ts,
            emoji: reaction,
        })
    };

    Some(SlackDispatch {
        inbound: InboundMessage {
            channel: SLACK_CHANNEL_NAME.to_string(),
            sender_id: user,
            chat_id,
            thread_id: thread_ts,
            content: payload_text,
            attachments: Vec::new(),
            metadata,
        },
        reaction,
    })
}

fn should_process_event(event: &Value, bot_user_id: Option<&str>) -> bool {
    let ev_type = event["type"].as_str().unwrap_or_default();
    if ev_type != "message" && ev_type != "app_mention" {
        return false;
    }

    if event.get("bot_id").is_some() || event.get("subtype").is_some() {
        return false;
    }

    if ev_type == "message" {
        if let Some(uid) = bot_user_id {
            let text = event["text"].as_str().unwrap_or_default();
            if text.contains(&format!("<@{uid}>")) {
                return false;
            }
        }
    }

    true
}

fn strip_bot_mention(text: &str, bot_user_id: Option<&str>) -> String {
    let mut stripped_text = text.to_string();
    if let Some(uid) = bot_user_id {
        let mention = format!("<@{uid}>");
        if stripped_text.contains(&mention) {
            stripped_text = stripped_text.replace(&mention, "").trim().to_string();
        }
    } else if let Some(idx) = stripped_text.find("> ") {
        if stripped_text.starts_with("<@") {
            stripped_text = stripped_text[idx + 2..].to_string();
        }
    }
    stripped_text
}

fn normalize_webhook_path(path: Option<&str>) -> String {
    match path.map(str::trim).filter(|value| !value.is_empty()) {
        Some(path) if path.starts_with('/') => path.to_string(),
        Some(path) => format!("/{}", path),
        None => DEFAULT_WEBHOOK_PATH.to_string(),
    }
}

fn slack_profile_display_name(json: &Value) -> Option<String> {
    json["user"]["profile"]["display_name"]
        .as_str()
        .filter(|value| !value.is_empty())
        .or_else(|| json["user"]["profile"]["real_name"].as_str())
        .map(ToOwned::to_owned)
}

fn slack_user_profile_is_fresh(profile: &StoredSlackUserProfile, now_unix_secs: i64) -> bool {
    now_unix_secs.saturating_sub(profile.fetched_at_unix_secs) <= USER_NAME_CACHE_TTL_SECS as i64
}

fn verify_slack_signature(
    signing_secret: &str,
    headers: &HeaderMap,
    body: &[u8],
    now_timestamp: i64,
    tolerance_secs: i64,
) -> Result<(), &'static str> {
    let signature = headers
        .get("X-Slack-Signature")
        .and_then(|value| value.to_str().ok())
        .ok_or("missing_signature")?;
    let timestamp = headers
        .get("X-Slack-Request-Timestamp")
        .and_then(|value| value.to_str().ok())
        .ok_or("missing_timestamp")?;
    let ts = timestamp.parse::<i64>().map_err(|_| "invalid_timestamp")?;
    if (now_timestamp - ts).abs() > tolerance_secs {
        return Err("stale_timestamp");
    }

    let provided = signature.strip_prefix("v0=").ok_or("invalid_signature")?;
    let provided = hex::decode(provided).map_err(|_| "invalid_signature")?;

    let mut mac = HmacSha256::new_from_slice(signing_secret.as_bytes())
        .map_err(|_| "invalid_signing_secret")?;
    mac.update(format!("v0:{}:", timestamp).as_bytes());
    mac.update(body);
    mac.verify_slice(&provided).map_err(|_| "invalid_signature")
}

fn current_unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}

fn log_slack(logger_tx: &LoggerHandle, event: LogEvent) {
    let _ = logger_tx.send(BusMessage::Log(event));
}

impl SlackIngressMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Socket => "socket",
            Self::Webhook => "webhook",
        }
    }
}

async fn wait_for_shutdown_or_timeout(
    shutdown_rx: &mut watch::Receiver<bool>,
    duration: Duration,
) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(duration) => false,
        changed = shutdown_rx.changed() => changed.is_ok() && *shutdown_rx.borrow(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex as StdMutex;
    use tower::ServiceExt;

    use crate::logging::create_logger_channel;

    #[test]
    fn verify_signature_accepts_valid_signature() {
        let secret = "signing-secret";
        let body = br#"{"type":"url_verification","challenge":"abc"}"#;
        let timestamp = "1710000000";
        let signature = sign_for_test(secret, timestamp, body);
        let mut headers = HeaderMap::new();
        headers.insert("X-Slack-Signature", signature.parse().unwrap());
        headers.insert("X-Slack-Request-Timestamp", timestamp.parse().unwrap());

        let result = verify_slack_signature(secret, &headers, body, 1710000000, 300);

        assert!(result.is_ok());
    }

    #[test]
    fn verify_signature_rejects_invalid_signature() {
        let secret = "signing-secret";
        let body = br#"{"type":"url_verification","challenge":"abc"}"#;
        let mut headers = HeaderMap::new();
        headers.insert("X-Slack-Signature", "v0=deadbeef".parse().unwrap());
        headers.insert("X-Slack-Request-Timestamp", "1710000000".parse().unwrap());

        let result = verify_slack_signature(secret, &headers, body, 1710000000, 300);

        assert!(result.is_err());
    }

    #[test]
    fn verify_signature_rejects_missing_signature() {
        let secret = "signing-secret";
        let body = br#"{}"#;
        let mut headers = HeaderMap::new();
        headers.insert("X-Slack-Request-Timestamp", "1710000000".parse().unwrap());

        let result = verify_slack_signature(secret, &headers, body, 1710000000, 300);

        assert_eq!(result, Err("missing_signature"));
    }

    #[test]
    fn verify_signature_rejects_stale_timestamp() {
        let secret = "signing-secret";
        let body = br#"{}"#;
        let timestamp = "1710000000";
        let signature = sign_for_test(secret, timestamp, body);
        let mut headers = HeaderMap::new();
        headers.insert("X-Slack-Signature", signature.parse().unwrap());
        headers.insert("X-Slack-Request-Timestamp", timestamp.parse().unwrap());

        let result = verify_slack_signature(secret, &headers, body, 1710000400, 300);

        assert_eq!(result, Err("stale_timestamp"));
    }

    #[test]
    fn normalize_event_builds_app_mention_inbound_message() {
        let config = test_slack_config();
        let event = json!({
            "type": "app_mention",
            "user": "U123",
            "channel": "C123",
            "text": "<@B123> build status?",
            "ts": "1710.0001"
        });

        let dispatch = normalize_slack_event(
            &config,
            &event,
            Some("B123"),
            "Jane",
            Some("Ev123"),
            SlackIngressMode::Webhook,
        )
        .unwrap();

        assert_eq!(dispatch.inbound.channel, "slack");
        assert_eq!(dispatch.inbound.sender_id, "U123");
        assert_eq!(dispatch.inbound.chat_id, "C123");
        assert_eq!(dispatch.inbound.thread_id.as_deref(), Some("1710.0001"));
        assert_eq!(dispatch.inbound.content, "(Slack User: Jane) build status?");
        assert_eq!(
            dispatch.inbound.metadata["slack_event_id"],
            Value::String("Ev123".to_string())
        );
        assert_eq!(
            dispatch.inbound.metadata["slack_ingress_mode"],
            Value::String("webhook".to_string())
        );
    }

    #[test]
    fn normalize_event_ignores_bot_and_subtype_events() {
        let config = test_slack_config();
        let bot_event = json!({
            "type": "message",
            "bot_id": "B999",
            "user": "U123",
            "channel": "C123",
            "text": "hi",
            "ts": "1"
        });
        let subtype_event = json!({
            "type": "message",
            "subtype": "message_changed",
            "user": "U123",
            "channel": "C123",
            "text": "hi",
            "ts": "1"
        });

        assert!(normalize_slack_event(
            &config,
            &bot_event,
            Some("B123"),
            "Jane",
            Some("Ev1"),
            SlackIngressMode::Socket
        )
        .is_none());
        assert!(normalize_slack_event(
            &config,
            &subtype_event,
            Some("B123"),
            "Jane",
            Some("Ev2"),
            SlackIngressMode::Socket
        )
        .is_none());
    }

    #[test]
    fn normalize_event_uses_dm_and_thread_rules() {
        let mut config = test_slack_config();
        config.reply_in_thread = Some(true);
        let dm_event = json!({
            "type": "message",
            "user": "U123",
            "channel": "D123",
            "text": "hello",
            "ts": "3.14"
        });
        let threaded_event = json!({
            "type": "message",
            "user": "U123",
            "channel": "C123",
            "text": "hello",
            "ts": "3.15",
            "thread_ts": "3.10"
        });

        let dm = normalize_slack_event(
            &config,
            &dm_event,
            Some("B123"),
            "Jane",
            Some("EvDm"),
            SlackIngressMode::Socket,
        )
        .unwrap();
        let threaded = normalize_slack_event(
            &config,
            &threaded_event,
            Some("B123"),
            "Jane",
            Some("EvThread"),
            SlackIngressMode::Socket,
        )
        .unwrap();

        assert_eq!(dm.inbound.thread_id, None);
        assert_eq!(threaded.inbound.thread_id.as_deref(), Some("3.10"));
    }

    #[tokio::test]
    async fn webhook_handler_returns_challenge() {
        let state = test_webhook_state();
        let body = br#"{"type":"url_verification","challenge":"abc123"}"#;
        let app = build_webhook_router(state, DEFAULT_WEBHOOK_PATH);
        let request = signed_request(body, "secret");

        let response = app.oneshot(request).await.unwrap();
        let response_body = to_bytes(response.into_body(), usize::MAX).await.unwrap();

        assert_eq!(response_body, Bytes::from_static(b"abc123"));
    }

    #[tokio::test]
    async fn webhook_handler_rejects_invalid_signature() {
        let state = test_webhook_state();
        let app = build_webhook_router(state, DEFAULT_WEBHOOK_PATH);
        let request = Request::builder()
            .method("POST")
            .uri(DEFAULT_WEBHOOK_PATH)
            .header("content-type", "application/json")
            .body(Body::from(r#"{"type":"event_callback"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn webhook_handler_dispatches_once_for_duplicate_event() {
        let (state, mut inbound_rx) = test_webhook_state_with_channel();
        state
            .shared
            .user_names_cache
            .insert("U123".into(), "Jane".into());
        *state.shared.bot_user_id.write().await = Some("B123".into());
        let app = build_webhook_router(state.clone(), DEFAULT_WEBHOOK_PATH);
        let body = br#"{
            "type":"event_callback",
            "event_id":"Ev123",
            "event":{"type":"app_mention","user":"U123","channel":"C123","text":"<@B123> ping","ts":"1710.1"}
        }"#;

        let response = app
            .clone()
            .oneshot(signed_request(body, "secret"))
            .await
            .unwrap();
        let duplicate = app.oneshot(signed_request(body, "secret")).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(duplicate.status(), StatusCode::OK);

        let msg = tokio::time::timeout(Duration::from_millis(200), inbound_rx.recv())
            .await
            .unwrap()
            .unwrap();
        let crate::bus::BusMessage::Inbound(inbound) = msg else {
            panic!("expected BusMessage::Inbound");
        };
        assert_eq!(inbound.metadata["slack_event_id"], "Ev123");
        assert!(
            tokio::time::timeout(Duration::from_millis(100), inbound_rx.recv())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn send_fails_when_slack_returns_ok_false() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_clone = attempts.clone();
        let result = execute_post_message_with_retries(2, Duration::from_millis(1), move |_| {
            attempts_clone.fetch_add(1, Ordering::SeqCst);
            async {
                Ok(SlackHttpResponse {
                    status: StatusCode::OK,
                    body: r#"{"ok":false,"error":"not_in_channel"}"#.to_string(),
                    retry_after: None,
                })
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn send_retries_rate_limits() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let responses = Arc::new(StdMutex::new(VecDeque::from(vec![
            Ok(SlackHttpResponse {
                status: StatusCode::TOO_MANY_REQUESTS,
                body: "rate limited".to_string(),
                retry_after: Some(Duration::from_secs(0)),
            }),
            Ok(SlackHttpResponse {
                status: StatusCode::OK,
                body: r#"{"ok":true}"#.to_string(),
                retry_after: None,
            }),
        ])));
        let attempts_clone = attempts.clone();
        let responses_clone = responses.clone();

        execute_post_message_with_retries(3, Duration::from_millis(1), move |_| {
            attempts_clone.fetch_add(1, Ordering::SeqCst);
            let next = responses_clone.lock().unwrap().pop_front().unwrap();
            async move { next }
        })
        .await
        .unwrap();

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn send_retries_server_errors() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let responses = Arc::new(StdMutex::new(VecDeque::from(vec![
            Ok(SlackHttpResponse {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: "boom".to_string(),
                retry_after: None,
            }),
            Ok(SlackHttpResponse {
                status: StatusCode::OK,
                body: r#"{"ok":true}"#.to_string(),
                retry_after: None,
            }),
        ])));
        let attempts_clone = attempts.clone();
        let responses_clone = responses.clone();

        execute_post_message_with_retries(3, Duration::from_millis(1), move |_| {
            attempts_clone.fetch_add(1, Ordering::SeqCst);
            let next = responses_clone.lock().unwrap().pop_front().unwrap();
            async move { next }
        })
        .await
        .unwrap();

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn validate_simple_slack_api_response_rejects_ok_false() {
        let result = validate_simple_slack_api_response(
            "Slack reaction",
            StatusCode::OK,
            r#"{"ok":false,"error":"missing_scope"}"#,
        );

        assert_eq!(
            result,
            Err("Slack reaction failed with ok=false: missing_scope".to_string())
        );
    }

    #[test]
    fn validate_simple_slack_api_response_accepts_ok_true() {
        let result =
            validate_simple_slack_api_response("Slack reaction", StatusCode::OK, r#"{"ok":true}"#);

        assert!(result.is_ok());
    }

    #[test]
    fn build_post_message_body_serializes_thread_ts_only_for_channels() {
        let channel_body = build_post_message_body(&OutboundMessage {
            channel: "slack".into(),
            chat_id: "C123".into(),
            thread_id: Some("1710.2".into()),
            content: "hello".into(),
            metadata: HashMap::new(),
        });
        let dm_body = build_post_message_body(&OutboundMessage {
            channel: "slack".into(),
            chat_id: "D123".into(),
            thread_id: Some("1710.2".into()),
            content: "hello".into(),
            metadata: HashMap::new(),
        });

        assert_eq!(channel_body["thread_ts"], "1710.2");
        assert!(dm_body.get("thread_ts").is_none());
    }

    fn test_slack_config() -> SlackConfig {
        SlackConfig {
            enabled: Some(true),
            mode: Some(SlackMode::Webhook),
            app_token: Some("xapp-test".into()),
            bot_token: "xoxb-test".into(),
            signing_secret: Some("secret".into()),
            webhook_port: Some(8081),
            webhook_path: Some(DEFAULT_WEBHOOK_PATH.into()),
            reply_in_thread: Some(true),
            reaction_emoji: Some(String::new()),
        }
    }

    fn test_webhook_state() -> SlackWebhookState {
        let (state, _rx) = test_webhook_state_with_channel();
        state
    }

    fn test_webhook_state_with_channel(
    ) -> (SlackWebhookState, tokio::sync::mpsc::Receiver<BusMessage>) {
        let (logger_tx, _logger_rx) = create_logger_channel(8);
        let shared = Arc::new(SlackRuntimeState::new(
            logger_tx,
            "http://localhost".into(),
            Arc::new(SlackUserProfileStore::new(":memory:").unwrap()),
        ));
        let (bus_tx, bus_rx) = tokio::sync::mpsc::channel(8);
        (
            SlackWebhookState {
                shared,
                bus_tx,
                config: test_slack_config(),
                signing_secret: "secret".into(),
                timestamp_tolerance_secs: DEFAULT_TIMESTAMP_TOLERANCE_SECS,
            },
            bus_rx,
        )
    }

    fn signed_request(body: &[u8], secret: &str) -> Request<Body> {
        let timestamp = current_unix_timestamp().to_string();
        let signature = sign_for_test(secret, &timestamp, body);
        Request::builder()
            .method("POST")
            .uri(DEFAULT_WEBHOOK_PATH)
            .header("content-type", "application/json")
            .header("X-Slack-Signature", signature)
            .header("X-Slack-Request-Timestamp", &timestamp)
            .body(Body::from(body.to_vec()))
            .unwrap()
    }

    fn sign_for_test(secret: &str, timestamp: &str, body: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(format!("v0:{}:", timestamp).as_bytes());
        mac.update(body);
        format!("v0={}", hex::encode(mac.finalize().into_bytes()))
    }
}
