use async_trait::async_trait;
use crate::bus::{BusMessage, InboundMessage, LogEvent, OutboundMessage};
use crate::channels::Channel;
use crate::config::SlackConfig;
use crate::logging::LoggerHandle;
use futures_util::{SinkExt, StreamExt};
use log::{error, info, warn};
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::{mpsc::Sender, watch};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

pub struct SlackChannel {
    config: SlackConfig,
    client: Client,
    logger_tx: LoggerHandle,
    shutdown_tx: watch::Sender<bool>,
    task_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl SlackChannel {
    pub fn new(config: SlackConfig, logger_tx: LoggerHandle) -> Self {
        let (shutdown_tx, _) = watch::channel(false);
        Self {
            config,
            client: crate::utils::build_reqwest_client(),
            logger_tx,
            shutdown_tx,
            task_handle: Mutex::new(None),
        }
    }
}

#[async_trait]
impl Channel for SlackChannel {
    fn name(&self) -> &str {
        "slack"
    }

    async fn start(&self, inbound_tx: Sender<InboundMessage>) -> Result<(), String> {
        let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info("SlackChannel", "Starting Slack channel...")));

        let app_token = self.config.app_token.clone();
        let client = self.client.clone();
        let channel_name = self.name().to_string();
        let config = self.config.clone();
        let logger_tx = self.logger_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        let handle = tokio::spawn(async move {
            let mut backoff_secs = 2;
            let mut bot_user_id: Option<String> = None;
            let mut user_names_cache: HashMap<String, String> = HashMap::new();

            'outer: loop {
                if *shutdown_rx.borrow() {
                    break;
                }

                if bot_user_id.is_none() {
                    info!("Requesting Slack bot user ID...");
                    let auth_res = client
                        .post("https://slack.com/api/auth.test")
                        .header("Authorization", format!("Bearer {}", config.bot_token))
                        .send()
                        .await;

                    if let Ok(r) = auth_res {
                        if let Ok(json) = r.json::<Value>().await {
                            if json["ok"].as_bool() == Some(true) {
                                if let Some(uid) = json["user_id"].as_str() {
                                    bot_user_id = Some(uid.to_string());
                                    let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                                        "SlackChannel",
                                        &format!("Slack bot connected as {}", uid),
                                    )));
                                }
                            } else {
                                let _ = logger_tx.send(BusMessage::Log(LogEvent::warn(
                                    "SlackChannel",
                                    &format!("Slack auth.test failed: {:?}", json),
                                )));
                            }
                        }
                    } else {
                        let _ = logger_tx.send(BusMessage::Log(LogEvent::warn(
                            "SlackChannel",
                            "Failed to request Slack auth.test",
                        )));
                    }
                }

                let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                    "SlackChannel",
                    "Requesting Slack Socket Mode URL...",
                )));
                let res = client
                    .post("https://slack.com/api/apps.connections.open")
                    .header("Authorization", format!("Bearer {}", app_token))
                    .header("Content-type", "application/x-www-form-urlencoded")
                    .send()
                    .await;

                let ws_url = match res {
                    Ok(r) => {
                        if let Ok(json) = r.json::<Value>().await {
                            if json["ok"].as_bool() == Some(true) {
                                json["url"].as_str().unwrap_or_default().to_string()
                            } else {
                                let _ = logger_tx.send(BusMessage::Log(LogEvent::error(
                                    "SlackChannel",
                                    &format!("Slack apps.connections.open failed: {:?}", json),
                                )));
                                if wait_for_shutdown_or_timeout(
                                    &mut shutdown_rx,
                                    tokio::time::Duration::from_secs(backoff_secs),
                                )
                                .await
                                {
                                    break;
                                }
                                backoff_secs = std::cmp::min(backoff_secs * 2, 60);
                                continue;
                            }
                        } else {
                            let _ = logger_tx.send(BusMessage::Log(LogEvent::error(
                                "SlackChannel",
                                "Failed to parse Slack response",
                            )));
                            if wait_for_shutdown_or_timeout(
                                &mut shutdown_rx,
                                tokio::time::Duration::from_secs(backoff_secs),
                            )
                            .await
                            {
                                break;
                            }
                            backoff_secs = std::cmp::min(backoff_secs * 2, 60);
                            continue;
                        }
                    }
                    Err(e) => {
                        let _ = logger_tx.send(BusMessage::Log(LogEvent::error(
                            "SlackChannel",
                            &format!("Failed to request Slack websockets URL: {}", e),
                        )));
                        if wait_for_shutdown_or_timeout(
                            &mut shutdown_rx,
                            tokio::time::Duration::from_secs(backoff_secs),
                        )
                        .await
                        {
                            break;
                        }
                        backoff_secs = std::cmp::min(backoff_secs * 2, 60);
                        continue;
                    }
                };

                let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                    "SlackChannel",
                    "Connecting to Slack Socket Mode...",
                )));
                let (ws_stream, _) = match connect_async(&ws_url).await {
                    Ok(stream) => {
                        backoff_secs = 2;
                        stream
                    }
                    Err(e) => {
                        let _ = logger_tx.send(BusMessage::Log(LogEvent::error(
                            "SlackChannel",
                            &format!("WebSocket connection failed: {}", e),
                        )));
                        if wait_for_shutdown_or_timeout(
                            &mut shutdown_rx,
                            tokio::time::Duration::from_secs(backoff_secs),
                        )
                        .await
                        {
                            break;
                        }
                        backoff_secs = std::cmp::min(backoff_secs * 2, 60);
                        continue;
                    }
                };

                let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                    "SlackChannel",
                    "Slack Socket Mode connected successfully.",
                )));
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
                        if let Ok(payload) = serde_json::from_str::<Value>(&text) {
                            if let Some(envelope_id) = payload.get("envelope_id").and_then(|v| v.as_str()) {
                                let ack = json!({ "envelope_id": envelope_id });
                                if let Err(e) = write.send(Message::Text(ack.to_string().into())).await {
                                    error!("Failed to ack Slack envelope: {}", e);
                                }
                            }

                            if payload["type"].as_str() != Some("events_api") {
                                continue;
                            }

                            let event = &payload["payload"]["event"];
                            let ev_type = event["type"].as_str().unwrap_or_default();

                            if ev_type != "message" && ev_type != "app_mention" {
                                continue;
                            }

                            if event.get("bot_id").is_some() || event.get("subtype").is_some() {
                                continue;
                            }

                            let text = event["text"].as_str().unwrap_or_default().to_string();
                            if ev_type == "message" {
                                if let Some(ref uid) = bot_user_id {
                                    if text.contains(&format!("<@{uid}>")) {
                                        continue;
                                    }
                                }
                            }

                            let user = event["user"].as_str().unwrap_or_default().to_string();
                            let mut display_name = user.clone();
                            if let Some(name) = user_names_cache.get(&user) {
                                display_name = name.clone();
                            } else if !user.is_empty() {
                                let info_url = format!("https://slack.com/api/users.info?user={}", user);
                                let info_res = client
                                    .get(&info_url)
                                    .header("Authorization", format!("Bearer {}", config.bot_token))
                                    .send()
                                    .await;

                                if let Ok(r) = info_res {
                                    if let Ok(json) = r.json::<Value>().await {
                                        if json["ok"].as_bool() == Some(true) {
                                            if let Some(name) = json["user"]["profile"]["display_name"]
                                                .as_str()
                                                .filter(|s| !s.is_empty())
                                                .or_else(|| json["user"]["profile"]["real_name"].as_str())
                                            {
                                                user_names_cache.insert(user.clone(), name.to_string());
                                                display_name = name.to_string();
                                            }
                                        }
                                    }
                                }
                            }

                            let chat_id = event["channel"].as_str().unwrap_or_default().to_string();
                            let ts = event.get("ts").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                            let mut thread_ts =
                                event.get("thread_ts").and_then(|v| v.as_str()).map(|s| s.to_string());

                            if thread_ts.is_none()
                                && config.reply_in_thread.unwrap_or(false)
                                && !chat_id.starts_with('D')
                            {
                                thread_ts = Some(ts.clone());
                            }

                            let mut stripped_text = text.clone();
                            if let Some(ref uid) = bot_user_id {
                                let mention = format!("<@{uid}>");
                                if stripped_text.contains(&mention) {
                                    stripped_text = stripped_text.replace(&mention, "").trim().to_string();
                                }
                            } else if let Some(idx) = stripped_text.find("> ") {
                                if stripped_text.starts_with("<@") {
                                    stripped_text = stripped_text[idx + 2..].to_string();
                                }
                            }

                            let payload_text =
                                format!("(Slack User: {}) {}", display_name, stripped_text).trim().to_string();

                            let msg = InboundMessage {
                                channel: channel_name.clone(),
                                sender_id: user,
                                chat_id: chat_id.clone(),
                                thread_id: thread_ts,
                                content: payload_text,
                                metadata: HashMap::new(),
                            };

                            if let Err(e) = inbound_tx.send(msg).await {
                                warn!("Failed to route InboundMessage from Slack: {}", e);
                            } else {
                                let emoji = config
                                    .reaction_emoji
                                    .clone()
                                    .unwrap_or_else(|| "eyes".to_string());
                                if !emoji.is_empty() && !ts.is_empty() {
                                    let bot_token = config.bot_token.clone();
                                    let channel = chat_id.clone();
                                    let ts_clone = ts.clone();

                                    tokio::spawn(async move {
                                        let req_client = crate::utils::build_reqwest_client();
                                        let body = json!({
                                            "channel": channel,
                                            "timestamp": ts_clone,
                                            "name": emoji
                                        });
                                        let res = req_client
                                            .post("https://slack.com/api/reactions.add")
                                            .header("Authorization", format!("Bearer {}", bot_token))
                                            .json(&body)
                                            .send()
                                            .await;

                                        match res {
                                            Ok(response) => {
                                                let status = response.status();
                                                if !status.is_success() {
                                                    let err_text = response.text().await.unwrap_or_default();
                                                    warn!(
                                                        "Failed to add Slack emoji reaction. API returned: {} - {}",
                                                        status, err_text
                                                    );
                                                }
                                            }
                                            Err(e) => {
                                                warn!("Network error adding Slack emoji reaction: {}", e);
                                            }
                                        }
                                    });
                                }
                            }
                        }
                    }
                }

                warn!("Slack Socket Mode disconnected. Reconnecting in {} seconds...", backoff_secs);
                if wait_for_shutdown_or_timeout(
                    &mut shutdown_rx,
                    tokio::time::Duration::from_secs(backoff_secs),
                )
                .await
                {
                    break;
                }
                backoff_secs = std::cmp::min(backoff_secs * 2, 60);
            }

            let _ = logger_tx.send(BusMessage::Log(LogEvent::info("SlackChannel", "Slack channel stopped.")));
        });

        *self.task_handle.lock().unwrap() = Some(handle);
        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        info!("Stopping Slack channel...");
        let _ = self.shutdown_tx.send(true);
        let handle = self.task_handle.lock().unwrap().take();
        if let Some(handle) = handle {
            let _ = handle.await;
        }
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        let bot_token = &self.config.bot_token;

        let mut body = json!({
            "channel": msg.chat_id,
            "text": msg.content,
        });

        if let Some(ts) = msg.thread_id {
            if !msg.chat_id.starts_with('D') {
                body.as_object_mut()
                    .unwrap()
                    .insert("thread_ts".to_string(), Value::String(ts));
            }
        }

        let max_retries = 3;
        let mut backoff_secs = 2;

        for attempt in 1..=max_retries {
            let res = self
                .client
                .post("https://slack.com/api/chat.postMessage")
                .header("Authorization", format!("Bearer {}", bot_token))
                .json(&body)
                .send()
                .await;

            match res {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return Ok(());
                    } else if status.is_server_error() {
                        let text = response.text().await.unwrap_or_default();
                        error!("Slack postMessage 5xx error (attempt {}): {} - {}", attempt, status, text);
                    } else {
                        let text = response.text().await.unwrap_or_default();
                        error!("Slack postMessage fatal 4xx error: {} - {}", status, text);
                        return Err("Slack API returned fatal 4xx error".to_string());
                    }
                }
                Err(e) => {
                    error!("Slack postMessage network error (attempt {}): {}", attempt, e);
                }
            }

            if attempt < max_retries {
                tokio::time::sleep(tokio::time::Duration::from_secs(backoff_secs)).await;
                backoff_secs *= 2;
            }
        }

        error!("Slack send failed after {} attempts.", max_retries);
        Err("Slack send max retries exceeded".to_string())
    }
}

async fn wait_for_shutdown_or_timeout(
    shutdown_rx: &mut watch::Receiver<bool>,
    duration: tokio::time::Duration,
) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(duration) => false,
        changed = shutdown_rx.changed() => changed.is_ok() && *shutdown_rx.borrow(),
    }
}
