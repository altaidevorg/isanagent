use async_trait::async_trait;
use crate::channels::Channel;
use crate::bus::{InboundMessage, OutboundMessage};
use crate::config::SlackConfig;
use tokio::sync::mpsc::Sender;
use log::{info, error, warn};
use reqwest::Client;
use serde_json::{Value, json};
use futures_util::{StreamExt, SinkExt};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

pub struct SlackChannel {
    config: SlackConfig,
    client: Client,
}

impl SlackChannel {
    pub fn new(config: SlackConfig) -> Self {
        Self { config, client: Client::new() }
    }
}

#[async_trait]
impl Channel for SlackChannel {
    fn name(&self) -> &str {
        "slack"
    }

    async fn start(&self, inbound_tx: Sender<InboundMessage>) -> Result<(), String> {
        info!("Starting Slack channel...");
        
        let app_token = self.config.app_token.clone();
        let client = self.client.clone();
        let channel_name = self.name().to_string();
        let config = self.config.clone();

        tokio::spawn(async move {
            let res = client.post("https://slack.com/api/apps.connections.open")
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
                            error!("Slack apps.connections.open failed: {:?}", json);
                            return;
                        }
                    } else {
                        error!("Failed to parse Slack response");
                        return;
                    }
                }
                Err(e) => {
                    error!("Failed to request Slack websockets URL: {}", e);
                    return;
                }
            };

            info!("Connecting to Slack Socket Mode...");
            let (ws_stream, _) = match connect_async(&ws_url).await {
                Ok(stream) => stream,
                Err(e) => {
                    error!("WebSocket connection failed: {}", e);
                    return;
                }
            };
            
            info!("Slack Socket Mode connected successfully.");
            let (mut write, mut read) = ws_stream.split();

            while let Some(msg) = read.next().await {
                let msg = match msg {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Slack websocket read error: {}", e);
                        continue;
                    }
                };

                if let Message::Text(text) = msg {
                    if let Ok(payload) = serde_json::from_str::<Value>(&text) {
                        // 1. Always Acknowledge the envelope ASAP
                        if let Some(envelope_id) = payload.get("envelope_id").and_then(|v| v.as_str()) {
                            let ack = json!({ "envelope_id": envelope_id });
                            if let Err(e) = write.send(Message::Text(ack.to_string().into())).await {
                                error!("Failed to ack Slack envelope: {}", e);
                            }
                        }

                        // 2. Parse payload type
                        if payload["type"].as_str() != Some("events_api") {
                            continue;
                        }

                        let event = &payload["payload"]["event"];
                        let ev_type = event["type"].as_str().unwrap_or_default();
                        
                        if ev_type == "message" || ev_type == "app_mention" {
                            // Ignore bot messages
                            if event.get("bot_id").is_some() || event.get("subtype").is_some() {
                                continue;
                            }

                            let text = event["text"].as_str().unwrap_or_default().to_string();
                            let user = event["user"].as_str().unwrap_or_default().to_string();
                            let chat_id = event["channel"].as_str().unwrap_or_default().to_string();
                            
                            let mut thread_ts = event.get("thread_ts").and_then(|v| v.as_str()).map(|s| s.to_string());
                            
                            // Apply `reply_in_thread` logic for non-DM channels. DM channels in Slack start with 'D'.
                            if thread_ts.is_none() && config.reply_in_thread.unwrap_or(false) && !chat_id.starts_with('D') {
                                thread_ts = event.get("ts").and_then(|v| v.as_str()).map(|s| s.to_string());
                            }

                            let mut stripped_text = text.clone();
                            // Very naive strip of bot mention. E.g. <@U123456>
                            if let Some(idx) = stripped_text.find("> ") {
                                if stripped_text.starts_with("<@") {
                                    stripped_text = stripped_text[idx + 2..].to_string();
                                }
                            }

                            let msg = InboundMessage {
                                channel: channel_name.clone(),
                                sender_id: user,
                                chat_id: chat_id.clone(),
                                thread_id: thread_ts,
                                content: stripped_text,
                                metadata: std::collections::HashMap::new(),
                            };

                            if let Err(e) = inbound_tx.send(msg).await {
                                warn!("Failed to route InboundMessage from Slack: {}", e);
                            }
                        }
                    }
                }
            }
            error!("Slack Socket Mode disconnected.");
        });

        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        info!("Stopping Slack channel...");
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
                body.as_object_mut().unwrap().insert("thread_ts".to_string(), Value::String(ts));
            }
        }

        let res = self.client.post("https://slack.com/api/chat.postMessage")
            .header("Authorization", format!("Bearer {}", bot_token))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Reqwest error: {}", e))?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
            error!("Slack postMessage failed: {} - {}", status, text);
            return Err("Slack API returned error".to_string());
        }

        Ok(())
    }
}
