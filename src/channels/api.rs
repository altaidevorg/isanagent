use async_trait::async_trait;
use crate::channels::Channel;
use crate::bus::{InboundMessage, OutboundMessage};
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;
use log::{info, error};
use std::sync::Arc;
use dashmap::DashMap;
use axum::{
    routing::post,
    Router, Json,
    extract::State,
};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
struct ApiState {
    inbound_tx: Sender<InboundMessage>,
    pending_requests: Arc<DashMap<String, oneshot::Sender<OutboundMessage>>>,
    channel_name: String,
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

pub struct ApiChannel {
    port: u16,
    pending_requests: Arc<DashMap<String, oneshot::Sender<OutboundMessage>>>,
}

impl ApiChannel {
    pub fn new(port: u16) -> Self {
        Self { 
            port,
            pending_requests: Arc::new(DashMap::new()),
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
        let state = ApiState {
            inbound_tx,
            pending_requests: self.pending_requests.clone(),
            channel_name: self.name().to_string(),
        };

        tokio::spawn(async move {
            let app = Router::new()
                .route("/v1/chat/completions", post(handle_chat))
                .with_state(state);

            let addr = format!("0.0.0.0:{}", port);
            info!("API channel listening on http://{}", addr);

            let listener = tokio::net::TcpListener::bind(&addr).await.expect("Failed to bind API port");
            axum::serve(listener, app).await.expect("API server crashed");
        });

        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        info!("Stopping API channel...");
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        let chat_id = &msg.chat_id;
        if let Some((_, sender)) = self.pending_requests.remove(chat_id) {
            let _ = sender.send(msg);
        } else {
            error!("ApiChannel: No pending request found for chat_id: {}", chat_id);
        }
        Ok(())
    }
}

async fn handle_chat(
    State(state): State<ApiState>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    // We need a unique ID for this synchronous request so we can map the asynchronous response back to it.
    let chat_id = uuid::Uuid::new_v4().to_string();
    let sender_id = payload.user.unwrap_or_else(|| "api_user".to_string());

    let (tx, rx) = oneshot::channel();
    state.pending_requests.insert(chat_id.clone(), tx);

    let msg = InboundMessage {
        channel: state.channel_name,
        sender_id,
        chat_id: chat_id.clone(),
        thread_id: None,
        content: payload.message,
        metadata: Default::default(),
    };

    if let Err(e) = state.inbound_tx.send(msg).await {
        error!("API Handler failed to send inbound message: {}", e);
        state.pending_requests.remove(&chat_id);
        return Json(ChatResponse { response: "Internal agent error (queue full)".to_string() });
    }

    // Wait up to 60 seconds for the agent to reply
    match tokio::time::timeout(std::time::Duration::from_secs(60), rx).await {
        Ok(Ok(outbound)) => {
            Json(ChatResponse {
                response: outbound.content,
            })
        }
        Ok(Err(_)) => {
            Json(ChatResponse { response: "Agent channel closed unexpectedly.".to_string() })
        }
        Err(_) => {
            state.pending_requests.remove(&chat_id);
            Json(ChatResponse { response: "Request timed out waiting for Agent.".to_string() })
        }
    }
}
