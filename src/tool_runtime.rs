//! Per–tool-call context for tools that need session identity (e.g. [`crate::tools::workflow::AskUserTool`]).
//!
//! Set by the agent around each tool invocation via [`with_tool_exec_scope`] or
//! [`with_tool_exec_and_progress_scope`].

use crate::bus::{clarification_session_key, BusMessage, TelemetryEvent};
use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Identity for the session executing a tool (matches `AgentLogic` memory key format).
#[derive(Clone, Debug)]
pub struct ToolExecCtx {
    pub session_key: String,
    pub channel: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
    /// Optional tool call ID from the LLM; used by tools like ask_user to link
    /// background tickets back to the specific function call.
    pub tool_call_id: Option<String>,
    /// True when execution is detached/background and should avoid blocking user-interaction waits.
    pub is_background: bool,
    /// Cancellation token for the **current** reasoning loop (parent or sub-agent), when set.
    /// Used by harness tools to link child work to parent cancellation policy.
    pub reasoning_cancel: Option<tokio_util::sync::CancellationToken>,
    /// Metadata from the inbound message that triggered the reasoning loop.
    pub inbound_metadata: Arc<HashMap<String, serde_json::Value>>,
}

impl ToolExecCtx {
    pub fn new(
        channel: impl Into<String>,
        chat_id: impl Into<String>,
        thread_id: Option<String>,
    ) -> Self {
        let channel = channel.into();
        let chat_id = chat_id.into();
        let session_key = clarification_session_key(&channel, &chat_id, thread_id.as_deref());
        Self {
            session_key,
            channel,
            chat_id,
            thread_id,
            tool_call_id: None,
            is_background: false,
            reasoning_cancel: None,
            inbound_metadata: Arc::new(HashMap::new()),
        }
    }

    pub fn with_tool_call_id(mut self, tool_call_id: Option<String>) -> Self {
        self.tool_call_id = tool_call_id;
        self
    }

    pub fn with_reasoning_cancel(mut self, token: tokio_util::sync::CancellationToken) -> Self {
        self.reasoning_cancel = Some(token);
        self
    }

    pub fn with_background(mut self, is_background: bool) -> Self {
        self.is_background = is_background;
        self
    }

    pub fn with_metadata(mut self, metadata: Arc<HashMap<String, serde_json::Value>>) -> Self {
        self.inbound_metadata = metadata;
        self
    }
}

/// Bus + routing fields for [`emit_tool_progress_message`] during a tool call.
#[derive(Clone, Debug)]
pub struct ToolProgressEmitter {
    pub outbound_tx: mpsc::Sender<BusMessage>,
    pub channel: String,
    pub chat_id: String,
    pub tool_name: String,
    pub tool_call_id: Option<String>,
    pub background_job_id: Option<String>,
}

tokio::task_local! {
    static TOOL_EXEC_CTX: RefCell<Option<ToolExecCtx>>;
}

tokio::task_local! {
    static TOOL_PROGRESS_EMITTER: RefCell<Option<ToolProgressEmitter>>;
}

/// Run `fut` with tool execution context installed for the current async task.
pub async fn with_tool_exec_scope<Fut, T>(ctx: ToolExecCtx, fut: Fut) -> T
where
    Fut: Future<Output = T>,
{
    let cell = RefCell::new(Some(ctx));
    TOOL_EXEC_CTX.scope(cell, fut).await
}

/// Run `fut` with tool execution context and optional progress emitter (nested task-locals).
pub async fn with_tool_exec_and_progress_scope<Fut, T>(
    ctx: ToolExecCtx,
    progress: ToolProgressEmitter,
    fut: Fut,
) -> T
where
    Fut: Future<Output = T>,
{
    let prog_cell = RefCell::new(Some(progress));
    let ctx_cell = RefCell::new(Some(ctx));
    TOOL_PROGRESS_EMITTER
        .scope(prog_cell, async {
            TOOL_EXEC_CTX.scope(ctx_cell, fut).await
        })
        .await
}

/// Current tool call context, if any.
pub fn current_tool_exec_ctx() -> Option<ToolExecCtx> {
    TOOL_EXEC_CTX
        .try_with(|c| c.borrow().clone())
        .ok()
        .flatten()
}

fn current_tool_progress_emitter() -> Option<ToolProgressEmitter> {
    TOOL_PROGRESS_EMITTER
        .try_with(|c| c.borrow().clone())
        .ok()
        .flatten()
}

/// Emit a mid–tool-call status line (telemetry) when a progress emitter is installed.
pub async fn emit_tool_progress_message(message: &str) {
    let Some(emitter) = current_tool_progress_emitter() else {
        return;
    };
    let msg = message.trim();
    if msg.is_empty() {
        return;
    }
    let _ = emitter
        .outbound_tx
        .send(BusMessage::Telemetry(TelemetryEvent::ToolProgress {
            chat_id: emitter.chat_id.clone(),
            channel: emitter.channel.clone(),
            tool_name: emitter.tool_name.clone(),
            tool_call_id: emitter.tool_call_id.clone(),
            message: msg.to_string(),
            background_job_id: emitter.background_job_id.clone(),
        }))
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn emit_tool_progress_sends_telemetry_under_nested_scope() {
        let (tx, mut rx) = mpsc::channel::<BusMessage>(8);
        let ctx = ToolExecCtx::new("api", "chat-xyz", None);
        let emitter = ToolProgressEmitter {
            outbound_tx: tx,
            channel: "api".to_string(),
            chat_id: "chat-xyz".to_string(),
            tool_name: "execution_session_create".to_string(),
            tool_call_id: Some("call-1".to_string()),
            background_job_id: None,
        };
        with_tool_exec_and_progress_scope(ctx, emitter, async {
            emit_tool_progress_message("Creating Python environment with uv…").await;
        })
        .await;

        let msg = rx.recv().await.expect("one message");
        match msg {
            BusMessage::Telemetry(TelemetryEvent::ToolProgress {
                chat_id,
                channel,
                tool_name,
                tool_call_id,
                message,
                ..
            }) => {
                assert_eq!(chat_id, "chat-xyz");
                assert_eq!(channel, "api");
                assert_eq!(tool_name, "execution_session_create");
                assert_eq!(tool_call_id.as_deref(), Some("call-1"));
                assert!(message.contains("uv"));
            }
            other => panic!("unexpected message: {:?}", other),
        }
    }
}
