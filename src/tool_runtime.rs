//! Per–tool-call context for tools that need session identity (e.g. [`crate::tools::workflow::AskUserTool`]).
//!
//! Set by the agent around each tool invocation via [`with_tool_exec_scope`].

use crate::bus::clarification_session_key;
use std::cell::RefCell;
use std::future::Future;

/// Identity for the session executing a tool (matches `AgentLogic` memory key format).
#[derive(Clone, Debug)]
pub struct ToolExecCtx {
    pub session_key: String,
    pub channel: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
    /// Cancellation token for the **current** reasoning loop (parent or sub-agent), when set.
    /// Used by harness tools to link child work to parent cancellation policy.
    pub reasoning_cancel: Option<tokio_util::sync::CancellationToken>,
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
            reasoning_cancel: None,
        }
    }

    pub fn with_reasoning_cancel(mut self, token: tokio_util::sync::CancellationToken) -> Self {
        self.reasoning_cancel = Some(token);
        self
    }
}

tokio::task_local! {
    static TOOL_EXEC_CTX: RefCell<Option<ToolExecCtx>>;
}

/// Run `fut` with tool execution context installed for the current async task.
pub async fn with_tool_exec_scope<Fut, T>(ctx: ToolExecCtx, fut: Fut) -> T
where
    Fut: Future<Output = T>,
{
    let cell = RefCell::new(Some(ctx));
    TOOL_EXEC_CTX.scope(cell, fut).await
}

/// Current tool call context, if any.
pub fn current_tool_exec_ctx() -> Option<ToolExecCtx> {
    TOOL_EXEC_CTX
        .try_with(|c| c.borrow().clone())
        .ok()
        .flatten()
}
