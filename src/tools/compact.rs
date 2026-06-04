//! PR-10: agent-triggered compaction tool.
//!
//! The agent calls `compact_context` when it judges the moment is right to
//! free up context — typically after extracting a key result from a noisy
//! exploration. The tool posts a `BusMessage::TriggerCompaction` carrying the
//! `AgentSelf` reason so the eval pipeline can distinguish agent-driven
//! compactions from caller-driven `Manual` triggers (PR-5).
//!
//! ### Why deferred execution
//!
//! The tool does not run compaction synchronously. AGENTS.md's per-chat FIFO
//! invariant says compaction for chat X happens *between* X's turns — never
//! during. The tool fires while a turn is in flight, so the inner
//! `trigger_compaction_with_reason` would refuse with an in-flight error.
//! Instead, the tool enqueues the bus message; the agent's actor loop picks
//! it up after the current turn completes, at which point the per-chat FIFO
//! guard sees no in-flight turn and the compaction runs.
//!
//! The tool's return string tells the LLM that compaction is scheduled (not
//! immediate), so subsequent reasoning in the same turn shouldn't assume the
//! context has already shrunk.

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc::Sender;

use crate::bus::{BusMessage, CompactionTrigger};
use crate::tool_runtime::current_tool_exec_ctx;
use crate::traits::Tool;

/// Tool definition. Construct in `main.rs` with the global outbound bus sender
/// (same one passed to `MessageTool`, `AskUserTool`, etc.) and register it.
pub struct CompactContextTool {
    pub outbound_tx: Sender<BusMessage>,
}

#[async_trait]
impl Tool for CompactContextTool {
    fn name(&self) -> &str {
        "compact_context"
    }

    fn description(&self) -> &str {
        "Request a compaction of this chat's context. Use sparingly — only after \
         you have extracted a concrete result from a noisy exploration and want \
         to drop the exploration noise from future turns. The compaction runs \
         between turns (not during the current turn), so its effect is visible \
         to the next user message rather than to your next reasoning step. Pass \
         `focus_instructions` to bias which content the summarizer keeps."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "focus_instructions": {
                    "type": "string",
                    "description": "Optional natural-language guidance for the summarizer (e.g. \"keep the API design decisions, drop the file-listing exploration\"). Omit if no specific bias is needed."
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let ctx = current_tool_exec_ctx().ok_or_else(|| {
            "compact_context is only available during a live agent turn (missing tool runtime context)."
                .to_string()
        })?;

        let focus = args
            .get("focus_instructions")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        self.outbound_tx
            .send(BusMessage::TriggerCompaction {
                session_key: ctx.session_key.clone(),
                focus_instructions: focus,
                trigger: Some(CompactionTrigger::AgentSelf),
            })
            .await
            .map_err(|e| format!("Failed to enqueue compaction request: {}", e))?;

        Ok(
            "Compaction scheduled. It will run between this turn and the next user message, \
             so the next inbound will see a smaller context. The current reasoning step still \
             sees the full transcript."
                .to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_runtime::{with_tool_exec_scope, ToolExecCtx};
    use tokio::sync::mpsc;

    fn make_tool() -> (CompactContextTool, mpsc::Receiver<BusMessage>) {
        let (tx, rx) = mpsc::channel(4);
        (CompactContextTool { outbound_tx: tx }, rx)
    }

    #[tokio::test]
    async fn emits_trigger_compaction_with_agent_self_reason() {
        let (tool, mut rx) = make_tool();
        let ctx = ToolExecCtx::new("terminal", "u1", None);
        let session_key = ctx.session_key.clone();
        let result = with_tool_exec_scope(ctx, async { tool.execute(serde_json::json!({})).await })
            .await
            .expect("tool execute");
        assert!(result.contains("Compaction scheduled"));

        let msg = rx.recv().await.expect("bus message");
        match msg {
            BusMessage::TriggerCompaction {
                session_key: sk,
                focus_instructions,
                trigger,
            } => {
                assert_eq!(sk, session_key);
                assert_eq!(focus_instructions, None);
                assert!(matches!(trigger, Some(CompactionTrigger::AgentSelf)));
            }
            other => panic!("expected TriggerCompaction, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn passes_focus_instructions_through() {
        let (tool, mut rx) = make_tool();
        let ctx = ToolExecCtx::new("api", "c-abc", None);
        with_tool_exec_scope(ctx, async {
            tool.execute(serde_json::json!({
                "focus_instructions": "keep the API design talk"
            }))
            .await
        })
        .await
        .expect("tool execute");

        let msg = rx.recv().await.expect("bus message");
        match msg {
            BusMessage::TriggerCompaction {
                focus_instructions, ..
            } => {
                assert_eq!(
                    focus_instructions.as_deref(),
                    Some("keep the API design talk")
                );
            }
            other => panic!("expected TriggerCompaction, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn blank_focus_instructions_become_none() {
        let (tool, mut rx) = make_tool();
        let ctx = ToolExecCtx::new("api", "c-blank", None);
        with_tool_exec_scope(ctx, async {
            tool.execute(serde_json::json!({"focus_instructions": "   \n  "}))
                .await
        })
        .await
        .expect("tool execute");

        let msg = rx.recv().await.expect("bus message");
        match msg {
            BusMessage::TriggerCompaction {
                focus_instructions, ..
            } => assert_eq!(focus_instructions, None),
            other => panic!("expected TriggerCompaction, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn refuses_when_no_tool_exec_ctx_installed() {
        let (tool, _rx) = make_tool();
        let err = tool
            .execute(serde_json::json!({}))
            .await
            .expect_err("must require tool exec ctx");
        assert!(err.contains("live agent turn"));
    }
}
