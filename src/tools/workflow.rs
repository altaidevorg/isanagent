//! Session-scoped workflow tools (todos, tool discovery, user clarification).

use async_trait::async_trait;
use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

use crate::bus::{BusMessage, OutboundMessage};
use crate::clarification::{
    ClarificationHub, METADATA_CLARIFICATION, METADATA_CLARIFICATION_CHOICES,
};
use crate::memory::{MemoryMessage, SharedReply};
use crate::tool_runtime::current_tool_exec_ctx;
use crate::traits::Tool;
use crate::NodeHandle;

use super::search_tool_index;

pub use crate::memory::TodoRow;

const MAX_TODO_ITEMS: usize = 200;

/// Replace the structured todo list for this chat session (persists via [`SqliteMemoryActor`]).
pub struct TodoWriteTool {
    pub memory_node: NodeHandle<MemoryMessage>,
}

fn normalize_todo_status(s: &str) -> Result<(), String> {
    match s {
        "pending" | "in_progress" | "completed" => Ok(()),
        _ => Err(format!(
            "Invalid status {:?}; use pending, in_progress, or completed.",
            s
        )),
    }
}

fn format_todo_list(chat_id: &str, rows: &[TodoRow]) -> String {
    let mut out = String::from("# Todo list\n\n");
    if rows.is_empty() {
        out.push_str("(empty)\n");
        return out;
    }
    let icon = |s: &str| match s {
        "completed" => "[x]",
        "in_progress" => "[~]",
        _ => "[ ]",
    };
    for (i, row) in rows.iter().enumerate() {
        out.push_str(&format!(
            "{}. {} {}\n",
            i + 1,
            icon(&row.status),
            row.content
        ));
    }
    let done = rows.iter().filter(|r| r.status == "completed").count();
    out.push_str(&format!(
        "\nSession: {}\nProgress: {}/{}\n",
        chat_id,
        done,
        rows.len()
    ));
    out
}

#[async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &str {
        "todo_write"
    }

    fn description(&self) -> &str {
        "Replace the structured todo list for this chat. Todos are scoped per chat_id (from RUNTIME CONTEXT) and stored in the agent SQLite database (harness_todos) so they survive restarts. Use for multi-step work tracking."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "Chat/session id from RUNTIME CONTEXT (same as for the message tool)."
                },
                "items": {
                    "type": "array",
                    "description": "Complete new todo list (replaces any previous list for this chat).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Task state"
                            }
                        },
                        "required": ["content", "status"]
                    }
                }
            },
            "required": ["chat_id", "items"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let chat_id = args
            .get("chat_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'chat_id'")?;

        let items_val = args
            .get("items")
            .and_then(|v| v.as_array())
            .ok_or("Missing 'items' array")?;

        if items_val.len() > MAX_TODO_ITEMS {
            return Err(format!(
                "At most {} todo items allowed (got {}).",
                MAX_TODO_ITEMS,
                items_val.len()
            ));
        }

        let mut rows: Vec<TodoRow> = Vec::with_capacity(items_val.len());
        for (i, item) in items_val.iter().enumerate() {
            let content = item
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("items[{}]: missing content", i))?
                .to_string();
            let status_raw = item
                .get("status")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("items[{}]: missing status", i))?;
            normalize_todo_status(status_raw).map_err(|e| format!("items[{}]: {}", i, e))?;
            rows.push(TodoRow {
                content,
                status: status_raw.to_string(),
            });
        }

        let summary = format_todo_list(chat_id, &rows);
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.memory_node
            .send_packet(MemoryMessage::ReplaceHarnessTodos {
                chat_id: chat_id.to_string(),
                items: rows,
                reply: SharedReply::new(tx),
            })
            .await
            .map_err(|e| format!("todo_write: memory actor: {}", e))?;
        rx.await
            .map_err(|_| "Memory actor channel closed".to_string())?
            .map_err(|e| format!("Failed to save todos: {}", e))?;
        Ok(summary)
    }
}

/// Search registered tools by keywords (name + description).
pub struct ToolSearchTool {
    pub catalog: Arc<RwLock<Vec<(String, String)>>>,
}

#[async_trait]
impl Tool for ToolSearchTool {
    fn name(&self) -> &str {
        "search_tools"
    }

    fn description(&self) -> &str {
        "Find built-in tools by keyword or short phrase. Use when unsure which tool fits a task. Matches tool names and descriptions."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for (e.g. 'grep', 'schedule', 'memory')."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 12, max 40)."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query'")?;

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(12)
            .clamp(1, 40) as usize;

        let entries = self
            .catalog
            .read()
            .map_err(|e| format!("catalog lock: {}", e))?
            .clone();

        let hits = search_tool_index(&entries, query, limit);
        if hits.is_empty() {
            return Ok("No tools matched that query.".to_string());
        }

        let mut out = String::from("Matching tools:\n\n");
        for (name, score) in hits {
            let desc = entries
                .iter()
                .find(|(n, _)| n == &name)
                .map(|(_, d)| d.as_str())
                .unwrap_or("");
            let snippet: String = desc.chars().take(160).collect();
            let ellipses = if desc.len() > 160 { "…" } else { "" };
            out.push_str(&format!(
                "- **{}** (score {})\n  {}{}\n\n",
                name, score, snippet, ellipses
            ));
        }
        Ok(out.trim_end().to_string())
    }
}

struct ClarificationSlotGuard {
    hub: Arc<ClarificationHub>,
    session_key: String,
    armed: bool,
}

impl ClarificationSlotGuard {
    fn new(hub: Arc<ClarificationHub>, session_key: String) -> Self {
        Self {
            hub,
            session_key,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ClarificationSlotGuard {
    fn drop(&mut self) {
        if self.armed {
            self.hub.cancel_wait(&self.session_key);
        }
    }
}

/// Block until the user sends the next message in this session, then return it as the tool result.
///
/// Requires the agent to wrap tool execution with a [`crate::tool_runtime::ToolExecCtx`]. The
/// channel delivers an [`OutboundMessage`] tagged with [`METADATA_CLARIFICATION`] so terminals and
/// API clients can style the prompt; the following inbound on the same session completes the wait.
pub struct AskUserTool {
    pub clarification_hub: Arc<ClarificationHub>,
    pub outbound_tx: mpsc::Sender<BusMessage>,
    pub memory_node: Option<NodeHandle<MemoryMessage>>,
}

const ASK_USER_TIMEOUT_SECS_MIN: u64 = 10;
const ASK_USER_TIMEOUT_SECS_MAX: u64 = 86_400;
const ASK_USER_MAX_CHOICES: usize = 8;

/// Exact option text wins; otherwise `1`..=`choices.len()` selects by 1-based index.
fn resolve_ask_user_choice(trimmed: &str, choices: &[String]) -> Option<String> {
    if choices.is_empty() {
        return None;
    }
    if choices.iter().any(|c| c.as_str() == trimmed) {
        return Some(trimmed.to_string());
    }
    let n: usize = trimmed.parse().ok()?;
    if n >= 1 && n <= choices.len() {
        return Some(choices[n - 1].clone());
    }
    None
}

#[async_trait]
impl Tool for AskUserTool {
    fn name(&self) -> &str {
        "ask_user"
    }

    fn description(&self) -> &str {
        "Ask the human a focused question and wait for their next reply in this chat. Use when you need a decision, missing detail, or confirmation before continuing. The user’s following message becomes this tool’s return value (not a new agent turn). Works in terminal and API channels when inbound messages reach the same session. When `choices` is set, the user may answer with the exact option text or a 1-based index (1 = first choice)."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Clear question for the user (plain text)."
                },
                "choices": {
                    "type": "array",
                    "description": "Optional short list of allowed answers (max 8); UIs may show numbered options. The user may reply with exact text or 1-based index (1 = first item).",
                    "items": { "type": "string" },
                    "maxItems": 8
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Optional max seconds to wait (10–86400). If omitted, waits without timeout."
                },
                "allow_empty": {
                    "type": "boolean",
                    "description": "If false (default), treat whitespace-only replies as invalid and keep waiting until timeout."
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional UI-only structured metadata attached to the clarification event. Do not put secrets here."
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let ctx = current_tool_exec_ctx().ok_or_else(|| {
            "ask_user is only available during a live agent turn (missing tool runtime context)."
                .to_string()
        })?;

        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'prompt'".to_string())?
            .trim();
        if prompt.is_empty() {
            return Err("prompt must be non-empty".to_string());
        }

        let allow_empty = args
            .get("allow_empty")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .map(|v| v.clamp(ASK_USER_TIMEOUT_SECS_MIN, ASK_USER_TIMEOUT_SECS_MAX));

        let mut choices: Vec<String> = Vec::new();
        if let Some(arr) = args.get("choices").and_then(|v| v.as_array()) {
            if arr.len() > ASK_USER_MAX_CHOICES {
                return Err(format!(
                    "At most {} choices allowed (got {}).",
                    ASK_USER_MAX_CHOICES,
                    arr.len()
                ));
            }
            for (i, v) in arr.iter().enumerate() {
                let s = v
                    .as_str()
                    .ok_or_else(|| format!("choices[{}]: expected string", i))?
                    .trim();
                if s.is_empty() {
                    return Err(format!("choices[{}]: must be non-empty", i));
                }
                choices.push(s.to_string());
            }
        }

        if ctx.is_background {
            let memory_node = self.memory_node.as_ref().ok_or_else(|| {
                "Background ask_user requires a memory node for clarification tickets".to_string()
            })?;

            let ticket_id = uuid::Uuid::new_v4().to_string();
            let now = Utc::now().timestamp_millis();

            let job_id = ctx
                .inbound_metadata
                .get(crate::bus::METADATA_BACKGROUND_JOB_ID)
                .and_then(|v| v.as_str())
                .unwrap_or(&ctx.chat_id)
                .to_string();

            {
                let (tx, rx) = tokio::sync::oneshot::channel();
                memory_node
                    .send_packet(MemoryMessage::UpsertClarificationTicket {
                        record: crate::memory::ClarificationTicketRecord {
                            ticket_id: ticket_id.clone(),
                            job_id: job_id.clone(),
                            chat_id: ctx.chat_id.clone(),
                            channel: ctx.channel.clone(),
                            thread_id: ctx.thread_id.clone(),
                            tool_call_id: ctx.tool_call_id.clone(),
                            prompt: prompt.to_string(),
                            choices_json: if choices.is_empty() {
                                None
                            } else {
                                Some(
                                    serde_json::to_string(&choices)
                                        .map_err(|e| format!("serialize choices: {}", e))?,
                                )
                            },
                            response: None,
                            status: "waiting".to_string(),
                            created_at_ms: now,
                            updated_at_ms: now,
                        },
                        reply: SharedReply::new(tx),
                    })
                    .await
                    .map_err(|e| format!("clarification ticket enqueue: {}", e))?;
                rx.await
                    .map_err(|_| "clarification ticket actor channel closed".to_string())?
                    .map_err(|e| format!("clarification ticket: {}", e))?;

                let notification_id = uuid::Uuid::new_v4().to_string();
                let (ntx, nrx) = tokio::sync::oneshot::channel();
                memory_node
                    .send_packet(MemoryMessage::InsertNotification {
                        record: crate::memory::NotificationRecord {
                            notification_id: notification_id.clone(),
                            chat_id: ctx.chat_id.clone(),
                            channel: ctx.channel.clone(),
                            thread_id: ctx.thread_id.clone(),
                            kind: "clarification_ticket".to_string(),
                            title: "Background input required".to_string(),
                            body: prompt.to_string(),
                            action_kind: Some("reply_ticket".to_string()),
                            action_payload: Some(ticket_id.clone()),
                            seen_at_ms: None,
                            resolved_at_ms: None,
                            created_at_ms: now,
                        },
                        reply: SharedReply::new(ntx),
                    })
                    .await
                    .map_err(|e| format!("notification enqueue: {}", e))?;
                nrx.await
                    .map_err(|_| "notification actor channel closed".to_string())?
                    .map_err(|e| format!("notification: {}", e))?;
                let _ = self
                    .outbound_tx
                    .send(BusMessage::Telemetry(
                        crate::bus::TelemetryEvent::NotificationCreated {
                            notification_id: notification_id.clone(),
                            chat_id: ctx.chat_id.clone(),
                            channel: ctx.channel.clone(),
                            kind: "clarification_ticket".to_string(),
                            title: "Background input required".to_string(),
                        },
                    ))
                    .await;
            }
            let mut metadata = HashMap::new();
            metadata.insert(
                "isanagent_notification".to_string(),
                serde_json::Value::Bool(true),
            );
            metadata.insert(
                "isanagent_notification_kind".to_string(),
                serde_json::Value::String("clarification_ticket".to_string()),
            );
            metadata.insert(
                crate::bus::METADATA_CLARIFICATION_TICKET_ID.to_string(),
                serde_json::Value::String(ticket_id.clone()),
            );
            if let Some(jid) = ctx
                .inbound_metadata
                .get(crate::bus::METADATA_BACKGROUND_JOB_ID)
            {
                metadata.insert(
                    crate::bus::METADATA_BACKGROUND_JOB_ID.to_string(),
                    jid.clone(),
                );
            }
            let outbound = OutboundMessage {
                channel: ctx.channel.clone(),
                chat_id: ctx.chat_id.clone(),
                thread_id: ctx.thread_id.clone(),
                content: format!(
                    "Background task needs input and has been paused.\n\nQuestion: {}\nTicket: {}",
                    prompt, ticket_id
                ),
                metadata,
            };
            self.outbound_tx
                .send(BusMessage::Outbound(outbound))
                .await
                .map_err(|e| format!("failed to send clarification ticket notification: {}", e))?;

            // Update job state to waiting
            let (tx, rx) = tokio::sync::oneshot::channel();
            let _ = memory_node
                .send_packet(crate::memory::MemoryMessage::UpdateBackgroundJobState {
                    job_id: job_id.clone(),
                    state: "waiting".to_string(),
                    last_error: None,
                    reply: crate::memory::SharedReply::new(tx),
                })
                .await;
            let _ = rx.await;

            return Err(format!("{}{}", crate::agent::WAIT_SIGNAL_PREFIX, ticket_id));
        }

        let rx = self
            .clarification_hub
            .begin_wait(&ctx.session_key)
            .map_err(|e| e.to_string())?;

        let mut guard = ClarificationSlotGuard::new(
            Arc::clone(&self.clarification_hub),
            ctx.session_key.clone(),
        );

        let mut body = String::from("The agent needs your input:\n\n");
        body.push_str(prompt);

        let mut metadata = HashMap::new();
        if let Some(extra) = args.get("metadata").and_then(|v| v.as_object()) {
            // Metadata is deliberately restricted to a JSON object and is carried only on
            // the outbound clarification event. Callers must keep it bounded and secret-free.
            metadata.extend(
                extra
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
        }
        metadata.insert(
            METADATA_CLARIFICATION.to_string(),
            serde_json::Value::Bool(true),
        );
        if !choices.is_empty() {
            metadata.insert(
                METADATA_CLARIFICATION_CHOICES.to_string(),
                serde_json::json!(choices),
            );
        }

        let outbound = OutboundMessage {
            channel: ctx.channel.clone(),
            chat_id: ctx.chat_id.clone(),
            thread_id: ctx.thread_id.clone(),
            content: body,
            metadata,
        };

        self.outbound_tx
            .send(BusMessage::Outbound(outbound))
            .await
            .map_err(|e| format!("failed to send clarification prompt: {}", e))?;

        let reply = match timeout_secs {
            Some(timeout_secs) => {
                let wait = tokio::time::Duration::from_secs(timeout_secs);
                match tokio::time::timeout(wait, rx).await {
                    Err(_) => {
                        return Err(format!(
                            "Timed out after {}s waiting for a user reply to ask_user.",
                            timeout_secs
                        ));
                    }
                    Ok(Err(_)) => {
                        return Err(
                            "Clarification wait ended without a reply (session cancelled or reset)."
                                .to_string(),
                        );
                    }
                    Ok(Ok(text)) => text,
                }
            }
            None => rx.await.map_err(|_| {
                "Clarification wait ended without a reply (session cancelled or reset).".to_string()
            })?,
        };

        guard.disarm();

        let trimmed = reply.trim();
        if !allow_empty && trimmed.is_empty() {
            return Err(
                "User reply was empty (allow_empty is false). Call ask_user again if you still need input."
                    .to_string(),
            );
        }

        let canonical_reply = if choices.is_empty() {
            reply.clone()
        } else {
            match resolve_ask_user_choice(trimmed, &choices) {
                Some(s) => s,
                None => {
                    return Ok(format!(
                        "User reply (not among listed choices): {}\n\nListed options were: {:?}",
                        reply, choices
                    ));
                }
            }
        };

        Ok(format!("User reply:\n{}", canonical_reply))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::BusMessage;
    use crate::clarification::METADATA_CLARIFICATION_CHOICES;
    use crate::memory::{MemoryMessage, SharedReply, SqliteMemoryActor};
    use crate::tool_runtime::{with_tool_exec_scope, ToolExecCtx};
    use crate::NodeHandle;
    use serde_json::json;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    fn spawn_memory_node(db: &Path) -> NodeHandle<MemoryMessage> {
        let actor =
            SqliteMemoryActor::new(db.to_str().expect("utf8 db path")).expect("memory actor");
        NodeHandle::new(actor, 100, 1, Duration::from_millis(5))
    }

    async fn load_todos(node: &NodeHandle<MemoryMessage>, chat_id: &str) -> Option<Vec<TodoRow>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        node.send_packet(MemoryMessage::LoadHarnessTodos {
            chat_id: chat_id.to_string(),
            reply: SharedReply::new(tx),
        })
        .await
        .expect("send LoadHarnessTodos");
        rx.await.expect("LoadHarnessTodos reply").expect("sqlite")
    }

    #[tokio::test]
    async fn ask_user_outbound_and_reply() {
        let hub = Arc::new(ClarificationHub::new());
        let (ob_tx, mut ob_rx) = mpsc::channel(8);
        let tool = AskUserTool {
            clarification_hub: hub.clone(),
            outbound_tx: ob_tx,
            memory_node: None,
        };
        let hub_signal = hub.clone();
        let join = tokio::spawn(async move {
            with_tool_exec_scope(ToolExecCtx::new("terminal", "u1", None), async move {
                tool.execute(json!({
                    "prompt": "Which?",
                    "choices": ["Red", "Blue"],
                    "timeout_secs": 30,
                    "allow_empty": true
                }))
                .await
            })
            .await
        });

        let ob = ob_rx.recv().await.expect("outbound");
        match ob {
            BusMessage::Outbound(out) => {
                assert_eq!(
                    out.metadata.get(METADATA_CLARIFICATION),
                    Some(&serde_json::Value::Bool(true))
                );
                assert_eq!(
                    out.metadata.get(METADATA_CLARIFICATION_CHOICES),
                    Some(&json!(["Red", "Blue"]))
                );
                assert!(out.content.contains("Which?"));
                assert!(!out.content.contains("Options:"));
            }
            _ => panic!("expected Outbound"),
        }
        assert!(hub_signal.try_deliver_reply("terminal:u1:", "Red".into()));
        let answer = join.await.expect("join").expect("tool ok");
        assert!(answer.contains("Red"));
    }

    #[tokio::test]
    async fn ask_user_accepts_numeric_choice_index() {
        let hub = Arc::new(ClarificationHub::new());
        let (ob_tx, mut ob_rx) = mpsc::channel(8);
        let tool = AskUserTool {
            clarification_hub: hub.clone(),
            outbound_tx: ob_tx,
            memory_node: None,
        };
        let hub_signal = hub.clone();
        let join = tokio::spawn(async move {
            with_tool_exec_scope(ToolExecCtx::new("terminal", "u2", None), async move {
                tool.execute(json!({
                    "prompt": "Pick",
                    "choices": ["Red", "Blue"],
                    "timeout_secs": 30,
                    "allow_empty": true
                }))
                .await
            })
            .await
        });

        let _ = ob_rx.recv().await.expect("outbound");
        assert!(hub_signal.try_deliver_reply("terminal:u2:", "2".into()));
        let answer = join.await.expect("join").expect("tool ok");
        assert!(answer.contains("Blue"), "{answer}");
    }

    #[test]
    fn resolve_ask_user_choice_exact_wins_over_index() {
        let c = vec!["1".into(), "Red".into()];
        assert_eq!(resolve_ask_user_choice("1", &c), Some("1".into()));
    }

    #[test]
    fn resolve_ask_user_choice_by_one_based_index() {
        let c = vec!["Red".into(), "Blue".into()];
        assert_eq!(resolve_ask_user_choice("2", &c), Some("Blue".into()));
    }

    #[test]
    fn resolve_ask_user_choice_rejects_invalid_index() {
        let c = vec!["A".into(), "B".into()];
        assert!(resolve_ask_user_choice("3", &c).is_none());
        assert!(resolve_ask_user_choice("0", &c).is_none());
    }

    fn temp_todo_db(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "isanagent_todos_{}_{}.sqlite",
            name,
            uuid::Uuid::new_v4()
        ))
    }

    #[tokio::test]
    async fn todo_write_isolated_per_chat_id() {
        let db = temp_todo_db("isolate");
        let memory_node = spawn_memory_node(&db);
        let tool = TodoWriteTool {
            memory_node: memory_node.clone(),
        };

        tool.execute(json!({
            "chat_id": "chat-a",
            "items": [
                {"content": "A1", "status": "pending"},
                {"content": "A2", "status": "completed"}
            ]
        }))
        .await
        .unwrap();

        tool.execute(json!({
            "chat_id": "chat-b",
            "items": [{"content": "B1", "status": "in_progress"}]
        }))
        .await
        .unwrap();

        let a = load_todos(&memory_node, "chat-a").await.expect("chat-a");
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].content, "A1");
        assert_eq!(a[1].status, "completed");

        let b = load_todos(&memory_node, "chat-b").await.expect("chat-b");
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].content, "B1");

        let _ = fs::remove_file(&db);
    }

    #[tokio::test]
    async fn todo_persists_across_new_store() {
        let db = temp_todo_db("persist");
        {
            let memory_node = spawn_memory_node(&db);
            let tool = TodoWriteTool {
                memory_node: memory_node.clone(),
            };
            tool.execute(json!({
                "chat_id": "session-xyz",
                "items": [{"content": "survive restart", "status": "pending"}]
            }))
            .await
            .unwrap();
        }

        let memory_node2 = spawn_memory_node(&db);
        let loaded = load_todos(&memory_node2, "session-xyz")
            .await
            .expect("sqlite");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "survive restart");

        let _ = fs::remove_file(&db);
    }

    #[tokio::test]
    async fn todo_write_rejects_bad_status() {
        let db = temp_todo_db("badstatus");
        let tool = TodoWriteTool {
            memory_node: spawn_memory_node(&db),
        };
        let err = tool
            .execute(json!({
                "chat_id": "c",
                "items": [{"content": "x", "status": "done"}]
            }))
            .await
            .unwrap_err();
        assert!(err.contains("Invalid status"), "{}", err);
        let _ = fs::remove_file(&db);
    }

    #[tokio::test]
    async fn tool_search_ranks_name_matches() {
        let cat = Arc::new(RwLock::new(vec![
            (
                "read_file".to_string(),
                "Read a local file from disk.".to_string(),
            ),
            (
                "glob_files".to_string(),
                "Find paths by glob pattern.".to_string(),
            ),
            (
                "search_memory".to_string(),
                "Search session memory summaries.".to_string(),
            ),
        ]));
        let tool = ToolSearchTool {
            catalog: Arc::clone(&cat),
        };

        let out = tool
            .execute(json!({"query": "memory", "limit": 5}))
            .await
            .unwrap();
        assert!(
            out.contains("search_memory"),
            "expected search_memory in:\n{}",
            out
        );
        assert!(out.contains("score"));

        let out2 = tool
            .execute(json!({"query": "glob", "limit": 2}))
            .await
            .unwrap();
        assert!(out2.contains("glob_files"));
    }
}
