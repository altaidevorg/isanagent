//! Sub-agent task harness (Phase 5): background reasoning loops keyed by parent chat.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use serde::Deserialize;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::ReasoningLoopCtx;
use crate::bus::{BusMessage, InboundMessage};
use crate::clarification::ClarificationHub;
use crate::logging::LoggerHandle;
use crate::session::SessionManager;
use crate::skills::SkillRegistry;
use crate::tool_activity::SharedToolExecutionActivity;
use crate::tool_runtime::ToolExecCtx;
use crate::tools::ToolRegistry;
use crate::traits::{Provider, Tool};

/// Arguments for [`SubagentHarness::spawn`].
pub struct SubagentSpawnSpec {
    pub parent_channel: String,
    pub parent_chat_id: String,
    pub parent_thread_id: Option<String>,
    pub parent_reasoning_cancel: Option<CancellationToken>,
    pub prompt: String,
    pub wait: bool,
    pub display_name: Option<String>,
}

/// Shared wiring for each spawned sub-agent run.
pub struct SubagentSpawnDeps {
    pub agent_name: String,
    pub provider_template: Box<dyn Provider>,
    pub session_manager: Arc<SessionManager>,
    pub skills: Arc<SkillRegistry>,
    pub system_prompt: String,
    pub max_iterations: usize,
    pub max_tool_output_chars: usize,
    pub max_recent_summaries: usize,
    pub short_term_threshold_turns: usize,
    pub short_term_threshold_tokens: usize,
    pub tool_execution_activity: Option<SharedToolExecutionActivity>,
    pub outbound_tx: tokio::sync::mpsc::Sender<BusMessage>,
    pub logger_tx: LoggerHandle,
    pub clarification_hub: Arc<ClarificationHub>,
    pub cancel_children_on_parent_cancel: bool,
    pub default_allowlist: Option<Arc<HashSet<String>>>,
    pub max_tasks: usize,
    pub max_wait_secs: u64,
}

struct TaskRecord {
    parent_chat_id: String,
    child_chat_id: String,
    prompt: String,
    cancel: CancellationToken,
    status: std::sync::atomic::AtomicU8,
    result: tokio::sync::RwLock<Option<String>>,
    error: tokio::sync::RwLock<Option<String>>,
    done: tokio::sync::Notify,
}

const ST_PENDING: u8 = 0;
const ST_RUNNING: u8 = 1;
const ST_COMPLETED: u8 = 2;
const ST_FAILED: u8 = 3;
const ST_CANCELLED: u8 = 4;

impl TaskRecord {
    fn is_terminal(&self) -> bool {
        let s = self.status.load(std::sync::atomic::Ordering::Acquire);
        s >= ST_COMPLETED
    }

    fn snapshot_json(&self) -> String {
        let status = match self.status.load(std::sync::atomic::Ordering::Acquire) {
            ST_PENDING => "pending",
            ST_RUNNING => "running",
            ST_COMPLETED => "completed",
            ST_FAILED => "failed",
            ST_CANCELLED => "cancelled",
            _ => "unknown",
        };
        let result = self
            .result
            .try_read()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_default();
        let error = self
            .error
            .try_read()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_default();
        serde_json::json!({
            "child_chat_id": self.child_chat_id,
            "parent_chat_id": self.parent_chat_id,
            "status": status,
            "prompt": self.prompt,
            "result": result,
            "error": error,
        })
        .to_string()
    }
}

struct Inner {
    deps: SubagentSpawnDeps,
    tasks: DashMap<String, Arc<TaskRecord>>,
    tools: std::sync::OnceLock<Arc<ToolRegistry>>,
}

/// Task registry + spawn helper; tools call into this type.
#[derive(Clone)]
pub struct SubagentHarness {
    inner: Arc<Inner>,
}

impl SubagentHarness {
    pub fn new(deps: SubagentSpawnDeps) -> Self {
        Self {
            inner: Arc::new(Inner {
                deps,
                tasks: DashMap::new(),
                tools: std::sync::OnceLock::new(),
            }),
        }
    }

    pub fn bind_tools(&self, tools: Arc<ToolRegistry>) -> Result<(), String> {
        self.inner
            .tools
            .set(tools)
            .map_err(|_| "subagent bind_tools: tools already bound".to_string())
    }

    pub fn cancel_children_on_parent_cancel(&self) -> bool {
        self.inner.deps.cancel_children_on_parent_cancel
    }

    pub fn cancel_children_for_parent(&self, parent_chat_id: &str) {
        for e in self.inner.tasks.iter() {
            if e.value().parent_chat_id == parent_chat_id {
                e.value().cancel.cancel();
            }
        }
    }

    fn tools(&self) -> Result<Arc<ToolRegistry>, String> {
        self.inner
            .tools
            .get()
            .cloned()
            .ok_or_else(|| "Subagent harness is not wired to tools yet".to_string())
    }

    pub async fn spawn(&self, spec: SubagentSpawnSpec) -> Result<String, String> {
        let SubagentSpawnSpec {
            parent_channel,
            parent_chat_id,
            parent_thread_id,
            parent_reasoning_cancel,
            prompt,
            wait,
            display_name,
        } = spec;

        if self.inner.tasks.len() >= self.inner.deps.max_tasks {
            return Err(format!(
                "Maximum number of sub-agent tasks ({}) reached",
                self.inner.deps.max_tasks
            ));
        }

        let tools = self.tools()?;
        let task_id = uuid::Uuid::new_v4().simple().to_string();
        let child_chat_id = format!("subagent-{}", &task_id[..12.min(task_id.len())]);

        let task_cancel = match (
            &parent_reasoning_cancel,
            self.inner.deps.cancel_children_on_parent_cancel,
        ) {
            (Some(p), true) => p.child_token(),
            _ => CancellationToken::new(),
        };

        let record = Arc::new(TaskRecord {
            parent_chat_id: parent_chat_id.clone(),
            child_chat_id: child_chat_id.clone(),
            prompt: prompt.clone(),
            cancel: task_cancel.clone(),
            status: std::sync::atomic::AtomicU8::new(ST_PENDING),
            result: tokio::sync::RwLock::new(None),
            error: tokio::sync::RwLock::new(None),
            done: tokio::sync::Notify::new(),
        });

        self.inner.tasks.insert(task_id.clone(), record.clone());

        let provider = dyn_clone::clone_box(&*self.inner.deps.provider_template);
        let inbound = InboundMessage {
            channel: parent_channel.clone(),
            sender_id: parent_chat_id.clone(),
            chat_id: child_chat_id.clone(),
            thread_id: parent_thread_id.clone(),
            content: if let Some(ref n) = display_name.as_ref().filter(|s| !s.is_empty()) {
                format!("[Sub-agent: {}]\n\n{}", n, prompt)
            } else {
                format!(
                    "[Sub-agent task {}]\n\n{}",
                    &task_id[..8.min(task_id.len())],
                    prompt
                )
            },
            attachments: vec![],
            metadata: HashMap::new(),
        };

        let tool_exec_ctx = ToolExecCtx::new(
            parent_channel.clone(),
            child_chat_id.clone(),
            parent_thread_id.clone(),
        )
        .with_reasoning_cancel(task_cancel.clone());

        let ctx = ReasoningLoopCtx {
            name: self.inner.deps.agent_name.clone(),
            provider,
            session_manager: self.inner.deps.session_manager.clone(),
            tools,
            skills: self.inner.deps.skills.clone(),
            system_prompt: self.inner.deps.system_prompt.clone(),
            max_iterations: self.inner.deps.max_iterations,
            max_tool_output_chars: self.inner.deps.max_tool_output_chars,
            max_recent_summaries: self.inner.deps.max_recent_summaries,
            short_term_threshold_turns: self.inner.deps.short_term_threshold_turns,
            short_term_threshold_tokens: self.inner.deps.short_term_threshold_tokens,
            tool_execution_activity: self.inner.deps.tool_execution_activity.clone(),
            outbound_tx: self.inner.deps.outbound_tx.clone(),
            logger_tx: self.inner.deps.logger_tx.clone(),
            inbound,
            cancel_token: task_cancel.clone(),
            clarification_hub: self.inner.deps.clarification_hub.clone(),
            tool_exec_ctx,
            is_subagent: true,
            subagent_allowlist: self.inner.deps.default_allowlist.clone(),
        };

        record
            .status
            .store(ST_RUNNING, std::sync::atomic::Ordering::Release);

        let tasks = self.inner.tasks.clone();
        let tid = task_id.clone();
        let rec = record.clone();
        tokio::spawn(async move {
            let outcome = super::AgentLogic::run_reasoning_loop(ctx).await;
            match outcome {
                Ok(text) => {
                    if rec.cancel.is_cancelled() {
                        rec.status
                            .store(ST_CANCELLED, std::sync::atomic::Ordering::Release);
                    } else {
                        {
                            let mut r = rec.result.write().await;
                            *r = Some(text);
                        }
                        rec.status
                            .store(ST_COMPLETED, std::sync::atomic::Ordering::Release);
                    }
                }
                Err(e) => {
                    *rec.error.write().await = Some(e);
                    rec.status
                        .store(ST_FAILED, std::sync::atomic::Ordering::Release);
                }
            }
            // Drop from the index before waking `wait=true` callers so `task_list` does not
            // briefly show a finished task after a blocking spawn returns.
            tasks.remove(&tid);
            rec.done.notify_waiters();
        });

        if wait {
            let max_wait = self.inner.deps.max_wait_secs;
            let wait_fut = async {
                loop {
                    let notified = record.done.notified();
                    if record.is_terminal() {
                        break;
                    }
                    notified.await;
                }
            };
            tokio::time::timeout(std::time::Duration::from_secs(max_wait), wait_fut)
                .await
                .map_err(|_| {
                    record.cancel.cancel();
                    format!("subagent_spawn timed out after {}s (wait=true)", max_wait)
                })?;
        }

        let mut body = serde_json::json!({
            "task_id": task_id,
            "child_chat_id": child_chat_id,
            "wait": wait,
        });
        if wait {
            let result_text = record.result.read().await.clone().unwrap_or_default();
            body["result"] = serde_json::Value::String(result_text);
        }
        Ok(body.to_string())
    }

    pub fn list_for_parent(&self, parent_chat_id: &str) -> String {
        let mut rows: Vec<String> = Vec::new();
        for e in self.inner.tasks.iter() {
            if e.value().parent_chat_id == parent_chat_id {
                rows.push(e.value().snapshot_json());
            }
        }
        if rows.is_empty() {
            return "No active sub-agent tasks for this chat.".to_string();
        }
        rows.join("\n")
    }

    fn find_task(&self, task_id: &str, parent_chat_id: &str) -> Result<Arc<TaskRecord>, String> {
        let t = self
            .inner
            .tasks
            .get(task_id)
            .ok_or_else(|| format!("Task '{}' not found", task_id))?;
        if t.parent_chat_id != parent_chat_id {
            return Err("Task is not owned by the current chat".to_string());
        }
        Ok(Arc::clone(t.value()))
    }

    pub fn get_task(&self, task_id: &str, parent_chat_id: &str) -> Result<String, String> {
        let t = self.find_task(task_id, parent_chat_id)?;
        Ok(t.snapshot_json())
    }

    pub fn cancel_task(&self, task_id: &str, parent_chat_id: &str) -> Result<String, String> {
        let t = self.find_task(task_id, parent_chat_id)?;
        t.cancel.cancel();
        Ok(format!("Cancellation requested for task {}", task_id))
    }
}

fn current_parent_ids(
) -> Result<(String, String, Option<String>, Option<CancellationToken>), String> {
    let ctx = crate::tool_runtime::current_tool_exec_ctx()
        .ok_or_else(|| "subagent tools require an active agent tool scope".to_string())?;
    Ok((
        ctx.channel,
        ctx.chat_id,
        ctx.thread_id,
        ctx.reasoning_cancel,
    ))
}

pub struct SubagentSpawnTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for SubagentSpawnTool {
    fn name(&self) -> &str {
        "subagent_spawn"
    }

    fn description(&self) -> &str {
        "Spawn a background sub-agent that runs a separate reasoning loop with its own chat id (prefixed subagent-). Use wait=false for fire-and-forget; wait=true blocks until completion or timeout and includes the assistant-facing final text in the JSON field \"result\". Sub-agents inherit config allowlists and cannot spawn nested sub-agents."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": { "type": "string", "description": "User task for the sub-agent" },
                "wait": { "type": "boolean", "description": "If true, block until the run finishes or times out (see harness max_wait_secs)." },
                "name": { "type": "string", "description": "Optional short label for logs / context." }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or("Missing prompt")?
            .to_string();
        let wait = args.get("wait").and_then(|v| v.as_bool()).unwrap_or(false);
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let (ch, parent_chat, thread, parent_cancel) = current_parent_ids()?;
        self.harness
            .spawn(SubagentSpawnSpec {
                parent_channel: ch,
                parent_chat_id: parent_chat,
                parent_thread_id: thread,
                parent_reasoning_cancel: parent_cancel,
                prompt,
                wait,
                display_name: name,
            })
            .await
    }
}

pub struct TaskListTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for TaskListTool {
    fn name(&self) -> &str {
        "task_list"
    }

    fn description(&self) -> &str {
        "List active sub-agent tasks spawned from the current chat."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn execute(&self, _: Value) -> Result<String, String> {
        let (_, parent_chat, _, _) = current_parent_ids()?;
        Ok(self.harness.list_for_parent(&parent_chat))
    }
}

pub struct TaskGetTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for TaskGetTool {
    fn name(&self) -> &str {
        "task_get"
    }

    fn description(&self) -> &str {
        "Get status snapshot JSON for a sub-agent task still listed for this chat. Completed tasks are dropped from the table quickly; use subagent_spawn with wait=true for the `result` text in the spawn response."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_id": { "type": "string", "description": "Task id returned by subagent_spawn" }
            },
            "required": ["task_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let id = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing task_id")?;
        let (_, parent_chat, _, _) = current_parent_ids()?;
        self.harness.get_task(id, &parent_chat)
    }
}

pub struct TaskCancelTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for TaskCancelTool {
    fn name(&self) -> &str {
        "task_cancel"
    }

    fn description(&self) -> &str {
        "Cancel a running sub-agent task owned by the current chat."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_id": { "type": "string" }
            },
            "required": ["task_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let id = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing task_id")?;
        let (_, parent_chat, _, _) = current_parent_ids()?;
        self.harness.cancel_task(id, &parent_chat)
    }
}

#[derive(Deserialize)]
struct PlanStep {
    id: String,
    #[serde(default)]
    depends_on: Vec<String>,
    prompt: String,
}

#[derive(Deserialize)]
struct PlanInput {
    steps: Vec<PlanStep>,
}

pub struct SubagentPlanTool {
    pub harness: Arc<SubagentHarness>,
}

#[async_trait]
impl Tool for SubagentPlanTool {
    fn name(&self) -> &str {
        "subagent_plan_execute"
    }

    fn description(&self) -> &str {
        "Run a multi-step plan: JSON object {\"steps\":[{\"id\":\"1\",\"depends_on\":[],\"prompt\":\"...\"}, ...]}. Steps run in dependency order (sequential). Each step sees prior steps' assistant-facing final output (subagent_spawn result text) in its prompt prefix."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "JSON string: { \"steps\": [ { \"id\", \"depends_on\" (array of ids), \"prompt\" } ] }"
                }
            },
            "required": ["plan"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let plan_str = args
            .get("plan")
            .and_then(|v| v.as_str())
            .ok_or("Missing plan (JSON string)")?;
        let plan: PlanInput =
            serde_json::from_str(plan_str).map_err(|e| format!("Invalid plan JSON: {}", e))?;
        if plan.steps.is_empty() {
            return Err("plan.steps is empty".to_string());
        }

        let mut deps: HashMap<String, Vec<String>> = HashMap::new();
        let mut prompts: HashMap<String, String> = HashMap::new();
        for s in &plan.steps {
            if s.id.is_empty() || s.prompt.is_empty() {
                return Err("Each step needs non-empty id and prompt".to_string());
            }
            deps.insert(s.id.clone(), s.depends_on.clone());
            prompts.insert(s.id.clone(), s.prompt.clone());
        }

        let mut done: HashSet<String> = HashSet::new();
        let mut results: HashMap<String, String> = HashMap::new();
        let mut out = String::from("# Plan execution\n\n");

        while done.len() < plan.steps.len() {
            let mut ready: Vec<String> = Vec::new();
            for s in &plan.steps {
                if done.contains(&s.id) {
                    continue;
                }
                let ok = deps[&s.id].iter().all(|d| done.contains(d));
                if ok {
                    ready.push(s.id.clone());
                }
            }
            if ready.is_empty() {
                return Err("Plan deadlock or missing dependency ids".to_string());
            }
            ready.sort();
            for step_id in ready {
                let mut body = String::new();
                for d in &deps[&step_id] {
                    if let Some(r) = results.get(d) {
                        body.push_str(&format!("## Prior step {} result:\n{}\n\n", d, r));
                    }
                }
                body.push_str(&prompts[&step_id]);
                let (ch, parent_chat, thread, parent_cancel) = current_parent_ids()?;
                let label = format!("plan-{}", step_id);
                let spawn_json = self
                    .harness
                    .spawn(SubagentSpawnSpec {
                        parent_channel: ch,
                        parent_chat_id: parent_chat,
                        parent_thread_id: thread,
                        parent_reasoning_cancel: parent_cancel,
                        prompt: body,
                        wait: true,
                        display_name: Some(label),
                    })
                    .await?;
                let step_output = serde_json::from_str::<serde_json::Value>(&spawn_json)
                    .ok()
                    .and_then(|v| {
                        v.get("result")
                            .and_then(|r| r.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or(spawn_json.clone());
                results.insert(step_id.clone(), step_output.clone());
                out.push_str(&format!("## Step {}\n{}\n\n", step_id, step_output));
                done.insert(step_id);
            }
        }

        Ok(out)
    }
}

pub fn register_subagent_tools(registry: &mut ToolRegistry, harness: Arc<SubagentHarness>) {
    registry.register(Box::new(SubagentSpawnTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(TaskListTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(TaskGetTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(TaskCancelTool {
        harness: harness.clone(),
    }));
    registry.register(Box::new(SubagentPlanTool { harness }));
}
