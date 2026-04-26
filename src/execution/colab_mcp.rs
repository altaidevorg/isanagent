//! Colab MCP-backed execution provider (Phase 5 MVP).
//!
//! This provider launches a local `colab-mcp` stdio server (typically via `uvx`) and
//! forwards `execution_run` as MCP `tools/call` requests to a notebook-execution tool.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use log::{trace, warn};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

use super::capabilities::{
    NetworkPolicy, ProviderCapabilities, ProviderCapabilitiesSnapshot, SessionCapabilities,
};
use super::error::ExecutionError;
use super::ids::SessionId;
use super::provider::ExecutionProvider;
use super::run::{CwdPolicy, RunResult, RunSpec, SessionCreateRequest, SessionHandle};

#[derive(Debug, Clone)]
pub struct ColabMcpExecutionProviderConfig {
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub startup_timeout_secs: u64,
    pub connect_tool_name: String,
    pub execute_tool_name: Option<String>,
    pub execute_code_arg_keys: Vec<String>,
    pub max_sessions: usize,
    pub max_output_bytes: usize,
}

pub struct ColabMcpExecutionProvider {
    config: ColabMcpExecutionProviderConfig,
    caps: ProviderCapabilities,
    sessions: DashMap<SessionId, Arc<ColabMcpSession>>,
}

#[derive(Debug, Clone)]
enum ColabExecutionMode {
    Direct {
        execute_tool_name: String,
        execute_code_arg_key: String,
    },
    NotebookCells {
        add_code_cell_tool_name: String,
        add_code_arg_key: String,
        add_cell_index_arg_key: Option<String>,
        run_code_cell_tool_name: String,
        run_cell_id_arg_key: String,
    },
}

#[derive(Debug, Clone)]
struct McpToolDef {
    name: String,
    input_schema: Option<Value>,
}

/// Max buffered out-of-band JSON-RPC messages (notifications / stray responses) while waiting
/// for a matching response id. Oldest entries are dropped when full.
const MCP_INCOMING_BUFFER_CAP: usize = 64;

struct McpProcessClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    /// Set when the server sends MCP `notifications/tools/list_changed`.
    tools_list_dirty: Arc<AtomicBool>,
    /// Messages read from stdout that were not the JSON-RPC response for the in-flight request.
    incoming_buffer: VecDeque<Value>,
    stderr_tail: Arc<Mutex<String>>,
    stderr_reader: JoinHandle<()>,
}

struct ColabMcpSession {
    client: Mutex<McpProcessClient>,
    execution_mode: ColabExecutionMode,
    /// Last `tools/list` snapshot for this session (refreshed on `notifications/tools/list_changed`).
    cached_tools: Arc<RwLock<Vec<McpToolDef>>>,
}

async fn append_stderr_lines(tail: Arc<Mutex<String>>, stderr: ChildStderr) {
    let mut r = BufReader::new(stderr);
    let mut line = String::new();
    const MAX_CHARS: usize = 8192;
    loop {
        line.clear();
        match r.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {
                let mut s = tail.lock().await;
                s.push_str(&line);
                if s.len() > MAX_CHARS {
                    let over = s.len() - MAX_CHARS;
                    let cut = s.char_indices().nth(over).map(|(i, _)| i).unwrap_or(0);
                    s.drain(..cut);
                }
            }
            Err(_) => break,
        }
    }
}

impl ColabMcpExecutionProvider {
    pub fn new(config: ColabMcpExecutionProviderConfig) -> Result<Self, ExecutionError> {
        if config.command.trim().is_empty() {
            return Err(ExecutionError::InvalidArgument(
                "colab_mcp.command must be non-empty".to_string(),
            ));
        }
        if config.args.is_empty() {
            return Err(ExecutionError::InvalidArgument(
                "colab_mcp.args must contain at least one argument".to_string(),
            ));
        }
        let mut caps = ProviderCapabilities::minimal("colab_mcp");
        caps.languages = vec!["python".into()];
        caps.supports_persistent_sessions = true;
        caps.supports_interrupt = false;
        caps.supports_package_install = false;
        caps.supports_remote_shell = false;
        caps.jupyter_kernel = false;
        caps.network_policy = NetworkPolicy::Full;
        caps.max_output_bytes_default = Some(config.max_output_bytes as u64);
        caps.extensions.insert(
            "transport".into(),
            Value::String("mcp_stdio_colab_proxy".to_string()),
        );
        caps.extensions.insert(
            "connect_tool_name".into(),
            Value::String(config.connect_tool_name.clone()),
        );
        if let Some(name) = config.execute_tool_name.as_ref() {
            caps.extensions
                .insert("execute_tool_name_hint".into(), Value::String(name.clone()));
        }

        Ok(Self {
            config,
            caps,
            sessions: DashMap::new(),
        })
    }

    /// Best-effort: gracefully shut down every active MCP session. Used by the agent's exit
    /// path to avoid leaking child MCP processes. Failures per session are logged but do not
    /// stop the sweep, since this runs while the process is already on its way down.
    pub async fn shutdown_all_sessions(&self) {
        let ids: Vec<SessionId> = self.sessions.iter().map(|r| r.key().clone()).collect();
        for sid in ids {
            if let Some((_, sess)) = self.sessions.remove(&sid) {
                let mut guard = sess.client.lock().await;
                guard.shutdown().await;
            }
        }
    }

    /// Number of currently registered MCP sessions (for tests and runtime summaries).
    pub fn active_session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Call a proxied Colab MCP tool by name (used by `colab_mcp_tool_call` after allowlist checks).
    pub async fn call_mcp_tool_raw(
        &self,
        session_id: &SessionId,
        tool_name: &str,
        arguments: serde_json::Map<String, Value>,
    ) -> Result<Value, ExecutionError> {
        let sess = self
            .sessions
            .get(session_id)
            .map(|r| r.value().clone())
            .ok_or_else(|| ExecutionError::InvalidSession(session_id.to_string()))?;
        let mut guard = sess.client.lock().await;
        if guard.take_tools_list_dirty() {
            let t = guard.list_tools().await?;
            *sess.cached_tools.write().await = t;
        }
        guard.call_tool_raw(tool_name, arguments).await
    }

    /// Tool names from the last `tools/list` (or last refresh after `notifications/tools/list_changed`).
    pub async fn list_cached_mcp_tool_names(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<String>, ExecutionError> {
        let sess = self
            .sessions
            .get(session_id)
            .map(|r| r.value().clone())
            .ok_or_else(|| ExecutionError::InvalidSession(session_id.to_string()))?;
        let g = sess.cached_tools.read().await;
        Ok(g.iter().map(|t| t.name.clone()).collect())
    }
}

#[async_trait]
impl ExecutionProvider for ColabMcpExecutionProvider {
    fn provider_id(&self) -> &str {
        "colab_mcp"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.caps.clone()
    }

    async fn create_session(
        &self,
        _req: SessionCreateRequest,
    ) -> Result<SessionHandle, ExecutionError> {
        if self.config.max_sessions > 0 && self.sessions.len() >= self.config.max_sessions {
            return Err(ExecutionError::limit_exceeded(
                "sessions",
                format!("max_sessions={} reached", self.config.max_sessions),
            ));
        }
        let mut cmd = Command::new(&self.config.command);
        cmd.args(&self.config.args);
        if let Some(cwd) = self.config.cwd.as_ref() {
            cmd.current_dir(cwd);
        }
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.as_std_mut().creation_flags(CREATE_NO_WINDOW);
        }
        let child = cmd.spawn().map_err(|e| {
            ExecutionError::Provider(format!("spawn colab_mcp command failed: {e}"))
        })?;
        let mut client = McpProcessClient::new_from_child(child)?;

        let startup = Duration::from_secs(self.config.startup_timeout_secs);
        match tokio::time::timeout(startup, client.initialize()).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(client.wrap_provider_err(e).await),
            Err(_) => {
                let h = client.stderr_hint().await;
                if !h.is_empty() {
                    warn!(
                        "colab_mcp: initialize timed out after {}s:{}",
                        self.config.startup_timeout_secs, h
                    );
                }
                return Err(ExecutionError::Timeout {
                    timeout_secs: self.config.startup_timeout_secs,
                });
            }
        }

        let mut tools = match tokio::time::timeout(startup, client.list_tools()).await {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => return Err(client.wrap_provider_err(e).await),
            Err(_) => {
                let h = client.stderr_hint().await;
                if !h.is_empty() {
                    warn!(
                        "colab_mcp: tools/list timed out after {}s:{}",
                        self.config.startup_timeout_secs, h
                    );
                }
                return Err(ExecutionError::Timeout {
                    timeout_secs: self.config.startup_timeout_secs,
                });
            }
        };

        // Some Colab MCP installs expose notebook execution tools only after
        // browser authorization is established; retry discovery after connect.
        let execute_tool_name =
            match pick_execute_tool_name(&tools, self.config.execute_tool_name.as_deref()) {
                Ok(name) => name,
                Err(_e) if self.config.execute_tool_name.is_none() => {
                    let _ = tokio::time::timeout(
                        startup,
                        client.call_tool(
                            &self.config.connect_tool_name,
                            serde_json::Map::new(),
                            self.config.max_output_bytes,
                        ),
                    )
                    .await;
                    tools = match tokio::time::timeout(startup, client.list_tools()).await {
                        Ok(Ok(t)) => t,
                        Ok(Err(err)) => return Err(client.wrap_provider_err(err).await),
                        Err(_) => {
                            let h = client.stderr_hint().await;
                            if !h.is_empty() {
                                warn!(
                                    "colab_mcp: tools/list (post-connect) timed out after {}s:{}",
                                    self.config.startup_timeout_secs, h
                                );
                            }
                            return Err(ExecutionError::Timeout {
                                timeout_secs: self.config.startup_timeout_secs,
                            });
                        }
                    };
                    match pick_execute_tool_name(&tools, None) {
                        Ok(name) => name,
                        Err(e2) => return Err(client.wrap_provider_err(e2).await),
                    }
                }
                Err(e) => return Err(client.wrap_provider_err(e).await),
            };

        let execute_code_arg_key = pick_execute_code_arg_key(
            &tools,
            &execute_tool_name,
            &self.config.execute_code_arg_keys,
        );
        let execution_mode = match detect_execution_mode(
            &tools,
            &execute_tool_name,
            &execute_code_arg_key,
            &self.config.execute_code_arg_keys,
        ) {
            Ok(m) => m,
            Err(e) => return Err(client.wrap_provider_err(e).await),
        };

        // Best effort: this call opens/requests browser connection on disconnected clients.
        let _ = tokio::time::timeout(
            startup,
            client.call_tool(
                &self.config.connect_tool_name,
                serde_json::Map::new(),
                self.config.max_output_bytes,
            ),
        )
        .await;

        let sid = SessionId::new(uuid::Uuid::new_v4().to_string());
        let mut ext = std::collections::BTreeMap::new();
        match &execution_mode {
            ColabExecutionMode::Direct {
                execute_tool_name,
                execute_code_arg_key,
            } => {
                ext.insert(
                    "colab_mcp_execute_tool".into(),
                    Value::String(execute_tool_name.clone()),
                );
                ext.insert(
                    "colab_mcp_code_arg_key".into(),
                    Value::String(execute_code_arg_key.clone()),
                );
                ext.insert(
                    "colab_mcp_execution_mode".into(),
                    Value::String("direct".to_string()),
                );
            }
            ColabExecutionMode::NotebookCells {
                add_code_cell_tool_name,
                add_code_arg_key,
                add_cell_index_arg_key,
                run_code_cell_tool_name,
                run_cell_id_arg_key,
            } => {
                ext.insert(
                    "colab_mcp_execution_mode".into(),
                    Value::String("notebook_cells".to_string()),
                );
                ext.insert(
                    "colab_mcp_add_code_cell_tool".into(),
                    Value::String(add_code_cell_tool_name.clone()),
                );
                ext.insert(
                    "colab_mcp_add_code_arg_key".into(),
                    Value::String(add_code_arg_key.clone()),
                );
                if let Some(idx_key) = add_cell_index_arg_key {
                    ext.insert(
                        "colab_mcp_add_cell_index_arg_key".into(),
                        Value::String(idx_key.clone()),
                    );
                }
                ext.insert(
                    "colab_mcp_run_code_cell_tool".into(),
                    Value::String(run_code_cell_tool_name.clone()),
                );
                ext.insert(
                    "colab_mcp_run_cell_id_arg_key".into(),
                    Value::String(run_cell_id_arg_key.clone()),
                );
            }
        }
        ext.insert(
            "colab_mcp_connect_tool".into(),
            Value::String(self.config.connect_tool_name.clone()),
        );
        let caps = SessionCapabilities {
            session_id: sid.clone(),
            schema_version: 1,
            provider_id: "colab_mcp".to_string(),
            active_language: Some("python".to_string()),
            gpu_visible: None,
            working_directory_display: Some("colab-browser-runtime".to_string()),
            provider_snapshot: ProviderCapabilitiesSnapshot {
                supports_interrupt: false,
                supports_package_install: false,
                supports_remote_shell: false,
                jupyter_kernel: false,
                network_policy: NetworkPolicy::Full,
            },
            extensions: ext,
        };

        let cached_tools = Arc::new(RwLock::new(tools.clone()));
        let session = Arc::new(ColabMcpSession {
            client: Mutex::new(client),
            execution_mode,
            cached_tools,
        });
        self.sessions.insert(sid.clone(), session);
        Ok(SessionHandle {
            id: sid,
            capabilities: caps,
        })
    }

    async fn close_session(&self, session_id: &SessionId) -> Result<(), ExecutionError> {
        if let Some((_, sess)) = self.sessions.remove(session_id) {
            let mut guard = sess.client.lock().await;
            guard.shutdown().await;
            return Ok(());
        }
        Err(ExecutionError::InvalidSession(session_id.to_string()))
    }

    async fn run(
        &self,
        session_id: &SessionId,
        spec: RunSpec,
    ) -> Result<RunResult, ExecutionError> {
        if !matches!(spec.cwd, CwdPolicy::SessionDefault) {
            return Err(ExecutionError::unsupported(
                "cwd_mode",
                "colab_mcp provider supports only session_default cwd",
            ));
        }
        let sess = self
            .sessions
            .get(session_id)
            .map(|r| r.value().clone())
            .ok_or_else(|| ExecutionError::InvalidSession(session_id.to_string()))?;
        let mut guard = sess.client.lock().await;
        if guard.take_tools_list_dirty() {
            let t = guard.list_tools().await?;
            *sess.cached_tools.write().await = t;
        }
        let run_timeout = Duration::from_secs(spec.timeout_secs.max(1));
        let output = match &sess.execution_mode {
            ColabExecutionMode::Direct {
                execute_tool_name,
                execute_code_arg_key,
            } => {
                let mut args = serde_json::Map::new();
                args.insert(
                    execute_code_arg_key.clone(),
                    Value::String(spec.code.to_string()),
                );
                tokio::time::timeout(
                    run_timeout,
                    guard.call_tool(execute_tool_name, args, self.config.max_output_bytes / 2),
                )
                .await
                .map_err(|_| ExecutionError::Timeout {
                    timeout_secs: spec.timeout_secs,
                })??
            }
            ColabExecutionMode::NotebookCells {
                add_code_cell_tool_name,
                add_code_arg_key,
                add_cell_index_arg_key,
                run_code_cell_tool_name,
                run_cell_id_arg_key,
            } => {
                let mut add_args = serde_json::Map::new();
                add_args.insert(
                    add_code_arg_key.clone(),
                    Value::String(spec.code.to_string()),
                );
                if let Some(idx_key) = add_cell_index_arg_key {
                    // Insert at top by default for tools requiring explicit index.
                    add_args.insert(idx_key.clone(), Value::Number(serde_json::Number::from(0)));
                }
                // Hint for Colab editors that expose language selection.
                add_args.insert("language".to_string(), Value::String("python".to_string()));
                let add_resp = tokio::time::timeout(
                    run_timeout,
                    guard.call_tool_raw(add_code_cell_tool_name, add_args),
                )
                .await
                .map_err(|_| ExecutionError::Timeout {
                    timeout_secs: spec.timeout_secs,
                })??;
                let cell_id = extract_cell_id_from_tool_result(&add_resp).ok_or_else(|| {
                    ExecutionError::Provider(format!(
                        "colab_mcp add_code_cell did not return a cell id. response={}",
                        truncate_json_for_error(&add_resp, 800)
                    ))
                })?;
                let mut run_args = serde_json::Map::new();
                run_args.insert(run_cell_id_arg_key.clone(), Value::String(cell_id));
                let run_resp = tokio::time::timeout(
                    run_timeout,
                    guard.call_tool_raw(run_code_cell_tool_name, run_args),
                )
                .await
                .map_err(|_| ExecutionError::Timeout {
                    timeout_secs: spec.timeout_secs,
                })??;
                extract_tool_text(&run_resp, self.config.max_output_bytes / 2)
            }
        };
        Ok(RunResult::new(output, String::new(), Some(0)))
    }

    async fn cancel(&self, session_id: &SessionId) -> Result<(), ExecutionError> {
        if !self.sessions.contains_key(session_id) {
            return Err(ExecutionError::InvalidSession(session_id.to_string()));
        }
        Err(ExecutionError::unsupported(
            "execution_cancel",
            "colab_mcp provider does not support interrupt in MVP",
        ))
    }
}

impl McpProcessClient {
    fn new_from_child(mut child: Child) -> Result<Self, ExecutionError> {
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| ExecutionError::Provider("colab_mcp missing stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ExecutionError::Provider("colab_mcp missing stdout".to_string()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| ExecutionError::Provider("colab_mcp missing stderr".to_string()))?;

        let stderr_tail = Arc::new(Mutex::new(String::new()));
        let tail_for_reader = Arc::clone(&stderr_tail);
        let stderr_reader = tokio::spawn(async move {
            append_stderr_lines(tail_for_reader, stderr).await;
        });

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
            tools_list_dirty: Arc::new(AtomicBool::new(false)),
            incoming_buffer: VecDeque::new(),
            stderr_tail,
            stderr_reader,
        })
    }

    fn take_tools_list_dirty(&self) -> bool {
        self.tools_list_dirty.swap(false, Ordering::AcqRel)
    }

    async fn stderr_hint(&self) -> String {
        let g = self.stderr_tail.lock().await;
        if g.is_empty() {
            return String::new();
        }
        format!("\ncolab-mcp stderr (recent):\n{}", g.as_str())
    }

    async fn wrap_provider_err(&self, e: ExecutionError) -> ExecutionError {
        match e {
            ExecutionError::Provider(s) => {
                let h = self.stderr_hint().await;
                if h.is_empty() {
                    ExecutionError::Provider(s)
                } else {
                    ExecutionError::Provider(format!("{s}{h}"))
                }
            }
            other => other,
        }
    }

    fn push_incoming_buffered(&mut self, msg: Value) {
        if self.incoming_buffer.len() >= MCP_INCOMING_BUFFER_CAP {
            self.incoming_buffer.pop_front();
            trace!(
                "colab_mcp: dropped oldest buffered MCP message (cap={MCP_INCOMING_BUFFER_CAP})"
            );
        }
        self.incoming_buffer.push_back(msg);
    }

    fn consume_tools_list_changed_if_notification(dirty: &AtomicBool, msg: &Value) -> bool {
        let Some(Value::String(method)) = msg.get("method") else {
            return false;
        };
        let m = method.as_str();
        let hit = m == "notifications/tools/list_changed"
            || m == "tools/list_changed"
            || m.ends_with("/tools/list_changed");
        if hit {
            dirty.store(true, Ordering::Release);
            trace!("colab_mcp: tools/list changed notification ({m})");
        }
        hit
    }

    async fn next_decoded_message(&mut self) -> Result<Value, ExecutionError> {
        loop {
            if let Some(msg) = self.incoming_buffer.pop_front() {
                if Self::consume_tools_list_changed_if_notification(&self.tools_list_dirty, &msg) {
                    continue;
                }
                return Ok(msg);
            }
            let msg = self.read_message_from_wire().await?;
            if Self::consume_tools_list_changed_if_notification(&self.tools_list_dirty, &msg) {
                continue;
            }
            return Ok(msg);
        }
    }

    async fn initialize(&mut self) -> Result<(), ExecutionError> {
        let _ = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": { "name": "isanagent", "version": env!("CARGO_PKG_VERSION") }
                }),
            )
            .await?;
        self.notify("notifications/initialized", json!({})).await
    }

    async fn list_tools(&mut self) -> Result<Vec<McpToolDef>, ExecutionError> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                ExecutionError::Provider("MCP tools/list missing tools[]".to_string())
            })?;
        let mut out = Vec::with_capacity(tools.len());
        for t in tools {
            let name = t
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ExecutionError::Provider("MCP tool missing name".to_string()))?;
            out.push(McpToolDef {
                name: name.to_string(),
                input_schema: t.get("inputSchema").cloned(),
            });
        }
        Ok(out)
    }

    async fn call_tool(
        &mut self,
        tool_name: &str,
        args: serde_json::Map<String, Value>,
        max_text: usize,
    ) -> Result<String, ExecutionError> {
        let result = self.call_tool_raw(tool_name, args).await?;
        Ok(extract_tool_text(&result, max_text))
    }

    async fn call_tool_raw(
        &mut self,
        tool_name: &str,
        args: serde_json::Map<String, Value>,
    ) -> Result<Value, ExecutionError> {
        self.request(
            "tools/call",
            json!({
                "name": tool_name,
                "arguments": Value::Object(args),
            }),
        )
        .await
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<(), ExecutionError> {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_message(&msg).await
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, ExecutionError> {
        let id = self.next_id;
        self.next_id += 1;
        let req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_message(&req).await?;
        loop {
            let msg = self.next_decoded_message().await?;
            let matches_id = msg.get("id").and_then(|v| v.as_u64()) == Some(id);
            if matches_id {
                if let Some(err) = msg.get("error") {
                    return Err(ExecutionError::Provider(format!(
                        "MCP {method} error: {}",
                        err
                    )));
                }
                return msg.get("result").cloned().ok_or_else(|| {
                    ExecutionError::Provider(format!("MCP {method} missing result"))
                });
            }
            if Self::consume_tools_list_changed_if_notification(&self.tools_list_dirty, &msg) {
                continue;
            }
            // Preserve notifications and unrelated responses for a later request (same client).
            self.push_incoming_buffered(msg);
        }
    }

    async fn write_message(&mut self, msg: &Value) -> Result<(), ExecutionError> {
        let mut payload = serde_json::to_vec(msg)?;
        payload.push(b'\n');
        self.stdin
            .write_all(&payload)
            .await
            .map_err(|e| ExecutionError::Provider(format!("mcp stdin write body: {e}")))?;
        self.stdin
            .flush()
            .await
            .map_err(|e| ExecutionError::Provider(format!("mcp stdin flush: {e}")))?;
        Ok(())
    }

    async fn read_message_from_wire(&mut self) -> Result<Value, ExecutionError> {
        loop {
            let mut line = String::new();
            let n =
                self.stdout.read_line(&mut line).await.map_err(|e| {
                    ExecutionError::Provider(format!("mcp stdout read header: {e}"))
                })?;
            if n == 0 {
                return Err(ExecutionError::Provider(
                    "colab_mcp process closed stdout".to_string(),
                ));
            }
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(t) {
                return Ok(v);
            }
            let lower = t.to_ascii_lowercase();
            // Compatibility fallback: MCP Content-Length framing (may include extra header lines).
            if let Some(rest) = lower.strip_prefix("content-length:") {
                let parsed = rest.trim().parse::<usize>().map_err(|e| {
                    ExecutionError::Provider(format!("invalid MCP Content-Length header: {e}"))
                })?;
                loop {
                    let mut hdr = String::new();
                    let hn = self.stdout.read_line(&mut hdr).await.map_err(|e| {
                        ExecutionError::Provider(format!("mcp stdout read framing header: {e}"))
                    })?;
                    if hn == 0 {
                        return Err(ExecutionError::Provider(
                            "colab_mcp EOF before Content-Length body".to_string(),
                        ));
                    }
                    if hdr.trim().is_empty() {
                        break;
                    }
                }
                let mut buf = vec![0u8; parsed];
                self.stdout
                    .read_exact(&mut buf)
                    .await
                    .map_err(|e| ExecutionError::Provider(format!("mcp stdout read body: {e}")))?;
                let v: Value = serde_json::from_slice(&buf)?;
                return Ok(v);
            }
        }
    }

    async fn shutdown(&mut self) {
        self.stderr_reader.abort();
        let _ = self.child.kill().await;
    }
}

/// Best-effort kill at drop time. `close_session` (or `ColabMcpExecutionProvider::shutdown_all`)
/// is the primary teardown path; this Drop catches the case where a session leaks past its
/// graceful close (e.g. process exit without a /exit shutdown sweep) and prevents zombie
/// MCP child processes.
impl Drop for McpProcessClient {
    fn drop(&mut self) {
        self.stderr_reader.abort();
        // `start_kill` is synchronous; the actual SIGKILL/TerminateProcess may race with the
        // OS, but on Windows (where this agent typically runs) it does not block.
        let _ = self.child.start_kill();
    }
}

fn pick_execute_tool_name(
    tools: &[McpToolDef],
    configured: Option<&str>,
) -> Result<String, ExecutionError> {
    if let Some(name) = configured {
        if tools.iter().any(|t| t.name == name) {
            return Ok(name.to_string());
        }
        return Err(ExecutionError::Provider(format!(
            "configured colab_mcp execute_tool_name {name:?} not found in tools/list"
        )));
    }
    const CANDIDATES: &[&str] = &[
        "execute_python",
        "run_python",
        "run_python_cell",
        "run_code_cell",
        "execute_cell",
        "run_code",
    ];
    for c in CANDIDATES {
        if let Some(t) = tools.iter().find(|t| t.name.eq_ignore_ascii_case(c)) {
            return Ok(t.name.clone());
        }
    }
    for t in tools {
        let n = t.name.to_ascii_lowercase();
        if (n.contains("run") || n.contains("execute"))
            && (n.contains("python") || n.contains("code") || n.contains("cell"))
        {
            return Ok(t.name.clone());
        }
    }
    Err(ExecutionError::Provider(
        "could not auto-detect a Colab execution tool from MCP tools/list; configure [harness.execution.colab_mcp].execute_tool_name".to_string(),
    ))
}

fn pick_execute_code_arg_key(
    tools: &[McpToolDef],
    tool_name: &str,
    preferred: &[String],
) -> String {
    let Some(tool) = tools.iter().find(|t| t.name == tool_name) else {
        return preferred
            .first()
            .cloned()
            .unwrap_or_else(|| "code".to_string());
    };
    let props = tool
        .input_schema
        .as_ref()
        .and_then(|s| s.get("properties"))
        .and_then(|p| p.as_object());
    if let Some(props) = props {
        for key in preferred {
            if props.contains_key(key) {
                return key.clone();
            }
        }
        for k in props.keys() {
            let lower = k.to_ascii_lowercase();
            if lower.contains("code") || lower.contains("source") || lower.contains("cell") {
                return k.clone();
            }
        }
    }
    preferred
        .first()
        .cloned()
        .unwrap_or_else(|| "code".to_string())
}

fn pick_arg_key_from_tool_schema(
    tools: &[McpToolDef],
    tool_name: &str,
    preferred: &[&str],
) -> Option<String> {
    let tool = tools.iter().find(|t| t.name == tool_name)?;
    let props = tool
        .input_schema
        .as_ref()
        .and_then(|s| s.get("properties"))
        .and_then(|p| p.as_object())?;
    for key in preferred {
        if props.contains_key(*key) {
            return Some((*key).to_string());
        }
    }
    None
}

fn detect_execution_mode(
    tools: &[McpToolDef],
    execute_tool_name: &str,
    execute_code_arg_key: &str,
    configured_code_keys: &[String],
) -> Result<ColabExecutionMode, ExecutionError> {
    let key_lower = execute_code_arg_key.to_ascii_lowercase();
    if !matches!(key_lower.as_str(), "cellid" | "cell_id" | "id") {
        return Ok(ColabExecutionMode::Direct {
            execute_tool_name: execute_tool_name.to_string(),
            execute_code_arg_key: execute_code_arg_key.to_string(),
        });
    }
    let add_tool = "add_code_cell";
    if !tools.iter().any(|t| t.name == add_tool) {
        return Ok(ColabExecutionMode::Direct {
            execute_tool_name: execute_tool_name.to_string(),
            execute_code_arg_key: execute_code_arg_key.to_string(),
        });
    }
    let configured_pref: Vec<&str> = configured_code_keys.iter().map(String::as_str).collect();
    let add_key = pick_arg_key_from_tool_schema(tools, add_tool, &configured_pref)
        .or_else(|| {
            pick_arg_key_from_tool_schema(tools, add_tool, &["code", "source", "cell", "input"])
        })
        .ok_or_else(|| {
            ExecutionError::Provider(
                "colab_mcp: add_code_cell found but no usable code argument key in schema"
                    .to_string(),
            )
        })?;
    let add_cell_index_arg_key =
        pick_arg_key_from_tool_schema(tools, add_tool, &["cellIndex", "cell_index", "index"]);
    Ok(ColabExecutionMode::NotebookCells {
        add_code_cell_tool_name: add_tool.to_string(),
        add_code_arg_key: add_key,
        add_cell_index_arg_key,
        run_code_cell_tool_name: execute_tool_name.to_string(),
        run_cell_id_arg_key: execute_code_arg_key.to_string(),
    })
}

fn extract_cell_id_from_tool_result(value: &Value) -> Option<String> {
    fn walk(v: &Value) -> Option<String> {
        match v {
            Value::Object(map) => {
                for key in ["newCellId", "cellId", "cell_id", "id"] {
                    if let Some(Value::String(s)) = map.get(key) {
                        if !s.trim().is_empty() {
                            return Some(s.trim().to_string());
                        }
                    }
                }
                for nested in map.values() {
                    if let Some(found) = walk(nested) {
                        return Some(found);
                    }
                }
                None
            }
            Value::Array(arr) => {
                for item in arr {
                    if let Some(found) = walk(item) {
                        return Some(found);
                    }
                }
                None
            }
            _ => None,
        }
    }
    walk(value)
}

fn truncate_json_for_error(value: &Value, max_chars: usize) -> String {
    let mut s = value.to_string();
    if s.len() <= max_chars {
        return s;
    }
    let mut end = max_chars;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s.push('…');
    s
}

fn extract_tool_text(result: &Value, max_text: usize) -> String {
    let mut chunks: Vec<String> = Vec::new();
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        for item in content {
            if item.get("type").and_then(|v| v.as_str()) == Some("text") {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    chunks.push(text.to_string());
                }
            }
        }
    }
    if chunks.is_empty() {
        if let Some(s) = result.get("structuredContent") {
            chunks.push(s.to_string());
        } else {
            chunks.push(result.to_string());
        }
    }
    let mut out = chunks.join("\n");
    if out.len() > max_text {
        let mut end = max_text;
        while end > 0 && !out.is_char_boundary(end) {
            end -= 1;
        }
        out.truncate(end);
        out.push_str("\n... (truncated)");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execute_tool_autodetect_prefers_named_candidates() {
        let tools = vec![McpToolDef {
            name: "execute_python".to_string(),
            input_schema: None,
        }];
        let picked = pick_execute_tool_name(&tools, None).expect("pick");
        assert_eq!(picked, "execute_python");
    }

    fn provider_with_dummy_command() -> ColabMcpExecutionProvider {
        ColabMcpExecutionProvider::new(ColabMcpExecutionProviderConfig {
            command: "isanagent-colab-mcp-fixture-do-not-spawn".to_string(),
            args: vec!["--noop".to_string()],
            cwd: None,
            startup_timeout_secs: 1,
            connect_tool_name: "connect".to_string(),
            execute_tool_name: None,
            execute_code_arg_keys: vec!["code".to_string()],
            max_sessions: 4,
            max_output_bytes: 4096,
        })
        .expect("provider builds with valid config (does not spawn)")
    }

    #[tokio::test]
    async fn shutdown_all_sessions_is_safe_when_empty() {
        let p = provider_with_dummy_command();
        assert_eq!(p.active_session_count(), 0);
        // Must not panic, must not deadlock.
        p.shutdown_all_sessions().await;
        assert_eq!(p.active_session_count(), 0);
    }

    #[tokio::test]
    async fn execution_harness_shutdown_without_colab_mcp_is_noop() {
        // Build a no-op harness whose colab_mcp slot is None; shutdown should return
        // immediately without any side effects.
        use crate::execution::harness::ExecutionHarness;
        use crate::execution::provider::ExecutionProvider;
        use async_trait::async_trait;

        struct StubProvider {
            caps: ProviderCapabilities,
        }
        #[async_trait]
        impl ExecutionProvider for StubProvider {
            fn provider_id(&self) -> &str {
                "stub"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                self.caps.clone()
            }
            async fn create_session(
                &self,
                _req: crate::execution::run::SessionCreateRequest,
            ) -> Result<crate::execution::run::SessionHandle, ExecutionError> {
                Err(ExecutionError::Provider("stub".into()))
            }
            async fn close_session(
                &self,
                _id: &crate::execution::ids::SessionId,
            ) -> Result<(), ExecutionError> {
                Ok(())
            }
            async fn run(
                &self,
                _id: &crate::execution::ids::SessionId,
                _spec: crate::execution::run::RunSpec,
            ) -> Result<crate::execution::run::RunResult, ExecutionError> {
                Err(ExecutionError::Provider("stub".into()))
            }
            async fn cancel(
                &self,
                _id: &crate::execution::ids::SessionId,
            ) -> Result<(), ExecutionError> {
                Ok(())
            }
        }

        let prov: Arc<dyn ExecutionProvider> = Arc::new(StubProvider {
            caps: ProviderCapabilities::minimal("stub"),
        });
        let h = ExecutionHarness::new(
            prov,
            "python",
            std::path::PathBuf::from("."),
            std::path::PathBuf::from("."),
            crate::execution::artifacts::ArtifactLimits::default(),
            30,
            300,
            120,
        );
        // No-op; just must not panic.
        h.shutdown().await;
    }

    #[test]
    fn code_arg_key_prefers_schema_match() {
        let tools = vec![McpToolDef {
            name: "run_code".to_string(),
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "source": { "type": "string" }
                }
            })),
        }];
        let key = pick_execute_code_arg_key(
            &tools,
            "run_code",
            &["code".to_string(), "source".to_string()],
        );
        assert_eq!(key, "source");
    }
}
