//! Model Context Protocol (MCP) Stdio Client & Tool Adapter.
//!
//! Conforms to the MCP specification and Agent Plugins 1.0 `mcp.json` definition.
//! Launches stdio servers, exchanges initialize and tools/list requests, and exposes
//! discovered tools as native [`crate::traits::Tool`] instances.
//!
//! ## Transport
//!
//! One background task owns the server's stdout. It routes each JSON-RPC response to the
//! request waiting for that `id`, answers server-to-client requests (`ping`, `roots/list`),
//! and skips lines that are not JSON (a server that prints logs to stdout breaks the protocol,
//! but should not take the whole client down with it). Every request has a deadline; a request
//! that times out or whose caller is dropped (turn cancelled) sends `notifications/cancelled`.
//! The server's stderr is captured to the log; its last lines are attached to launch errors.
//!
//! ## Tool names
//!
//! Tools are exposed to the model as `{server}__{tool}` (see [`namespaced_tool_name`]) so two
//! servers — or a server and a built-in tool — can never shadow each other. The original name
//! is kept on [`McpProxyTool`] and used for `tools/call`.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::{oneshot, Mutex};

use crate::traits::{Tool, ToolPolicy};

fn default_mcp_type() -> String {
    "stdio".to_string()
}

/// Default deadline for `initialize` + `tools/list` (first runs of `uv run` may install
/// dependencies: raise it per server with `startupTimeoutSecs`).
pub const DEFAULT_MCP_STARTUP_TIMEOUT_SECS: u64 = 60;
/// Default deadline for one `tools/call` / `resources/read`.
pub const DEFAULT_MCP_TOOL_TIMEOUT_SECS: u64 = 300;
/// Longest tool name every supported provider accepts (`^[a-zA-Z0-9_-]{1,64}$`).
pub const MAX_PROVIDER_TOOL_NAME_LEN: usize = 64;
/// Separator between server and tool in exposed tool names.
pub const MCP_TOOL_NAME_SEPARATOR: &str = "__";
/// Env var every MCP server receives with the agent's sandbox directory (when known).
pub const MCP_SANDBOX_ENV: &str = "ISANAGENT_SANDBOX_DIR";

const STDERR_TAIL_LINES: usize = 20;
const LOG_LINE_PREVIEW_CHARS: usize = 200;

/// Configuration schema for `mcp.json` conforming to Agent Plugins 1.0.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct McpConfigFile {
    #[serde(rename = "$schema", default)]
    pub schema: Option<String>,
    /// Ordered by server name so tool registration (and the provider `tools` array, which
    /// prompt caches key on) is the same on every start.
    #[serde(rename = "mcpServers", default)]
    pub mcp_servers: BTreeMap<String, McpServerEntry>,
}

/// Single server definition inside `mcpServers`.
///
/// `command`, `args`, `env` and `cwd` expand `${VAR}` / `${VAR:-default}` (see
/// [`expand_mcp_vars`]). The camelCase fields after `cwd` are isanagent extensions; other
/// clients ignore them, and `[harness.mcp.servers.<name>]` in `config.toml` overrides them.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct McpServerEntry {
    #[serde(rename = "type", default = "default_mcp_type")]
    pub server_type: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub cwd: Option<String>,
    /// Deadline in seconds for `initialize` + `tools/list`.
    #[serde(
        rename = "startupTimeoutSecs",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub startup_timeout_secs: Option<u64>,
    /// Deadline in seconds for each `tools/call`.
    #[serde(
        rename = "toolTimeoutSecs",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub tool_timeout_secs: Option<u64>,
    /// Only these tools are registered (original names; a trailing `*` matches a prefix).
    #[serde(
        rename = "includeTools",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub include_tools: Option<Vec<String>>,
    /// These tools are never registered (original names; a trailing `*` matches a prefix).
    #[serde(
        rename = "excludeTools",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub exclude_tools: Vec<String>,
    /// `true`: keep this server's tool schemas out of every request until `search_tools` or a
    /// skill's `allowed-tools` loads them. `false`: always send them. Unset: `[harness.mcp]
    /// defer_tools` decides.
    #[serde(
        rename = "deferTools",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub defer_tools: Option<bool>,
}

/// Discovered MCP tool definition from `tools/list`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Option<Value>,
    /// MCP Apps linkage (`_meta.ui.resourceUri`) and visibility.
    #[serde(rename = "_meta", default)]
    pub meta: Option<Value>,
}

// ---------------------------------------------------------------------------------------------
// Tool names
// ---------------------------------------------------------------------------------------------

fn sanitize_name_part(raw: &str) -> String {
    let mut out: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        out.push_str("mcp");
    }
    out
}

/// Name a tool is exposed under: `{server}__{tool}`, restricted to `[A-Za-z0-9_-]`, starting
/// with a letter or `_` (Gemini), at most [`MAX_PROVIDER_TOOL_NAME_LEN`] characters. Longer
/// names keep a readable prefix plus a stable hash of the original pair, so the same tool gets
/// the same name on every start.
pub fn namespaced_tool_name(server: &str, tool: &str) -> String {
    let mut full = format!(
        "{}{MCP_TOOL_NAME_SEPARATOR}{}",
        sanitize_name_part(server),
        sanitize_name_part(tool)
    );
    if !full
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    {
        full.insert_str(0, "mcp_");
    }
    if full.len() <= MAX_PROVIDER_TOOL_NAME_LEN {
        return full;
    }
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(server.as_bytes());
    hasher.update([0u8]);
    hasher.update(tool.as_bytes());
    let digest = hex::encode(hasher.finalize());
    let hash = &digest[..8];
    // ASCII only after sanitizing, so byte slicing is safe.
    let keep = MAX_PROVIDER_TOOL_NAME_LEN - 1 - hash.len();
    format!("{}_{hash}", &full[..keep])
}

/// Exact match, or prefix match when `pattern` ends with `*` (`"*"` matches everything).
pub fn tool_pattern_matches(pattern: &str, name: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => pattern == name,
    }
}

/// Applies `includeTools` / `excludeTools` to an original MCP tool name.
pub fn tool_passes_filters(name: &str, include: Option<&[String]>, exclude: &[String]) -> bool {
    if exclude.iter().any(|p| tool_pattern_matches(p, name)) {
        return false;
    }
    match include {
        Some(list) => list.iter().any(|p| tool_pattern_matches(p, name)),
        None => true,
    }
}

// ---------------------------------------------------------------------------------------------
// Variable expansion
// ---------------------------------------------------------------------------------------------

/// Values available to `${VAR}` in `mcp.json`.
#[derive(Clone, Debug, Default)]
pub struct McpExpansionContext {
    pub plugin_root: PathBuf,
    pub sandbox_dir: Option<PathBuf>,
    /// Secret-looking host variables (`*_KEY`, `*_TOKEN`, …) a server may read via `${VAR}`.
    /// Everything else the host has is readable; secrets need the user's opt-in in
    /// `[harness.mcp] env_passthrough` (or the server's own `env_passthrough`).
    pub env_passthrough: Vec<String>,
}

impl McpExpansionContext {
    fn builtin(&self, name: &str) -> Option<String> {
        match name {
            "PLUGIN_DIR" | "plugin_dir" | "CLAUDE_PLUGIN_DIR" | "CLAUDE_PLUGIN_ROOT" => {
                Some(self.plugin_root.to_string_lossy().to_string())
            }
            "SANDBOX_DIR" => self
                .sandbox_dir
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            _ => None,
        }
    }

    fn host_var(
        &self,
        name: &str,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Option<String>, String> {
        let policy = crate::environment::ExecutionEnvironmentPolicy::default_safe();
        if policy.is_secret_var(name) && !self.env_passthrough.iter().any(|k| k == name) {
            return Err(format!(
                "'{name}' looks like a credential and is not listed in [harness.mcp] env_passthrough"
            ));
        }
        Ok(lookup(name))
    }
}

/// Expands `${VAR}` and `${VAR:-default}`. Built-ins: `PLUGIN_DIR` (aliases `plugin_dir`,
/// `CLAUDE_PLUGIN_DIR`, `CLAUDE_PLUGIN_ROOT`) and `SANDBOX_DIR`; any other name reads the host
/// environment, where credential-looking names need an explicit opt-in (see
/// [`McpExpansionContext::env_passthrough`]). An unset variable without a default expands to
/// the empty string; the returned warnings say which.
pub fn expand_mcp_vars(input: &str, ctx: &McpExpansionContext) -> (String, Vec<String>) {
    expand_mcp_vars_with(input, ctx, &|name| std::env::var(name).ok())
}

/// [`expand_mcp_vars`] with an explicit host-environment lookup (tests pass a map instead of
/// mutating the process environment, which is not thread-safe).
fn expand_mcp_vars_with(
    input: &str,
    ctx: &McpExpansionContext,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> (String, Vec<String>) {
    let mut out = String::with_capacity(input.len());
    let mut warnings = Vec::new();
    let mut rest = input;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find('}') else {
            out.push_str(&rest[start..]);
            rest = "";
            break;
        };
        let body = &after[..end];
        let (name, default) = match body.split_once(":-") {
            Some((n, d)) => (n, Some(d)),
            None => (body, None),
        };
        let valid_name =
            !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !valid_name {
            out.push_str(&rest[start..start + 2 + end + 1]);
        } else {
            let value = match ctx.builtin(name) {
                Some(v) => Some(v),
                None => match ctx.host_var(name, lookup) {
                    Ok(v) => v,
                    Err(reason) => {
                        warnings.push(reason);
                        None
                    }
                },
            };
            match (value, default) {
                (Some(v), _) => out.push_str(&v),
                (None, Some(d)) => out.push_str(d),
                (None, None) => warnings.push(format!("'{name}' is not set; expanded to empty")),
            }
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    (out, warnings)
}

fn resolve_relative(plugin_root: &Path, raw: &str) -> PathBuf {
    if raw.starts_with("./") || raw.starts_with(".\\") {
        let rel = raw
            .trim_start_matches('.')
            .trim_start_matches('/')
            .trim_start_matches('\\');
        plugin_root.join(rel)
    } else {
        PathBuf::from(raw)
    }
}

/// `file://` URI for a local directory (MCP `roots/list`).
pub fn file_uri(path: &Path) -> String {
    let raw = path.to_string_lossy().replace('\\', "/");
    let encoded: Vec<String> = raw
        .split('/')
        .map(|seg| {
            // Keep a Windows drive (`C:`) readable.
            if seg.len() == 2 && seg.ends_with(':') {
                seg.to_string()
            } else {
                urlencoding::encode(seg).into_owned()
            }
        })
        .collect();
    let joined = encoded.join("/");
    if joined.starts_with('/') {
        format!("file://{joined}")
    } else {
        format!("file:///{joined}")
    }
}

/// Everything [`McpClient::launch`] needs besides the `mcp.json` entry.
#[derive(Clone, Debug)]
pub struct McpLaunchOptions {
    /// Server name from `mcpServers` (logs and error messages).
    pub server_name: String,
    pub expansion: McpExpansionContext,
    pub uv_binary: String,
    pub startup_timeout: Duration,
    pub tool_timeout: Duration,
}

impl McpLaunchOptions {
    pub fn new(server_name: impl Into<String>, plugin_root: &Path, uv_binary: &str) -> Self {
        Self {
            server_name: server_name.into(),
            expansion: McpExpansionContext {
                plugin_root: plugin_root.to_path_buf(),
                ..McpExpansionContext::default()
            },
            uv_binary: uv_binary.to_string(),
            startup_timeout: Duration::from_secs(DEFAULT_MCP_STARTUP_TIMEOUT_SECS),
            tool_timeout: Duration::from_secs(DEFAULT_MCP_TOOL_TIMEOUT_SECS),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------------------------

type PendingMap = HashMap<u64, oneshot::Sender<Result<Value, String>>>;

/// State shared between the client handle and its stdout / stderr tasks.
struct McpShared {
    server_name: String,
    stdin: Mutex<ChildStdin>,
    /// `None` once the server's stdout closed: new requests fail fast.
    pending: std::sync::Mutex<Option<PendingMap>>,
    stderr_tail: std::sync::Mutex<VecDeque<String>>,
    /// Set when stderr reached EOF, so a crash report can wait for the last stderr lines.
    stderr_closed: std::sync::atomic::AtomicBool,
    stderr_closed_notify: tokio::sync::Notify,
    /// `roots/list` answer (the agent sandbox, when known).
    roots: Vec<Value>,
}

impl McpShared {
    async fn write_message(&self, payload: &Value) -> Result<(), String> {
        let mut text = serde_json::to_string(payload).map_err(|e| e.to_string())?;
        text.push('\n');
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(text.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to MCP server '{}': {e}", self.server_name))?;
        stdin
            .flush()
            .await
            .map_err(|e| format!("Failed to flush MCP server '{}': {e}", self.server_name))
    }

    fn stderr_tail(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|tail| tail.iter().cloned().collect::<Vec<_>>().join("\n"))
            .unwrap_or_default()
    }

    fn closed_error(&self) -> String {
        let tail = self.stderr_tail();
        if tail.is_empty() {
            format!("MCP server '{}' exited (stdout closed)", self.server_name)
        } else {
            format!(
                "MCP server '{}' exited (stdout closed). Last stderr lines:\n{tail}",
                self.server_name
            )
        }
    }

    /// Marks the transport closed and fails every waiting request.
    fn close(&self) {
        let drained = self.pending.lock().ok().and_then(|mut guard| guard.take());
        if let Some(pending) = drained {
            let error = self.closed_error();
            for (_, tx) in pending {
                let _ = tx.send(Err(error.clone()));
            }
        }
    }

    fn complete(&self, id: u64, result: Result<Value, String>) {
        let tx = self
            .pending
            .lock()
            .ok()
            .and_then(|mut guard| guard.as_mut().and_then(|map| map.remove(&id)));
        match tx {
            Some(tx) => {
                let _ = tx.send(result);
            }
            None => debug!(
                "MCP server '{}' answered unknown or abandoned request id {id}",
                self.server_name
            ),
        }
    }

    async fn handle_message(&self, msg: Value) {
        if let Some(method) = msg.get("method").and_then(|m| m.as_str()) {
            match msg.get("id").cloned() {
                Some(id) => {
                    let response = match method {
                        "ping" => json!({"jsonrpc": "2.0", "id": id, "result": {}}),
                        "roots/list" => json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {"roots": self.roots}
                        }),
                        other => {
                            debug!(
                                "MCP server '{}' sent unsupported request '{other}'",
                                self.server_name
                            );
                            json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": {"code": -32601, "message": format!("Method not found: {other}")}
                            })
                        }
                    };
                    if let Err(e) = self.write_message(&response).await {
                        warn!("{e}");
                    }
                }
                None => debug!("MCP server '{}' notification: {method}", self.server_name),
            }
            return;
        }
        let Some(id) = msg.get("id").and_then(|v| v.as_u64()) else {
            debug!(
                "MCP server '{}' sent a message without a usable id",
                self.server_name
            );
            return;
        };
        if let Some(err) = msg.get("error") {
            self.complete(id, Err(format!("MCP error: {err}")));
        } else {
            self.complete(id, Ok(msg.get("result").cloned().unwrap_or(Value::Null)));
        }
    }
}

fn preview(line: &str) -> String {
    let mut s: String = line.chars().take(LOG_LINE_PREVIEW_CHARS).collect();
    if line.chars().count() > LOG_LINE_PREVIEW_CHARS {
        s.push('…');
    }
    s
}

async fn stdout_loop(shared: Arc<McpShared>, stdout: ChildStdout) {
    let mut reader = BufReader::new(stdout);
    let mut buf = Vec::new();
    let mut warned_non_json = false;
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf).await {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                warn!(
                    "MCP server '{}' stdout read failed: {e}",
                    shared.server_name
                );
                break;
            }
        }
        let line = String::from_utf8_lossy(&buf);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(trimmed) {
            Ok(Value::Array(batch)) => {
                for msg in batch {
                    shared.handle_message(msg).await;
                }
            }
            Ok(msg) if msg.is_object() => shared.handle_message(msg).await,
            _ => {
                // Servers must keep stdout for JSON-RPC; tolerate stray prints (import-time
                // banners, print() debugging) instead of failing the request in flight.
                if !warned_non_json {
                    warned_non_json = true;
                    warn!(
                        "MCP server '{}' wrote non-JSON to stdout (ignored; servers should log to stderr): {}",
                        shared.server_name,
                        preview(trimmed)
                    );
                } else {
                    debug!(
                        "MCP server '{}' non-JSON stdout: {}",
                        shared.server_name,
                        preview(trimmed)
                    );
                }
            }
        }
    }
    // A crashing server usually explains itself on stderr; give that reader a moment to catch
    // up so the error carries the reason.
    if !shared.stderr_closed.load(Ordering::SeqCst) {
        let notified = shared.stderr_closed_notify.notified();
        if !shared.stderr_closed.load(Ordering::SeqCst) {
            let _ = tokio::time::timeout(Duration::from_millis(500), notified).await;
        }
    }
    shared.close();
}

async fn stderr_loop(shared: Arc<McpShared>, stderr: ChildStderr) {
    let mut reader = BufReader::new(stderr);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&buf).trim_end().to_string();
        if line.is_empty() {
            continue;
        }
        debug!("[mcp:{}] {}", shared.server_name, preview(&line));
        if let Ok(mut tail) = shared.stderr_tail.lock() {
            if tail.len() >= STDERR_TAIL_LINES {
                tail.pop_front();
            }
            tail.push_back(line);
        }
    }
    shared.stderr_closed.store(true, Ordering::SeqCst);
    shared.stderr_closed_notify.notify_waiters();
}

/// Removes a request from the pending map when its caller goes away; when the request had
/// not completed, tells the server to stop working on it (`notifications/cancelled`).
struct PendingGuard {
    shared: Arc<McpShared>,
    id: u64,
    cancel_on_drop: bool,
    reason: &'static str,
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        let was_pending = self
            .shared
            .pending
            .lock()
            .ok()
            .and_then(|mut guard| guard.as_mut().and_then(|map| map.remove(&self.id)))
            .is_some();
        if was_pending && self.cancel_on_drop {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                let shared = Arc::clone(&self.shared);
                let payload = json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": self.id, "reason": self.reason}
                });
                handle.spawn(async move {
                    let _ = shared.write_message(&payload).await;
                });
            }
        }
    }
}

/// Process handle managing bidirectional stdio JSON-RPC 2.0 communication.
pub struct McpClient {
    shared: Arc<McpShared>,
    next_id: AtomicU64,
    startup_timeout: Duration,
    tool_timeout: Duration,
    /// Killed when the client is dropped (`kill_on_drop`).
    _child: std::sync::Mutex<Child>,
    stdout_task: tokio::task::JoinHandle<()>,
    stderr_task: tokio::task::JoinHandle<()>,
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.stdout_task.abort();
        self.stderr_task.abort();
    }
}

impl McpClient {
    /// Launches an MCP stdio server from its `mcp.json` entry (expanding `${VAR}`, resolving
    /// `./` paths against the plugin root and `uv`/`uvx` to the configured binary) and runs
    /// the `initialize` handshake within the startup timeout.
    pub async fn launch(entry: &McpServerEntry, opts: &McpLaunchOptions) -> Result<Self, String> {
        let server = opts.server_name.as_str();
        let plugin_root = opts.expansion.plugin_root.as_path();
        let mut warnings = Vec::new();
        let mut expand = |s: &str| -> String {
            let (value, w) = expand_mcp_vars(s, &opts.expansion);
            warnings.extend(w);
            value
        };

        // 1. Resolve executable command
        let raw_cmd = expand(&entry.command);
        let mut args: Vec<String> = Vec::new();
        let command_path = if raw_cmd == "uv" {
            opts.uv_binary.clone()
        } else if raw_cmd == "uvx" {
            // `uvx` is `uv tool run`; run it through the configured uv binary.
            args.extend(["tool".to_string(), "run".to_string()]);
            opts.uv_binary.clone()
        } else if raw_cmd.starts_with("./") || raw_cmd.starts_with(".\\") {
            resolve_relative(plugin_root, &raw_cmd)
                .to_string_lossy()
                .to_string()
        } else {
            raw_cmd
        };

        // 2. Arguments, working directory, environment
        args.extend(entry.args.iter().map(|a| expand(a)));
        let cwd_path = match entry.cwd.as_deref() {
            Some(cwd) => resolve_relative(plugin_root, &expand(cwd)),
            None => plugin_root.to_path_buf(),
        };
        let env: Vec<(String, String)> = entry
            .env
            .iter()
            .map(|(k, v)| (k.clone(), expand(v)))
            .collect();
        for w in warnings {
            warn!("MCP server '{server}' (mcp.json): {w}");
        }

        let mut cmd = Command::new(&command_path);
        cmd.args(&args);
        cmd.current_dir(&cwd_path);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        // Captured (not inherited): a server's stderr must not draw over the terminal UI.
        cmd.stderr(Stdio::piped());
        cmd.kill_on_drop(true);

        // MCP servers are agent-controlled child processes: start from the scrubbed safe
        // environment (never the host's master credentials), then layer the server's declared
        // env on top so per-server configuration still applies.
        crate::environment::ExecutionEnvironmentPolicy::default_safe()
            .apply_to_tokio_command(&mut cmd);
        let mut roots = Vec::new();
        if let Some(sandbox) = opts.expansion.sandbox_dir.as_ref() {
            cmd.env(MCP_SANDBOX_ENV, sandbox);
            roots.push(json!({"uri": file_uri(sandbox), "name": "workspace"}));
        }
        for (k, v) in &env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().map_err(|e| {
            format!("Failed to spawn MCP server '{server}' ('{command_path}'): {e}")
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Failed to capture MCP process stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture MCP process stdout".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "Failed to capture MCP process stderr".to_string())?;

        let shared = Arc::new(McpShared {
            server_name: server.to_string(),
            stdin: Mutex::new(stdin),
            pending: std::sync::Mutex::new(Some(HashMap::new())),
            stderr_tail: std::sync::Mutex::new(VecDeque::new()),
            stderr_closed: std::sync::atomic::AtomicBool::new(false),
            stderr_closed_notify: tokio::sync::Notify::new(),
            roots,
        });
        let stdout_task = tokio::spawn(stdout_loop(Arc::clone(&shared), stdout));
        let stderr_task = tokio::spawn(stderr_loop(Arc::clone(&shared), stderr));

        let client = Self {
            shared,
            next_id: AtomicU64::new(1),
            startup_timeout: opts.startup_timeout,
            tool_timeout: opts.tool_timeout,
            _child: std::sync::Mutex::new(child),
            stdout_task,
            stderr_task,
        };

        // Handshake: initialize
        client.initialize().await?;

        Ok(client)
    }

    /// Server name from `mcpServers`.
    pub fn server_name(&self) -> &str {
        &self.shared.server_name
    }

    async fn send_request_with_timeout(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, String> {
        let req_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        {
            let mut guard = self
                .shared
                .pending
                .lock()
                .map_err(|_| "MCP pending map poisoned".to_string())?;
            match guard.as_mut() {
                Some(map) => {
                    map.insert(req_id, tx);
                }
                None => return Err(self.shared.closed_error()),
            }
        }
        let _guard = PendingGuard {
            shared: Arc::clone(&self.shared),
            id: req_id,
            // The spec forbids cancelling `initialize`.
            cancel_on_drop: method != "initialize",
            reason: "client timeout or cancellation",
        };

        let payload = json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        });
        self.shared.write_message(&payload).await?;

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(self.shared.closed_error()),
            Err(_) => {
                let tail = self.shared.stderr_tail();
                let mut msg = format!(
                    "MCP server '{}' did not answer '{method}' within {}s",
                    self.shared.server_name,
                    timeout.as_secs()
                );
                if !tail.is_empty() {
                    msg.push_str(&format!(". Last stderr lines:\n{tail}"));
                }
                Err(msg)
            }
        }
    }

    async fn send_notification(&self, method: &str, params: Value) -> Result<(), String> {
        let payload = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        self.shared.write_message(&payload).await
    }

    async fn initialize(&self) -> Result<(), String> {
        let mut capabilities = json!({
            "extensions": {
                "io.modelcontextprotocol/ui": {
                    "mimeTypes": ["text/html;profile=mcp-app"]
                }
            }
        });
        if !self.shared.roots.is_empty() {
            capabilities["roots"] = json!({"listChanged": false});
        }
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": capabilities,
            "clientInfo": {
                "name": "isanagent",
                "version": env!("CARGO_PKG_VERSION")
            }
        });
        self.send_request_with_timeout("initialize", params, self.startup_timeout)
            .await?;
        self.send_notification("notifications/initialized", json!({}))
            .await?;
        Ok(())
    }

    /// Fetches tool definitions from the server (follows `nextCursor` pages).
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>, String> {
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = match cursor.as_deref() {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let res = self
                .send_request_with_timeout("tools/list", params, self.startup_timeout)
                .await?;
            let tools_val = res.get("tools").cloned().unwrap_or(Value::Array(vec![]));
            let page: Vec<McpToolDefinition> = serde_json::from_value(tools_val)
                .map_err(|e| format!("Failed to parse tools list: {e}"))?;
            out.extend(page);
            match res.get("nextCursor").and_then(|v| v.as_str()) {
                Some(next) if !next.is_empty() && cursor.as_deref() != Some(next) => {
                    cursor = Some(next.to_string());
                }
                _ => break,
            }
        }
        Ok(out)
    }

    /// Invokes a tool and returns the raw `tools/call` result.
    pub async fn call_tool_raw(&self, name: &str, arguments: Value) -> Result<Value, String> {
        let params = json!({
            "name": name,
            "arguments": arguments
        });
        self.send_request_with_timeout("tools/call", params, self.tool_timeout)
            .await
    }

    /// Invokes a tool by name with arguments. The model sees text content only.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<String, String> {
        let res = self.call_tool_raw(name, arguments).await?;
        tool_result_text(name, &res)
    }

    /// Loads an MCP App HTML resource (`text/html;profile=mcp-app`).
    pub async fn read_ui_resource(&self, uri: &str) -> Result<McpUiResource, String> {
        let res = self
            .send_request_with_timeout("resources/read", json!({ "uri": uri }), self.tool_timeout)
            .await?;
        let contents = res
            .get("contents")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("MCP resource '{uri}' returned no contents"))?;
        let first = contents
            .first()
            .ok_or_else(|| format!("MCP resource '{uri}' returned empty contents"))?;
        let mime_type = first
            .get("mimeType")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !mime_type.starts_with("text/html") {
            return Err(format!(
                "MCP resource '{uri}' is '{mime_type}', not an HTML MCP App"
            ));
        }
        let html = first
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("MCP resource '{uri}' has no text body"))?;
        if html.len() > MCP_APP_HTML_MAX_BYTES {
            return Err(format!(
                "MCP App HTML for '{uri}' is {} bytes, over the {} byte limit",
                html.len(),
                MCP_APP_HTML_MAX_BYTES
            ));
        }
        let csp = first.pointer("/_meta/ui/csp").cloned();
        Ok(McpUiResource {
            html: html.to_string(),
            mime_type,
            csp,
        })
    }
}

/// HTML document served by an MCP App resource.
#[derive(Debug, Clone)]
pub struct McpUiResource {
    pub html: String,
    pub mime_type: String,
    pub csp: Option<Value>,
}

fn tool_result_text(name: &str, res: &Value) -> Result<String, String> {
    if res.get("isError").and_then(|v| v.as_bool()) == Some(true) {
        let err_msg = extract_content_text(res);
        return Err(if err_msg.is_empty() {
            format!("MCP tool '{name}' reported an error")
        } else {
            err_msg
        });
    }
    Ok(extract_content_text(res))
}

fn extract_content_text(val: &Value) -> String {
    if let Some(content) = val.get("content").and_then(|v| v.as_array()) {
        let mut out = String::new();
        for item in content {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(text);
            }
        }
        out
    } else if let Some(text) = val.as_str() {
        text.to_string()
    } else {
        val.to_string()
    }
}

/// Upper bound for one MCP App HTML document kept in memory.
pub const MCP_APP_HTML_MAX_BYTES: usize = 1_500_000;

const MCP_APP_VIEW_CAP: usize = 64;
const MCP_APP_TOOL_RESULT_MAX_BYTES: usize = 256_000;

/// `ui://` resource declared on a tool or a `tools/call` result.
pub fn ui_resource_uri(meta: Option<&Value>) -> Option<String> {
    let meta = meta?;
    let raw = meta
        .pointer("/ui/resourceUri")
        .or_else(|| meta.get("ui/resourceUri"))
        .and_then(|value| value.as_str())?;
    let uri = raw.trim();
    if uri.starts_with("ui://") && uri.len() <= 512 {
        Some(uri.to_string())
    } else {
        None
    }
}

/// Tools whose MCP Apps visibility list omits `model` stay callable by the
/// widget but are not offered to the LLM.
pub fn tool_visible_to_model(meta: Option<&Value>) -> bool {
    let Some(list) = meta
        .and_then(|value| value.pointer("/ui/visibility"))
        .and_then(|value| value.as_array())
    else {
        return true;
    };
    list.is_empty() || list.iter().any(|value| value.as_str() == Some("model"))
}

fn valid_mcp_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

struct McpAppView {
    resource_uri: String,
    html: String,
    mime_type: String,
    csp: Option<Value>,
    tool_args: Value,
    tool_result: Value,
    client: Arc<McpClient>,
}

struct McpAppBridgeInner {
    order: std::collections::VecDeque<String>,
    views: HashMap<String, McpAppView>,
    /// Keeps stdio servers alive even when every tool is app-only.
    clients: Vec<Arc<McpClient>>,
}

/// In-memory MCP App views. The web UI fetches HTML by `view_id` and proxies
/// widget `tools/call` requests back to the same stdio client.
#[derive(Clone)]
pub struct McpAppBridge {
    inner: Arc<std::sync::Mutex<McpAppBridgeInner>>,
}

/// JSON body for `GET /v1/mcp/apps/{view_id}`.
#[derive(Debug, Clone, Serialize)]
pub struct McpAppSnapshot {
    pub view_id: String,
    pub resource_uri: String,
    pub html: String,
    pub mime_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub csp: Option<Value>,
    pub tool_args: Value,
    pub tool_result: Value,
}

impl McpAppBridge {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(McpAppBridgeInner {
                order: std::collections::VecDeque::new(),
                views: HashMap::new(),
                clients: Vec::new(),
            })),
        }
    }

    pub fn retain_client(&self, client: Arc<McpClient>) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.clients.push(client);
        }
    }

    pub fn register_view(
        &self,
        client: Arc<McpClient>,
        resource_uri: String,
        resource: McpUiResource,
        tool_args: Value,
        tool_result: Value,
    ) -> crate::bus::McpAppRef {
        let view_id = uuid::Uuid::new_v4().to_string();
        let tool_result = cap_tool_result(tool_result);
        let view = McpAppView {
            resource_uri: resource_uri.clone(),
            html: resource.html,
            mime_type: resource.mime_type,
            csp: resource.csp,
            tool_args,
            tool_result,
            client,
        };
        if let Ok(mut guard) = self.inner.lock() {
            guard.order.push_back(view_id.clone());
            guard.views.insert(view_id.clone(), view);
            while guard.order.len() > MCP_APP_VIEW_CAP {
                if let Some(old) = guard.order.pop_front() {
                    guard.views.remove(&old);
                }
            }
        }
        crate::bus::McpAppRef {
            view_id,
            resource_uri,
        }
    }

    pub fn snapshot(&self, view_id: &str) -> Option<McpAppSnapshot> {
        let guard = self.inner.lock().ok()?;
        let view = guard.views.get(view_id)?;
        Some(McpAppSnapshot {
            view_id: view_id.to_string(),
            resource_uri: view.resource_uri.clone(),
            html: view.html.clone(),
            mime_type: view.mime_type.clone(),
            csp: view.csp.clone(),
            tool_args: view.tool_args.clone(),
            tool_result: view.tool_result.clone(),
        })
    }

    pub async fn call_tool(
        &self,
        view_id: &str,
        name: &str,
        arguments: Value,
    ) -> Result<Value, String> {
        if !valid_mcp_tool_name(name) {
            return Err(format!("Rejected MCP App tool name '{name}'"));
        }
        let arguments = if arguments.is_null() {
            json!({})
        } else if arguments.is_object() {
            arguments
        } else {
            return Err("MCP App tool arguments must be a JSON object".to_string());
        };
        let client = {
            let guard = self
                .inner
                .lock()
                .map_err(|_| "MCP App bridge lock poisoned".to_string())?;
            guard
                .views
                .get(view_id)
                .map(|view| Arc::clone(&view.client))
                .ok_or_else(|| format!("MCP App view '{view_id}' not found"))?
        };
        client.call_tool_raw(name, arguments).await
    }
}

impl Default for McpAppBridge {
    fn default() -> Self {
        Self::new()
    }
}

fn cap_tool_result(result: Value) -> Value {
    let encoded = serde_json::to_string(&result).unwrap_or_default();
    if encoded.len() <= MCP_APP_TOOL_RESULT_MAX_BYTES {
        return result;
    }
    json!({
        "content": [{
            "type": "text",
            "text": "[MCP App tool result omitted: larger than the host cache]"
        }]
    })
}

/// A dynamic proxy tool routing calls to a running MCP server client.
///
/// The model sees [`namespaced_tool_name`] (`{server}__{tool}`); `tools/call` uses the
/// server's own name.
pub struct McpProxyTool {
    name: String,
    remote_name: String,
    description: String,
    parameters: Value,
    ui_resource_uri: Option<String>,
    client: Arc<McpClient>,
    apps: McpAppBridge,
}

impl McpProxyTool {
    pub fn new(def: McpToolDefinition, client: Arc<McpClient>, apps: McpAppBridge) -> Self {
        let ui_resource_uri = ui_resource_uri(def.meta.as_ref());
        let name = namespaced_tool_name(client.server_name(), &def.name);
        Self {
            name,
            remote_name: def.name,
            description: def.description.unwrap_or_default(),
            parameters: def
                .input_schema
                .unwrap_or_else(|| json!({"type": "object"})),
            ui_resource_uri,
            client,
            apps,
        }
    }

    /// The tool's name on its MCP server.
    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }
}

#[async_trait]
impl Tool for McpProxyTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::serial()
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        self.client.call_tool(&self.remote_name, args).await
    }

    async fn execute_with_approved_mutation_typed(
        &self,
        args: Value,
        _approved_preview: Option<&crate::traits::MutationPreview>,
    ) -> crate::traits::ToolResult {
        use crate::traits::{ToolErrorCode, ToolResult};

        let raw = match self
            .client
            .call_tool_raw(&self.remote_name, args.clone())
            .await
        {
            Ok(raw) => raw,
            Err(error) => return ToolResult::error(ToolErrorCode::ExecutionFailed, error),
        };
        let text = match tool_result_text(&self.remote_name, &raw) {
            Ok(text) => text,
            Err(error) => return ToolResult::error(ToolErrorCode::ExecutionFailed, error),
        };
        let resource_uri =
            ui_resource_uri(raw.get("_meta")).or_else(|| self.ui_resource_uri.clone());
        let Some(resource_uri) = resource_uri else {
            return ToolResult::success(text);
        };
        match self.client.read_ui_resource(&resource_uri).await {
            Ok(resource) => {
                let app_ref = self.apps.register_view(
                    Arc::clone(&self.client),
                    resource_uri,
                    resource,
                    args,
                    raw,
                );
                ToolResult::success(text).with_mcp_app(app_ref)
            }
            Err(error) => {
                warn!(
                    "MCP tool '{}' declared UI resource '{resource_uri}' but it could not be loaded: {error}",
                    self.name
                );
                ToolResult::success(text)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        namespaced_tool_name, tool_passes_filters, tool_visible_to_model, ui_resource_uri,
        McpAppBridge, McpClient, McpExpansionContext, McpLaunchOptions, McpServerEntry,
        MAX_PROVIDER_TOOL_NAME_LEN,
    };
    use serde_json::json;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn namespaced_names_are_provider_safe_and_stable() {
        assert_eq!(namespaced_tool_name("chem", "read_file"), "chem__read_file");
        assert_eq!(
            namespaced_tool_name("chem.agent", "rdkit/logp"),
            "chem_agent__rdkit_logp"
        );
        assert_eq!(namespaced_tool_name("1chem", "x"), "mcp_1chem__x");
        let long = namespaced_tool_name("chemistry-server", &"very_long_tool_name_".repeat(5));
        assert_eq!(long.len(), MAX_PROVIDER_TOOL_NAME_LEN);
        assert!(long.starts_with("chemistry-server__very_long"));
        assert_eq!(
            long,
            namespaced_tool_name("chemistry-server", &"very_long_tool_name_".repeat(5))
        );
        // Two long names sharing the readable prefix still differ.
        let other = namespaced_tool_name("chemistry-server", &"very_long_tool_name_".repeat(6));
        assert_ne!(long, other);
        let re = regex::Regex::new(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$").unwrap();
        assert!(re.is_match(&long) && re.is_match(&other));
    }

    #[test]
    fn include_and_exclude_filters_use_original_names() {
        let exclude = vec!["read_file".to_string(), "python_*".to_string()];
        assert!(!tool_passes_filters("read_file", None, &exclude));
        assert!(!tool_passes_filters("python_interpreter", None, &exclude));
        assert!(tool_passes_filters("rdkit_logp", None, &exclude));
        let include = vec!["rdkit_*".to_string()];
        assert!(tool_passes_filters("rdkit_logp", Some(&include), &exclude));
        assert!(!tool_passes_filters("stats_mean", Some(&include), &exclude));
    }

    #[test]
    fn expansion_handles_builtins_defaults_and_secret_opt_in() {
        let ctx = McpExpansionContext {
            plugin_root: PathBuf::from("/plugins/chem"),
            sandbox_dir: Some(PathBuf::from("/ws/workspace")),
            env_passthrough: vec!["ISANAGENT_TEST_NCBI_API_KEY".to_string()],
        };
        // A fake host environment: mutating the real one (`set_var`) races other test threads.
        let host: std::collections::HashMap<&str, &str> = [
            ("ISANAGENT_TEST_NCBI_API_KEY", "k-allowed"),
            ("ISANAGENT_TEST_OTHER_API_KEY", "k-hidden"),
            ("MCPTEST_PLAIN_VALUE", "plain"),
        ]
        .into_iter()
        .collect();
        let lookup = |name: &str| host.get(name).map(|v| v.to_string());
        let expand_mcp_vars = |input: &str, ctx: &McpExpansionContext| {
            super::expand_mcp_vars_with(input, ctx, &lookup)
        };

        let (v, w) = expand_mcp_vars("${PLUGIN_DIR}/server.py ${SANDBOX_DIR}", &ctx);
        assert_eq!(v, "/plugins/chem/server.py /ws/workspace");
        assert!(w.is_empty());
        let (v, _) = expand_mcp_vars("${ISANAGENT_TEST_NCBI_API_KEY}", &ctx);
        assert_eq!(v, "k-allowed");
        let (v, w) = expand_mcp_vars("${ISANAGENT_TEST_OTHER_API_KEY:-none}", &ctx);
        assert_eq!(v, "none");
        assert_eq!(w.len(), 1, "secret without opt-in must warn");
        let (v, _) = expand_mcp_vars("a-${MCPTEST_PLAIN_VALUE}-b", &ctx);
        assert_eq!(v, "a-plain-b");
        let (v, w) = expand_mcp_vars("${MCPTEST_UNSET_XYZ}|${MCPTEST_UNSET_XYZ:-d}", &ctx);
        assert_eq!(v, "|d");
        assert_eq!(w.len(), 1);
        let (v, _) = expand_mcp_vars("${not valid} ${unterminated", &ctx);
        assert_eq!(v, "${not valid} ${unterminated");
    }

    fn fake_server() -> Option<PathBuf> {
        which::which("python3").ok()?;
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_mcp_server.py");
        path.is_file().then_some(path)
    }

    async fn launch_fake(tool_timeout: Duration, sandbox: Option<PathBuf>) -> Option<McpClient> {
        let server = fake_server()?;
        let root = server.parent().unwrap().to_path_buf();
        let mut opts = McpLaunchOptions::new("fake", &root, "uv");
        opts.tool_timeout = tool_timeout;
        opts.startup_timeout = Duration::from_secs(20);
        opts.expansion.sandbox_dir = sandbox;
        let entry = McpServerEntry {
            command: "python3".to_string(),
            args: vec![
                "-u".to_string(),
                "${PLUGIN_DIR}/fake_mcp_server.py".to_string(),
            ],
            env: [(
                "FAKE_SANDBOX".to_string(),
                "${SANDBOX_DIR:-none}".to_string(),
            )]
            .into_iter()
            .collect(),
            ..McpServerEntry::default()
        };
        Some(
            McpClient::launch(&entry, &opts)
                .await
                .expect("launch fake server"),
        )
    }

    #[tokio::test]
    async fn noisy_stdout_paging_and_out_of_order_responses() {
        let Some(client) = launch_fake(Duration::from_secs(20), None).await else {
            return;
        };
        let client = std::sync::Arc::new(client);
        // The banner printed before the handshake did not break it, and both pages arrive.
        let tools = client.list_tools().await.expect("tools/list");
        assert_eq!(tools.len(), 6);

        // A slow call does not block (or swallow the response of) a fast one.
        let slow = {
            let c = client.clone();
            tokio::spawn(async move { c.call_tool("slow", json!({"secs": 1.0})).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        let started = std::time::Instant::now();
        let echoed = client
            .call_tool("echo", json!({"text": "hi"}))
            .await
            .unwrap();
        assert_eq!(echoed, "hi");
        assert!(started.elapsed() < Duration::from_millis(900));
        assert_eq!(slow.await.unwrap().unwrap(), "slept");
    }

    #[tokio::test]
    async fn timeouts_cancel_and_the_client_stays_usable() {
        let Some(client) = launch_fake(Duration::from_millis(300), None).await else {
            return;
        };
        let err = client
            .call_tool("slow", json!({"secs": 2.0}))
            .await
            .expect_err("must time out");
        assert!(err.contains("did not answer 'tools/call'"), "{err}");
        assert_eq!(
            client
                .call_tool("echo", json!({"text": "ok"}))
                .await
                .unwrap(),
            "ok"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
        let cancelled = client.call_tool("cancelled", json!({})).await.unwrap();
        assert_ne!(
            cancelled, "[]",
            "server must have received notifications/cancelled"
        );
    }

    #[tokio::test]
    async fn roots_and_sandbox_env_reach_the_server() {
        let dir = tempfile::tempdir().unwrap();
        let Some(client) =
            launch_fake(Duration::from_secs(20), Some(dir.path().to_path_buf())).await
        else {
            return;
        };
        let roots = client.call_tool("roots", json!({})).await.unwrap();
        let roots: serde_json::Value = serde_json::from_str(&roots).unwrap();
        let uri = roots["roots"][0]["uri"].as_str().unwrap();
        assert!(uri.starts_with("file://"), "{uri}");
        assert!(uri.ends_with(dir.path().file_name().unwrap().to_str().unwrap()));
        let env = client
            .call_tool("env", json!({"key": super::MCP_SANDBOX_ENV}))
            .await
            .unwrap();
        assert_eq!(PathBuf::from(env), dir.path());
        let expanded = client
            .call_tool("env", json!({"key": "FAKE_SANDBOX"}))
            .await
            .unwrap();
        assert_eq!(PathBuf::from(expanded), dir.path());
    }

    #[tokio::test]
    async fn a_crashing_server_fails_fast_with_its_stderr() {
        let Some(client) = launch_fake(Duration::from_secs(20), None).await else {
            return;
        };
        let err = client
            .call_tool("crash", json!({}))
            .await
            .expect_err("crash");
        assert!(err.contains("exited"), "{err}");
        assert!(err.contains("rdkit import failed"), "{err}");
        let again = client
            .call_tool("echo", json!({"text": "x"}))
            .await
            .expect_err("closed");
        assert!(again.contains("exited"), "{again}");
    }

    #[test]
    fn ui_resource_uri_reads_nested_and_flat_meta() {
        let nested = json!({ "ui": { "resourceUri": "ui://mcp-ui-tester-app/dashboard" } });
        assert_eq!(
            ui_resource_uri(Some(&nested)).as_deref(),
            Some("ui://mcp-ui-tester-app/dashboard")
        );
        let flat = json!({ "ui/resourceUri": "ui://widgets/picker" });
        assert_eq!(
            ui_resource_uri(Some(&flat)).as_deref(),
            Some("ui://widgets/picker")
        );
        let file = json!({ "ui": { "resourceUri": "file:///etc/passwd" } });
        assert!(ui_resource_uri(Some(&file)).is_none());
    }

    #[test]
    fn app_only_tools_are_hidden_from_the_model() {
        assert!(tool_visible_to_model(None));
        let both = json!({ "ui": { "visibility": ["model", "app"] } });
        assert!(tool_visible_to_model(Some(&both)));
        let app_only = json!({ "ui": { "visibility": ["app"] } });
        assert!(!tool_visible_to_model(Some(&app_only)));
    }

    fn simple_mcp_app_server() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("ISANAGENT_SIMPLE_MCP_APP") {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Some(path);
            }
        }
        let sibling = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../simple-mcp-app/server.js");
        sibling.is_file().then_some(sibling)
    }

    #[tokio::test]
    async fn simple_mcp_app_dashboard_round_trip() {
        let Some(server) = simple_mcp_app_server() else {
            return;
        };
        let root = server.parent().expect("server parent").to_path_buf();
        let client = McpClient::launch(
            &McpServerEntry {
                command: "node".to_string(),
                args: vec![server.to_string_lossy().to_string()],
                cwd: Some(root.to_string_lossy().to_string()),
                ..McpServerEntry::default()
            },
            &super::McpLaunchOptions::new("simple-mcp-app", &root, "uv"),
        )
        .await
        .expect("launch simple-mcp-app");
        let client = std::sync::Arc::new(client);
        let tools = client.list_tools().await.expect("tools/list");
        let dashboard = tools
            .iter()
            .find(|tool| tool.name == "open_dashboard")
            .expect("open_dashboard");
        assert_eq!(
            ui_resource_uri(dashboard.meta.as_ref()).as_deref(),
            Some("ui://mcp-ui-tester-app/dashboard")
        );

        let raw = client
            .call_tool_raw("open_dashboard", json!({}))
            .await
            .expect("tools/call");
        let text = super::tool_result_text("open_dashboard", &raw).expect("text result");
        assert!(text.contains("dashboard"));
        let uri = ui_resource_uri(raw.get("_meta")).expect("result resource uri");
        let resource = client.read_ui_resource(&uri).await.expect("resources/read");
        assert!(resource.html.contains("ui/initialize"));
        assert!(resource.mime_type.starts_with("text/html"));

        let bridge = McpAppBridge::new();
        let app = bridge.register_view(
            std::sync::Arc::clone(&client),
            uri,
            resource,
            json!({}),
            raw,
        );
        let items = bridge
            .call_tool(&app.view_id, "get_items", json!({}))
            .await
            .expect("widget tools/call");
        let body = items["content"][0]["text"].as_str().expect("items text");
        assert!(body.contains("Review MCP Apps spec"));
        assert!(bridge
            .call_tool(&app.view_id, "../exec", json!({}))
            .await
            .is_err());
    }
}
