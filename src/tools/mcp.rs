//! Model Context Protocol (MCP) Stdio Client & Tool Adapter.
//!
//! Conforms to the MCP specification and Agent Plugins 1.0 `mcp.json` definition.
//! Launches stdio servers, exchanges initialize and tools/list requests, and exposes
//! discovered tools as native [`crate::traits::Tool`] instances.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use log::warn;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::traits::{Tool, ToolPolicy};

fn default_mcp_type() -> String {
    "stdio".to_string()
}

/// Configuration schema for `mcp.json` conforming to Agent Plugins 1.0.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct McpConfigFile {
    #[serde(rename = "$schema", default)]
    pub schema: Option<String>,
    #[serde(rename = "mcpServers", default)]
    pub mcp_servers: HashMap<String, McpServerEntry>,
}

/// Single server definition inside `mcpServers`.
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

/// Process handle managing bidirectional stdio JSON-RPC 2.0 communication.
pub struct McpClient {
    #[allow(dead_code)]
    child: Child,
    stdin: Mutex<ChildStdin>,
    reader: Mutex<BufReader<ChildStdout>>,
    next_id: AtomicU64,
}

impl McpClient {
    /// Launches an MCP stdio server process from configured entry, resolving paths and UV if applicable.
    pub async fn launch(
        plugin_root: &Path,
        entry: &McpServerEntry,
        uv_binary: &str,
    ) -> Result<Self, String> {
        let plugin_str = plugin_root.to_string_lossy().to_string();

        let expand_vars = |s: &str| -> String {
            s.replace("${PLUGIN_DIR}", &plugin_str)
                .replace("${plugin_dir}", &plugin_str)
                .replace("${CLAUDE_PLUGIN_DIR}", &plugin_str)
        };

        // 1. Resolve executable command
        let raw_cmd = expand_vars(&entry.command);
        let command_path = if raw_cmd == "uv" || raw_cmd == "uvx" {
            uv_binary.to_string()
        } else if raw_cmd.starts_with("./") || raw_cmd.starts_with(".\\") {
            let rel = raw_cmd
                .trim_start_matches('.')
                .trim_start_matches('/')
                .trim_start_matches('\\');
            let resolved = plugin_root.join(rel);
            resolved.to_string_lossy().to_string()
        } else {
            raw_cmd
        };

        // 2. Resolve arguments
        let args: Vec<String> = entry.args.iter().map(|a| expand_vars(a)).collect();

        // 3. Resolve working directory
        let cwd_path = if let Some(ref cwd) = entry.cwd {
            let expanded = expand_vars(cwd);
            if expanded.starts_with("./") || expanded.starts_with(".\\") {
                let rel = expanded
                    .trim_start_matches('.')
                    .trim_start_matches('/')
                    .trim_start_matches('\\');
                plugin_root.join(rel)
            } else {
                PathBuf::from(expanded)
            }
        } else {
            plugin_root.to_path_buf()
        };

        let mut cmd = Command::new(&command_path);
        cmd.args(&args);
        cmd.current_dir(&cwd_path);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::inherit());

        // MCP servers are agent-controlled child processes: start from the scrubbed safe
        // environment (never the host's master credentials), then layer the server's declared
        // env on top so per-server configuration still applies.
        crate::environment::ExecutionEnvironmentPolicy::default_safe()
            .apply_to_tokio_command(&mut cmd);
        for (k, v) in &entry.env {
            cmd.env(k, expand_vars(v));
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn MCP process '{command_path}': {e}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Failed to capture MCP process stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture MCP process stdout".to_string())?;

        let client = Self {
            child,
            stdin: Mutex::new(stdin),
            reader: Mutex::new(BufReader::new(stdout)),
            next_id: AtomicU64::new(1),
        };

        // Handshake: initialize
        client.initialize().await?;

        Ok(client)
    }

    async fn send_request(&self, method: &str, params: Value) -> Result<Value, String> {
        let req_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let payload = json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        });

        let mut text = serde_json::to_string(&payload).map_err(|e| e.to_string())?;
        text.push('\n');

        {
            let mut stdin = self.stdin.lock().await;
            stdin
                .write_all(text.as_bytes())
                .await
                .map_err(|e| format!("Failed to write to MCP stdin: {e}"))?;
            stdin
                .flush()
                .await
                .map_err(|e| format!("Failed to flush MCP stdin: {e}"))?;
        }

        let mut reader = self.reader.lock().await;
        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = reader
                .read_line(&mut line)
                .await
                .map_err(|e| format!("Failed to read from MCP stdout: {e}"))?;
            if bytes_read == 0 {
                return Err("MCP process closed stdout unexpectedly".to_string());
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let resp: Value = serde_json::from_str(trimmed).map_err(|e| {
                format!("Failed to parse MCP JSON-RPC response: {e} (raw: {trimmed})")
            })?;

            if resp.get("id").and_then(|v| v.as_u64()) == Some(req_id) {
                if let Some(err) = resp.get("error") {
                    return Err(format!("MCP error: {err}"));
                }
                return Ok(resp.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }

    async fn send_notification(&self, method: &str, params: Value) -> Result<(), String> {
        let payload = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });

        let mut text = serde_json::to_string(&payload).map_err(|e| e.to_string())?;
        text.push('\n');

        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(text.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to MCP stdin: {e}"))?;
        stdin
            .flush()
            .await
            .map_err(|e| format!("Failed to flush MCP stdin: {e}"))?;
        Ok(())
    }

    async fn initialize(&self) -> Result<(), String> {
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "extensions": {
                    "io.modelcontextprotocol/ui": {
                        "mimeTypes": ["text/html;profile=mcp-app"]
                    }
                }
            },
            "clientInfo": {
                "name": "isanagent",
                "version": env!("CARGO_PKG_VERSION")
            }
        });
        self.send_request("initialize", params).await?;
        self.send_notification("notifications/initialized", json!({}))
            .await?;
        Ok(())
    }

    /// Fetches tool definitions from the server.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>, String> {
        let res = self.send_request("tools/list", json!({})).await?;
        let tools_val = res.get("tools").cloned().unwrap_or(Value::Array(vec![]));
        serde_json::from_value(tools_val).map_err(|e| format!("Failed to parse tools list: {e}"))
    }

    /// Invokes a tool and returns the raw `tools/call` result.
    pub async fn call_tool_raw(&self, name: &str, arguments: Value) -> Result<Value, String> {
        let params = json!({
            "name": name,
            "arguments": arguments
        });
        self.send_request("tools/call", params).await
    }

    /// Invokes a tool by name with arguments. The model sees text content only.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<String, String> {
        let res = self.call_tool_raw(name, arguments).await?;
        tool_result_text(name, &res)
    }

    /// Loads an MCP App HTML resource (`text/html;profile=mcp-app`).
    pub async fn read_ui_resource(&self, uri: &str) -> Result<McpUiResource, String> {
        let res = self
            .send_request("resources/read", json!({ "uri": uri }))
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
pub struct McpProxyTool {
    name: String,
    description: String,
    parameters: Value,
    ui_resource_uri: Option<String>,
    client: Arc<McpClient>,
    apps: McpAppBridge,
}

impl McpProxyTool {
    pub fn new(def: McpToolDefinition, client: Arc<McpClient>, apps: McpAppBridge) -> Self {
        let ui_resource_uri = ui_resource_uri(def.meta.as_ref());
        Self {
            name: def.name,
            description: def.description.unwrap_or_default(),
            parameters: def
                .input_schema
                .unwrap_or_else(|| json!({"type": "object"})),
            ui_resource_uri,
            client,
            apps,
        }
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
        self.client.call_tool(&self.name, args).await
    }

    async fn execute_with_approved_mutation_typed(
        &self,
        args: Value,
        _approved_preview: Option<&crate::traits::MutationPreview>,
    ) -> crate::traits::ToolResult {
        use crate::traits::{ToolErrorCode, ToolResult};

        let raw = match self.client.call_tool_raw(&self.name, args.clone()).await {
            Ok(raw) => raw,
            Err(error) => return ToolResult::error(ToolErrorCode::ExecutionFailed, error),
        };
        let text = match tool_result_text(&self.name, &raw) {
            Ok(text) => text,
            Err(error) => return ToolResult::error(ToolErrorCode::ExecutionFailed, error),
        };
        let resource_uri = ui_resource_uri(raw.get("_meta")).or_else(|| self.ui_resource_uri.clone());
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
        tool_visible_to_model, ui_resource_uri, McpAppBridge, McpClient, McpServerEntry,
    };
    use serde_json::json;
    use std::path::PathBuf;

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
            &root,
            &McpServerEntry {
                command: "node".to_string(),
                args: vec![server.to_string_lossy().to_string()],
                cwd: Some(root.to_string_lossy().to_string()),
                ..McpServerEntry::default()
            },
            "uv",
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
        let body = items["content"][0]["text"]
            .as_str()
            .expect("items text");
        assert!(body.contains("Review MCP Apps spec"));
        assert!(bridge
            .call_tool(&app.view_id, "../exec", json!({}))
            .await
            .is_err());
    }
}
