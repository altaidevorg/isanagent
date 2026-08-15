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
            "capabilities": {},
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

    /// Invokes a tool by name with arguments.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<String, String> {
        let params = json!({
            "name": name,
            "arguments": arguments
        });
        let res = self.send_request("tools/call", params).await?;

        if let Some(is_error) = res.get("isError").and_then(|v| v.as_bool()) {
            if is_error {
                let err_msg = extract_content_text(&res);
                return Err(if err_msg.is_empty() {
                    format!("MCP tool '{name}' reported an error")
                } else {
                    err_msg
                });
            }
        }

        Ok(extract_content_text(&res))
    }
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

/// A dynamic proxy tool routing calls to a running MCP server client.
pub struct McpProxyTool {
    name: String,
    description: String,
    parameters: Value,
    client: Arc<McpClient>,
}

impl McpProxyTool {
    pub fn new(def: McpToolDefinition, client: Arc<McpClient>) -> Self {
        Self {
            name: def.name,
            description: def.description.unwrap_or_default(),
            parameters: def
                .input_schema
                .unwrap_or_else(|| json!({"type": "object"})),
            client,
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
}
