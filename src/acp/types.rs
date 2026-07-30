use serde::{Deserialize, Serialize};
use serde_json::Value;

/// ACP Protocol Version supported by this agent.
pub const ACP_PROTOCOL_VERSION: u32 = 1;

/// JSON-RPC 2.0 Request envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 Response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// JSON-RPC 2.0 Notification envelope (no `id`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcNotification {
    pub fn new(method: impl Into<String>, params: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params: Some(params),
        }
    }
}

/// JSON-RPC 2.0 Error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

// Common JSON-RPC error codes.
pub const ERR_PARSE_ERROR: i64 = -32700;
pub const ERR_INVALID_REQUEST: i64 = -32600;
pub const ERR_METHOD_NOT_FOUND: i64 = -32601;
pub const ERR_INVALID_PARAMS: i64 = -32602;
pub const ERR_INTERNAL_ERROR: i64 = -32603;

// --- ACP Capabilities & Client/Agent Info ---

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fs: Option<FsCapabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elicitation: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct FsCapabilities {
    #[serde(default)]
    pub read_text_file: bool,
    #[serde(default)]
    pub write_text_file: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    #[serde(default)]
    pub load_session: bool,
    pub prompt_capabilities: PromptCapabilities,
    pub mcp_capabilities: McpCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delete: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_directories: Option<Value>,
    pub session_capabilities: SessionCapabilities,
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self {
            load_session: true,
            prompt_capabilities: PromptCapabilities {
                image: true,
                audio: false,
                embedded_context: true,
            },
            mcp_capabilities: McpCapabilities {
                http: true,
                sse: false,
            },
            delete: Some(serde_json::json!({})),
            additional_directories: Some(serde_json::json!({})),
            session_capabilities: SessionCapabilities {
                resume: Some(serde_json::json!({})),
                close: Some(serde_json::json!({})),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptCapabilities {
    pub image: bool,
    pub audio: bool,
    pub embedded_context: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpCapabilities {
    pub http: bool,
    pub sse: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub close: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImplementationInfo {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub version: String,
}

// --- Request & Response Payloads ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: u32,
    #[serde(default)]
    pub client_capabilities: ClientCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_info: Option<ImplementationInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: u32,
    pub agent_capabilities: AgentCapabilities,
    pub agent_info: ImplementationInfo,
    pub auth_methods: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNewParams {
    pub cwd: String,
    #[serde(default)]
    pub mcp_servers: Vec<AcpMcpServerConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_directories: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNewResult {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionLoadParams {
    pub session_id: String,
    pub cwd: String,
    #[serde(default)]
    pub mcp_servers: Vec<AcpMcpServerConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_directories: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionResumeParams {
    pub session_id: String,
    pub cwd: String,
    #[serde(default)]
    pub mcp_servers: Vec<AcpMcpServerConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_directories: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCloseParams {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionDeleteParams {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpMcpServerConfig {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<Vec<AcpEnvVar>>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub server_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcpEnvVar {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionPromptParams {
    pub session_id: String,
    pub prompt: Vec<AcpContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionPromptResult {
    pub stop_reason: AcpStopReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcpStopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCancelParams {
    pub session_id: String,
}

// --- Content Blocks ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AcpContentBlock {
    Text {
        text: String,
    },
    ResourceLink {
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    Resource {
        resource: AcpResource,
    },
    Image {
        data: String,
        mime_type: String,
    },
    Audio {
        data: String,
        mime_type: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpResource {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

// --- Session Update Notifications ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionUpdateParams {
    pub session_id: String,
    pub update: AcpSessionUpdate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
pub enum AcpSessionUpdate {
    AgentMessageChunk {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message_id: Option<String>,
        content: AcpTextContentPart,
    },
    UserMessageChunk {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message_id: Option<String>,
        content: AcpTextContentPart,
    },
    Plan {
        entries: Vec<AcpPlanEntry>,
    },
    ToolCall {
        tool_call_id: String,
        title: String,
        kind: AcpToolKind,
        status: AcpToolCallStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw_input: Option<Value>,
    },
    ToolCallUpdate {
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<AcpToolCallStatus>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Vec<AcpToolCallContent>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        locations: Option<Vec<AcpToolLocation>>,
    },
    UsageUpdate {
        used: u64,
        size: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost: Option<AcpCost>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AcpTextContentPart {
    Text { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpPlanEntry {
    pub content: String,
    pub priority: String,
    pub status: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcpToolKind {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Think,
    Fetch,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcpToolCallStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AcpToolCallContent {
    Content {
        content: AcpContentBlock,
    },
    Diff {
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        old_text: Option<String>,
        new_text: String,
    },
    Terminal {
        terminal_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpToolLocation {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcpCost {
    pub amount: f64,
    pub currency: String,
}

// --- Request Permission ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionParams {
    pub session_id: String,
    pub tool_call: AcpToolCallHeader,
    pub options: Vec<AcpPermissionOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpToolCallHeader {
    pub tool_call_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpPermissionOption {
    pub option_id: String,
    pub name: String,
    pub kind: AcpPermissionOptionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcpPermissionOptionKind {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionResult {
    pub outcome: AcpPermissionOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum AcpPermissionOutcome {
    Selected { option_id: String },
    Cancelled,
}
