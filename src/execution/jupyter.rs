//! Jupyter Server kernel execution (Phase 3): REST kernel lifecycle + default binary WebSocket
//! multiplexing channel (`/api/kernels/{id}/channels`).
//!
//! Phase 6: `display_data` / `execute_result` may materialize binary MIME payloads (PNG, JPEG,
//! large CSV/JSON) under the sandbox [`.execution_artifacts`](super::artifacts::ARTIFACT_ROOT_DIR).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tokio_util::sync::CancellationToken;

use super::artifacts::{
    decode_jupyter_base64_data, jupyter_data_utf8_string, materialize_run_artifacts,
    sanitize_session_segment, ArtifactCollector, ArtifactLimits, LARGE_TEXT_SPILL_THRESHOLD,
};
use super::capabilities::{
    NetworkPolicy, ProviderCapabilities, ProviderCapabilitiesSnapshot, SessionCapabilities,
};
use super::error::ExecutionError;
use super::ids::SessionId;
use super::jupyter_notebook;
use super::provider::ExecutionProvider;
use super::run::{CwdPolicy, RunResult, RunSpec, SessionCreateRequest, SessionHandle};
use super::run_events::{RunEvent, RunEventThrottle};

type ExecWsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// Build-time config for [`JupyterExecutionProvider`] (assembled from `AppConfig` in the harness).
#[derive(Debug, Clone)]
pub struct JupyterExecutionProviderConfig {
    pub base_url: String,
    pub token: Option<String>,
    pub default_kernel_name: String,
    pub max_run_timeout_secs: u64,
    pub max_output_bytes: usize,
    pub max_sessions: usize,
    /// Agent sandbox root; artifacts are written under [`super::artifacts::ARTIFACT_ROOT_DIR`].
    pub artifact_sandbox_dir: PathBuf,
    pub artifact_limits: ArtifactLimits,
    /// When set (e.g. `isanagent/{session_id}.ipynb`), append each successful run as a code cell via Contents API.
    pub notebook_sync_path_template: Option<String>,
}

/// Jupyter HTTP + kernel WebSocket backend.
pub struct JupyterExecutionProvider {
    config: JupyterExecutionProviderConfig,
    client: reqwest::Client,
    caps: ProviderCapabilities,
    sessions: DashMap<SessionId, Arc<JupyterSession>>,
}

/// Per-run state for Phase 6 artifact materialization (WS loop → disk).
struct JupyterRunArtifactSink {
    sandbox_dir: PathBuf,
    session_id: SessionId,
    run_id: String,
    limits: ArtifactLimits,
    collector: ArtifactCollector,
}

struct JupyterSession {
    kernel_id: String,
    ws: Mutex<Option<ExecWsStream>>,
    run_cancel: Mutex<Option<CancellationToken>>,
    /// Resolved server-side `.ipynb` path for optional Contents sync (from template at session create).
    notebook_server_path: Option<String>,
}

impl JupyterExecutionProvider {
    pub fn new(config: JupyterExecutionProviderConfig) -> Result<Self, ExecutionError> {
        let base = config.base_url.trim().trim_end_matches('/').to_string();
        if base.is_empty() {
            return Err(ExecutionError::InvalidArgument(
                "jupyter base_url is empty".into(),
            ));
        }
        if !base.starts_with("http://") && !base.starts_with("https://") {
            return Err(ExecutionError::InvalidArgument(format!(
                "jupyter base_url must start with http:// or https:// (got {base:?})"
            )));
        }

        let mut caps = ProviderCapabilities::minimal("jupyter");
        caps.languages = vec!["python".into(), "r".into()];
        caps.supports_persistent_sessions = true;
        caps.supports_interrupt = true;
        caps.supports_package_install = false;
        caps.supports_remote_shell = false;
        caps.jupyter_kernel = true;
        caps.network_policy = NetworkPolicy::Full;
        caps.max_output_bytes_default = Some(config.max_output_bytes as u64);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| ExecutionError::Provider(e.to_string()))?;

        Ok(Self {
            config: JupyterExecutionProviderConfig {
                base_url: base,
                ..config
            },
            client,
            caps,
            sessions: DashMap::new(),
        })
    }

    fn rest_url(&self, path: &str) -> String {
        format!("{}{}", self.config.base_url, path)
    }

    fn apply_auth(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref t) = self.config.token {
            req = req.header("Authorization", format!("token {t}"));
        }
        req
    }

    async fn kernel_exists(&self, kernel_id: &str) -> Result<bool, ExecutionError> {
        let url = self.rest_url(&format!("/api/kernels/{kernel_id}"));
        let resp = self
            .apply_auth(self.client.get(url))
            .send()
            .await
            .map_err(|e| ExecutionError::Provider(format!("jupyter GET kernel: {e}")))?;
        Ok(resp.status().is_success())
    }

    fn ws_channels_url(&self, kernel_id: &str) -> Result<String, ExecutionError> {
        let mut base = self.config.base_url.clone();
        if base.starts_with("https://") {
            base = base.replacen("https://", "wss://", 1);
        } else if base.starts_with("http://") {
            base = base.replacen("http://", "ws://", 1);
        } else {
            return Err(ExecutionError::Provider("internal: base_url scheme".into()));
        }
        let mut url = format!("{base}/api/kernels/{kernel_id}/channels");
        if let Some(ref t) = self.config.token {
            url.push_str(&format!("?token={}", urlencoding::encode(t)));
        }
        Ok(url)
    }

    fn pick_kernel_name(&self, req: &SessionCreateRequest) -> Result<String, ExecutionError> {
        let lang = req
            .language
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        match lang {
            None | Some("python") | Some("py") => Ok(self.config.default_kernel_name.clone()),
            Some("r") | Some("R") => Ok("ir".into()),
            Some(other) => Err(ExecutionError::InvalidArgument(format!(
                "unsupported language for jupyter provider: {other} (supported: python, r)"
            ))),
        }
    }

    fn session_caps(
        &self,
        id: &SessionId,
        kernel_name: &str,
        kernel_id: &str,
        notebook_server_path: Option<&str>,
    ) -> SessionCapabilities {
        let active_language = if kernel_name == "ir" {
            Some("r".into())
        } else {
            Some("python".into())
        };
        let mut ext = BTreeMap::new();
        ext.insert(
            "jupyter_kernel_id".into(),
            Value::String(kernel_id.to_string()),
        );
        if let Some(p) = notebook_server_path {
            ext.insert(
                "jupyter_notebook_sync_path".into(),
                Value::String(p.to_string()),
            );
        }
        SessionCapabilities {
            session_id: id.clone(),
            schema_version: 1,
            provider_id: "jupyter".into(),
            active_language,
            gpu_visible: None,
            working_directory_display: Some(format!(
                "jupyter @ {} (kernel {})",
                self.config.base_url, kernel_id
            )),
            provider_snapshot: ProviderCapabilitiesSnapshot {
                supports_interrupt: self.caps.supports_interrupt,
                supports_package_install: self.caps.supports_package_install,
                supports_remote_shell: self.caps.supports_remote_shell,
                jupyter_kernel: self.caps.jupyter_kernel,
                network_policy: self.caps.network_policy,
            },
            extensions: ext,
        }
    }

    async fn post_kernel(&self, kernel_name: &str) -> Result<String, ExecutionError> {
        let url = self.rest_url("/api/kernels");
        let body = json!({ "name": kernel_name });
        let resp = self
            .apply_auth(self.client.post(url).json(&body))
            .send()
            .await
            .map_err(|e| ExecutionError::Provider(format!("jupyter POST /api/kernels: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ExecutionError::Provider(format!(
                "jupyter POST /api/kernels failed: {status} {text}"
            )));
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| ExecutionError::Provider(format!("jupyter kernel JSON: {e}")))?;
        let id = v
            .get("id")
            .and_then(|x| x.as_str())
            .ok_or_else(|| ExecutionError::Provider("jupyter kernel response missing id".into()))?;
        Ok(id.to_string())
    }

    async fn delete_kernel(&self, kernel_id: &str) -> Result<(), ExecutionError> {
        let url = self.rest_url(&format!("/api/kernels/{kernel_id}"));
        let resp = self
            .apply_auth(self.client.delete(url))
            .send()
            .await
            .map_err(|e| ExecutionError::Provider(format!("jupyter DELETE kernel: {e}")))?;
        if !resp.status().is_success() && resp.status() != reqwest::StatusCode::NOT_FOUND {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ExecutionError::Provider(format!(
                "jupyter DELETE kernel failed: {status} {text}"
            )));
        }
        Ok(())
    }

    async fn interrupt_kernel(&self, kernel_id: &str) -> Result<(), ExecutionError> {
        let url = self.rest_url(&format!("/api/kernels/{kernel_id}/interrupt"));
        let resp = self
            .apply_auth(self.client.post(url).json(&json!({})))
            .send()
            .await
            .map_err(|e| ExecutionError::Provider(format!("jupyter interrupt: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ExecutionError::Provider(format!(
                "jupyter interrupt failed: {status} {text}"
            )));
        }
        Ok(())
    }

    async fn ensure_ws(&self, session: &JupyterSession) -> Result<(), ExecutionError> {
        let mut slot = session.ws.lock().await;
        if slot.is_some() {
            return Ok(());
        }
        let url = self.ws_channels_url(&session.kernel_id)?;
        let ws = connect_kernel_channels_ws(&url).await?;
        *slot = Some(ws);
        Ok(())
    }
}

/// Preferred Jupyter Server kernel WebSocket subprotocol (binary v1 layout); matches Lab/Notebook 3.x.
const JUPYTER_KERNEL_WS_SUBPROTOCOL: &str = "v1.kernel.websocket.jupyter.org";

async fn connect_kernel_channels_ws(url: &str) -> Result<ExecWsStream, ExecutionError> {
    let req = Request::builder()
        .uri(url)
        .header("Sec-WebSocket-Protocol", JUPYTER_KERNEL_WS_SUBPROTOCOL)
        .body(())
        .map_err(|e| ExecutionError::Provider(format!("jupyter websocket request build: {e}")))?;

    match connect_async(req).await {
        Ok((ws, resp)) => {
            if log::log_enabled!(log::Level::Debug) {
                if let Some(negotiated) = resp.headers().get("Sec-WebSocket-Protocol") {
                    log::debug!(
                        "jupyter kernel ws negotiated subprotocol: {}",
                        negotiated.to_str().unwrap_or("non-utf8")
                    );
                } else {
                    log::debug!("jupyter kernel ws: server did not echo Sec-WebSocket-Protocol");
                }
            }
            Ok(ws)
        }
        Err(e1) => {
            log::debug!(
                "jupyter kernel ws connect with {} failed: {}; retrying without subprotocol",
                JUPYTER_KERNEL_WS_SUBPROTOCOL,
                e1
            );
            let (ws, _) = connect_async(url).await.map_err(|e2| {
                ExecutionError::Provider(format!(
                    "jupyter websocket connect: {e2} (after subprotocol attempt: {e1})"
                ))
            })?;
            Ok(ws)
        }
    }
}

#[async_trait]
impl ExecutionProvider for JupyterExecutionProvider {
    fn provider_id(&self) -> &str {
        "jupyter"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.caps.clone()
    }

    fn session_journal_extensions(
        &self,
        session_id: &SessionId,
    ) -> Option<BTreeMap<String, Value>> {
        self.sessions.get(session_id).map(|sess| {
            let s = sess.value();
            let mut ext = BTreeMap::new();
            ext.insert(
                "jupyter_kernel_id".into(),
                Value::String(s.kernel_id.clone()),
            );
            if let Some(ref p) = s.notebook_server_path {
                ext.insert(
                    "jupyter_notebook_sync_path".into(),
                    Value::String(p.clone()),
                );
            }
            ext
        })
    }

    async fn create_session(
        &self,
        req: SessionCreateRequest,
    ) -> Result<SessionHandle, ExecutionError> {
        if self.config.max_sessions > 0 && self.sessions.len() >= self.config.max_sessions {
            return Err(ExecutionError::limit_exceeded(
                "sessions",
                format!("max_sessions={} reached", self.config.max_sessions),
            ));
        }
        let kernel_name = self.pick_kernel_name(&req)?;
        let id = SessionId::new(uuid::Uuid::new_v4().to_string());
        let kernel_id = if let Some(ref kid) = req.resume_jupyter_kernel_id {
            let kid = kid.trim();
            if kid.is_empty() {
                return Err(ExecutionError::InvalidArgument(
                    "resume_jupyter_kernel_id must not be empty".into(),
                ));
            }
            if !self.kernel_exists(kid).await? {
                return Err(ExecutionError::InvalidArgument(format!(
                    "resume_jupyter_kernel_id not found on Jupyter server: {kid}"
                )));
            }
            kid.to_string()
        } else {
            self.post_kernel(&kernel_name).await?
        };
        let notebook_server_path = self
            .config
            .notebook_sync_path_template
            .as_deref()
            .map(|t| t.replace("{session_id}", &sanitize_session_segment(&id)));
        let caps = self.session_caps(
            &id,
            &kernel_name,
            &kernel_id,
            notebook_server_path.as_deref(),
        );
        let session = Arc::new(JupyterSession {
            kernel_id,
            ws: Mutex::new(None),
            run_cancel: Mutex::new(None),
            notebook_server_path,
        });
        self.sessions.insert(id.clone(), session);
        Ok(SessionHandle {
            id,
            capabilities: caps,
        })
    }

    async fn close_session(&self, session_id: &SessionId) -> Result<(), ExecutionError> {
        if let Some((_, sess)) = self.sessions.remove(session_id) {
            if let Some(t) = sess.run_cancel.lock().await.take() {
                t.cancel();
            }
            *sess.ws.lock().await = None;
            let _ = self.delete_kernel(&sess.kernel_id).await;
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
                "run",
                "jupyter provider only supports cwd_mode session_default (no per-run sandbox cwd); use %cd in code if needed",
            ));
        }

        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| ExecutionError::InvalidSession(session_id.to_string()))?;
        let session = session.value().clone();

        let cancel = CancellationToken::new();
        {
            let mut slot = session.run_cancel.lock().await;
            if slot.is_some() {
                return Err(ExecutionError::unsupported(
                    "run",
                    "session already has an active run",
                ));
            }
            *slot = Some(cancel.clone());
        }

        let timeout_secs = spec
            .timeout_secs
            .min(self.config.max_run_timeout_secs)
            .max(1);

        self.ensure_ws(&session).await?;

        let notebook_code = spec.code.clone();
        let nb_path = session.notebook_server_path.clone();
        let auth_token = self.config.token.clone();

        let result = tokio::time::timeout(Duration::from_secs(timeout_secs), async {
            let exec_msg_id = uuid::Uuid::new_v4().to_string();
            let session_key = uuid::Uuid::new_v4().to_string();
            let run_id = spec
                .run_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let msg = build_execute_request(&exec_msg_id, &session_key, &spec.code);
            let payload = encode_kernel_ws_frame(&msg)?;

            let throttle = spec
                .run_event_tx
                .as_ref()
                .map(|_| Arc::new(RunEventThrottle::new(Duration::from_millis(40))));

            let mut ws_guard = session.ws.lock().await;
            let ws = ws_guard
                .as_mut()
                .ok_or_else(|| ExecutionError::Provider("jupyter ws not connected".into()))?;

            ws.send(Message::Binary(payload.into()))
                .await
                .map_err(|e| ExecutionError::Provider(format!("jupyter ws send: {e}")))?;

            let io = JupyterWsIoContext {
                client: &self.client,
                base_http: &self.config.base_url,
                token: self.config.token.as_deref(),
                kernel_id: &session.kernel_id,
            };
            let artifact_sink = JupyterRunArtifactSink {
                sandbox_dir: self.config.artifact_sandbox_dir.clone(),
                session_id: session_id.clone(),
                run_id,
                limits: self.config.artifact_limits,
                collector: ArtifactCollector::new(self.config.artifact_limits),
            };
            collect_execute_output(
                ws,
                &exec_msg_id,
                self.config.max_output_bytes,
                cancel,
                io,
                artifact_sink,
                spec.run_event_tx.clone(),
                throttle,
            )
            .await
        })
        .await;

        *session.run_cancel.lock().await = None;

        let out = match result {
            Err(_) => Err(ExecutionError::Timeout { timeout_secs }),
            Ok(inner) => inner,
        };

        if let (Ok(_), Some(path)) = (&out, nb_path.as_ref()) {
            let token = auth_token.clone();
            let r = jupyter_notebook::append_code_cell(
                &self.client,
                &self.config.base_url,
                self.config.token.as_deref(),
                path,
                &notebook_code,
                |mut b: reqwest::RequestBuilder| {
                    if let Some(ref t) = token {
                        b = b.header("Authorization", format!("token {t}"));
                    }
                    b
                },
            )
            .await;
            if let Err(e) = r {
                log::warn!("jupyter notebook sync {}: {}", path, e);
            }
        }

        out
    }

    async fn cancel(&self, session_id: &SessionId) -> Result<(), ExecutionError> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| ExecutionError::InvalidSession(session_id.to_string()))?;
        let session = session.value().clone();
        if let Some(t) = session.run_cancel.lock().await.clone() {
            t.cancel();
        }
        self.interrupt_kernel(&session.kernel_id).await?;
        Ok(())
    }
}

fn build_execute_request(msg_id: &str, session: &str, code: &str) -> Value {
    let header = json!({
        "msg_id": msg_id,
        "session": session,
        "username": "isanagent",
        "version": "5.3",
        "date": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        "msg_type": "execute_request"
    });
    let content = json!({
        "code": code,
        "silent": false,
        "store_history": true,
        "user_expressions": serde_json::Map::new(),
        "allow_stdin": false,
        "stop_on_error": true
    });
    json!({
        "channel": "shell",
        "header": header,
        "parent_header": json!({}),
        "metadata": json!({}),
        "content": content
    })
}

/// Jupyter Server legacy binary WebSocket serialization (matches `jupyter_server` /
/// `deserialize_binary_message`): **big-endian** `u32` header (`nbufs`, then `nbufs` offsets),
/// then UTF-8 JSON (first segment) and optional raw buffers.
pub(crate) fn encode_kernel_ws_frame(msg: &Value) -> Result<Vec<u8>, ExecutionError> {
    let json_str = serde_json::to_string(msg)?;
    // One JSON “buffer”: `nbufs == 1`, single offset points past 8-byte header (4 + 4*nbufs).
    let nbufs: u32 = 1;
    let offset0: u32 = 4 * (nbufs + 1);
    let mut out = Vec::with_capacity(offset0 as usize + json_str.len());
    out.extend_from_slice(&nbufs.to_be_bytes());
    out.extend_from_slice(&offset0.to_be_bytes());
    out.extend_from_slice(json_str.as_bytes());
    Ok(out)
}

/// Decode a **server → client** WebSocket binary payload. Jupyter Server uses
/// `serialize_msg_to_ws_v1` when the `v1.kernel.websocket.jupyter.org` subprotocol is active;
/// otherwise it often sends JSON as **text** frames (handled separately). Legacy
/// `deserialize_binary_message` blobs still appear when ZMQ messages carry buffers.
pub(crate) fn decode_incoming_ws_bytes(bytes: &[u8]) -> Result<Value, ExecutionError> {
    if let Some(v) = try_decode_ws_v1(bytes) {
        return Ok(v);
    }
    decode_kernel_ws_frame(bytes)
}

/// `deserialize_msg_from_ws_v1` from `jupyter_server` (64-bit little-endian offset table).
fn try_decode_ws_v1(ws_msg: &[u8]) -> Option<Value> {
    if ws_msg.len() < 8 + 16 {
        return None;
    }
    let offset_number = u64::from_le_bytes(ws_msg.get(0..8)?.try_into().ok()?) as usize;
    // Channel + header + parent_header + metadata + content ⇒ at least 6 offsets.
    if !(6..=64).contains(&offset_number) {
        return None;
    }
    let header_end = 8 + 8 * offset_number;
    if ws_msg.len() < header_end {
        return None;
    }
    let mut offsets: Vec<usize> = Vec::with_capacity(offset_number);
    for i in 0..offset_number {
        let start = 8 * (i + 1);
        let end = 8 * (i + 2);
        let o = u64::from_le_bytes(ws_msg.get(start..end)?.try_into().ok()?) as usize;
        if o > ws_msg.len() {
            return None;
        }
        offsets.push(o);
    }
    for w in offsets.windows(2) {
        if w[1] < w[0] {
            return None;
        }
    }
    let channel = std::str::from_utf8(ws_msg.get(offsets[0]..offsets[1])?).ok()?;
    let header: Value = serde_json::from_slice(ws_msg.get(offsets[1]..offsets[2])?).ok()?;
    let parent_header: Value = serde_json::from_slice(ws_msg.get(offsets[2]..offsets[3])?).ok()?;
    let metadata: Value = serde_json::from_slice(ws_msg.get(offsets[3]..offsets[4])?).ok()?;
    let content: Value = serde_json::from_slice(ws_msg.get(offsets[4]..offsets[5])?).ok()?;
    Some(json!({
        "channel": channel,
        "header": header,
        "parent_header": parent_header,
        "metadata": metadata,
        "content": content,
    }))
}

pub(crate) fn decode_kernel_ws_frame(bytes: &[u8]) -> Result<Value, ExecutionError> {
    if bytes.len() < 8 {
        return Err(ExecutionError::Provider(format!(
            "jupyter ws frame too short: {} bytes",
            bytes.len()
        )));
    }
    // Server uses `struct.unpack("!i", ...)` — signed count; positive values match `u32` BE.
    let nbufs = i32::from_be_bytes(bytes[0..4].try_into().unwrap());
    if nbufs < 1 {
        return Err(ExecutionError::Provider(format!(
            "jupyter ws invalid nbufs: {nbufs}"
        )));
    }
    let n = nbufs as usize;
    let header_len = 4 + 4 * n;
    if bytes.len() < header_len {
        return Err(ExecutionError::Provider(
            "jupyter ws frame header truncated".into(),
        ));
    }
    let mut offsets = Vec::with_capacity(n);
    for i in 0..n {
        let o = u32::from_be_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        offsets.push(o);
    }
    let msg_start = offsets[0];
    let msg_end = if n >= 2 { offsets[1] } else { bytes.len() };
    if msg_start > bytes.len() || msg_end > bytes.len() || msg_end < msg_start {
        return Err(ExecutionError::Provider(
            "jupyter ws frame offsets out of range".into(),
        ));
    }
    let slice = &bytes[msg_start..msg_end];
    let s = std::str::from_utf8(slice)
        .map_err(|e| ExecutionError::Provider(format!("jupyter ws frame utf-8: {e}")))?;
    serde_json::from_str(s).map_err(Into::into)
}

/// Shared HTTP context for cancel → interrupt during a WebSocket read loop.
struct JupyterWsIoContext<'a> {
    client: &'a reqwest::Client,
    base_http: &'a str,
    token: Option<&'a str>,
    kernel_id: &'a str,
}

fn stream_text(content: &Value) -> String {
    match content.get("text") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn text_plain_from_data(data: &Value) -> Option<String> {
    data.get("text/plain").map(|tp| match tp {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

fn append_truncated(buf: &mut String, chunk: &str, budget: &mut usize) {
    if *budget == 0 {
        return;
    }
    let take = (*budget).min(chunk.len());
    if take < chunk.len() {
        buf.push_str(&chunk[..take]);
        buf.push_str("\n... (truncated)");
        *budget = 0;
    } else {
        buf.push_str(chunk);
        *budget -= take;
    }
}

/// Mutable state while folding Jupyter execute WebSocket messages for one `execute_request`.
struct ExecuteFoldCtx<'a> {
    stdout: &'a mut String,
    stderr: &'a mut String,
    budget: &'a mut usize,
    exit_code: &'a mut Option<i32>,
    got_execute_reply: &'a mut bool,
    got_iopub_idle: &'a mut bool,
    artifact_limits: ArtifactLimits,
    artifact_collector: Option<&'a mut ArtifactCollector>,
}

fn fold_display_data_payload(v: &Value, ctx: &mut ExecuteFoldCtx<'_>) {
    let Some(data) = v["content"].get("data") else {
        return;
    };

    if let Some(collector) = ctx.artifact_collector.as_mut() {
        let lim = ctx.artifact_limits.max_file_bytes;
        for (mime, ext) in [
            ("image/png", ".png"),
            ("image/jpeg", ".jpg"),
            ("image/jpg", ".jpg"),
        ] {
            if let Some(bytes) = decode_jupyter_base64_data(data, mime, lim) {
                let _ = collector.try_push(mime.to_string(), bytes, ext);
            }
        }
        for (mime, ext) in [("text/csv", ".csv"), ("application/json", ".json")] {
            if let Some(s) = jupyter_data_utf8_string(data, mime) {
                if s.len() > LARGE_TEXT_SPILL_THRESHOLD {
                    let bytes = s.into_bytes();
                    let _ = collector.try_push(mime.to_string(), bytes, ext);
                } else if !s.is_empty() {
                    append_truncated(ctx.stdout, &s, ctx.budget);
                }
            }
        }
    }

    if let Some(s) = text_plain_from_data(data) {
        append_truncated(ctx.stdout, &s, ctx.budget);
    }
}

/// Apply one Jupyter message (decoded JSON envelope) to stdout/stderr / completion state.
fn fold_execute_ws_message(v: &Value, exec_msg_id: &str, ctx: &mut ExecuteFoldCtx<'_>) {
    let msg_type = v["header"]["msg_type"].as_str().unwrap_or("");
    let parent_id = v["parent_header"]["msg_id"].as_str();
    let matches_parent = parent_id == Some(exec_msg_id);
    if !matches_parent {
        return;
    }
    match msg_type {
        "stream" => {
            let name = v["content"]["name"].as_str().unwrap_or("stdout");
            let text = stream_text(&v["content"]);
            if name == "stderr" {
                append_truncated(ctx.stderr, &text, ctx.budget);
            } else {
                append_truncated(ctx.stdout, &text, ctx.budget);
            }
        }
        "execute_result" | "display_data" | "update_display_data" => {
            fold_display_data_payload(v, ctx);
        }
        "error" => {
            let ename = v["content"]["ename"].as_str().unwrap_or("Error");
            let evalue = v["content"]["evalue"].as_str().unwrap_or("");
            let trace = v["content"]["traceback"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();
            let block = format!("{ename}: {evalue}\n{trace}\n");
            append_truncated(ctx.stderr, &block, ctx.budget);
            *ctx.exit_code = Some(1);
        }
        "execute_reply" => {
            // Shell reply: sync status / exit only. Traceback belongs on iopub `error` to avoid duplicates.
            let status = v["content"]["status"].as_str().unwrap_or("");
            match status {
                "error" if ctx.exit_code.is_none() || *ctx.exit_code == Some(0) => {
                    *ctx.exit_code = Some(1);
                }
                "abort" => *ctx.exit_code = Some(130),
                "ok" if ctx.exit_code.is_none() || *ctx.exit_code == Some(0) => {
                    *ctx.exit_code = Some(0);
                }
                _ => {}
            }
            *ctx.got_execute_reply = true;
        }
        "status" => {
            let parent_id = v["parent_header"]["msg_id"].as_str();
            if parent_id == Some(exec_msg_id) {
                let state = v["content"]["execution_state"].as_str().unwrap_or("");
                if state == "idle" {
                    *ctx.got_iopub_idle = true;
                }
            }
        }
        _ => {}
    }
}

fn try_emit_run_event(tx: &Option<tokio::sync::mpsc::Sender<RunEvent>>, ev: RunEvent) {
    if let Some(t) = tx {
        let _ = t.try_send(ev);
    }
}

fn emit_jupyter_ws_run_events(
    v: &Value,
    exec_msg_id: &str,
    tx: &Option<tokio::sync::mpsc::Sender<RunEvent>>,
    throttle: &Option<Arc<RunEventThrottle>>,
) {
    let parent_id = v["parent_header"]["msg_id"].as_str();
    if parent_id != Some(exec_msg_id) {
        return;
    }
    let msg_type = v["header"]["msg_type"].as_str().unwrap_or("");
    match msg_type {
        "stream" => {
            if throttle.as_ref().is_some_and(|th| !th.should_emit()) {
                return;
            }
            let name = v["content"]["name"].as_str().unwrap_or("stdout");
            let text = stream_text(&v["content"]);
            if text.is_empty() {
                return;
            }
            let chunk: String = text.chars().take(2000).collect();
            let chunk = if text.chars().nth(2000).is_some() {
                format!("{chunk}…")
            } else {
                chunk
            };
            if name == "stderr" {
                try_emit_run_event(tx, RunEvent::StderrChunk { text: chunk });
            } else {
                try_emit_run_event(tx, RunEvent::StdoutChunk { text: chunk });
            }
        }
        "status" => {
            let state = v["content"]["execution_state"].as_str().unwrap_or("");
            if state == "busy" {
                try_emit_run_event(tx, RunEvent::KernelBusy);
            }
            if state == "idle" {
                try_emit_run_event(tx, RunEvent::KernelIdle);
            }
        }
        "display_data" | "execute_result" => {
            if throttle.as_ref().is_some_and(|th| !th.should_emit()) {
                return;
            }
            if let Some(data) = v["content"].get("data").and_then(|d| d.as_object()) {
                let mime = data
                    .keys()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| "?".to_string());
                try_emit_run_event(tx, RunEvent::DisplayDataSummary { mime });
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
async fn collect_execute_output(
    ws: &mut ExecWsStream,
    exec_msg_id: &str,
    max_output_bytes: usize,
    cancel: CancellationToken,
    io: JupyterWsIoContext<'_>,
    artifact_sink: JupyterRunArtifactSink,
    event_tx: Option<tokio::sync::mpsc::Sender<RunEvent>>,
    throttle: Option<Arc<RunEventThrottle>>,
) -> Result<RunResult, ExecutionError> {
    let mut stdout = String::new();
    let mut stderr = String::new();
    let mut budget = max_output_bytes;
    let mut exit_code = Some(0i32);
    let mut got_execute_reply = false;
    let mut got_iopub_idle = false;
    let artifact_limits = artifact_sink.limits;
    let mut artifact_sink = artifact_sink;

    loop {
        if got_execute_reply && got_iopub_idle {
            break;
        }
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                let _ = interrupt_kernel_rest(io.client, io.base_http, io.token, io.kernel_id).await;
                return Err(ExecutionError::Cancelled);
            }
            next = ws.next() => {
                let frame = match next {
                    None => {
                        if got_execute_reply {
                            // Connection closed before idle; return best-effort output.
                            break;
                        }
                        return Err(ExecutionError::Provider(
                            "jupyter websocket closed before execute_reply".into(),
                        ));
                    }
                    Some(Err(e)) => {
                        return Err(ExecutionError::Provider(format!("jupyter ws read: {e}")));
                    }
                    Some(Ok(m)) => m,
                };
                match frame {
                    Message::Binary(b) => {
                        let v = decode_incoming_ws_bytes(&b)?;
                        emit_jupyter_ws_run_events(&v, exec_msg_id, &event_tx, &throttle);
                        let mut ctx = ExecuteFoldCtx {
                            stdout: &mut stdout,
                            stderr: &mut stderr,
                            budget: &mut budget,
                            exit_code: &mut exit_code,
                            got_execute_reply: &mut got_execute_reply,
                            got_iopub_idle: &mut got_iopub_idle,
                            artifact_limits,
                            artifact_collector: Some(&mut artifact_sink.collector),
                        };
                        fold_execute_ws_message(&v, exec_msg_id, &mut ctx);
                    }
                    Message::Ping(p) => {
                        ws.send(Message::Pong(p)).await.map_err(|e| {
                            ExecutionError::Provider(format!("jupyter ws pong: {e}"))
                        })?;
                    }
                    Message::Close(f) => {
                        if got_execute_reply {
                            break;
                        }
                        return Err(ExecutionError::Provider(format!(
                            "jupyter websocket closed: {f:?}"
                        )));
                    }
                    Message::Pong(_) | Message::Frame(_) => {}
                    Message::Text(t) => {
                        if let Ok(v) = serde_json::from_str::<Value>(&t) {
                            emit_jupyter_ws_run_events(&v, exec_msg_id, &event_tx, &throttle);
                            let mut ctx = ExecuteFoldCtx {
                                stdout: &mut stdout,
                                stderr: &mut stderr,
                                budget: &mut budget,
                                exit_code: &mut exit_code,
                                got_execute_reply: &mut got_execute_reply,
                                got_iopub_idle: &mut got_iopub_idle,
                                artifact_limits,
                                artifact_collector: Some(&mut artifact_sink.collector),
                            };
                            fold_execute_ws_message(&v, exec_msg_id, &mut ctx);
                        }
                    }
                }
            }
        }
    }

    let sandbox_dir = artifact_sink.sandbox_dir;
    let session_id = artifact_sink.session_id;
    let run_id = artifact_sink.run_id;
    let pending = artifact_sink.collector.pending;
    let attachments =
        materialize_run_artifacts(&sandbox_dir, &session_id, &run_id, pending).await?;

    let mut summary_budget = max_output_bytes.saturating_sub(stdout.len() + stderr.len());
    if summary_budget > 0 {
        for a in &attachments {
            if let Some(p) = &a.path {
                let m = a.mime_hint.as_deref().unwrap_or("?");
                let line = format!("[execution artifact] {p} ({m})\n");
                append_truncated(&mut stdout, &line, &mut summary_budget);
            }
        }
    }

    try_emit_run_event(&event_tx, RunEvent::RunFinished);

    let mut result = RunResult::new(stdout, stderr, exit_code);
    result.attachments = attachments;
    Ok(result)
}

async fn interrupt_kernel_rest(
    client: &reqwest::Client,
    base_http: &str,
    token: Option<&str>,
    kernel_id: &str,
) -> Result<(), ExecutionError> {
    let base = base_http.trim().trim_end_matches('/');
    let url = format!("{base}/api/kernels/{kernel_id}/interrupt");
    let mut req = client.post(url).json(&json!({}));
    if let Some(t) = token {
        req = req.header("Authorization", format!("token {t}"));
    }
    let _ = req.send().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let msg = json!({"channel":"shell","header":{"msg_type":"execute_request"},"parent_header":{},"metadata":{},"content":{}});
        let bytes = encode_kernel_ws_frame(&msg).unwrap();
        let back = decode_kernel_ws_frame(&bytes).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn decode_two_segment_frame() {
        let json = r#"{"channel":"iopub","header":{"msg_type":"stream"}}"#;
        let off0: u32 = 12; // 4 + 4 * nbufs, nbufs == 2
        let off1 = off0 + json.len() as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_be_bytes());
        bytes.extend_from_slice(&off0.to_be_bytes());
        bytes.extend_from_slice(&off1.to_be_bytes());
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(b"buf2"); // second wire segment (ignored by our JSON parser)
        let v = decode_kernel_ws_frame(&bytes).unwrap();
        assert_eq!(v["header"]["msg_type"], "stream");
    }

    /// Matches `jupyter_server.serialize_msg_to_ws_v1` layout (tests only).
    fn encode_ws_v1_fixture(channel: &str, parts: [&str; 4]) -> Vec<u8> {
        let msg_list: Vec<Vec<u8>> = parts.iter().map(|s| s.as_bytes().to_vec()).collect();
        let n = msg_list.len();
        let mut offsets: Vec<u64> = Vec::new();
        offsets.push(((n + 3) * 8) as u64);
        offsets.push(offsets[0] + channel.len() as u64);
        for m in &msg_list {
            offsets.push(offsets.last().unwrap() + m.len() as u64);
        }
        let offset_number = offsets.len() as u64;
        let mut out = Vec::new();
        out.extend_from_slice(&offset_number.to_le_bytes());
        for o in &offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out.extend_from_slice(channel.as_bytes());
        for m in &msg_list {
            out.extend_from_slice(m);
        }
        out
    }

    #[test]
    fn decode_incoming_ws_v1_stream_stdout() {
        let pid = "parent-exec-id";
        let h = r#"{"msg_type":"stream","msg_id":"s1"}"#;
        let ph = format!(r#"{{"msg_id":"{pid}"}}"#);
        let raw = encode_ws_v1_fixture(
            "iopub",
            [h, ph.as_str(), "{}", r#"{"name":"stdout","text":"42"}"#],
        );
        let v = decode_incoming_ws_bytes(&raw).unwrap();
        assert_eq!(v["header"]["msg_type"], "stream");
        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut budget = 1000;
        let mut exit_code = Some(0i32);
        let mut got_execute_reply = false;
        let mut got_iopub_idle = false;
        let mut ctx = ExecuteFoldCtx {
            stdout: &mut stdout,
            stderr: &mut stderr,
            budget: &mut budget,
            exit_code: &mut exit_code,
            got_execute_reply: &mut got_execute_reply,
            got_iopub_idle: &mut got_iopub_idle,
            artifact_limits: ArtifactLimits::default(),
            artifact_collector: None,
        };
        fold_execute_ws_message(&v, pid, &mut ctx);
        assert_eq!(stdout, "42");
        assert!(!got_execute_reply);
        assert!(!got_iopub_idle);
    }

    #[test]
    fn text_json_envelope_folds_stream() {
        let v: Value = serde_json::from_str(
            r#"{
            "channel":"iopub",
            "header":{"msg_type":"stream"},
            "parent_header":{"msg_id":"abc"},
            "metadata":{},
            "content":{"name":"stdout","text":"hello"}
        }"#,
        )
        .unwrap();
        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut budget = 1000;
        let mut exit_code = Some(0i32);
        let mut got_execute_reply = false;
        let mut got_iopub_idle = false;
        let mut ctx = ExecuteFoldCtx {
            stdout: &mut stdout,
            stderr: &mut stderr,
            budget: &mut budget,
            exit_code: &mut exit_code,
            got_execute_reply: &mut got_execute_reply,
            got_iopub_idle: &mut got_iopub_idle,
            artifact_limits: ArtifactLimits::default(),
            artifact_collector: None,
        };
        fold_execute_ws_message(&v, "abc", &mut ctx);
        assert_eq!(stdout, "hello");
    }

    #[test]
    fn iopub_error_then_execute_reply_error_no_duplicate_stderr() {
        let pid = "exec-1";
        let err = serde_json::from_str::<Value>(&format!(
            r#"{{
            "channel":"iopub",
            "header":{{"msg_type":"error"}},
            "parent_header":{{"msg_id":"{pid}"}},
            "content":{{
                "ename":"ValueError",
                "evalue":"bad",
                "traceback":["line-A","line-B"]
            }}
        }}"#
        ))
        .unwrap();
        let reply = serde_json::from_str::<Value>(&format!(
            r#"{{
            "channel":"shell",
            "header":{{"msg_type":"execute_reply"}},
            "parent_header":{{"msg_id":"{pid}"}},
            "content":{{
                "status":"error",
                "ename":"ValueError",
                "evalue":"bad",
                "traceback":["line-A","line-B"]
            }}
        }}"#
        ))
        .unwrap();
        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut budget = 10_000;
        let mut exit_code = Some(0i32);
        let mut got_execute_reply = false;
        let mut got_iopub_idle = false;
        let mut ctx = ExecuteFoldCtx {
            stdout: &mut stdout,
            stderr: &mut stderr,
            budget: &mut budget,
            exit_code: &mut exit_code,
            got_execute_reply: &mut got_execute_reply,
            got_iopub_idle: &mut got_iopub_idle,
            artifact_limits: ArtifactLimits::default(),
            artifact_collector: None,
        };
        fold_execute_ws_message(&err, pid, &mut ctx);
        assert_eq!(exit_code, Some(1));
        let after_error = stderr.clone();
        let mut ctx = ExecuteFoldCtx {
            stdout: &mut stdout,
            stderr: &mut stderr,
            budget: &mut budget,
            exit_code: &mut exit_code,
            got_execute_reply: &mut got_execute_reply,
            got_iopub_idle: &mut got_iopub_idle,
            artifact_limits: ArtifactLimits::default(),
            artifact_collector: None,
        };
        fold_execute_ws_message(&reply, pid, &mut ctx);
        assert_eq!(
            stderr, after_error,
            "execute_reply must not append a second traceback"
        );
        assert!(got_execute_reply);
        assert!(!got_iopub_idle);
    }

    #[test]
    fn fold_status_idle_marks_iopub_complete() {
        let pid = "exec-2";
        let idle: Value = serde_json::from_str(&format!(
            r#"{{
            "channel":"iopub",
            "header":{{"msg_type":"status"}},
            "parent_header":{{"msg_id":"{pid}"}},
            "content":{{"execution_state":"idle"}}
        }}"#
        ))
        .unwrap();
        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut budget = 1000;
        let mut exit_code = Some(0i32);
        let mut got_execute_reply = false;
        let mut got_iopub_idle = false;
        let mut ctx = ExecuteFoldCtx {
            stdout: &mut stdout,
            stderr: &mut stderr,
            budget: &mut budget,
            exit_code: &mut exit_code,
            got_execute_reply: &mut got_execute_reply,
            got_iopub_idle: &mut got_iopub_idle,
            artifact_limits: ArtifactLimits::default(),
            artifact_collector: None,
        };
        fold_execute_ws_message(&idle, pid, &mut ctx);
        assert!(got_iopub_idle);
        assert!(!got_execute_reply);
    }

    #[test]
    fn display_data_png_with_collector_queues_bytes() {
        let pid = "exec-png";
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let v: Value = serde_json::from_str(&format!(
            r#"{{
            "channel":"iopub",
            "header":{{"msg_type":"display_data"}},
            "parent_header":{{"msg_id":"{pid}"}},
            "content":{{"data":{{"image/png":"{b64}"}}}}
        }}"#
        ))
        .unwrap();
        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut budget = 10_000;
        let mut exit_code = Some(0i32);
        let mut got_execute_reply = false;
        let mut got_iopub_idle = false;
        let limits = ArtifactLimits::default();
        let mut collector = ArtifactCollector::new(limits);
        let mut ctx = ExecuteFoldCtx {
            stdout: &mut stdout,
            stderr: &mut stderr,
            budget: &mut budget,
            exit_code: &mut exit_code,
            got_execute_reply: &mut got_execute_reply,
            got_iopub_idle: &mut got_iopub_idle,
            artifact_limits: limits,
            artifact_collector: Some(&mut collector),
        };
        fold_execute_ws_message(&v, pid, &mut ctx);
        assert_eq!(collector.pending.len(), 1);
        assert_eq!(collector.pending[0].mime, "image/png");
        assert!(!collector.pending[0].bytes.is_empty());
    }
}
