use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::Sha256;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

use crate::bus::TelemetryEvent;

type HmacSha256 = Hmac<Sha256>;

const DEFAULT_QUEUE: usize = 256;

#[derive(Clone, Debug)]
pub struct HookObservationMeta<'a> {
    pub channel: &'a str,
    pub chat_id: &'a str,
    pub thread_id: Option<&'a str>,
    pub is_subagent: bool,
    pub metadata: &'a HashMap<String, Value>,
}

#[derive(Clone)]
pub struct ObservationHooksHandle {
    tx: mpsc::Sender<Value>,
    metadata_keys: Arc<Vec<String>>,
}

impl ObservationHooksHandle {
    /// Non-blocking: drops when the bounded queue is full.
    pub fn try_emit(&self, telemetry: TelemetryEvent, meta: HookObservationMeta<'_>) {
        let envelope = build_envelope_with_keys(telemetry, meta, self.metadata_keys.as_slice());
        let _ = self.tx.try_send(envelope);
    }
}

#[derive(Debug, Clone)]
pub struct ObservationHooksParams {
    pub jsonl_path: Option<PathBuf>,
    pub webhook_url: Option<String>,
    pub webhook_hmac_secret: Option<String>,
    pub metadata_keys: Vec<String>,
    pub queue_capacity: usize,
}

/// Start background observation delivery. Returns `None` if nothing to do (no sinks).
pub fn start_observation_hooks(params: ObservationHooksParams) -> Option<ObservationHooksHandle> {
    if params.jsonl_path.is_none() && params.webhook_url.is_none() {
        return None;
    }
    let cap = params.queue_capacity.clamp(1, 65_536);
    let (tx, mut rx) = mpsc::channel::<Value>(cap);
    let jsonl_path = params.jsonl_path.clone();
    let webhook_url = params.webhook_url.clone();
    let hmac_secret = params.webhook_hmac_secret.clone();
    let metadata_keys = Arc::new(params.metadata_keys);
    // Reuse the process-wide static redactor instead of re-scanning the env and recompiling every
    // pattern here. Applied in the background consumer (off the agent hot path) so a secret that
    // lands in tool output never reaches the JSONL journal or the third-party webhook. Redacting
    // here does NOT affect what the executed child or the model sees.
    let redactor = crate::redact::shared();

    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .ok();

        while let Some(mut envelope) = rx.recv().await {
            redactor.redact_json(&mut envelope);
            if let Some(ref path) = jsonl_path {
                if let Err(e) = append_jsonl(path, &envelope).await {
                    log::warn!("hooks observation jsonl: {}", e);
                }
            }
            if let Some(ref url) = webhook_url {
                if let Some(ref c) = client {
                    if let Err(e) = post_webhook(c, url, &hmac_secret, &envelope).await {
                        log::warn!("hooks observation webhook: {}", e);
                    }
                }
            }
        }
    });

    Some(ObservationHooksHandle { tx, metadata_keys })
}

fn extract_hook_metadata(
    metadata: &HashMap<String, Value>,
    keys: &[String],
) -> HashMap<String, Value> {
    let mut out = HashMap::new();
    for k in keys {
        if let Some(v) = metadata.get(k) {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

fn build_envelope_with_keys(
    telemetry: TelemetryEvent,
    meta: HookObservationMeta<'_>,
    metadata_keys: &[String],
) -> Value {
    let tele_json = serde_json::to_value(&telemetry)
        .unwrap_or_else(|_| json!({"error": "telemetry_serialize"}));
    json!({
        "schema_version": 1,
        "at": Utc::now().to_rfc3339(),
        "channel": meta.channel,
        "chat_id": meta.chat_id,
        "thread_id": meta.thread_id,
        "is_subagent": meta.is_subagent,
        "hook_metadata": extract_hook_metadata(meta.metadata, metadata_keys),
        "telemetry": tele_json,
    })
}

async fn append_jsonl(path: &Path, envelope: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("create_dir_all {}: {}", parent.display(), e))?;
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let line = serde_json::to_string(envelope).map_err(|e| format!("encode: {}", e))?;
    file.write_all(line.as_bytes())
        .await
        .map_err(|e| format!("write: {}", e))?;
    file.write_all(b"\n")
        .await
        .map_err(|e| format!("write nl: {}", e))?;
    Ok(())
}

/// Milliseconds to wait after webhook attempt `attempt_index` fails (`0` = first backoff).
/// Uses exponential backoff (capped) plus jitter from wall-clock nanos (no extra RNG dependency).
fn webhook_retry_delay_ms(attempt_index: usize) -> u64 {
    const BASE_MS: u64 = 250;
    const MAX_BACKOFF_MS: u64 = 15_000;
    let shift = attempt_index.min(16) as u32;
    let exp = BASE_MS
        .saturating_mul(1u64.checked_shl(shift).unwrap_or(u64::MAX))
        .min(MAX_BACKOFF_MS);
    let jitter_span = (exp / 5).clamp(25, 5_000);
    let jitter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| {
            let mix =
                (d.subsec_nanos() as u64).wrapping_add(d.as_secs().wrapping_mul(1_000_000_037));
            mix % (jitter_span + 1)
        })
        .unwrap_or(0);
    exp.saturating_add(jitter)
}

async fn post_webhook(
    client: &reqwest::Client,
    url: &str,
    hmac_secret: &Option<String>,
    envelope: &Value,
) -> Result<(), String> {
    const MAX_ATTEMPTS: usize = 3;
    let body = serde_json::to_vec(envelope).map_err(|e| format!("encode body: {}", e))?;

    for attempt in 0..MAX_ATTEMPTS {
        let mut req = client.post(url).body(body.clone());
        if let Some(secret) = hmac_secret {
            if !secret.is_empty() {
                let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
                    .map_err(|e| format!("hmac key: {}", e))?;
                mac.update(&body);
                let sig = hex::encode(mac.finalize().into_bytes());
                req = req.header("X-Isanagent-Hook-Signature", format!("sha256={sig}"));
            }
        }
        match req.send().await {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => {
                if attempt + 1 >= MAX_ATTEMPTS {
                    return Err(format!("webhook status {}", response.status()));
                }
            }
            Err(err) => {
                if attempt + 1 >= MAX_ATTEMPTS {
                    return Err(format!("webhook request: {}", err));
                }
            }
        }
        let delay_ms = webhook_retry_delay_ms(attempt);
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }
    Err("webhook: exhausted retries".to_string())
}

/// Build params with workspace-relative jsonl path resolution.
pub fn observation_params_from_config(
    workspace_dir: &Path,
    jsonl_path: Option<&str>,
    webhook_url: Option<&str>,
    webhook_hmac_secret: Option<&str>,
    metadata_keys: Vec<String>,
    queue_capacity: Option<usize>,
) -> ObservationHooksParams {
    let jsonl_path = jsonl_path.and_then(|p| {
        let p = p.trim();
        if p.is_empty() {
            None
        } else {
            Some(workspace_dir.join(p))
        }
    });
    ObservationHooksParams {
        jsonl_path,
        webhook_url: webhook_url
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        webhook_hmac_secret: webhook_hmac_secret
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        metadata_keys,
        queue_capacity: queue_capacity.unwrap_or(DEFAULT_QUEUE),
    }
}
