//! Sandbox paths and limits for execution run artifacts (Phase 6).

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::error::ExecutionError;
use super::ids::SessionId;
use super::run::RunAttachmentRef;

/// Root directory under the agent sandbox for materialized execution artifacts.
pub const ARTIFACT_ROOT_DIR: &str = ".execution_artifacts";

/// Default max bytes per artifact file.
pub const DEFAULT_MAX_FILE_BYTES: usize = 4 * 1024 * 1024;
/// Default max total bytes for all attachments in one run.
pub const DEFAULT_MAX_TOTAL_BYTES_PER_RUN: usize = 32 * 1024 * 1024;
/// Default max attachment files per run.
pub const DEFAULT_MAX_FILES_PER_RUN: usize = 64;
/// Inline text longer than this (UTF-8) may be written as `text/csv` or `application/json` files.
pub const LARGE_TEXT_SPILL_THRESHOLD: usize = 8192;

/// Caps for writing run artifacts (Jupyter `display_data`, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactLimits {
    pub max_file_bytes: usize,
    pub max_total_bytes_per_run: usize,
    pub max_files_per_run: usize,
}

impl Default for ArtifactLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_total_bytes_per_run: DEFAULT_MAX_TOTAL_BYTES_PER_RUN,
            max_files_per_run: DEFAULT_MAX_FILES_PER_RUN,
        }
    }
}

/// One decoded blob queued for disk write after the WS loop (sync collection, async flush).
#[derive(Debug)]
pub struct PendingArtifact {
    pub mime: String,
    pub bytes: Vec<u8>,
    pub ext: &'static str,
}

/// Collects pending attachments while folding Jupyter messages; enforces caps.
#[derive(Debug)]
pub struct ArtifactCollector {
    pub pending: Vec<PendingArtifact>,
    limits: ArtifactLimits,
    total_bytes: usize,
}

impl ArtifactCollector {
    pub fn new(limits: ArtifactLimits) -> Self {
        Self {
            pending: Vec::new(),
            limits,
            total_bytes: 0,
        }
    }

    /// Returns false if caps would be exceeded (caller should skip or trim).
    pub fn try_push(&mut self, mime: String, bytes: Vec<u8>, ext: &'static str) -> bool {
        if self.pending.len() >= self.limits.max_files_per_run {
            return false;
        }
        if bytes.len() > self.limits.max_file_bytes {
            return false;
        }
        if self.total_bytes.saturating_add(bytes.len()) > self.limits.max_total_bytes_per_run {
            return false;
        }
        self.total_bytes += bytes.len();
        self.pending.push(PendingArtifact { mime, bytes, ext });
        true
    }
}

/// Sanitize `session_id` for use as a single path segment (no `..`, no separators).
pub fn sanitize_session_segment(session_id: &SessionId) -> String {
    let s = session_id.as_str();
    let mut out = String::with_capacity(s.len().min(128));
    for ch in s.chars().take(128) {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "session".into()
    } else {
        out
    }
}

/// Relative directory (under sandbox) for one run: `.execution_artifacts/{session}/{run_id}`.
pub fn artifact_run_rel_dir(session_id: &SessionId, run_id: &str) -> String {
    let seg = sanitize_session_segment(session_id);
    let run_seg = sanitize_run_id_segment(run_id);
    format!("{ARTIFACT_ROOT_DIR}/{seg}/{run_seg}")
}

fn sanitize_run_id_segment(run_id: &str) -> String {
    let mut out = String::with_capacity(run_id.len().min(128));
    for ch in run_id.chars().take(128) {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "run".into()
    } else {
        out
    }
}

/// Write pending blobs to disk; returns [`RunAttachmentRef`] with paths relative to sandbox root.
pub async fn materialize_run_artifacts(
    sandbox_dir: &Path,
    session_id: &SessionId,
    run_id: &str,
    pending: Vec<PendingArtifact>,
) -> Result<Vec<RunAttachmentRef>, ExecutionError> {
    if pending.is_empty() {
        return Ok(Vec::new());
    }
    let rel_dir = artifact_run_rel_dir(session_id, run_id);
    let abs_dir = sandbox_dir.join(&rel_dir);
    tokio::fs::create_dir_all(&abs_dir)
        .await
        .map_err(|e| ExecutionError::Provider(format!("artifact mkdir: {e}")))?;

    let mut attachments = Vec::with_capacity(pending.len());
    for (i, p) in pending.into_iter().enumerate() {
        let name = format!("artifact_{i:03}{}", p.ext);
        let rel_path = format!("{rel_dir}/{name}");
        let abs_path = sandbox_dir.join(&rel_path);
        tokio::fs::write(&abs_path, &p.bytes)
            .await
            .map_err(|e| ExecutionError::Provider(format!("artifact write {rel_path}: {e}")))?;
        attachments.push(RunAttachmentRef {
            id: format!("{i:03}"),
            path: Some(rel_path),
            mime_hint: Some(p.mime),
        });
    }
    Ok(attachments)
}

/// Decode Jupyter `data` MIME payload: string or array of strings (base64), capped.
pub fn decode_jupyter_base64_data(
    data: &serde_json::Value,
    mime: &str,
    max_raw: usize,
) -> Option<Vec<u8>> {
    let v = data.get(mime)?;
    let mut pieces: Vec<&str> = Vec::new();
    match v {
        serde_json::Value::String(s) => {
            if !s.is_empty() {
                pieces.push(s.as_str());
            }
        }
        serde_json::Value::Array(a) => {
            for x in a {
                if let Some(s) = x.as_str() {
                    if !s.is_empty() {
                        pieces.push(s);
                    }
                }
            }
        }
        _ => return None,
    }
    if pieces.is_empty() {
        return None;
    }
    use base64::Engine as _;
    let engine = base64::engine::general_purpose::STANDARD;
    let mut out = Vec::new();
    for piece in pieces {
        let dec = engine.decode(piece).ok()?;
        if out.len().saturating_add(dec.len()) > max_raw {
            return None;
        }
        out.extend_from_slice(&dec);
    }
    Some(out)
}

/// UTF-8 text from `data["mime"]` when it is a string or JSON-encoded string form.
pub fn jupyter_data_utf8_string(data: &serde_json::Value, mime: &str) -> Option<String> {
    let v = data.get(mime)?;
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(a) => {
            let mut s = String::new();
            for x in a {
                if let Some(t) = x.as_str() {
                    s.push_str(t);
                } else {
                    s.push_str(&x.to_string());
                }
            }
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_session_no_path_sep() {
        let sid = SessionId::new("../../etc/passwd");
        let s = sanitize_session_segment(&sid);
        assert!(!s.contains('/'));
        assert!(!s.contains('\\'));
    }

    #[test]
    fn decode_base64_png_fixture() {
        use serde_json::json;
        // 1x1 PNG
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let data = json!({ "image/png": b64 });
        let bytes = decode_jupyter_base64_data(&data, "image/png", 1_000_000).expect("decode");
        assert!(bytes.len() >= 67);
        assert_eq!(
            &bytes[0..8],
            &[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        );
    }
}
