//! Capability snapshots for providers and sessions (serde-stable, forward-compatible).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use super::ids::SessionId;

/// Network exposure for user code (policy hint; executor may enforce stricter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NetworkPolicy {
    #[default]
    Off,
    WorkspaceOnly,
    Full,
}

/// Static or slowly-changing capabilities for a provider implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Stable id (`local`, `jupyter`, `ssh`, …).
    pub provider_id: String,
    /// Bump when adding required fields to this struct (tooling may branch on it).
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Languages or kernels offered (free-form strings for Phase 0).
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub supports_persistent_sessions: bool,
    #[serde(default)]
    pub supports_interrupt: bool,
    #[serde(default)]
    pub supports_package_install: bool,
    #[serde(default)]
    pub supports_remote_shell: bool,
    #[serde(default)]
    pub jupyter_kernel: bool,
    #[serde(default)]
    pub network_policy: NetworkPolicy,
    /// Default cap for stdout+stderr combined (bytes); executor may override.
    #[serde(default)]
    pub max_output_bytes_default: Option<u64>,
    /// Forward-compatible key/value bag; survives round-trip through JSON.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, JsonValue>,
}

fn default_schema_version() -> u32 {
    1
}

impl ProviderCapabilities {
    pub fn minimal(provider_id: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            schema_version: 1,
            languages: Vec::new(),
            supports_persistent_sessions: false,
            supports_interrupt: false,
            supports_package_install: false,
            supports_remote_shell: false,
            jupyter_kernel: false,
            network_policy: NetworkPolicy::default(),
            max_output_bytes_default: None,
            extensions: BTreeMap::new(),
        }
    }
}

/// Per-session view (may differ from provider defaults, e.g. GPU visible).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCapabilities {
    pub session_id: SessionId,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub provider_id: String,
    #[serde(default)]
    pub active_language: Option<String>,
    /// `Some(true/false)` only when the provider actually probed GPU visibility. Omitted from
    /// serialized snapshots when unknown, so a permanently-null field is not surfaced to the
    /// model as if it were a real (negative) capability signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_visible: Option<bool>,
    #[serde(default)]
    pub working_directory_display: Option<String>,
    #[serde(flatten)]
    pub provider_snapshot: ProviderCapabilitiesSnapshot,
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, JsonValue>,
}

/// Subset of provider flags often copied into session context (avoid huge snapshots).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProviderCapabilitiesSnapshot {
    #[serde(default)]
    pub supports_interrupt: bool,
    #[serde(default)]
    pub supports_package_install: bool,
    #[serde(default)]
    pub supports_remote_shell: bool,
    #[serde(default)]
    pub jupyter_kernel: bool,
    #[serde(default)]
    pub network_policy: NetworkPolicy,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn provider_capabilities_roundtrip_with_unknown_top_level_keys() {
        let v = json!({
            "provider_id": "local",
            "schema_version": 1,
            "languages": ["python"],
            "supports_interrupt": true,
            "future_flag_from_newer_agent": true,
            "nested": { "x": 1 }
        });
        let cap: ProviderCapabilities = serde_json::from_value(v).expect("deserialize");
        assert_eq!(cap.provider_id, "local");
        assert_eq!(cap.languages, vec!["python".to_string()]);
        assert!(cap.supports_interrupt);
        assert_eq!(
            cap.extensions.get("future_flag_from_newer_agent"),
            Some(&json!(true))
        );
        assert_eq!(cap.extensions.get("nested"), Some(&json!({ "x": 1 })));

        let again = serde_json::to_value(&cap).expect("serialize");
        assert_eq!(again["provider_id"], json!("local"));
        assert_eq!(again["future_flag_from_newer_agent"], json!(true));
    }

    #[test]
    fn session_capabilities_roundtrip() {
        let s = SessionCapabilities {
            session_id: SessionId::new("sess-1"),
            schema_version: 1,
            provider_id: "jupyter".into(),
            active_language: Some("python3".into()),
            gpu_visible: Some(true),
            working_directory_display: Some("/workspace/project".into()),
            provider_snapshot: ProviderCapabilitiesSnapshot {
                supports_interrupt: true,
                supports_package_install: true,
                supports_remote_shell: false,
                jupyter_kernel: true,
                network_policy: NetworkPolicy::Full,
            },
            extensions: BTreeMap::from([("quota_tier".into(), json!("pro"))]),
        };
        let v = serde_json::to_value(&s).unwrap();
        let back: SessionCapabilities = serde_json::from_value(v).unwrap();
        assert_eq!(back, s);
    }
}
