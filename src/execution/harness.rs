//! Shared runtime for execution tools (Phase 2): one [`ExecutionProvider`] per process + capability summaries.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::config::AppConfig;

use super::error::ExecutionError;
use super::jupyter::{JupyterExecutionProvider, JupyterExecutionProviderConfig};
use super::local::{LocalExecutionConfig, LocalExecutionProvider};
use super::provider::ExecutionProvider;

/// Owns the active [`ExecutionProvider`] and small bits of host config tools need (e.g. `-V` probe).
pub struct ExecutionHarness {
    provider: Arc<dyn ExecutionProvider>,
    python_executable: String,
}

impl ExecutionHarness {
    pub fn new(provider: Arc<dyn ExecutionProvider>, python_executable: impl Into<String>) -> Self {
        Self {
            provider,
            python_executable: python_executable.into(),
        }
    }

    pub fn provider(&self) -> &Arc<dyn ExecutionProvider> {
        &self.provider
    }

    pub fn python_executable(&self) -> &str {
        &self.python_executable
    }

    /// Bounded JSON for injection into tool results (omits `extensions` map).
    pub fn capabilities_summary(&self) -> Value {
        let c = self.provider.capabilities();
        json!({
            "provider_id": c.provider_id,
            "schema_version": c.schema_version,
            "languages": c.languages,
            "supports_persistent_sessions": c.supports_persistent_sessions,
            "supports_interrupt": c.supports_interrupt,
            "supports_package_install": c.supports_package_install,
            "supports_remote_shell": c.supports_remote_shell,
            "jupyter_kernel": c.jupyter_kernel,
            "network_policy": serde_json::to_value(c.network_policy)
                .unwrap_or_else(|_| json!("unknown")),
            "max_output_bytes_default": c.max_output_bytes_default,
        })
    }
}

/// Build the harness from workspace + app config. Call only when `[harness.execution] enabled = true`.
pub fn build_execution_harness(
    sandbox_dir: PathBuf,
    restrict_to_workspace: bool,
    config: &AppConfig,
) -> Result<Arc<ExecutionHarness>, String> {
    let pid = config.execution_default_provider();
    if !config.execution_provider_allowed(&pid) {
        return Err(format!(
            "execution provider {pid:?} is not listed in [harness.execution].allowed_providers"
        ));
    }
    match pid.as_str() {
        "local" => {
            let lc = LocalExecutionConfig {
                sandbox_dir,
                restrict_to_workspace,
                max_run_timeout_secs: config.execution_max_wall_secs(),
                max_output_bytes: config.execution_max_output_bytes(),
                max_sessions: config.execution_max_sessions(),
                python_executable: config.execution_python_executable(),
            };
            let p = LocalExecutionProvider::new(lc).map_err(|e: ExecutionError| e.to_string())?;
            Ok(Arc::new(ExecutionHarness::new(
                Arc::new(p),
                config.execution_python_executable(),
            )))
        }
        "jupyter" => {
            let base_url = config.execution_jupyter_base_url().ok_or_else(|| {
                "[harness.execution.jupyter].base_url is required when default_provider = \"jupyter\""
                    .to_string()
            })?;
            let jc = JupyterExecutionProviderConfig {
                base_url,
                token: config.execution_jupyter_token(),
                default_kernel_name: config.execution_jupyter_kernel_name(),
                max_run_timeout_secs: config.execution_max_wall_secs(),
                max_output_bytes: config.execution_max_output_bytes(),
                max_sessions: config.execution_max_sessions(),
            };
            let p =
                JupyterExecutionProvider::new(jc).map_err(|e: ExecutionError| e.to_string())?;
            Ok(Arc::new(ExecutionHarness::new(
                Arc::new(p),
                config.execution_python_executable(),
            )))
        }
        other => Err(format!(
            "unknown [harness.execution] default_provider: {other} (supported: \"local\", \"jupyter\")"
        )),
    }
}
