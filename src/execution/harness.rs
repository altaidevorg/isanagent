//! Shared runtime for execution tools (Phase 2): one [`ExecutionProvider`] per process + capability summaries.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::config::AppConfig;

use super::artifacts::ArtifactLimits;
use super::colab_mcp::{ColabMcpExecutionProvider, ColabMcpExecutionProviderConfig};
use super::error::ExecutionError;
use super::jupyter::{JupyterExecutionProvider, JupyterExecutionProviderConfig};
use super::local::{LocalExecutionConfig, LocalExecutionProvider, LocalPythonRuntime};
use super::provider::ExecutionProvider;
use super::ssh::{SshExecutionProvider, SshExecutionProviderConfig};

/// Owns the active [`ExecutionProvider`] and small bits of host config tools need (e.g. `-V` probe).
pub struct ExecutionHarness {
    provider: Arc<dyn ExecutionProvider>,
    /// Set when `default_provider = "colab_mcp"`: typed access for `colab_mcp_tool_call` and catalog refresh.
    colab_mcp: Option<Arc<ColabMcpExecutionProvider>>,
    python_executable: String,
    workspace_dir: PathBuf,
    sandbox_dir: PathBuf,
    artifact_limits: ArtifactLimits,
    /// When `execution_run` omits `timeout_secs`, use this (clamped to provider max at tool call time).
    pub default_run_timeout_secs: u64,
    /// Upper bound for per-run `timeout_secs` from `[harness.execution] max_wall_secs`.
    pub max_wall_secs: u64,
    /// Short bound after which a synchronous run auto-promotes to a background job (`0` disables).
    pub auto_promote_after_secs: u64,
}

impl ExecutionHarness {
    pub fn new(
        provider: Arc<dyn ExecutionProvider>,
        python_executable: impl Into<String>,
        workspace_dir: PathBuf,
        sandbox_dir: PathBuf,
        artifact_limits: ArtifactLimits,
        default_run_timeout_secs: u64,
        max_wall_secs: u64,
        auto_promote_after_secs: u64,
    ) -> Self {
        Self {
            provider,
            colab_mcp: None,
            python_executable: python_executable.into(),
            workspace_dir,
            sandbox_dir,
            artifact_limits,
            default_run_timeout_secs,
            max_wall_secs,
            auto_promote_after_secs,
        }
    }

    /// Colab MCP harness only (`default_provider = "colab_mcp"`).
    pub fn new_colab_mcp(
        colab: Arc<ColabMcpExecutionProvider>,
        python_executable: impl Into<String>,
        workspace_dir: PathBuf,
        sandbox_dir: PathBuf,
        artifact_limits: ArtifactLimits,
        default_run_timeout_secs: u64,
        max_wall_secs: u64,
        auto_promote_after_secs: u64,
    ) -> Self {
        let provider: Arc<dyn ExecutionProvider> = colab.clone();
        Self {
            provider,
            colab_mcp: Some(colab),
            python_executable: python_executable.into(),
            workspace_dir,
            sandbox_dir,
            artifact_limits,
            default_run_timeout_secs,
            max_wall_secs,
            auto_promote_after_secs,
        }
    }

    pub fn colab_mcp(&self) -> Option<&Arc<ColabMcpExecutionProvider>> {
        self.colab_mcp.as_ref()
    }

    pub fn provider(&self) -> &Arc<dyn ExecutionProvider> {
        &self.provider
    }

    pub fn python_executable(&self) -> &str {
        &self.python_executable
    }

    pub fn workspace_dir(&self) -> &PathBuf {
        &self.workspace_dir
    }

    pub fn sandbox_dir(&self) -> &PathBuf {
        &self.sandbox_dir
    }

    pub fn artifact_limits(&self) -> ArtifactLimits {
        self.artifact_limits
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

/// Build the harness from workspace + app config. Call only when `AppConfig::execution_harness_enabled()` is true.
pub fn build_execution_harness(
    workspace_dir: PathBuf,
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
    let artifact_limits = ArtifactLimits {
        max_file_bytes: config.execution_artifact_max_file_bytes(),
        max_total_bytes_per_run: config.execution_artifact_max_total_bytes_per_run(),
        max_files_per_run: config.execution_artifact_max_files_per_run(),
    };
    match pid.as_str() {
        "local" => {
            let runtime = match config.execution_local_python_runtime().as_str() {
                "uv_managed" | "uvmanaged" | "uv" => LocalPythonRuntime::UvManaged,
                _ => LocalPythonRuntime::System,
            };
            let python_executable = match runtime {
                LocalPythonRuntime::System => config
                    .execution_python_executable_configured()
                    .ok_or_else(|| {
                        "[harness.execution].python_executable is required when local_python_runtime = \"system\"".to_string()
                    })?,
                LocalPythonRuntime::UvManaged => config.execution_python_executable(),
            };
            let lc = LocalExecutionConfig {
                sandbox_dir: sandbox_dir.clone(),
                restrict_to_workspace,
                max_run_timeout_secs: config.execution_max_wall_secs(),
                max_output_bytes: config.execution_max_output_bytes(),
                max_sessions: config.execution_max_sessions(),
                python_executable: python_executable.clone(),
                python_repl: config.execution_local_python_repl_enabled(),
                python_runtime: runtime,
                uv_binary: config.execution_uv_binary(),
                uv_python: config.execution_uv_python(),
                uv_requirements: config.execution_uv_requirements(),
                uv_env_root: workspace_dir
                    .join(".system_generated")
                    .join("uv")
                    .join("envs"),
            };
            let p = LocalExecutionProvider::new(lc).map_err(|e: ExecutionError| e.to_string())?;
            Ok(Arc::new(ExecutionHarness::new(
                Arc::new(p),
                python_executable,
                workspace_dir,
                sandbox_dir,
                artifact_limits,
                config.execution_default_run_timeout_secs(),
                config.execution_max_wall_secs(),
                config.execution_auto_promote_after_secs(),
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
                artifact_sandbox_dir: sandbox_dir.clone(),
                artifact_limits,
                notebook_sync_path_template: config
                    .execution_jupyter_notebook_sync_path_template(),
            };
            let p =
                JupyterExecutionProvider::new(jc).map_err(|e: ExecutionError| e.to_string())?;
            Ok(Arc::new(ExecutionHarness::new(
                Arc::new(p),
                config.execution_python_executable(),
                workspace_dir,
                sandbox_dir,
                artifact_limits,
                config.execution_default_run_timeout_secs(),
                config.execution_max_wall_secs(),
                config.execution_auto_promote_after_secs(),
            )))
        }
        "ssh" => {
            let host = config.execution_ssh_host().ok_or_else(|| {
                "[harness.execution.ssh].host is required when default_provider = \"ssh\"".to_string()
            })?;
            let user = config.execution_ssh_user().ok_or_else(|| {
                "[harness.execution.ssh].user is required when default_provider = \"ssh\"".to_string()
            })?;
            let remote_workdir = config.execution_ssh_remote_workdir().ok_or_else(|| {
                "[harness.execution.ssh].remote_workdir is required when default_provider = \"ssh\""
                    .to_string()
            })?;
            let sc = SshExecutionProviderConfig {
                host,
                port: config.execution_ssh_port(),
                user,
                remote_workdir,
                remote_python: config.execution_ssh_remote_python(),
                identity_path: config.execution_ssh_identity_file(),
                accept_unknown_host_keys: config.execution_ssh_accept_unknown_host_keys(),
                max_run_timeout_secs: config.execution_max_wall_secs(),
                max_output_bytes: config.execution_max_output_bytes(),
                max_sessions: config.execution_max_sessions(),
            };
            let p = SshExecutionProvider::new(sc).map_err(|e: ExecutionError| e.to_string())?;
            Ok(Arc::new(ExecutionHarness::new(
                Arc::new(p),
                config.execution_python_executable(),
                workspace_dir,
                sandbox_dir,
                artifact_limits,
                config.execution_default_run_timeout_secs(),
                config.execution_max_wall_secs(),
                config.execution_auto_promote_after_secs(),
            )))
        }
        "colab_mcp" => {
            let cc = ColabMcpExecutionProviderConfig {
                command: config.execution_colab_mcp_command(),
                args: config.execution_colab_mcp_args(),
                cwd: config.execution_colab_mcp_cwd(),
                startup_timeout_secs: config.execution_colab_mcp_startup_timeout_secs(),
                connect_tool_name: config.execution_colab_mcp_connect_tool_name(),
                execute_tool_name: config.execution_colab_mcp_execute_tool_name(),
                execute_code_arg_keys: config.execution_colab_mcp_execute_code_arg_keys(),
                max_sessions: config.execution_max_sessions(),
                max_output_bytes: config.execution_max_output_bytes(),
            };
            let p = Arc::new(
                ColabMcpExecutionProvider::new(cc).map_err(|e: ExecutionError| e.to_string())?,
            );
            Ok(Arc::new(ExecutionHarness::new_colab_mcp(
                p,
                config.execution_python_executable(),
                workspace_dir,
                sandbox_dir,
                artifact_limits,
                config.execution_default_run_timeout_secs(),
                config.execution_max_wall_secs(),
                config.execution_auto_promote_after_secs(),
            )))
        }
        other => Err(format!(
            "unknown [harness.execution] default_provider: {other} (supported: \"local\", \"jupyter\", \"ssh\", \"colab_mcp\")"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_system_runtime_requires_python_executable() {
        let cfg: AppConfig = toml::from_str(
            r#"
[harness.execution]
enabled = true
default_provider = "local"
local_python_runtime = "system"
"#,
        )
        .expect("parse");
        let root =
            std::env::temp_dir().join(format!("isanagent-harness-test-{}", uuid::Uuid::new_v4()));
        let sandbox = root.join("workspace");
        std::fs::create_dir_all(&sandbox).expect("mkdir");
        let res = build_execution_harness(root.clone(), sandbox.clone(), true, &cfg);
        assert!(res.is_err(), "expected missing python_executable error");
        let err = match res {
            Ok(_) => String::new(),
            Err(e) => e,
        };
        assert!(err.contains("python_executable"), "err={err}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
