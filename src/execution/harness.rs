//! Shared runtime for execution tools (Phase 2): one [`ExecutionProvider`] per process + capability summaries.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use serde_json::{json, Value};

use crate::config::AppConfig;

use super::artifacts::ArtifactLimits;
use super::error::ExecutionError;
use super::jupyter::{JupyterExecutionProvider, JupyterExecutionProviderConfig};
use super::local::{LocalExecutionConfig, LocalExecutionProvider, LocalPythonRuntime};
use super::provider::ExecutionProvider;
use super::ssh::{SshExecutionProvider, SshExecutionProviderConfig};

/// Owns one or more [`ExecutionProvider`] instances and resolves them per-session.
///
/// The harness is built once per process from `[harness.execution]` config. Each candidate
/// provider in `allowed_providers` (or all three implemented ids when unset) is constructed
/// independently; ones that fail to build are pruned with a warn so a single bad config
/// (e.g. missing SSH host) cannot block the rest. The default provider — used when a tool
/// call omits an explicit `provider` argument — is whichever id is named in
/// `default_provider` if it survived pruning, otherwise the first available id alphabetically.
///
/// Sessions are bound to the provider that created them via `session_providers`; subsequent
/// `execution_run` / `execution_session_close` calls are routed back to that provider, so a
/// process can hold local + jupyter sessions concurrently.
pub struct ExecutionHarness {
    /// All built providers, keyed by `provider_id` (`"local"`, `"jupyter"`, `"ssh"`).
    providers: HashMap<String, Arc<dyn ExecutionProvider>>,
    /// Default provider id used when a tool call omits `provider`. Always present in `providers`.
    default_provider_id: String,
    /// Maps `session_id` → owning `provider_id`. Populated by
    /// `execution_session_create` so per-session ops route back to the right provider.
    session_providers: Arc<DashMap<String, String>>,
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
    /// Build a harness from a pre-constructed provider map. `default_provider_id` must be a
    /// key in `providers`.
    #[allow(clippy::too_many_arguments)] // Constructor matches harness config surface area.
    pub fn new_with_providers(
        providers: HashMap<String, Arc<dyn ExecutionProvider>>,
        default_provider_id: String,
        python_executable: impl Into<String>,
        workspace_dir: PathBuf,
        sandbox_dir: PathBuf,
        artifact_limits: ArtifactLimits,
        default_run_timeout_secs: u64,
        max_wall_secs: u64,
        auto_promote_after_secs: u64,
    ) -> Self {
        debug_assert!(
            providers.contains_key(&default_provider_id),
            "default_provider_id {} must be a key in providers",
            default_provider_id
        );
        Self {
            providers,
            default_provider_id,
            session_providers: Arc::new(DashMap::new()),
            python_executable: python_executable.into(),
            workspace_dir,
            sandbox_dir,
            artifact_limits,
            default_run_timeout_secs,
            max_wall_secs,
            auto_promote_after_secs,
        }
    }

    /// Convenience wrapper for tests / single-provider call sites: builds a harness with a
    /// single provider that is also the default. Equivalent to the legacy `new` constructor.
    #[allow(clippy::too_many_arguments)]
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
        let id = provider.provider_id().to_string();
        let mut providers: HashMap<String, Arc<dyn ExecutionProvider>> = HashMap::new();
        providers.insert(id.clone(), provider);
        Self::new_with_providers(
            providers,
            id,
            python_executable,
            workspace_dir,
            sandbox_dir,
            artifact_limits,
            default_run_timeout_secs,
            max_wall_secs,
            auto_promote_after_secs,
        )
    }

    /// Default provider (for callers that don't know — or don't need — the per-session
    /// owner). Most session-bound ops should call [`provider_for_session`] instead.
    pub fn provider(&self) -> &Arc<dyn ExecutionProvider> {
        self.providers
            .get(&self.default_provider_id)
            .expect("default_provider_id is invariant in providers map")
    }

    /// Default provider id used when `execution_session_create` omits `provider`.
    pub fn default_provider_id(&self) -> &str {
        &self.default_provider_id
    }

    /// Sorted list of available provider ids (those that built successfully).
    pub fn available_providers(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.providers.keys().map(String::as_str).collect();
        ids.sort();
        ids
    }

    /// Resolve a provider by optional id. `None` (or empty / `"default"`) returns the default.
    /// Errors out with the available ids when `id` is unknown so the model can self-correct.
    pub fn provider_for(&self, id: Option<&str>) -> Result<Arc<dyn ExecutionProvider>, String> {
        let trimmed = id.map(str::trim).unwrap_or("");
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("default") {
            return Ok(self.provider().clone());
        }
        self.providers.get(trimmed).cloned().ok_or_else(|| {
            format!(
                "unknown execution provider {trimmed:?}; available providers: [{}]",
                self.available_providers().join(", ")
            )
        })
    }

    /// Look up the provider that owns `session_id`. Falls back to the default provider when
    /// the mapping is missing — preserves correctness for legacy single-provider sessions
    /// and for tests that bypass `register_session`.
    pub fn provider_for_session(&self, session_id: &str) -> Arc<dyn ExecutionProvider> {
        if let Some(entry) = self.session_providers.get(session_id) {
            if let Some(p) = self.providers.get(entry.value()) {
                return p.clone();
            }
        }
        self.provider().clone()
    }

    /// Record the provider that owns `session_id`. Called from `execution_session_create`.
    pub fn register_session(&self, session_id: &str, provider_id: &str) {
        self.session_providers
            .insert(session_id.to_string(), provider_id.to_string());
    }

    /// Forget the session→provider mapping after a successful close.
    pub fn unregister_session(&self, session_id: &str) {
        self.session_providers.remove(session_id);
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

    /// Best-effort graceful teardown of provider-side child processes / sessions before the
    /// agent exits. Failures are swallowed: this runs at exit.
    pub async fn shutdown(&self) {}

    /// Bounded JSON for injection into tool results (omits `extensions` map).
    /// Reflects the **default** provider; per-session ops should query the session's
    /// owning provider directly when they need provider-specific capabilities.
    pub fn capabilities_summary(&self) -> Value {
        let c = self.provider().capabilities();
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

/// All implemented provider ids (alphabetical). Any subset can be allowed via
/// `[harness.execution].allowed_providers`; unset = all three are candidates.
const IMPLEMENTED_PROVIDERS: &[&str] = &["jupyter", "local", "ssh"];

/// Result of a successful provider construction: the trait object.
type BuiltProvider = Arc<dyn ExecutionProvider>;

/// Try to construct one provider by id. Returns `Ok(provider)` or an
/// `Err` describing why the build failed (missing config, network, etc.). Never panics.
fn try_build_provider(
    pid: &str,
    workspace_dir: &std::path::Path,
    sandbox_dir: &std::path::Path,
    restrict_to_workspace: bool,
    artifact_limits: ArtifactLimits,
    config: &AppConfig,
) -> Result<BuiltProvider, String> {
    match pid {
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
                sandbox_dir: sandbox_dir.to_path_buf(),
                workspace_dir: workspace_dir.to_path_buf(),
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
            Ok(Arc::new(p))
        }
        "jupyter" => {
            let base_url = config.execution_jupyter_base_url().ok_or_else(|| {
                "[harness.execution.jupyter].base_url is required to build the jupyter provider"
                    .to_string()
            })?;
            let jc = JupyterExecutionProviderConfig {
                base_url,
                token: config.execution_jupyter_token(),
                default_kernel_name: config.execution_jupyter_kernel_name(),
                max_run_timeout_secs: config.execution_max_wall_secs(),
                max_output_bytes: config.execution_max_output_bytes(),
                max_sessions: config.execution_max_sessions(),
                artifact_sandbox_dir: sandbox_dir.to_path_buf(),
                artifact_limits,
                notebook_sync_path_template: config.execution_jupyter_notebook_sync_path_template(),
            };
            let p = JupyterExecutionProvider::new(jc).map_err(|e: ExecutionError| e.to_string())?;
            Ok(Arc::new(p))
        }
        "ssh" => {
            let host = config.execution_ssh_host().ok_or_else(|| {
                "[harness.execution.ssh].host is required to build the ssh provider".to_string()
            })?;
            let user = config.execution_ssh_user().ok_or_else(|| {
                "[harness.execution.ssh].user is required to build the ssh provider".to_string()
            })?;
            let remote_workdir = config.execution_ssh_remote_workdir().ok_or_else(|| {
                "[harness.execution.ssh].remote_workdir is required to build the ssh provider"
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
                known_hosts_path: workspace_dir
                    .join(".system_generated")
                    .join("ssh")
                    .join("known_hosts"),
                max_run_timeout_secs: config.execution_max_wall_secs(),
                max_output_bytes: config.execution_max_output_bytes(),
                max_sessions: config.execution_max_sessions(),
            };
            let p = SshExecutionProvider::new(sc).map_err(|e: ExecutionError| e.to_string())?;
            Ok(Arc::new(p))
        }
        other => Err(format!(
            "unknown provider id {other:?} (supported: \"local\", \"jupyter\", \"ssh\")"
        )),
    }
}

/// Build the harness from workspace + app config. Call only when
/// `AppConfig::execution_harness_enabled()` is true.
pub fn build_execution_harness(
    workspace_dir: PathBuf,
    sandbox_dir: PathBuf,
    restrict_to_workspace: bool,
    config: &AppConfig,
) -> Result<Arc<ExecutionHarness>, String> {
    let candidates = config.execution_allowed_providers().unwrap_or_else(|| {
        IMPLEMENTED_PROVIDERS
            .iter()
            .map(|s| s.to_string())
            .collect()
    });

    let configured_default = config.execution_default_provider();
    let default_explicit = config.execution_default_provider_explicit();
    let artifact_limits = ArtifactLimits {
        max_file_bytes: config.execution_artifact_max_file_bytes(),
        max_total_bytes_per_run: config.execution_artifact_max_total_bytes_per_run(),
        max_files_per_run: config.execution_artifact_max_files_per_run(),
    };

    let mut providers: HashMap<String, Arc<dyn ExecutionProvider>> = HashMap::new();
    let mut prune_log: Vec<(String, String)> = Vec::new();

    for pid in candidates {
        match try_build_provider(
            &pid,
            &workspace_dir,
            &sandbox_dir,
            restrict_to_workspace,
            artifact_limits,
            config,
        ) {
            Ok(provider) => {
                providers.insert(pid.to_string(), provider);
            }
            Err(reason) => {
                log::warn!(
                    "Execution provider {pid:?} disabled: {reason}; pruning from available set."
                );
                prune_log.push((pid.to_string(), reason));
            }
        }
    }

    if providers.is_empty() {
        let detail = if prune_log.is_empty() {
            "no provider candidates were configured".to_string()
        } else {
            prune_log
                .iter()
                .map(|(pid, why)| format!("{pid}: {why}"))
                .collect::<Vec<_>>()
                .join("; ")
        };
        return Err(format!(
            "[harness.execution] is enabled but no providers could be built. {detail}"
        ));
    }

    let default_provider_id = if providers.contains_key(&configured_default) {
        configured_default.clone()
    } else if default_explicit {
        return Err(format!(
            "execution default_provider {configured_default:?} failed to build and was pruned (see warnings); refusing to silently fall back. \
             Fix the provider config or change [harness.execution].default_provider to one of: [{}]",
            {
                let mut ids: Vec<&String> = providers.keys().collect();
                ids.sort();
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            }
        ));
    } else {
        let mut ids: Vec<&String> = providers.keys().collect();
        ids.sort();
        let auto = ids[0].clone();
        log::warn!(
            "Implicit default execution provider {configured_default:?} was pruned; \
             auto-picking {auto:?} (set [harness.execution].default_provider explicitly to silence this warning)."
        );
        auto
    };

    let built_summary = {
        let mut ids: Vec<&String> = providers.keys().collect();
        ids.sort();
        ids.iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    log::info!(
        "Execution harness ready: providers=[{built_summary}], default={default_provider_id}"
    );

    let python_executable = match default_provider_id.as_str() {
        "local" => match config.execution_local_python_runtime().as_str() {
            "uv_managed" | "uvmanaged" | "uv" => config.execution_python_executable(),
            _ => config
                .execution_python_executable_configured()
                .unwrap_or_else(|| config.execution_python_executable()),
        },
        _ => config.execution_python_executable(),
    };

    Ok(Arc::new(ExecutionHarness::new_with_providers(
        providers,
        default_provider_id,
        python_executable,
        workspace_dir,
        sandbox_dir,
        artifact_limits,
        config.execution_default_run_timeout_secs(),
        config.execution_max_wall_secs(),
        config.execution_auto_promote_after_secs(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn harness_resolves_providers_per_session() {
        let root =
            std::env::temp_dir().join(format!("isanagent-harness-test-{}", uuid::Uuid::new_v4()));
        let ws = root.clone();
        let sandbox = root.join("sandbox");
        let _ = std::fs::create_dir_all(&sandbox);

        let lc = LocalExecutionConfig::new(sandbox.clone(), ws.clone(), true);
        let lp = Arc::new(LocalExecutionProvider::new(lc).unwrap());

        let jc = JupyterExecutionProviderConfig {
            base_url: "http://127.0.0.1:8888".to_string(),
            token: None,
            default_kernel_name: "python3".to_string(),
            max_run_timeout_secs: 3600,
            max_output_bytes: 1024,
            max_sessions: 1,
            artifact_sandbox_dir: sandbox.clone(),
            artifact_limits: ArtifactLimits::default(),
            notebook_sync_path_template: None,
        };
        let jp = Arc::new(JupyterExecutionProvider::new(jc).unwrap());

        let mut providers: HashMap<String, Arc<dyn ExecutionProvider>> = HashMap::new();
        providers.insert("local".to_string(), lp.clone());
        providers.insert("jupyter".to_string(), jp.clone());

        let harness = ExecutionHarness::new_with_providers(
            providers,
            "local".to_string(),
            "python",
            ws.clone(),
            sandbox.clone(),
            ArtifactLimits::default(),
            60,
            3600,
            0,
        );

        assert_eq!(harness.available_providers(), vec!["jupyter", "local"]);
        assert_eq!(harness.provider_for(None).unwrap().provider_id(), "local");
        assert_eq!(
            harness.provider_for(Some("jupyter")).unwrap().provider_id(),
            "jupyter"
        );
        assert_eq!(
            harness.provider_for(Some("local")).unwrap().provider_id(),
            "local"
        );
        // Unknown id surfaces the available list so the model can self-correct.
        let err = harness
            .provider_for(Some("ssh"))
            .err()
            .expect("should be unknown");
        assert!(err.contains("available providers"), "err={err}");
        assert!(err.contains("local"), "err={err}");

        // Mapping persistence.
        harness.register_session("sid-1", "jupyter");
        assert_eq!(
            harness.provider_for_session("sid-1").provider_id(),
            "jupyter"
        );
        harness.unregister_session("sid-1");
        assert_eq!(
            harness.provider_for_session("sid-1").provider_id(),
            "local",
            "unregister must drop the mapping"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
