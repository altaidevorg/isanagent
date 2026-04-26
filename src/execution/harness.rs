//! Shared runtime for execution tools (Phase 2): one [`ExecutionProvider`] per process + capability summaries.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use serde_json::{json, Value};

use crate::config::AppConfig;

use super::artifacts::ArtifactLimits;
use super::colab_mcp::{ColabMcpExecutionProvider, ColabMcpExecutionProviderConfig};
use super::error::ExecutionError;
use super::jupyter::{JupyterExecutionProvider, JupyterExecutionProviderConfig};
use super::local::{LocalExecutionConfig, LocalExecutionProvider, LocalPythonRuntime};
use super::provider::ExecutionProvider;
use super::ssh::{SshExecutionProvider, SshExecutionProviderConfig};

/// Owns one or more [`ExecutionProvider`] instances and resolves them per-session.
///
/// The harness is built once per process from `[harness.execution]` config. Each candidate
/// provider in `allowed_providers` (or all four implemented ids when unset) is constructed
/// independently; ones that fail to build are pruned with a warn so a single bad config
/// (e.g. missing SSH host) cannot block the rest. The default provider — used when a tool
/// call omits an explicit `provider` argument — is whichever id is named in
/// `default_provider` if it survived pruning, otherwise the first available id alphabetically.
///
/// Sessions are bound to the provider that created them via `session_providers`; subsequent
/// `execution_run` / `execution_session_close` calls are routed back to that provider, so a
/// process can hold local + colab_mcp sessions concurrently.
pub struct ExecutionHarness {
    /// All built providers, keyed by `provider_id` (`"local"`, `"jupyter"`, `"ssh"`, `"colab_mcp"`).
    providers: HashMap<String, Arc<dyn ExecutionProvider>>,
    /// Default provider id used when a tool call omits `provider`. Always present in `providers`.
    default_provider_id: String,
    /// Set when the `colab_mcp` provider built successfully — typed access for
    /// `colab_mcp_tool_call`, catalog refresh, and `shutdown_all_sessions`.
    colab_mcp: Option<Arc<ColabMcpExecutionProvider>>,
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
    /// key in `providers`. `colab_mcp` (typed handle) is set when the corresponding entry was
    /// built successfully — used by `colab_mcp_tool_call` and `shutdown_all_sessions`.
    #[allow(clippy::too_many_arguments)] // Constructor matches harness config surface area.
    pub fn new_with_providers(
        providers: HashMap<String, Arc<dyn ExecutionProvider>>,
        default_provider_id: String,
        colab_mcp: Option<Arc<ColabMcpExecutionProvider>>,
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
            colab_mcp,
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
            None,
            python_executable,
            workspace_dir,
            sandbox_dir,
            artifact_limits,
            default_run_timeout_secs,
            max_wall_secs,
            auto_promote_after_secs,
        )
    }

    /// Colab MCP single-provider harness. Used by tests and callers that previously
    /// constructed a colab-only harness via the typed handle.
    #[allow(clippy::too_many_arguments)]
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
        let id = provider.provider_id().to_string();
        let mut providers: HashMap<String, Arc<dyn ExecutionProvider>> = HashMap::new();
        providers.insert(id.clone(), provider);
        Self::new_with_providers(
            providers,
            id,
            Some(colab),
            python_executable,
            workspace_dir,
            sandbox_dir,
            artifact_limits,
            default_run_timeout_secs,
            max_wall_secs,
            auto_promote_after_secs,
        )
    }

    pub fn colab_mcp(&self) -> Option<&Arc<ColabMcpExecutionProvider>> {
        self.colab_mcp.as_ref()
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
    /// agent exits. Today this only matters for Colab MCP (which spawns a long-running stdio
    /// proxy child); other providers are no-ops. Failures are swallowed: this runs at exit.
    pub async fn shutdown(&self) {
        if let Some(colab) = self.colab_mcp.as_ref() {
            colab.shutdown_all_sessions().await;
        }
    }

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
/// `[harness.execution].allowed_providers`; unset = all four are candidates.
const IMPLEMENTED_PROVIDERS: &[&str] = &["colab_mcp", "jupyter", "local", "ssh"];

/// Result of a successful provider construction: the trait object plus an optional concrete
/// `ColabMcpExecutionProvider` handle (only populated for the `colab_mcp` provider so the
/// harness can call its typed shutdown helpers).
type BuiltProvider = (
    Arc<dyn ExecutionProvider>,
    Option<Arc<ColabMcpExecutionProvider>>,
);

/// Try to construct one provider by id. Returns `Ok((provider, optional_typed_colab))` or an
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
            Ok((Arc::new(p), None))
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
                notebook_sync_path_template: config
                    .execution_jupyter_notebook_sync_path_template(),
            };
            let p =
                JupyterExecutionProvider::new(jc).map_err(|e: ExecutionError| e.to_string())?;
            Ok((Arc::new(p), None))
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
                max_run_timeout_secs: config.execution_max_wall_secs(),
                max_output_bytes: config.execution_max_output_bytes(),
                max_sessions: config.execution_max_sessions(),
            };
            let p = SshExecutionProvider::new(sc).map_err(|e: ExecutionError| e.to_string())?;
            Ok((Arc::new(p), None))
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
            let typed = p.clone();
            Ok((p, Some(typed)))
        }
        other => Err(format!(
            "unknown provider id {other:?} (supported: \"local\", \"jupyter\", \"ssh\", \"colab_mcp\")"
        )),
    }
}

/// Build the harness from workspace + app config. Call only when
/// `AppConfig::execution_harness_enabled()` is true.
///
/// Behavior:
/// - The candidate set is `allowed_providers` if non-empty, else all four implemented ids.
/// - Each candidate is built independently. Failures are logged at warn level and pruned;
///   they do not block the rest of the agent from starting.
/// - The default provider is chosen as follows:
///   - If user-configured `default_provider` survived pruning → use it.
///   - If user explicitly set `default_provider` and it was pruned → hard-fail (the user
///     pinned a now-broken provider; we'd rather refuse to start than silently swap).
///   - If `default_provider` was the implicit fallback (`"colab_mcp"`) and it was pruned →
///     warn and auto-pick the first available id alphabetically.
/// - If no provider builds successfully, return `Err` so the binary can refuse to start.
pub fn build_execution_harness(
    workspace_dir: PathBuf,
    sandbox_dir: PathBuf,
    restrict_to_workspace: bool,
    config: &AppConfig,
) -> Result<Arc<ExecutionHarness>, String> {
    let configured_default = config.execution_default_provider();
    let default_explicit = config.execution_default_provider_explicit();
    let allowed = config.execution_allowed_providers();

    if default_explicit && !config.execution_provider_allowed(&configured_default) {
        return Err(format!(
            "execution provider {configured_default:?} is not listed in [harness.execution].allowed_providers"
        ));
    }

    let candidates: Vec<&str> = match allowed.as_ref() {
        Some(list) => list.iter().map(String::as_str).collect(),
        None => IMPLEMENTED_PROVIDERS.to_vec(),
    };

    let artifact_limits = ArtifactLimits {
        max_file_bytes: config.execution_artifact_max_file_bytes(),
        max_total_bytes_per_run: config.execution_artifact_max_total_bytes_per_run(),
        max_files_per_run: config.execution_artifact_max_files_per_run(),
    };

    let mut providers: HashMap<String, Arc<dyn ExecutionProvider>> = HashMap::new();
    let mut typed_colab: Option<Arc<ColabMcpExecutionProvider>> = None;
    let mut prune_log: Vec<(String, String)> = Vec::new();

    for pid in candidates {
        match try_build_provider(
            pid,
            &workspace_dir,
            &sandbox_dir,
            restrict_to_workspace,
            artifact_limits,
            config,
        ) {
            Ok((provider, colab)) => {
                providers.insert(pid.to_string(), provider);
                if let Some(c) = colab {
                    typed_colab = Some(c);
                }
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
        ids.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
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
        typed_colab,
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SCRATCH_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn fresh_workspace(tag: &str) -> (PathBuf, PathBuf) {
        let n = SCRATCH_COUNTER.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!(
            "isanagent-harness-test-{tag}-{n}-{}",
            uuid::Uuid::new_v4()
        ));
        let sandbox = root.join("workspace");
        std::fs::create_dir_all(&sandbox).expect("mkdir");
        (root, sandbox)
    }

    #[test]
    fn local_system_runtime_requires_python_executable() {
        let cfg: AppConfig = toml::from_str(
            r#"
[harness.execution]
enabled = true
default_provider = "local"
local_python_runtime = "system"
allowed_providers = ["local"]
"#,
        )
        .expect("parse");
        let (root, sandbox) = fresh_workspace("system-runtime");
        let res = build_execution_harness(root.clone(), sandbox.clone(), true, &cfg);
        assert!(res.is_err(), "expected missing python_executable error");
        let err = match res {
            Ok(_) => String::new(),
            Err(e) => e,
        };
        // With allowed_providers = ["local"] and local broken, we get the "no providers" error
        // that quotes the underlying reason from the prune log.
        assert!(err.contains("python_executable"), "err={err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_execution_harness_prunes_misconfigured_and_keeps_local() {
        // ssh and jupyter are unconfigured (no host / base_url) and should be pruned. local builds.
        let cfg: AppConfig = toml::from_str(
            r#"
[harness.execution]
enabled = true
default_provider = "local"
local_python_runtime = "system"
python_executable = "python3"
allowed_providers = ["local", "ssh", "jupyter"]
"#,
        )
        .expect("parse");
        let (root, sandbox) = fresh_workspace("prune");
        let res = build_execution_harness(root.clone(), sandbox.clone(), true, &cfg);
        let harness = res.expect("harness should build with local intact");
        let avail = harness.available_providers();
        assert_eq!(avail, vec!["local"], "ssh/jupyter should be pruned");
        assert_eq!(harness.default_provider_id(), "local");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_execution_harness_hard_fails_when_explicit_default_pruned() {
        // User explicitly pinned ssh as default but ssh has no host: build must refuse, not
        // silently swap to local — surprises here would be hard to debug at runtime.
        let cfg: AppConfig = toml::from_str(
            r#"
[harness.execution]
enabled = true
default_provider = "ssh"
local_python_runtime = "system"
python_executable = "python3"
allowed_providers = ["ssh", "local"]
"#,
        )
        .expect("parse");
        let (root, sandbox) = fresh_workspace("explicit-default-pruned");
        let res = build_execution_harness(root.clone(), sandbox.clone(), true, &cfg);
        let err = res.err().expect("expected hard fail when explicit default pruned");
        assert!(
            err.contains("default_provider") && err.contains("ssh"),
            "err={err}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_execution_harness_empty_after_prune_returns_err() {
        // Only ssh is allowed and ssh has no host → empty map → hard error.
        let cfg: AppConfig = toml::from_str(
            r#"
[harness.execution]
enabled = true
default_provider = "ssh"
allowed_providers = ["ssh"]
"#,
        )
        .expect("parse");
        let (root, sandbox) = fresh_workspace("all-pruned");
        let res = build_execution_harness(root.clone(), sandbox.clone(), true, &cfg);
        let err = res.err().expect("expected empty-map failure");
        assert!(
            err.contains("no providers could be built") || err.contains("not listed"),
            "err={err}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn provider_for_resolves_optional_id_and_default() {
        let cfg: AppConfig = toml::from_str(
            r#"
[harness.execution]
enabled = true
default_provider = "local"
local_python_runtime = "system"
python_executable = "python3"
allowed_providers = ["local"]
"#,
        )
        .expect("parse");
        let (root, sandbox) = fresh_workspace("provider-for");
        let harness =
            build_execution_harness(root.clone(), sandbox.clone(), true, &cfg).expect("build");
        // None / empty / "default" all resolve to the default.
        assert_eq!(harness.provider_for(None).unwrap().provider_id(), "local");
        assert_eq!(harness.provider_for(Some("")).unwrap().provider_id(), "local");
        assert_eq!(
            harness.provider_for(Some("default")).unwrap().provider_id(),
            "local"
        );
        // Explicit known id resolves.
        assert_eq!(
            harness.provider_for(Some("local")).unwrap().provider_id(),
            "local"
        );
        // Unknown id surfaces the available list so the model can self-correct.
        let err = harness.provider_for(Some("colab_mcp")).err().expect("should be unknown");
        assert!(err.contains("available providers"), "err={err}");
        assert!(err.contains("local"), "err={err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn session_provider_mapping_round_trip() {
        let cfg: AppConfig = toml::from_str(
            r#"
[harness.execution]
enabled = true
default_provider = "local"
local_python_runtime = "system"
python_executable = "python3"
allowed_providers = ["local"]
"#,
        )
        .expect("parse");
        let (root, sandbox) = fresh_workspace("session-map");
        let harness =
            build_execution_harness(root.clone(), sandbox.clone(), true, &cfg).expect("build");

        // Unknown sid falls back to default.
        let p = harness.provider_for_session("does-not-exist");
        assert_eq!(p.provider_id(), "local");

        // After registration the mapping is honored. (Only one provider in this harness, but
        // the lookup path is what we want to exercise.)
        harness.register_session("sid-1", "local");
        assert_eq!(harness.provider_for_session("sid-1").provider_id(), "local");
        harness.unregister_session("sid-1");
        assert_eq!(
            harness.provider_for_session("sid-1").provider_id(),
            "local",
            "unregister must drop the mapping"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
