use log::warn;
use serde::{Deserialize, Serialize};

/// Local stdin/stdout chat. When `enabled` is omitted, defaults to `true`.
///
/// `max_iterations` / `max_tool_output_chars` here are used only when the root-level keys are
/// unset (see [`AppConfig::resolved_max_iterations`]) — prefer root keys above `[terminal]` in TOML.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct TerminalConfig {
    pub enabled: Option<bool>,
    pub max_iterations: Option<usize>,
    pub max_tool_output_chars: Option<usize>,
}

/// OpenSSH-style remote exec over TCP (`default_provider = "ssh"`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct SshExecutionConfig {
    /// Remote hostname or IP (required for `ssh`).
    pub host: Option<String>,
    /// TCP port (default 22).
    pub port: Option<u16>,
    /// Remote login name (required for `ssh`).
    pub user: Option<String>,
    /// Path to a private key file (OpenSSH PEM). Tilde expansion applied. Optional if
    /// **`SSH_PASSWORD`** is set in the environment.
    pub identity_file: Option<String>,
    /// Absolute path on the **remote** host used as the default cwd before running code (required).
    /// The SSH execution provider runs `mkdir -p` for the resolved cwd on each run so the path may be absent at first connect.
    pub remote_workdir: Option<String>,
    /// Remote Python interpreter for `language: python` (default `python3`).
    pub remote_python: Option<String>,
    /// When true (default), `check_server_key` accepts any host key (**MITM risk**). When false,
    /// host key verification fails until strict known-hosts support exists.
    pub accept_unknown_host_keys: Option<bool>,
}

/// Jupyter Server / Lab HTTP + kernel WebSocket (`default_provider = "jupyter"`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct JupyterExecutionConfig {
    /// e.g. `http://127.0.0.1:8888` (no trailing path).
    pub base_url: Option<String>,
    /// Optional token (prefer env `JUPYTER_TOKEN` in production; avoid committing secrets).
    pub token: Option<String>,
    /// Kernel spec name for `POST /api/kernels` (default `python3`).
    pub kernel_name: Option<String>,
    /// Optional server-side notebook path template for Contents API sync (e.g. `isanagent/{session_id}.ipynb`).
    /// `{session_id}` is replaced with the **sanitized** isanagent session id. Each `execution_run` appends a code cell.
    pub notebook_sync_path_template: Option<String>,
}

/// Code execution harness (`execution_*` tools). On by default; set `[harness.execution] enabled = false` to disable.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ExecutionHarnessConfig {
    /// When `Some(false)`, execution tools are not registered. Omitted or `Some(true)` keeps them on.
    pub enabled: Option<bool>,
    /// Provider id: `local` (subprocess), `jupyter` (remote kernel), or `ssh` (remote exec).
    pub default_provider: Option<String>,
    /// Max combined stdout+stderr bytes per run (default 262_144).
    pub max_output_bytes: Option<usize>,
    /// Upper bound on per-run `timeout_secs` (default 3600, clamped 1–86400).
    pub max_wall_secs: Option<u64>,
    /// Default `timeout_secs` when `execution_run` / `execution_run_background` omit it (default 600 when unset, clamped to `max_wall_secs`).
    pub default_execution_timeout_secs: Option<u64>,
    /// Short bound after which a synchronous run (`execution_run`)
    /// auto-promotes to a background job and returns a `job_id` envelope.
    /// Default 120 when unset; clamped to `5..=max_wall_secs`. Set to `0` to disable
    /// auto-promotion (synchronous calls then run up to their full `timeout_secs`).
    pub auto_promote_after_secs: Option<u64>,
    /// Max concurrent sessions (default 32, clamped 1–256).
    pub max_sessions: Option<usize>,
    /// If set and non-empty, only these provider ids may be constructed (e.g. `["local"]`).
    pub allowed_providers: Option<Vec<String>>,
    /// Interpreter for `language: python` (default `python`) — local provider and `execution_env_info`.
    pub python_executable: Option<String>,
    /// Local Python only: **`repl`** (default) keeps one interpreter per session so variables survive
    /// across `execution_run` calls; **`subprocess`** spawns a fresh `python -u -` per run (legacy).
    pub local_python_mode: Option<String>,
    /// Local Python runtime backend: `uv_managed` (default) or `system` (`uv`, `uv-managed` aliases accepted).
    pub local_python_runtime: Option<String>,
    /// Binary used when `local_python_runtime = "uv_managed"` (default `uv`).
    pub uv_binary: Option<String>,
    /// Python version request for `uv venv --python` (default `3.11`).
    pub uv_python: Option<String>,
    /// Optional package specs installed into the managed env (`uv pip install ...`) on first creation.
    pub uv_requirements: Option<Vec<String>>,
    /// Max bytes per execution artifact file (default 4MiB, clamped 64KiB–64MiB).
    pub artifact_max_file_bytes: Option<usize>,
    /// Max total bytes for all artifacts in one `execution_run` (default 32MiB).
    pub artifact_max_total_bytes_per_run: Option<usize>,
    /// Max artifact files per run (default 64, clamped 1–256).
    pub artifact_max_files_per_run: Option<usize>,
    /// Required when `default_provider = "jupyter"`.
    pub jupyter: Option<JupyterExecutionConfig>,
    /// Required when `default_provider = "ssh"`.
    pub ssh: Option<SshExecutionConfig>,
    /// When true (default), enqueue a synthetic inbound when a background execution job reaches a terminal state
    /// so the agent can call `execution_job_result` without waiting for the user. Set to false for API-only or
    /// headless runs that must not auto-continue the reasoning loop.
    pub wake_on_job_terminal: Option<bool>,
}

/// Sub-agent / task harness. Disabled unless `[harness.subagents] enabled = true`.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct SubagentHarnessConfig {
    pub enabled: Option<bool>,
    /// When true (default), cancelling the parent chat's reasoning (new inbound or `Cancel`) also
    /// cancels in-flight sub-agent tasks that were spawned from that chat.
    pub cancel_children_on_parent_cancel: Option<bool>,
    /// If set and non-empty, sub-agents may only call these tool names (main chat is unaffected).
    pub allowed_tools: Option<Vec<String>>,
    /// Max concurrent tasks per process (default 32, clamped 1–256).
    pub max_tasks: Option<usize>,
    /// Max seconds `subagent_spawn` may block when `wait` is true (default 300, clamped 10–3600).
    pub max_wait_secs: Option<u64>,
    /// When true (default), auto-enqueue a synthetic inbound when a subagent finishes so the
    /// parent agent can consume the result without polling.
    pub wake_on_completion: Option<bool>,
    /// Max completed tasks retained in SQLite per parent chat (default 200, clamped 10–2000).
    pub task_history_retention: Option<usize>,
}

/// A named agent definition from `[agents.<name>]` in config.toml.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AgentDefinition {
    /// Short description of when to use this agent.
    pub description: String,
    /// `subagent` (default) — invoked by the coordinator via tools.
    #[serde(default)]
    pub mode: AgentMode,
    /// Optional custom system prompt (inline or `{file:./path}` resolved at load time).
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Optional path to a markdown file containing the system prompt.
    #[serde(default)]
    pub system_prompt_file: Option<String>,
    /// Tool names this agent may call. `None` or `["*"]` means inherit the harness allowlist.
    /// An empty list means no tools (read-only manual reasoning).
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
    /// Optional model override (`provider/model-id`). When absent the parent model is used.
    #[serde(default)]
    pub model: Option<String>,
    /// Optional temperature override (0.0–1.0).
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Optional max reasoning iterations (clamped to parent max when larger).
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Hide from `@mention` autocomplete (still invokable via tools).
    #[serde(default)]
    pub hidden: bool,
    /// Hex colour or theme token for TUI rendering (e.g. `"#4CAF50"` or `"accent"`).
    #[serde(default)]
    pub color: Option<String>,
}

/// Agent visibility mode.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum AgentMode {
    /// Invoked by the coordinator via tools.
    #[default]
    Subagent,
    /// A deterministic, local-only code retrieval worker powered by Semble.
    /// It never invokes an LLM and may only search the configured workspace.
    #[serde(rename = "semble_scout")]
    SembleScout,
}

/// HF ml-intern–style ML policy overlay + optional autonomy hints (see `assets/ml_engineer_overlay.md`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct MlEngineerHarnessConfig {
    /// Append ML engineer policy to the compiled system prompt (default: false).
    pub enabled: Option<bool>,
    /// Append research-oriented instructions to **sub-agent** system prompts when `enabled` (default: true when enabled).
    pub subagent_research_overlay: Option<bool>,
    /// When true, if an inbound message sets no metadata override, autonomous sessions may still use config default (see inbound metadata `crate::bus::METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS`).
    pub forbid_final_without_tools: Option<bool>,
}

/// Shell safety policy for `exec` tool decisions before command execution.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ShellPolicyConfig {
    /// Interactive default mode: `ask`, `deny`, or `allow` (default `ask`).
    pub mode: Option<String>,
    /// Unattended/autonomous mode: `ask`, `deny`, or `allow` (default `deny`).
    pub unattended_default: Option<String>,
    /// Extra lowercase substrings that should require approval in `ask` mode.
    pub interactive_requires_approval_for: Option<Vec<String>>,
    /// File edit mode: `ask`, `deny`, or `allow` (default `ask`).
    pub edit_mode: Option<String>,
    /// File edit mode for unattended/autonomous sessions (default `deny`).
    pub edit_unattended_default: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellPolicyMode {
    Ask,
    Deny,
    Allow,
}

#[derive(Debug, Clone)]
pub struct ResolvedShellPolicy {
    pub interactive_mode: ShellPolicyMode,
    pub unattended_mode: ShellPolicyMode,
    /// File mutation policy, intentionally independent from shell execution.
    pub interactive_edit_mode: ShellPolicyMode,
    pub unattended_edit_mode: ShellPolicyMode,
    pub approval_patterns: Vec<String>,
}

/// Async JSONL / webhook observation (`[harness.hooks.observation]`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct HarnessHooksObservationConfig {
    pub enabled: Option<bool>,
    /// Path relative to workspace root; append-only JSONL.
    pub jsonl_path: Option<String>,
    pub webhook_url: Option<String>,
    pub webhook_hmac_secret: Option<String>,
    /// Bounded queue before events are dropped (default 256).
    pub queue_capacity: Option<usize>,
    /// Inbound metadata keys copied into each envelope (`hook_metadata`).
    pub metadata_keys: Option<Vec<String>>,
}

/// Synchronous command hooks — JSON on stdin, JSON on stdout (see `docs/hooks.md`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct HarnessHooksSteeringConfig {
    pub enabled: Option<bool>,
    pub max_stdout_bytes: Option<usize>,
    pub default_timeout_ms: Option<u64>,
    pub pre_tool: Option<Vec<HarnessHookCommandConfig>>,
    pub post_tool: Option<Vec<HarnessHookCommandConfig>>,
    pub user_prompt: Option<Vec<HarnessHookCommandConfig>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HarnessHookCommandConfig {
    /// Regex matched against tool name; omit or empty = all tools.
    pub matcher: Option<String>,
    pub command: String,
    pub timeout_ms: Option<u64>,
    /// Sandbox-relative working directory for the hook subprocess.
    pub cwd: Option<String>,
}

/// Lifecycle hooks for observability and policy (`[harness.hooks]`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct HarnessHooksConfig {
    pub observation: Option<HarnessHooksObservationConfig>,
    pub steering: Option<HarnessHooksSteeringConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct BackgroundJobsConfig {
    pub enabled: Option<bool>,
    pub auto_resume: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct NotificationsConfig {
    pub enabled: Option<bool>,
}

/// MaxEvolve kernel porting (`kernel_db_*` tools). See `docs/kernel-porting-user-guide.md`.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct KernelPortingHarnessConfig {
    pub enabled: Option<bool>,
    /// Sandbox-relative root for kernel projects (default `kernels/projects`).
    pub default_project_root: Option<String>,
    /// Sandbox-relative JSON schema path for MAP-Elites archives.
    pub map_elites_schema: Option<String>,
    /// Max elite entries retained per project archive (default 500).
    pub max_archive_entries: Option<usize>,
    /// Default mutation batch size hint for evolve orchestrator (default 4).
    pub mutation_batch_size: Option<usize>,
}

/// AutoTrainess autonomous post-training (`train_db_*` tools). See `docs/autotrainess-user-guide.md`.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct AutoTrainessHarnessConfig {
    pub enabled: Option<bool>,
    /// Sandbox-relative root for training projects (default `train/projects`).
    pub default_project_root: Option<String>,
    /// Max iteration entries retained per project ledger (default 500).
    pub max_log_entries: Option<usize>,
}

/// Optional harness features (see `docs/harness-implementation-plan.md`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct HarnessConfig {
    pub git_worktree: Option<GitWorktreeConfig>,
    /// Background sub-agents, task tools, and optional plan execution (Phase 5).
    pub subagents: Option<SubagentHarnessConfig>,
    /// Triton→Pallas porting and MAP-Elites evolution tools.
    pub kernel_porting: Option<KernelPortingHarnessConfig>,
    /// AutoTrainess experiment ledger and post-training workflow tools.
    pub autotrainess: Option<AutoTrainessHarnessConfig>,
    /// Named agent definitions loaded from `[agents.<name>]` (Phase 5b).
    #[serde(default)]
    pub agents: std::collections::HashMap<String, AgentDefinition>,
    /// Shell command policy (`exec`), including approval-vs-deny behavior.
    pub shell_policy: Option<ShellPolicyConfig>,
    /// Local / future execution providers (`execution_*` tools). See `docs/execution-implementation-plan.md`.
    pub execution: Option<ExecutionHarnessConfig>,
    /// ML engineer prompt overlay and related defaults.
    pub ml_engineer: Option<MlEngineerHarnessConfig>,
    /// Observation + steering hooks (disabled unless sub-tables set `enabled = true`).
    pub hooks: Option<HarnessHooksConfig>,
    pub background_jobs: Option<BackgroundJobsConfig>,
    pub notifications: Option<NotificationsConfig>,
}

/// Git worktree helpers (`git_worktree` tool). Disabled unless `[harness.git_worktree] enabled = true`.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct GitWorktreeConfig {
    /// Register the `git_worktree` tool when true (default: false).
    pub enabled: Option<bool>,
    /// When false (default), worktree paths must satisfy the same sandbox boundary as other tools
    /// whenever `restrict_to_workspace` is true. When true, worktree paths may resolve outside the
    /// sandbox (e.g. a host temp directory) after canonicalization.
    pub allow_path_outside_sandbox: Option<bool>,
}

/// Best-effort workspace diagnostic logging. These limits apply only to the
/// inspectable `.system_generated/logs/` files; SQLite remains the durable
/// source of truth for conversations and run state.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct LoggingConfig {
    /// Disable file-backed diagnostic logging entirely. Defaults to enabled.
    pub enabled: Option<bool>,
    /// Active-file byte limit for `conversation.jsonl`.
    pub conversation_max_bytes: Option<u64>,
    /// Active-file byte limit for `runtime.log`.
    pub runtime_max_bytes: Option<u64>,
    /// Number of rotated files retained for each diagnostic log.
    pub retained_generations: Option<usize>,
    /// Aggregate byte cap for recognized diagnostic log files in one workspace.
    pub max_total_bytes: Option<u64>,
}

/// Fully bounded diagnostic-log settings consumed by the logging actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveLoggingConfig {
    pub enabled: bool,
    pub conversation_max_bytes: u64,
    pub runtime_max_bytes: u64,
    pub retained_generations: usize,
    pub max_total_bytes: u64,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct AppConfig {
    pub restrict_to_workspace: Option<bool>,
    pub provider: Option<ProviderConfig>,
    /// Named alternative provider configs for runtime switching via `/model`.
    /// Keys are short labels (e.g. `"openai"`, `"claude"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub providers: Option<std::collections::HashMap<String, ProviderConfig>>,
    pub api: Option<ApiConfig>,
    pub slack: Option<SlackConfig>,
    pub email: Option<EmailConfig>,
    pub terminal: Option<TerminalConfig>,
    pub max_iterations: Option<usize>,
    /// When true (default), detect repeated identical tool calls and inject a corrective user message.
    pub doom_loop_enabled: Option<bool>,
    /// When true, back up each file before `edit_file`/`write_file` mutates it and register the
    /// `checkpoint` tool for one-step undo. Default false. Only touched files are backed up.
    pub checkpoint_enabled: Option<bool>,
    pub max_tool_output_chars: Option<usize>,
    /// Max characters returned by `web_search` / `web_fetch` (default 50_000). Separate from
    /// `max_tool_output_chars`, which caps tool output when passed to the model.
    pub max_web_tool_output_chars: Option<usize>,
    /// Wall-clock limit in seconds for the `search_text` ripgrep subprocess (default 30, clamped 1–3600).
    pub search_text_ripgrep_timeout_secs: Option<u64>,
    pub memory: Option<MemoryConfig>,
    pub multi_tenant_edge: Option<MultiTenantEdgeConfig>,
    /// When `enabled`, `web_search` / `web_fetch` use [Jina Reader](https://r.jina.ai/) and search (`s.jina.ai`).
    pub jina: Option<JinaConfig>,
    /// Bounded, file-backed diagnostic logs under `.system_generated/logs/`.
    pub logging: Option<LoggingConfig>,
    pub harness: Option<HarnessConfig>,
    /// Named agent definitions (`[agents.<name>]` in config.toml). Top-level alias for
    /// `harness.agents` when the user keeps agents in the root of config.toml.
    #[serde(default)]
    pub agents: std::collections::HashMap<String, AgentDefinition>,
}

/// Optional Jina Reader / Search backend for web tools (see https://jina.ai/reader ).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct JinaConfig {
    pub enabled: Option<bool>,
    pub api_key: Option<String>,
}

/// Resolved settings when `[jina].enabled = true` (for wiring into tools).
#[derive(Clone, Debug, Default)]
pub struct JinaWebBackend {
    pub api_key: Option<String>,
}

/// Heuristics to avoid sending obvious template values as `Authorization: Bearer`.
/// Language-agnostic: rejects non-ASCII, angle-bracket templates (e.g. `<changethis>`), and
/// common placeholder tokens (ASCII substrings only).
fn api_key_looks_like_placeholder(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() || t.starts_with('<') {
        return true;
    }
    if !t.is_ascii() {
        return true;
    }
    let lower = t.to_ascii_lowercase();
    if lower == "changethis" {
        return true;
    }
    ["optional", "placeholder", "replace_me", "replaceme"]
        .iter()
        .any(|pat| lower.contains(pat))
}

fn parse_shell_policy_mode(raw: Option<&str>, default_mode: ShellPolicyMode) -> ShellPolicyMode {
    let Some(value) = raw else {
        return default_mode;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "allow" => ShellPolicyMode::Allow,
        "deny" => ShellPolicyMode::Deny,
        "ask" => ShellPolicyMode::Ask,
        _ => default_mode,
    }
}

impl AppConfig {
    pub fn background_jobs_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.background_jobs.as_ref())
            .and_then(|b| b.enabled)
            .unwrap_or(true)
    }

    pub fn background_jobs_auto_resume(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.background_jobs.as_ref())
            .and_then(|b| b.auto_resume)
            .unwrap_or(true)
    }

    pub fn notifications_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.notifications.as_ref())
            .and_then(|n| n.enabled)
            .unwrap_or(true)
    }

    /// Expand family-format `[providers.*]` entries into flat per-model configs.
    ///
    /// Family entries (with `models = [...]`) are expanded into one `ProviderConfig` per model,
    /// inheriting `api_key`, `api_key_env`, and `base_url` from the family. The map key is used
    /// as `provider_name` when the field is omitted.
    ///
    /// Legacy single-model entries (with `model_name`) pass through unchanged.
    pub fn expanded_providers(&self) -> std::collections::HashMap<String, ProviderConfig> {
        let mut result = std::collections::HashMap::new();
        let providers = match &self.providers {
            Some(p) => p,
            None => return result,
        };
        let mut sorted_keys: Vec<_> = providers.keys().collect();
        sorted_keys.sort();
        for key in sorted_keys {
            let cfg = &providers[key];
            let provider_name = if cfg.provider_name.is_empty() {
                key.clone()
            } else {
                cfg.provider_name.clone()
            };

            if let Some(models) = &cfg.models {
                for model in models {
                    let expanded = ProviderConfig {
                        provider_name: provider_name.clone(),
                        model_name: model.clone(),
                        models: None,
                        api_key_env: cfg.api_key_env.clone(),
                        api_key: cfg.api_key.clone(),
                        base_url: cfg.base_url.clone(),
                    };
                    if result.contains_key(model) {
                        warn!(
                            "Duplicate model name \"{}\" in [providers.{}] -- \
                             overwriting previous entry. Use the legacy per-model \
                             format with unique map keys to distinguish the same \
                             model from different providers.",
                            model, key
                        );
                    }
                    result.insert(model.clone(), expanded);
                }
            } else if !cfg.model_name.is_empty() {
                let mut single = cfg.clone();
                if single.provider_name.is_empty() {
                    single.provider_name = provider_name;
                }
                single.models = None;
                if result.contains_key(key) {
                    warn!(
                        "Duplicate provider key \"{}\" in [providers] -- \
                         overwriting previous entry.",
                        key
                    );
                }
                result.insert(key.clone(), single);
            } else {
                warn!(
                    "[providers.{}] has neither \"models\" nor \
                     \"model_name\" -- entry is ignored. Add one to \
                     register a provider.",
                    key
                );
            }
        }
        result
    }

    /// Whether the stdin/stdout terminal channel is active (`[terminal].enabled`, default `true`).
    pub fn terminal_enabled(&self) -> bool {
        self.terminal
            .as_ref()
            .and_then(|t| t.enabled)
            .unwrap_or(true)
    }

    /// Root `max_iterations`, or `[terminal].max_iterations` when root is unset (common TOML mistake).
    pub fn resolved_max_iterations(&self) -> Option<usize> {
        self.max_iterations
            .or_else(|| self.terminal.as_ref().and_then(|t| t.max_iterations))
    }

    /// Root `max_tool_output_chars`, or `[terminal].max_tool_output_chars` when root is unset.
    pub fn resolved_max_tool_output_chars(&self) -> Option<usize> {
        self.max_tool_output_chars
            .or_else(|| self.terminal.as_ref().and_then(|t| t.max_tool_output_chars))
    }

    /// Resolved bounds for file-backed diagnostic logs. Invalid or missing
    /// values always resolve to bounded settings; they can never enable
    /// unbounded workspace log growth.
    pub fn effective_logging_config(&self) -> EffectiveLoggingConfig {
        const DEFAULT_CONVERSATION_MAX_BYTES: u64 = 20 * 1024 * 1024;
        const DEFAULT_RUNTIME_MAX_BYTES: u64 = 10 * 1024 * 1024;
        const DEFAULT_RETAINED_GENERATIONS: usize = 2;
        const DEFAULT_TOTAL_MAX_BYTES: u64 = 90 * 1024 * 1024;
        const MIN_ACTIVE_FILE_BYTES: u64 = 256;
        const MAX_ACTIVE_FILE_BYTES: u64 = 512 * 1024 * 1024;
        const MAX_RETAINED_GENERATIONS: usize = 32;
        const MAX_TOTAL_BYTES: u64 = 1024 * 1024 * 1024;

        let logging = self.logging.as_ref();
        let bounded_bytes = |configured: Option<u64>, default: u64| {
            configured
                .filter(|bytes| *bytes >= MIN_ACTIVE_FILE_BYTES)
                .map(|bytes| bytes.min(MAX_ACTIVE_FILE_BYTES))
                .unwrap_or(default)
        };
        let conversation_max_bytes = bounded_bytes(
            logging.and_then(|config| config.conversation_max_bytes),
            DEFAULT_CONVERSATION_MAX_BYTES,
        );
        let runtime_max_bytes = bounded_bytes(
            logging.and_then(|config| config.runtime_max_bytes),
            DEFAULT_RUNTIME_MAX_BYTES,
        );
        let retained_generations = logging
            .and_then(|config| config.retained_generations)
            .unwrap_or(DEFAULT_RETAINED_GENERATIONS)
            .min(MAX_RETAINED_GENERATIONS);
        let minimum_total = conversation_max_bytes.saturating_add(runtime_max_bytes);
        let default_total = DEFAULT_TOTAL_MAX_BYTES.max(minimum_total);
        let max_total_bytes = logging
            .and_then(|config| config.max_total_bytes)
            .filter(|bytes| *bytes >= minimum_total)
            .map(|bytes| bytes.min(MAX_TOTAL_BYTES))
            .filter(|bytes| *bytes >= minimum_total)
            .unwrap_or(default_total);

        EffectiveLoggingConfig {
            enabled: logging.and_then(|config| config.enabled).unwrap_or(true),
            conversation_max_bytes,
            runtime_max_bytes,
            retained_generations,
            max_total_bytes,
        }
    }

    /// At least one inbound channel other than terminal (API, Slack, or Email).
    pub fn has_non_terminal_inbound_channel(&self) -> bool {
        let api_on = self
            .api
            .as_ref()
            .is_some_and(|a| a.enabled.unwrap_or(false));
        let slack_on = self
            .slack
            .as_ref()
            .is_some_and(|s| s.enabled.unwrap_or(false));
        let email_on = self
            .email
            .as_ref()
            .is_some_and(|e| e.enabled.unwrap_or(false));
        api_on || slack_on || email_on
    }

    /// Returns `Some` when `[jina].enabled` is true so tools should call r.jina.ai / s.jina.ai.
    pub fn jina_web_backend(&self) -> Option<JinaWebBackend> {
        let j = self.jina.as_ref()?;
        if !j.enabled.unwrap_or(false) {
            return None;
        }
        let api_key = j
            .api_key
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !api_key_looks_like_placeholder(s));
        Some(JinaWebBackend { api_key })
    }

    /// Upper bound for `web_search` / `web_fetch` response bodies.
    pub fn effective_max_web_tool_output_chars(&self) -> usize {
        const DEFAULT: usize = 50_000;
        self.max_web_tool_output_chars.unwrap_or(DEFAULT)
    }

    /// Timeout for `search_text` when using ripgrep (workspace default; per-call override in tool args).
    pub fn effective_search_text_ripgrep_timeout_secs(&self) -> u64 {
        const DEFAULT: u64 = 30;
        const MIN: u64 = 1;
        const MAX: u64 = 3600;
        self.search_text_ripgrep_timeout_secs
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    /// ml-intern-style doom loop detection before each LLM call (default: enabled).
    pub fn doom_loop_enabled(&self) -> bool {
        self.doom_loop_enabled.unwrap_or(true)
    }

    /// Pre-edit file checkpointing for one-step undo (default: disabled).
    pub fn checkpoint_enabled(&self) -> bool {
        self.checkpoint_enabled.unwrap_or(false)
    }

    /// When true, `git_worktree` is registered (see `[harness.git_worktree]` in config).
    pub fn git_worktree_tool_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.git_worktree.as_ref())
            .and_then(|g| g.enabled)
            .unwrap_or(false)
    }

    /// When true with `git_worktree_tool_enabled`, worktree paths may lie outside the sandbox.
    pub fn git_worktree_allow_path_outside_sandbox(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.git_worktree.as_ref())
            .and_then(|g| g.allow_path_outside_sandbox)
            .unwrap_or(false)
    }

    /// `[harness.subagents] enabled = true` registers task / spawn / plan tools.
    pub fn subagent_harness_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.enabled)
            .unwrap_or(false)
    }

    /// Default true: parent chat cancel also cancels that chat's sub-agent tasks.
    pub fn subagent_cancel_children_on_parent_cancel(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.cancel_children_on_parent_cancel)
            .unwrap_or(true)
    }

    pub fn subagent_allowed_tools_set(&self) -> Option<std::collections::HashSet<String>> {
        let v = self
            .harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.allowed_tools.as_ref())?;
        if v.is_empty() {
            return None;
        }
        Some(v.iter().cloned().collect())
    }

    pub fn subagent_max_tasks(&self) -> usize {
        const DEFAULT: usize = 32;
        const MIN: usize = 1;
        const MAX: usize = 256;
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.max_tasks)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn subagent_max_wait_secs(&self) -> u64 {
        const DEFAULT: u64 = 300;
        const MIN: u64 = 10;
        const MAX: u64 = 3600;
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.max_wait_secs)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    /// When true (default), auto-enqueue a synthetic inbound when a subagent finishes.
    pub fn subagent_wake_on_completion(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.wake_on_completion)
            .unwrap_or(true)
    }

    /// Max completed tasks retained in SQLite per parent chat (default 200).
    pub fn subagent_task_history_retention(&self) -> usize {
        const DEFAULT: usize = 200;
        const MIN: usize = 10;
        const MAX: usize = 2000;
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.task_history_retention)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    /// Returns merged agent definitions from `[agents.<name>]` and `[harness.agents.<name>]`.
    /// Harness-level definitions override top-level ones of the same name.
    pub fn agent_definitions(&self) -> std::collections::HashMap<String, AgentDefinition> {
        let mut merged = self.agents.clone();
        if let Some(h) = self.harness.as_ref() {
            for (k, v) in &h.agents {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }

    /// When false under `[harness.execution]`, execution tools are not registered. Otherwise on (including when the table is omitted).
    pub fn execution_harness_enabled(&self) -> bool {
        match self.harness.as_ref().and_then(|h| h.execution.as_ref()) {
            None => true,
            Some(e) => e.enabled.unwrap_or(true),
        }
    }

    /// When true under `[harness.kernel_porting]`, `kernel_db_*` tools are registered.
    pub fn kernel_porting_harness_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.kernel_porting.as_ref())
            .and_then(|k| k.enabled)
            .unwrap_or(false)
    }

    pub fn kernel_porting_default_project_root(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.kernel_porting.as_ref())
            .and_then(|k| k.default_project_root.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "kernels/projects".to_string())
    }

    pub fn kernel_porting_map_elites_schema(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.kernel_porting.as_ref())
            .and_then(|k| k.map_elites_schema.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| ".agents/kernel-porting/map_elites.schema.json".to_string())
    }

    pub fn kernel_porting_max_archive_entries(&self) -> usize {
        self.harness
            .as_ref()
            .and_then(|h| h.kernel_porting.as_ref())
            .and_then(|k| k.max_archive_entries)
            .unwrap_or(500)
            .clamp(10, 10_000)
    }

    pub fn kernel_porting_mutation_batch_size(&self) -> usize {
        self.harness
            .as_ref()
            .and_then(|h| h.kernel_porting.as_ref())
            .and_then(|k| k.mutation_batch_size)
            .unwrap_or(4)
            .clamp(1, 64)
    }

    /// When true under `[harness.autotrainess]`, `train_db_*` tools are registered.
    pub fn autotrainess_harness_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.autotrainess.as_ref())
            .and_then(|a| a.enabled)
            .unwrap_or(false)
    }

    pub fn autotrainess_default_project_root(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.autotrainess.as_ref())
            .and_then(|a| a.default_project_root.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "train/projects".to_string())
    }

    pub fn autotrainess_max_log_entries(&self) -> usize {
        self.harness
            .as_ref()
            .and_then(|h| h.autotrainess.as_ref())
            .and_then(|a| a.max_log_entries)
            .unwrap_or(500)
            .clamp(10, 10_000)
    }

    /// `[harness.ml_engineer] enabled = true` appends ML policy overlay to the system prompt.
    pub fn ml_engineer_harness_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.ml_engineer.as_ref())
            .and_then(|m| m.enabled)
            .unwrap_or(false)
    }

    /// When ML harness is on, append research instructions to sub-agent system prompts (default true).
    pub fn ml_engineer_subagent_research_overlay(&self) -> bool {
        if !self.ml_engineer_harness_enabled() {
            return false;
        }
        self.harness
            .as_ref()
            .and_then(|h| h.ml_engineer.as_ref())
            .and_then(|m| m.subagent_research_overlay)
            .unwrap_or(true)
    }

    /// Config default for forbidding a final assistant message with no tool calls (overridable per inbound metadata).
    pub fn ml_engineer_forbid_final_without_tools(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.ml_engineer.as_ref())
            .and_then(|m| m.forbid_final_without_tools)
            .unwrap_or(false)
    }

    pub fn resolved_shell_policy(&self) -> ResolvedShellPolicy {
        let shell_cfg = self.harness.as_ref().and_then(|h| h.shell_policy.as_ref());
        let interactive_mode = parse_shell_policy_mode(
            shell_cfg.and_then(|s| s.mode.as_deref()),
            ShellPolicyMode::Ask,
        );
        let unattended_mode = parse_shell_policy_mode(
            shell_cfg.and_then(|s| s.unattended_default.as_deref()),
            ShellPolicyMode::Deny,
        );
        let interactive_edit_mode = parse_shell_policy_mode(
            shell_cfg.and_then(|s| s.edit_mode.as_deref()),
            ShellPolicyMode::Ask,
        );
        let unattended_edit_mode = parse_shell_policy_mode(
            shell_cfg.and_then(|s| s.edit_unattended_default.as_deref()),
            ShellPolicyMode::Deny,
        );
        let mut approval_patterns = vec![
            "rm -rf".to_string(),
            "rm -fr".to_string(),
            "del /f".to_string(),
            "del /q".to_string(),
            "rmdir /s".to_string(),
            "git clean -fd".to_string(),
            "git reset --hard".to_string(),
        ];
        if let Some(extra) = shell_cfg
            .and_then(|s| s.interactive_requires_approval_for.as_ref())
            .filter(|v| !v.is_empty())
        {
            approval_patterns.extend(
                extra
                    .iter()
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty()),
            );
        }
        approval_patterns.sort();
        approval_patterns.dedup();
        ResolvedShellPolicy {
            interactive_mode,
            unattended_mode,
            interactive_edit_mode,
            unattended_edit_mode,
            approval_patterns,
        }
    }

    /// Short lines for `[RUNTIME CONTEXT]` (token-frugal). Built in the binary and passed into the agent.
    pub fn runtime_harness_summary_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let shell_policy = self.resolved_shell_policy();
        let os_family = std::env::consts::OS;
        let shell_family = if cfg!(windows) {
            "powershell_or_cmd"
        } else {
            "sh_or_bash"
        };
        lines.push(format!(
            "host_os={} shell_family={} path_separator={}",
            os_family,
            shell_family,
            std::path::MAIN_SEPARATOR
        ));
        lines.push(format!(
            "execution_harness_enabled={}",
            self.execution_harness_enabled()
        ));
        if self.execution_harness_enabled() {
            lines.push(format!(
                "execution_default_provider={}",
                self.execution_default_provider()
            ));
            lines.push(format!(
                "execution_max_wall_secs={}",
                self.execution_max_wall_secs()
            ));
            lines.push(format!(
                "execution_default_run_timeout_secs={}",
                self.execution_default_run_timeout_secs()
            ));
            lines.push(format!(
                "execution_auto_promote_after_secs={}",
                self.execution_auto_promote_after_secs()
            ));
            lines.push(format!(
                "execution_max_output_bytes={}",
                self.execution_max_output_bytes()
            ));
            lines.push(format!(
                "execution_artifact_caps=file:{} total_per_run:{} max_files:{}",
                self.execution_artifact_max_file_bytes(),
                self.execution_artifact_max_total_bytes_per_run(),
                self.execution_artifact_max_files_per_run()
            ));
            lines.push(format!(
                "execution_wake_on_job_terminal={}",
                self.execution_wake_on_job_terminal()
            ));
        }
        lines.push(format!(
            "subagent_harness_enabled={}",
            self.subagent_harness_enabled()
        ));
        if self.subagent_harness_enabled() {
            lines.push(format!("subagent_max_tasks={}", self.subagent_max_tasks()));
            lines.push(format!(
                "subagent_max_wait_secs={}",
                self.subagent_max_wait_secs()
            ));
            lines.push(format!(
                "subagent_wake_on_completion={}",
                self.subagent_wake_on_completion()
            ));
            let allow = self.subagent_allowed_tools_set();
            lines.push(format!(
                "subagent_allowlist_active={} (count={})",
                allow.is_some(),
                allow.map(|s| s.len()).unwrap_or(0)
            ));
            let agent_count = self.agent_definitions().len();
            lines.push(format!("named_agents={}", agent_count));
        }
        lines.push(format!(
            "ml_engineer_harness_enabled={}",
            self.ml_engineer_harness_enabled()
        ));
        lines.push(format!(
            "ml_engineer_forbid_final_without_tools_default={}",
            self.ml_engineer_forbid_final_without_tools()
        ));
        lines.push(format!(
            "shell_policy_mode_interactive={:?} shell_policy_mode_unattended={:?} shell_policy_approval_patterns={}",
            shell_policy.interactive_mode,
            shell_policy.unattended_mode,
            shell_policy.approval_patterns.len()
        ));
        let hooks_obs = self
            .harness
            .as_ref()
            .and_then(|h| h.hooks.as_ref())
            .and_then(|x| x.observation.as_ref())
            .is_some_and(|o| o.enabled.unwrap_or(false));
        let hooks_steer = self
            .harness
            .as_ref()
            .and_then(|h| h.hooks.as_ref())
            .and_then(|x| x.steering.as_ref())
            .is_some_and(|s| s.enabled.unwrap_or(false));
        lines.push(format!(
            "hooks_observation_enabled={} hooks_steering_enabled={}",
            hooks_obs, hooks_steer
        ));
        lines
    }

    pub fn execution_default_provider(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.default_provider.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "local".to_string())
    }

    /// True iff the user explicitly set `[harness.execution].default_provider` to a non-empty
    /// string. Used by `build_execution_harness` to decide between auto-pick (implicit fallback
    /// happened to be misconfigured) and hard-fail (user pinned a now-pruned provider).
    pub fn execution_default_provider_explicit(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.default_provider.as_ref())
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
    }

    /// Returns the user-configured allowed-providers list (trimmed, non-empty entries).
    /// `None` means "no restriction" (all implemented providers may be tried).
    pub fn execution_allowed_providers(&self) -> Option<Vec<String>> {
        let raw = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.allowed_providers.as_ref())?;
        let cleaned: Vec<String> = raw
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if cleaned.is_empty() {
            None
        } else {
            Some(cleaned)
        }
    }

    pub fn execution_max_output_bytes(&self) -> usize {
        const DEFAULT: usize = 256 * 1024;
        const MIN: usize = 4096;
        const MAX: usize = 16 * 1024 * 1024;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.max_output_bytes)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn execution_max_wall_secs(&self) -> u64 {
        const DEFAULT: u64 = 3600;
        const MIN: u64 = 1;
        const MAX: u64 = 86400;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.max_wall_secs)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    /// Default per-run wall clock when the model omits `timeout_secs` (1..=max_wall_secs).
    pub fn execution_default_run_timeout_secs(&self) -> u64 {
        let cap = self.execution_max_wall_secs();
        const FALLBACK: u64 = 600;
        let v = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.default_execution_timeout_secs)
            .unwrap_or(FALLBACK);
        v.clamp(1, cap)
    }

    /// Short bound after which a synchronous run auto-promotes to a background job
    /// (`execution_run`).
    ///
    /// Returns `0` when auto-promote is explicitly disabled (`auto_promote_after_secs = 0`).
    /// Otherwise returns the configured value clamped to `5..=max_wall_secs`, defaulting to
    /// `min(120, max_wall_secs)` when unset.
    pub fn execution_auto_promote_after_secs(&self) -> u64 {
        const FALLBACK: u64 = 120;
        const MIN_NONZERO: u64 = 5;
        let cap = self.execution_max_wall_secs();
        let raw = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.auto_promote_after_secs);
        match raw {
            Some(0) => 0,
            Some(v) => v.clamp(MIN_NONZERO, cap),
            None => FALLBACK.min(cap).max(MIN_NONZERO.min(cap)),
        }
    }

    pub fn execution_max_sessions(&self) -> usize {
        const DEFAULT: usize = 32;
        const MIN: usize = 1;
        const MAX: usize = 256;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.max_sessions)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn execution_python_executable(&self) -> String {
        self.execution_python_executable_configured()
            .unwrap_or_else(|| "python".to_string())
    }

    /// Raw configured python executable (trimmed), if set.
    pub fn execution_python_executable_configured(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.python_executable.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Local provider Python: persistent REPL per session (default) vs one subprocess per run.
    pub fn execution_local_python_repl_enabled(&self) -> bool {
        let raw = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.local_python_mode.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        match raw {
            None => true,
            Some(s) => {
                let lower = s.to_ascii_lowercase();
                !matches!(
                    lower.as_str(),
                    "subprocess"
                        | "fresh"
                        | "stateless"
                        | "one_shot"
                        | "oneshot"
                        | "no"
                        | "false"
                        | "0"
                )
            }
        }
    }

    /// Local provider Python runtime backend.
    pub fn execution_local_python_runtime(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.local_python_runtime.as_ref())
            .map(|s| s.trim().to_ascii_lowercase().replace('-', "_"))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "uv_managed".to_string())
    }

    /// Binary for UV-managed local Python runtime.
    pub fn execution_uv_binary(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.uv_binary.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "uv".to_string())
    }

    /// Requested Python version for UV-managed local Python runtime.
    pub fn execution_uv_python(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.uv_python.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "3.11".to_string())
    }

    /// Optional package specs installed once for UV-managed local runtime.
    pub fn execution_uv_requirements(&self) -> Vec<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.uv_requirements.as_ref())
            .map(|items| {
                items
                    .iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    /// Caps for Phase 6 execution artifacts (Jupyter `display_data` materialization).
    pub fn execution_artifact_max_file_bytes(&self) -> usize {
        const DEFAULT: usize = 4 * 1024 * 1024;
        const MIN: usize = 64 * 1024;
        const MAX: usize = 64 * 1024 * 1024;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.artifact_max_file_bytes)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn execution_artifact_max_total_bytes_per_run(&self) -> usize {
        const DEFAULT: usize = 32 * 1024 * 1024;
        const MIN: usize = 256 * 1024;
        const MAX: usize = 128 * 1024 * 1024;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.artifact_max_total_bytes_per_run)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn execution_artifact_max_files_per_run(&self) -> usize {
        const DEFAULT: usize = 64;
        const MIN: usize = 1;
        const MAX: usize = 256;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.artifact_max_files_per_run)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    /// Default true: when `[harness.execution] wake_on_job_terminal` is omitted or true, background jobs enqueue a synthetic inbound at terminal state.
    pub fn execution_wake_on_job_terminal(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.wake_on_job_terminal)
            .unwrap_or(true)
    }

    /// When `allowed_providers` is missing or empty, any implemented provider id is allowed.
    pub fn execution_provider_allowed(&self, provider_id: &str) -> bool {
        let Some(list) = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.allowed_providers.as_ref())
        else {
            return true;
        };
        if list.is_empty() {
            return true;
        }
        list.iter().any(|s| s == provider_id)
    }

    /// `JUPYTER_TOKEN` env wins over `[harness.execution.jupyter].token`.
    pub fn execution_jupyter_token(&self) -> Option<String> {
        let from_env = std::env::var("JUPYTER_TOKEN")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        if from_env.is_some() {
            return from_env;
        }
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.jupyter.as_ref())
            .and_then(|j| j.token.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_jupyter_base_url(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.jupyter.as_ref())
            .and_then(|j| j.base_url.as_ref())
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_jupyter_kernel_name(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.jupyter.as_ref())
            .and_then(|j| j.kernel_name.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "python3".to_string())
    }

    pub fn execution_jupyter_notebook_sync_path_template(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.jupyter.as_ref())
            .and_then(|j| j.notebook_sync_path_template.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_ssh_host(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.host.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_ssh_port(&self) -> u16 {
        const DEFAULT: u16 = 22;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.port)
            .unwrap_or(DEFAULT)
    }

    pub fn execution_ssh_user(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.user.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Expanded filesystem path to a private key, when configured.
    pub fn execution_ssh_identity_file(&self) -> Option<String> {
        let raw = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.identity_file.as_ref())?;
        let t = raw.trim();
        if t.is_empty() {
            return None;
        }
        let expanded = shellexpand::tilde(t).into_owned();
        if expanded.trim().is_empty() {
            return None;
        }
        Some(expanded)
    }

    pub fn execution_ssh_remote_workdir(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.remote_workdir.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_ssh_remote_python(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.remote_python.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "python3".to_string())
    }

    pub fn execution_ssh_accept_unknown_host_keys(&self) -> bool {
        // Secure by default: verify host keys via the trust-on-first-use known_hosts store.
        // Opt in to the insecure "accept any key" behavior only by explicitly setting true.
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.accept_unknown_host_keys)
            .unwrap_or(false)
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct MemoryConfig {
    pub enabled: Option<bool>,
    pub short_term_threshold_turns: Option<usize>,
    pub short_term_threshold_tokens: Option<usize>,
    pub short_term_threshold_mins: Option<u64>,
    pub long_term_interval_mins: Option<u64>,
    pub max_recent_summaries: Option<usize>,
    pub long_term_threshold_summaries: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct MultiTenantEdgeConfig {
    pub activity_heartbeat_enabled: Option<bool>,
    pub cron_scheduling_enabled: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ApiConfig {
    pub enabled: Option<bool>,
    pub port: u16,
    pub serve_ui: Option<bool>,
    pub bind_address: Option<String>,
    /// Bearer token required on the `/v1` control surface when set. Also loadable via the
    /// `ISANAGENT_API_TOKEN` env var (config value wins). REQUIRED to bind a non-loopback
    /// address — the API exposes chat control and workspace file read/write with no other guard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_token: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ProviderConfig {
    /// One of `KNOWN_PROVIDERS` (e.g. `"gemini"`, `"openai"`, `"deepseek"`, `"openrouter"`,
    /// `"anthropic"`) or the `OPENAI_COMPATIBLE` sentinel for any third-party endpoint speaking
    /// the OpenAI Chat Completions protocol.
    /// In the family format, this is inferred from the map key when omitted.
    #[serde(default)]
    pub provider_name: String,
    /// Single model name (legacy per-model format).
    #[serde(default)]
    pub model_name: String,
    /// Multiple model names (family format). When present, the entry is expanded into one
    /// `ProviderConfig` per model by [`AppConfig::expanded_providers`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    /// Name of the environment variable holding the API key. Checked first during resolution.
    #[serde(default)]
    pub api_key_env: String,
    /// Direct API key value. Used as fallback when the env var named by `api_key_env` is not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Optional explicit chat-completions URL. Always wins when set, including for known
    /// provider names (lets users point a known provider at a proxy / Azure-OpenAI / self-hosted
    /// gateway). Required when `provider_name == "openai_compatible"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

impl ProviderConfig {
    /// Resolve the API key env var name. If `api_key_env` is explicitly set, use that.
    /// Otherwise, infer from `provider_name` (e.g. "openai" → "OPENAI_API_KEY").
    pub fn resolved_api_key_env(&self) -> String {
        if !self.api_key_env.is_empty() {
            return self.api_key_env.clone();
        }
        // Infer from provider_name
        format!("{}_API_KEY", self.provider_name.to_uppercase())
    }

    /// Resolve the API key: env var (from `resolved_api_key_env()`) first, then the inline
    /// `api_key` field in config.toml. Returns `Err` when neither source provides a non-empty key.
    pub fn resolve_api_key(&self) -> Result<String, String> {
        let env_var = self.resolved_api_key_env();
        if !env_var.is_empty() {
            if let Ok(key) = std::env::var(&env_var) {
                if !key.is_empty() {
                    return Ok(key);
                }
            }
        }
        if let Some(key) = &self.api_key {
            let trimmed = key.trim();
            if !trimmed.is_empty() && !api_key_looks_like_placeholder(trimmed) {
                return Ok(trimmed.to_string());
            }
        }
        Err(format!(
            "No API key found (checked env ${} and config api_key)",
            env_var
        ))
    }

    /// Resolve the chat-completions URL using the registry-then-override rules described on
    /// [`ProviderConfig::base_url`].
    ///
    /// Errors:
    /// - unknown `provider_name` (not in `KNOWN_PROVIDERS` and not `OPENAI_COMPATIBLE`)
    /// - `provider_name == OPENAI_COMPATIBLE` with no `base_url` set
    pub fn resolved_base_url(&self) -> Result<String, String> {
        if let Some(url) = self
            .base_url
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return Ok(url.to_string());
        }
        if let Some(url) = crate::provider_registry::lookup(self.provider_name.as_str()) {
            return Ok(url.to_string());
        }
        if self.provider_name == crate::provider_registry::OPENAI_COMPATIBLE {
            return Err(
                "[provider] provider_name = \"openai_compatible\" requires base_url".to_string(),
            );
        }
        let mut allowed = crate::provider_registry::known_names();
        allowed.push(crate::provider_registry::OPENAI_COMPATIBLE);
        Err(format!(
            "[provider] unknown provider_name '{}'; expected one of [{}]",
            self.provider_name,
            allowed.join(", ")
        ))
    }
}

#[cfg(test)]
mod provider_config_tests {
    use super::ProviderConfig;

    fn cfg(name: &str, base: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            provider_name: name.to_string(),
            model_name: "m".to_string(),
            models: None,
            api_key_env: "X".to_string(),
            api_key: None,
            base_url: base.map(str::to_string),
        }
    }

    #[test]
    fn resolves_known_name_from_registry() {
        let url = cfg("gemini", None).resolved_base_url().unwrap();
        assert!(
            url.contains("generativelanguage.googleapis.com"),
            "got {url}"
        );
    }

    #[test]
    fn explicit_base_url_overrides_known() {
        let url = cfg(
            "openai",
            Some("https://relay.example.com/v1/chat/completions"),
        )
        .resolved_base_url()
        .unwrap();
        assert_eq!(url, "https://relay.example.com/v1/chat/completions");
    }

    #[test]
    fn openai_compatible_requires_base_url() {
        let err = cfg("openai_compatible", None)
            .resolved_base_url()
            .unwrap_err();
        assert!(err.contains("openai_compatible"), "got {err}");
        assert!(err.contains("base_url"), "got {err}");

        let url = cfg(
            "openai_compatible",
            Some("https://my.host/v1/chat/completions"),
        )
        .resolved_base_url()
        .unwrap();
        assert_eq!(url, "https://my.host/v1/chat/completions");
    }

    #[test]
    fn unknown_provider_name_errors() {
        let err = cfg("totally-bogus", None).resolved_base_url().unwrap_err();
        assert!(err.contains("unknown provider_name"), "got {err}");
        assert!(err.contains("openai_compatible"), "got {err}");
    }

    #[test]
    fn empty_base_url_string_is_treated_as_unset() {
        let url = cfg("gemini", Some("   ")).resolved_base_url().unwrap();
        assert!(url.contains("generativelanguage.googleapis.com"));
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
#[clap(rename_all = "lower")]
pub enum SlackMode {
    #[default]
    Webhook,
    Socket,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SlackConfig {
    pub enabled: Option<bool>,
    pub mode: Option<SlackMode>,
    pub app_token: Option<String>,
    pub bot_token: String,
    pub signing_secret: Option<String>,
    pub webhook_port: Option<u16>,
    pub webhook_path: Option<String>,
    pub reply_in_thread: Option<bool>,
    pub reaction_emoji: Option<String>,
}

impl SlackConfig {
    pub fn mode(&self) -> SlackMode {
        self.mode.unwrap_or_default()
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmailConfig {
    pub enabled: Option<bool>,
    pub imap_host: String,
    pub imap_port: u16,
    pub imap_username: String,
    pub imap_password: String,
    pub smtp_host: String,
    pub smtp_port: u16,
    pub email_address: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logging_config_uses_bounded_defaults() {
        let config: AppConfig = toml::from_str("").expect("parse empty config");

        assert_eq!(
            config.effective_logging_config(),
            EffectiveLoggingConfig {
                enabled: true,
                conversation_max_bytes: 20 * 1024 * 1024,
                runtime_max_bytes: 10 * 1024 * 1024,
                retained_generations: 2,
                max_total_bytes: 90 * 1024 * 1024,
            }
        );
    }

    #[test]
    fn logging_config_parses_explicit_bounded_values() {
        let config: AppConfig = toml::from_str(
            r#"
[logging]
enabled = false
conversation_max_bytes = 1024
runtime_max_bytes = 2048
retained_generations = 3
max_total_bytes = 4096
"#,
        )
        .expect("parse logging config");

        assert_eq!(
            config.effective_logging_config(),
            EffectiveLoggingConfig {
                enabled: false,
                conversation_max_bytes: 1024,
                runtime_max_bytes: 2048,
                retained_generations: 3,
                max_total_bytes: 4096,
            }
        );
    }

    #[test]
    fn logging_config_invalid_values_stay_bounded() {
        let config: AppConfig = toml::from_str(
            r#"
[logging]
conversation_max_bytes = 0
runtime_max_bytes = 999999999999
retained_generations = 999
max_total_bytes = 1
"#,
        )
        .expect("parse logging config");
        let effective = config.effective_logging_config();

        assert_eq!(effective.conversation_max_bytes, 20 * 1024 * 1024);
        assert_eq!(effective.runtime_max_bytes, 512 * 1024 * 1024);
        assert_eq!(effective.retained_generations, 32);
        assert_eq!(effective.max_total_bytes, 532 * 1024 * 1024);
    }

    #[test]
    fn logging_config_rejects_integer_overflow() {
        let parsed = toml::from_str::<AppConfig>(
            r#"
[logging]
conversation_max_bytes = 18446744073709551616
"#,
        );

        assert!(parsed.is_err());
    }

    #[test]
    fn harness_execution_toml_roundtrip() {
        let s = r#"
[harness.execution]
enabled = true
default_provider = "local"
max_wall_secs = 90
max_output_bytes = 8192
allowed_providers = ["local"]
python_executable = "python3"
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(c.execution_harness_enabled());
        assert_eq!(c.execution_max_wall_secs(), 90);
        assert_eq!(c.execution_default_run_timeout_secs(), 90);
        assert_eq!(c.execution_max_output_bytes(), 8192);
        assert!(c.execution_provider_allowed("local"));
        assert!(!c.execution_provider_allowed("jupyter"));
        assert_eq!(c.execution_python_executable(), "python3");
    }

    #[test]
    fn harness_execution_on_by_default_without_harness_section() {
        let s = "restrict_to_workspace = true\n";
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(c.execution_harness_enabled());
        assert_eq!(c.execution_default_provider(), "local");
    }

    #[test]
    fn harness_execution_explicit_disabled() {
        let s = r#"
restrict_to_workspace = true
[harness.execution]
enabled = false
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(!c.execution_harness_enabled());
    }

    #[test]
    fn harness_execution_defaults_when_only_enabled() {
        let s = r#"
[harness.execution]
enabled = true
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_max_wall_secs(), 3600);
        assert_eq!(c.execution_default_run_timeout_secs(), 600);
        assert_eq!(c.execution_local_python_runtime(), "uv_managed");
        assert_eq!(c.execution_default_provider(), "local");
    }

    #[test]
    fn harness_execution_default_run_timeout_respects_max_wall() {
        let s = r#"
[harness.execution]
enabled = true
max_wall_secs = 120
default_execution_timeout_secs = 9999
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_default_run_timeout_secs(), 120);
        let s2 = r#"
[harness.execution]
enabled = true
max_wall_secs = 600
default_execution_timeout_secs = 45
"#;
        let c2: AppConfig = toml::from_str(s2).expect("parse");
        assert_eq!(c2.execution_default_run_timeout_secs(), 45);
    }

    #[test]
    fn harness_execution_artifact_limits_toml() {
        let s = r#"
[harness.execution]
enabled = true
artifact_max_file_bytes = 100000
artifact_max_total_bytes_per_run = 500000
artifact_max_files_per_run = 10
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_artifact_max_file_bytes(), 100_000);
        assert_eq!(c.execution_artifact_max_total_bytes_per_run(), 500_000);
        assert_eq!(c.execution_artifact_max_files_per_run(), 10);
    }

    #[test]
    fn doom_loop_enabled_toml() {
        let s = r#"doom_loop_enabled = false"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(!c.doom_loop_enabled());
        assert!(AppConfig::default().doom_loop_enabled());
    }

    #[test]
    fn resolved_max_iterations_falls_back_to_terminal_table() {
        let s = r#"
[terminal]
enabled = true
max_iterations = 999
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.resolved_max_iterations(), Some(999));
    }

    #[test]
    fn resolved_max_iterations_root_wins_over_terminal() {
        let s = r#"
max_iterations = 12
[terminal]
max_iterations = 999
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.resolved_max_iterations(), Some(12));
    }

    #[test]
    fn harness_execution_jupyter_toml() {
        let s = r#"
[harness.execution]
enabled = true
default_provider = "jupyter"
allowed_providers = ["jupyter", "local"]

[harness.execution.jupyter]
base_url = "http://127.0.0.1:8888"
token = "testtoken"
kernel_name = "python3"
notebook_sync_path_template = "scratch/{session_id}.ipynb"
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_default_provider(), "jupyter");
        assert!(c.execution_provider_allowed("jupyter"));
        assert_eq!(
            c.execution_jupyter_base_url().as_deref(),
            Some("http://127.0.0.1:8888")
        );
        assert_eq!(c.execution_jupyter_token().as_deref(), Some("testtoken"));
        assert_eq!(c.execution_jupyter_kernel_name(), "python3");
        assert_eq!(
            c.execution_jupyter_notebook_sync_path_template().as_deref(),
            Some("scratch/{session_id}.ipynb")
        );
    }

    #[test]
    fn harness_execution_ssh_toml() {
        let s = r#"
[harness.execution]
enabled = true
default_provider = "ssh"
allowed_providers = ["ssh", "local"]

[harness.execution.ssh]
host = "10.0.0.5"
port = 2222
user = "dev"
identity_file = "~/.ssh/id_ed25519"
remote_workdir = "/tmp/isanagent-exec"
remote_python = "python3"
accept_unknown_host_keys = false
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_default_provider(), "ssh");
        assert!(c.execution_provider_allowed("ssh"));
        assert_eq!(c.execution_ssh_host().as_deref(), Some("10.0.0.5"));
        assert_eq!(c.execution_ssh_port(), 2222);
        assert_eq!(c.execution_ssh_user().as_deref(), Some("dev"));
        assert!(c
            .execution_ssh_identity_file()
            .expect("identity")
            .replace('\\', "/")
            .ends_with("/.ssh/id_ed25519"));
        assert_eq!(
            c.execution_ssh_remote_workdir().as_deref(),
            Some("/tmp/isanagent-exec")
        );
        assert_eq!(c.execution_ssh_remote_python(), "python3");
        assert!(!c.execution_ssh_accept_unknown_host_keys());
    }

    #[test]
    fn ssh_accept_unknown_host_keys_defaults_false() {
        // 0.4: host-key verification must be ON by default. Omitting the key => secure default.
        let s = r#"
[harness.execution.ssh]
host = "10.0.0.9"
user = "dev"
remote_workdir = "/tmp/x"
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(
            !c.execution_ssh_accept_unknown_host_keys(),
            "default must verify host keys (accept_unknown_host_keys=false)"
        );
    }

    #[test]
    fn harness_execution_uv_local_runtime_toml() {
        let s = r#"
[harness.execution]
enabled = true
local_python_runtime = "uv_managed"
uv_binary = "uvx"
uv_python = "3.12"
uv_requirements = ["numpy", "pandas>=2.2"]
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_local_python_runtime(), "uv_managed");
        assert_eq!(c.execution_uv_binary(), "uvx");
        assert_eq!(c.execution_uv_python(), "3.12");
        assert_eq!(
            c.execution_uv_requirements(),
            vec!["numpy".to_string(), "pandas>=2.2".to_string()]
        );
    }

    #[test]
    fn agent_definitions_toml() {
        let s = r#"
[agents.researcher]
description = "Research topics and gather context"
allowed_tools = ["web_search", "web_fetch", "read_file"]
temperature = 0.1
hidden = false

[agents.coder]
description = "Implement code changes"
allowed_tools = ["*"]
model = "gemini-2.5-pro"
color = "4CAF50"
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        let agents = c.agent_definitions();
        assert_eq!(agents.len(), 2);
        let r = agents.get("researcher").expect("researcher");
        assert_eq!(r.description, "Research topics and gather context");
        assert_eq!(r.mode, AgentMode::Subagent);
        assert_eq!(
            r.allowed_tools.as_deref(),
            Some(
                &[
                    "web_search".to_string(),
                    "web_fetch".to_string(),
                    "read_file".to_string()
                ][..]
            )
        );
        assert!(!r.hidden);
        let c2 = agents.get("coder").expect("coder");
        assert_eq!(c2.model.as_deref(), Some("gemini-2.5-pro"));
        assert_eq!(c2.color.as_deref(), Some("4CAF50"));
    }

    #[test]
    fn agent_definitions_harness_merge() {
        let s = r#"
[agents.shared]
description = "Top-level agent"

[harness.agents.shared]
description = "Harness-level agent"
allowed_tools = ["read_file"]

[harness.agents.harness_only]
description = "Only in harness"
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        let agents = c.agent_definitions();
        // Harness-level wins over top-level for same name
        let shared = agents.get("shared").expect("shared");
        assert_eq!(shared.description, "Harness-level agent");
        assert_eq!(
            shared.allowed_tools.as_deref(),
            Some(&["read_file".to_string()][..])
        );
        // Harness-only entry surfaces
        assert!(agents.contains_key("harness_only"));
    }

    #[test]
    fn shell_policy_defaults_and_overrides() {
        let c = AppConfig::default();
        let p = c.resolved_shell_policy();
        assert_eq!(p.interactive_mode, ShellPolicyMode::Ask);
        assert_eq!(p.unattended_mode, ShellPolicyMode::Deny);
        assert!(!p.approval_patterns.is_empty());

        let s = r#"
[harness.shell_policy]
mode = "deny"
unattended_default = "allow"
interactive_requires_approval_for = ["terraform destroy"]
"#;
        let c2: AppConfig = toml::from_str(s).expect("parse");
        let p2 = c2.resolved_shell_policy();
        assert_eq!(p2.interactive_mode, ShellPolicyMode::Deny);
        assert_eq!(p2.unattended_mode, ShellPolicyMode::Allow);
        assert!(p2
            .approval_patterns
            .iter()
            .any(|s| s.contains("terraform destroy")));
    }
}

#[cfg(test)]
mod expanded_providers_tests {
    use super::*;

    #[test]
    fn family_format_expands_to_per_model_entries() {
        let toml_str = r#"
[providers.deepseek]
models = ["deepseek-v4-pro", "deepseek-v4-flash"]
api_key = "sk-test"
"#;
        let cfg: AppConfig = toml::from_str(toml_str).expect("parse");
        let expanded = cfg.expanded_providers();
        assert_eq!(expanded.len(), 2);

        let pro = expanded.get("deepseek-v4-pro").expect("pro entry");
        assert_eq!(pro.provider_name, "deepseek");
        assert_eq!(pro.model_name, "deepseek-v4-pro");
        assert_eq!(pro.api_key.as_deref(), Some("sk-test"));
        assert!(pro.models.is_none());

        let flash = expanded.get("deepseek-v4-flash").expect("flash entry");
        assert_eq!(flash.provider_name, "deepseek");
        assert_eq!(flash.model_name, "deepseek-v4-flash");
        assert_eq!(flash.api_key.as_deref(), Some("sk-test"));
    }

    #[test]
    fn legacy_single_model_format_passes_through() {
        let toml_str = r#"
[providers.my-custom]
provider_name = "openai_compatible"
model_name = "custom-model"
base_url = "https://example.com/v1/chat/completions"
"#;
        let cfg: AppConfig = toml::from_str(toml_str).expect("parse");
        let expanded = cfg.expanded_providers();
        assert_eq!(expanded.len(), 1);

        let entry = expanded.get("my-custom").expect("legacy entry");
        assert_eq!(entry.provider_name, "openai_compatible");
        assert_eq!(entry.model_name, "custom-model");
        assert_eq!(
            entry.base_url.as_deref(),
            Some("https://example.com/v1/chat/completions")
        );
    }

    #[test]
    fn infers_provider_name_from_map_key() {
        let toml_str = r#"
[providers.gemini]
models = ["gemini-2.5-flash"]
"#;
        let cfg: AppConfig = toml::from_str(toml_str).expect("parse");
        let expanded = cfg.expanded_providers();
        let entry = expanded.get("gemini-2.5-flash").expect("entry");
        assert_eq!(entry.provider_name, "gemini");
    }

    #[test]
    fn mixed_family_and_legacy_coexist() {
        let toml_str = r#"
[providers.anthropic]
models = ["claude-opus-4-7", "claude-sonnet-4-6"]

[providers.my-proxy]
provider_name = "openai_compatible"
model_name = "proxy-model"
base_url = "https://proxy.example/v1/chat/completions"
"#;
        let cfg: AppConfig = toml::from_str(toml_str).expect("parse");
        let expanded = cfg.expanded_providers();
        assert_eq!(expanded.len(), 3);
        assert!(expanded.contains_key("claude-opus-4-7"));
        assert!(expanded.contains_key("claude-sonnet-4-6"));
        assert!(expanded.contains_key("my-proxy"));
    }

    #[test]
    fn empty_providers_returns_empty_map() {
        let cfg = AppConfig::default();
        assert!(cfg.expanded_providers().is_empty());
    }

    #[test]
    fn duplicate_model_name_across_families_is_detected() {
        let toml_str = r#"
[providers.openai]
models = ["gpt-5.5"]
api_key = "sk-openai"

[providers.openrouter]
models = ["gpt-5.5", "claude-opus-4-7"]
api_key = "sk-or"
"#;
        let cfg: AppConfig = toml::from_str(toml_str).expect("parse");
        let expanded = cfg.expanded_providers();
        // Both families list "gpt-5.5"; last write wins in HashMap iteration order.
        // Only 2 unique keys survive: the overwritten "gpt-5.5" + "claude-opus-4-7".
        assert_eq!(expanded.len(), 2);
        assert!(expanded.contains_key("gpt-5.5"));
        assert!(expanded.contains_key("claude-opus-4-7"));
        // Both values are valid API keys from one of the providers.
        let gpt_key = expanded.get("gpt-5.5").unwrap().api_key.as_deref().unwrap();
        assert!(gpt_key == "sk-openai" || gpt_key == "sk-or");
    }

    #[test]
    fn family_and_legacy_with_same_model_name_is_detected() {
        // Family and legacy entry expanding to the same model name.
        // One overwrites the other depending on HashMap iteration order.
        let toml_str = r#"
[providers.openai]
models = ["gpt-5.5"]
api_key = "sk-family"

[providers.legacy-gpt]
provider_name = "openai"
model_name = "gpt-5.5"
api_key = "sk-legacy"
"#;
        let cfg: AppConfig = toml::from_str(toml_str).expect("parse");
        let expanded = cfg.expanded_providers();
        // "gpt-5.5" (overwritten) + "legacy-gpt" (pass-through)
        assert_eq!(expanded.len(), 2);
        assert!(expanded.contains_key("gpt-5.5"));
        assert!(expanded.contains_key("legacy-gpt"));
        // The "legacy-gpt" key always uses sk-legacy
        assert_eq!(
            expanded.get("legacy-gpt").unwrap().api_key.as_deref(),
            Some("sk-legacy")
        );
        // "gpt-5.5" gets one of the two keys (non-deterministic)
        let gpt_key = expanded.get("gpt-5.5").unwrap().api_key.as_deref().unwrap();
        assert!(gpt_key == "sk-family" || gpt_key == "sk-legacy");
    }

    #[test]
    fn empty_models_and_model_name_warns_and_skips() {
        let toml_str = r#"
[providers.broken]
# neither models nor model_name set
"#;
        let cfg: AppConfig = toml::from_str(toml_str).expect("parse");
        let expanded = cfg.expanded_providers();
        // entry is silently skipped (no model_name and no models)
        assert!(expanded.is_empty());
    }

    #[test]
    fn empty_models_list_skips_entry() {
        let toml_str = r#"
[providers.gemini]
models = []
"#;
        let cfg: AppConfig = toml::from_str(toml_str).expect("parse");
        let expanded = cfg.expanded_providers();
        // models is Some but empty vec: no expansions produced
        assert!(expanded.is_empty());
    }
}

#[cfg(test)]
mod placeholder_key_tests {
    use super::*;

    fn provider_with_key(key: &str) -> ProviderConfig {
        ProviderConfig {
            provider_name: "nonexistent-provider".to_string(),
            model_name: "some-model".to_string(),
            models: None,
            api_key_env: "".to_string(),
            api_key: Some(key.to_string()),
            base_url: None,
        }
    }

    #[test]
    fn rejects_angle_bracket_placeholder() {
        assert!(provider_with_key("<changethis>").resolve_api_key().is_err());
    }

    #[test]
    fn rejects_changethis_without_brackets() {
        assert!(provider_with_key("changethis").resolve_api_key().is_err());
    }

    #[test]
    fn rejects_replace_me() {
        assert!(provider_with_key("replace_me").resolve_api_key().is_err());
    }

    #[test]
    fn rejects_placeholder_keyword() {
        assert!(provider_with_key("my_placeholder_key")
            .resolve_api_key()
            .is_err());
    }

    #[test]
    fn accepts_real_api_key() {
        let result = provider_with_key("sk-abc123def456").resolve_api_key();
        assert_eq!(result.unwrap(), "sk-abc123def456");
    }
}
