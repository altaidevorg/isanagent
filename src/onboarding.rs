use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{AppConfig, SlackMode};
use crate::ml_engineer::HARNESS_OVERLAY;
use crate::skills::SkillRegistry;
use crate::workspace::{ensure_workspace_layout, WorkspaceLayout};
use clap::Args;
use include_dir::{include_dir, Dir};
use toml_edit::{value, DocumentMut};

/// Full skill tree (SKILL.md, reference.md, examples/) embedded at compile time.
static ONBOARD_SYNTHETIC_SKILL_DIR: Dir<'static> =
    include_dir!("$CARGO_MANIFEST_DIR/assets/onboarding/skills/synthetic-dataset-with-afterimage");
static ONBOARD_KERNEL_PORTING_SKILL_DIR: Dir<'static> =
    include_dir!("$CARGO_MANIFEST_DIR/assets/onboarding/skills/kernel-porting");
static ONBOARD_AUTOTRAINESS_SKILL_DIR: Dir<'static> =
    include_dir!("$CARGO_MANIFEST_DIR/assets/onboarding/skills/autotrainess");
static ONBOARD_KERNEL_AGENT_PROMPTS_DIR: Dir<'static> =
    include_dir!("$CARGO_MANIFEST_DIR/assets/onboarding/agents/prompts");
static ONBOARD_KERNEL_REFERENCE_DIR: Dir<'static> =
    include_dir!("$CARGO_MANIFEST_DIR/assets/onboarding/kernels/reference");
static ONBOARD_KERNEL_BENCHMARKS_DIR: Dir<'static> =
    include_dir!("$CARGO_MANIFEST_DIR/assets/onboarding/benchmarks");

const CONFIG_TEMPLATE: &str = include_str!("../assets/onboarding/config.toml");
const AGENTS_TEMPLATE: &str = include_str!("../assets/onboarding/AGENTS.md");
const USER_TEMPLATE: &str = include_str!("../assets/onboarding/USER.md");
const SOUL_TEMPLATE: &str = include_str!("../assets/onboarding/SOUL.md");
const CRON_SKILL_TEMPLATE: &str = include_str!("../assets/onboarding/skills/cron/SKILL.md");
const SKILL_CREATOR_TEMPLATE: &str =
    include_str!("../assets/onboarding/skills/skill-creator/SKILL.md");
const EXECUTION_RESEARCH_SKILL_TEMPLATE: &str =
    include_str!("../assets/onboarding/skills/execution-research/SKILL.md");
const JUPYTER_HEAVY_OUTPUT_SKILL_TEMPLATE: &str =
    include_str!("../assets/onboarding/skills/jupyter-heavy-output/SKILL.md");
const SCIENTIFIC_PYTHON_DEBUGGING_SKILL_TEMPLATE: &str =
    include_str!("../assets/onboarding/skills/scientific-python-debugging/SKILL.md");
const ML_EXECUTION_PREFLIGHT_SKILL_TEMPLATE: &str =
    include_str!("../assets/onboarding/skills/ml-execution-preflight/SKILL.md");
const LITERATURE_TO_RECIPE_SKILL_TEMPLATE: &str =
    include_str!("../assets/onboarding/skills/literature-to-recipe/SKILL.md");
const OOM_RECOVERY_PLAYBOOK_SKILL_TEMPLATE: &str =
    include_str!("../assets/onboarding/skills/oom-recovery-playbook/SKILL.md");
const COLAB_CLI_SKILL_TEMPLATE: &str =
    include_str!("../assets/onboarding/skills/colab-cli/SKILL.md");

struct TemplateFile {
    relative_path: &'static str,
    contents: &'static str,
}

#[derive(Clone, Debug, Default)]
pub struct BootstrapReport {
    pub root: PathBuf,
    pub created: Vec<PathBuf>,
    pub skipped: Vec<PathBuf>,
}

impl BootstrapReport {
    fn new(root: PathBuf) -> Self {
        Self {
            root,
            created: Vec::new(),
            skipped: Vec::new(),
        }
    }
}

/// Optional `config.toml` field overrides for [`onboard_workspace`].
///
/// Derives [`clap::Args`] so the binary can `flatten` these into `onboard` for scripting.
/// When any field is set, the written `config.toml` is normally produced by parse → merge →
/// serialize (comments omitted). `onboard --interactive` uses [`build_interactive_config_toml`]
/// instead so comments are kept. With all fields unset, the embedded template file is copied
/// verbatim (comments preserved).
#[derive(Debug, Clone, Default, Args)]
#[command(next_help_heading = "config.toml overrides")]
pub struct OnboardOptions {
    #[arg(long, help_heading = "Workspace / limits")]
    pub restrict_to_workspace: Option<bool>,
    #[arg(long, help_heading = "Workspace / limits")]
    pub max_iterations: Option<usize>,
    #[arg(long, help_heading = "Workspace / limits")]
    pub max_tool_output_chars: Option<usize>,
    #[arg(long, help_heading = "Workspace / limits")]
    pub max_web_tool_output_chars: Option<usize>,

    /// Sets `[terminal] enabled` (stdin/stdout chat).
    #[arg(long, help_heading = "Terminal channel")]
    pub terminal_enable: Option<bool>,

    /// One of the well-known names from `provider_registry::KNOWN_PROVIDERS` (e.g. `gemini`,
    /// `openai`, `deepseek`, `openrouter`) or `openai_compatible` for any third-party endpoint
    /// (which then requires `--provider-base-url`).
    #[arg(long, help_heading = "Provider")]
    pub provider_name: Option<String>,
    #[arg(long, help_heading = "Provider")]
    pub provider_model: Option<String>,
    #[arg(long, help_heading = "Provider")]
    pub provider_api_key_env: Option<String>,
    /// Explicit chat-completions URL. Required when `--provider-name openai_compatible`. For
    /// known names this is an optional override (e.g. to point at a proxy or self-hosted relay).
    #[arg(long, help_heading = "Provider")]
    pub provider_base_url: Option<String>,

    #[arg(long, help_heading = "HTTP API")]
    pub api_enabled: Option<bool>,
    #[arg(long, help_heading = "HTTP API")]
    pub api_port: Option<u16>,
    #[arg(long, help_heading = "HTTP API")]
    pub api_serve_ui: Option<bool>,
    #[arg(long, help_heading = "HTTP API")]
    pub api_bind_address: Option<String>,

    #[arg(long, help_heading = "Slack")]
    pub slack_enabled: Option<bool>,
    #[arg(long, value_enum, help_heading = "Slack")]
    pub slack_mode: Option<SlackMode>,

    #[arg(long, help_heading = "Email")]
    pub email_enabled: Option<bool>,

    #[arg(long, help_heading = "Jina / memory / multi-tenant")]
    pub jina_enabled: Option<bool>,
    #[arg(long, help_heading = "Jina / memory / multi-tenant")]
    pub memory_enabled: Option<bool>,
    #[arg(long, help_heading = "Jina / memory / multi-tenant")]
    pub multi_tenant_activity_heartbeat: Option<bool>,
    #[arg(long, help_heading = "Jina / memory / multi-tenant")]
    pub multi_tenant_cron_scheduling: Option<bool>,

    #[arg(long, help_heading = "Harness")]
    pub harness_git_worktree_enabled: Option<bool>,
    #[arg(long, help_heading = "Harness")]
    pub harness_subagents_enabled: Option<bool>,
    #[arg(long, help_heading = "Harness")]
    pub harness_ml_engineer_enabled: Option<bool>,
    #[arg(long, help_heading = "Harness")]
    pub harness_execution_enabled: Option<bool>,
    #[arg(long, help_heading = "Harness")]
    pub harness_background_jobs_enabled: Option<bool>,
    #[arg(long, help_heading = "Harness")]
    pub harness_notifications_enabled: Option<bool>,
}

impl OnboardOptions {
    /// True if any override is set (config must be emitted via merge + serialize).
    pub fn has_overrides(&self) -> bool {
        self.restrict_to_workspace.is_some()
            || self.max_iterations.is_some()
            || self.max_tool_output_chars.is_some()
            || self.max_web_tool_output_chars.is_some()
            || self.terminal_enable.is_some()
            || self.provider_name.is_some()
            || self.provider_model.is_some()
            || self.provider_api_key_env.is_some()
            || self.provider_base_url.is_some()
            || self.api_enabled.is_some()
            || self.api_port.is_some()
            || self.api_serve_ui.is_some()
            || self.api_bind_address.is_some()
            || self.slack_enabled.is_some()
            || self.slack_mode.is_some()
            || self.email_enabled.is_some()
            || self.jina_enabled.is_some()
            || self.memory_enabled.is_some()
            || self.multi_tenant_activity_heartbeat.is_some()
            || self.multi_tenant_cron_scheduling.is_some()
            || self.harness_git_worktree_enabled.is_some()
            || self.harness_subagents_enabled.is_some()
            || self.harness_ml_engineer_enabled.is_some()
            || self.harness_execution_enabled.is_some()
            || self.harness_background_jobs_enabled.is_some()
            || self.harness_notifications_enabled.is_some()
    }
}

fn apply_onboard_options(cfg: &mut AppConfig, opts: &OnboardOptions) {
    if let Some(v) = opts.restrict_to_workspace {
        cfg.restrict_to_workspace = Some(v);
    }
    if let Some(v) = opts.max_iterations {
        cfg.max_iterations = Some(v);
    }
    if let Some(v) = opts.max_tool_output_chars {
        cfg.max_tool_output_chars = Some(v);
    }
    if let Some(v) = opts.max_web_tool_output_chars {
        cfg.max_web_tool_output_chars = Some(v);
    }

    if let Some(v) = opts.terminal_enable {
        cfg.terminal.get_or_insert_with(Default::default).enabled = Some(v);
    }

    if opts.provider_name.is_some()
        || opts.provider_model.is_some()
        || opts.provider_api_key_env.is_some()
        || opts.provider_base_url.is_some()
    {
        let p = cfg.provider.get_or_insert_with(Default::default);
        if let Some(ref n) = opts.provider_name {
            p.provider_name = n.clone();
        }
        if let Some(ref m) = opts.provider_model {
            p.model_name = m.clone();
        }
        if let Some(ref e) = opts.provider_api_key_env {
            p.api_key_env = e.clone();
        }
        // Only persist `base_url` when explicitly set via the CLI flag. For known names we
        // intentionally leave the field unset so the registry stays the single source of truth.
        if let Some(ref u) = opts.provider_base_url {
            p.base_url = Some(u.clone());
        }
    }

    if let Some(a) = cfg.api.as_mut() {
        if let Some(v) = opts.api_enabled {
            a.enabled = Some(v);
        }
        if let Some(p) = opts.api_port {
            a.port = p;
        }
        if let Some(v) = opts.api_serve_ui {
            a.serve_ui = Some(v);
        }
        if let Some(ref addr) = opts.api_bind_address {
            a.bind_address = Some(addr.clone());
        }
    }

    if let Some(s) = cfg.slack.as_mut() {
        if let Some(v) = opts.slack_enabled {
            s.enabled = Some(v);
        }
        if let Some(m) = opts.slack_mode {
            s.mode = Some(m);
        }
    }

    if let Some(e) = cfg.email.as_mut() {
        if let Some(v) = opts.email_enabled {
            e.enabled = Some(v);
        }
    }

    if let Some(v) = opts.jina_enabled {
        cfg.jina.get_or_insert_with(Default::default).enabled = Some(v);
    }

    if let Some(v) = opts.memory_enabled {
        cfg.memory.get_or_insert_with(Default::default).enabled = Some(v);
    }

    if opts.multi_tenant_activity_heartbeat.is_some() || opts.multi_tenant_cron_scheduling.is_some()
    {
        let m = cfg.multi_tenant_edge.get_or_insert_with(Default::default);
        if let Some(v) = opts.multi_tenant_activity_heartbeat {
            m.activity_heartbeat_enabled = Some(v);
        }
        if let Some(v) = opts.multi_tenant_cron_scheduling {
            m.cron_scheduling_enabled = Some(v);
        }
    }

    if opts.harness_git_worktree_enabled.is_some()
        || opts.harness_subagents_enabled.is_some()
        || opts.harness_ml_engineer_enabled.is_some()
        || opts.harness_execution_enabled.is_some()
        || opts.harness_background_jobs_enabled.is_some()
        || opts.harness_notifications_enabled.is_some()
    {
        let h = cfg.harness.get_or_insert_with(Default::default);
        if let Some(v) = opts.harness_git_worktree_enabled {
            h.git_worktree.get_or_insert_with(Default::default).enabled = Some(v);
        }
        if let Some(v) = opts.harness_subagents_enabled {
            h.subagents.get_or_insert_with(Default::default).enabled = Some(v);
        }
        if let Some(v) = opts.harness_ml_engineer_enabled {
            h.ml_engineer.get_or_insert_with(Default::default).enabled = Some(v);
        }
        if let Some(v) = opts.harness_execution_enabled {
            h.execution.get_or_insert_with(Default::default).enabled = Some(v);
        }
        if let Some(v) = opts.harness_background_jobs_enabled {
            h.background_jobs
                .get_or_insert_with(Default::default)
                .enabled = Some(v);
        }
        if let Some(v) = opts.harness_notifications_enabled {
            h.notifications.get_or_insert_with(Default::default).enabled = Some(v);
        }
    }
}

fn build_config_toml(options: &OnboardOptions) -> Result<String, String> {
    let mut cfg: AppConfig = toml::from_str(CONFIG_TEMPLATE).map_err(|e| {
        format!(
            "Internal error: embedded config template is invalid TOML: {}",
            e
        )
    })?;
    apply_onboard_options(&mut cfg, options);
    toml::to_string_pretty(&cfg).map_err(|e| format!("Failed to serialize config.toml: {}", e))
}

/// Same overrides as [`apply_onboard_options`], applied in-place on the embedded template with
/// `toml_edit` so **comments are preserved** (used for `onboard --interactive`).
pub fn build_interactive_config_toml(options: &OnboardOptions) -> Result<String, String> {
    let mut doc: DocumentMut = CONFIG_TEMPLATE
        .parse()
        .map_err(|e| format!("interactive config template parse (toml_edit): {}", e))?;

    if let Some(v) = options.restrict_to_workspace {
        doc["restrict_to_workspace"] = value(v);
    }
    if let Some(v) = options.max_iterations {
        doc["max_iterations"] =
            value(i64::try_from(v).map_err(|_| "max_iterations does not fit i64".to_string())?);
    }
    if let Some(v) = options.max_tool_output_chars {
        doc["max_tool_output_chars"] = value(
            i64::try_from(v).map_err(|_| "max_tool_output_chars does not fit i64".to_string())?,
        );
    }
    if let Some(v) = options.max_web_tool_output_chars {
        doc["max_web_tool_output_chars"] = value(
            i64::try_from(v)
                .map_err(|_| "max_web_tool_output_chars does not fit i64".to_string())?,
        );
    }

    if let Some(v) = options.terminal_enable {
        doc["terminal"]["enabled"] = value(v);
    }

    if (options.provider_name.is_some()
        || options.provider_model.is_some()
        || options.provider_api_key_env.is_some()
        || options.provider_base_url.is_some())
        && doc.get("provider").is_none()
    {
        doc["provider"] = toml_edit::Item::Table(toml_edit::Table::new());
    }

    if let Some(ref n) = options.provider_name {
        doc["provider"]["provider_name"] = value(n.as_str());
    }
    if let Some(ref m) = options.provider_model {
        doc["provider"]["model_name"] = value(m.as_str());
    }
    if let Some(ref e) = options.provider_api_key_env {
        doc["provider"]["api_key_env"] = value(e.as_str());
    }
    // base_url is optional in the new schema. When the wizard hands us an explicit URL, write it
    // verbatim. When it doesn't, drop any base_url that may still be in the embedded template so
    // we never persist a stale URL alongside `provider_name`.
    if let Some(ref u) = options.provider_base_url {
        doc["provider"]["base_url"] = value(u.as_str());
    } else if doc
        .get("provider")
        .and_then(|t| t.as_table())
        .map(|t| t.contains_key("base_url"))
        .unwrap_or(false)
    {
        doc["provider"]
            .as_table_mut()
            .expect("provider table")
            .remove("base_url");
    }

    if let Some(v) = options.api_enabled {
        doc["api"]["enabled"] = value(v);
    }
    if let Some(p) = options.api_port {
        doc["api"]["port"] = value(i64::from(p));
    }
    if let Some(v) = options.api_serve_ui {
        doc["api"]["serve_ui"] = value(v);
    }
    if let Some(ref addr) = options.api_bind_address {
        doc["api"]["bind_address"] = value(addr.as_str());
    }

    if let Some(v) = options.slack_enabled {
        doc["slack"]["enabled"] = value(v);
    }
    if let Some(m) = options.slack_mode {
        let s = match m {
            SlackMode::Webhook => "webhook",
            SlackMode::Socket => "socket",
        };
        doc["slack"]["mode"] = value(s);
    }

    if let Some(v) = options.email_enabled {
        doc["email"]["enabled"] = value(v);
    }

    if let Some(v) = options.jina_enabled {
        doc["jina"]["enabled"] = value(v);
    }

    if let Some(v) = options.memory_enabled {
        doc["memory"]["enabled"] = value(v);
    }

    if let Some(v) = options.multi_tenant_activity_heartbeat {
        doc["multi_tenant_edge"]["activity_heartbeat_enabled"] = value(v);
    }
    if let Some(v) = options.multi_tenant_cron_scheduling {
        doc["multi_tenant_edge"]["cron_scheduling_enabled"] = value(v);
    }

    if let Some(v) = options.harness_git_worktree_enabled {
        doc["harness"]["git_worktree"]["enabled"] = value(v);
    }
    if let Some(v) = options.harness_subagents_enabled {
        doc["harness"]["subagents"]["enabled"] = value(v);
    }
    if let Some(v) = options.harness_ml_engineer_enabled {
        doc["harness"]["ml_engineer"]["enabled"] = value(v);
    }
    if let Some(v) = options.harness_execution_enabled {
        doc["harness"]["execution"]["enabled"] = value(v);
    }
    if let Some(v) = options.harness_background_jobs_enabled {
        doc["harness"]["background_jobs"]["enabled"] = value(v);
    }
    if let Some(v) = options.harness_notifications_enabled {
        doc["harness"]["notifications"]["enabled"] = value(v);
    }

    let out = doc.to_string();
    let _: AppConfig = toml::from_str(&out).map_err(|e| {
        format!(
            "interactive merged config.toml failed AppConfig validation: {}",
            e
        )
    })?;
    Ok(out)
}

/// Bootstrap a workspace at `root`, optionally overriding embedded `config.toml` values.
///
/// When `interactive_merged_config_toml` is `Some`, it is written as `config.toml` (instead of
/// serde-pretty output) so template **comments stay**; used for `onboard --interactive`.
pub fn onboard_workspace(
    root: &Path,
    options: &OnboardOptions,
    interactive_merged_config_toml: Option<&str>,
) -> Result<BootstrapReport, String> {
    let layout = ensure_workspace_layout(root)?;
    let report_root = fs::canonicalize(&layout.root).unwrap_or_else(|_| layout.root.clone());
    let mut report = BootstrapReport::new(report_root);

    write_all_templates(
        &layout,
        options,
        interactive_merged_config_toml,
        &mut report,
    )?;
    if options.has_overrides()
        && report
            .skipped
            .iter()
            .any(|p| p.as_os_str() == "config.toml")
    {
        return Err(
            "config.toml already exists in this workspace; onboard CLI overrides apply only when \
that file is first created. Remove or rename config.toml, or point --workspace at a new directory."
                .to_string(),
        );
    }
    validate_generated_files(&layout)?;

    Ok(report)
}

fn embedded_templates() -> &'static [TemplateFile] {
    static TEMPLATES: [TemplateFile; 13] = [
        TemplateFile {
            relative_path: "config.toml",
            contents: CONFIG_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/AGENTS.md",
            contents: AGENTS_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/USER.md",
            contents: USER_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/SOUL.md",
            contents: SOUL_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/skills/cron/SKILL.md",
            contents: CRON_SKILL_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/skills/skill-creator/SKILL.md",
            contents: SKILL_CREATOR_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/skills/execution-research/SKILL.md",
            contents: EXECUTION_RESEARCH_SKILL_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/skills/jupyter-heavy-output/SKILL.md",
            contents: JUPYTER_HEAVY_OUTPUT_SKILL_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/skills/scientific-python-debugging/SKILL.md",
            contents: SCIENTIFIC_PYTHON_DEBUGGING_SKILL_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/skills/ml-execution-preflight/SKILL.md",
            contents: ML_EXECUTION_PREFLIGHT_SKILL_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/skills/literature-to-recipe/SKILL.md",
            contents: LITERATURE_TO_RECIPE_SKILL_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/skills/oom-recovery-playbook/SKILL.md",
            contents: OOM_RECOVERY_PLAYBOOK_SKILL_TEMPLATE,
        },
        TemplateFile {
            relative_path: "workspace/skills/colab-cli/SKILL.md",
            contents: COLAB_CLI_SKILL_TEMPLATE,
        },
    ];

    &TEMPLATES
}

fn write_all_templates(
    layout: &WorkspaceLayout,
    options: &OnboardOptions,
    interactive_merged_config_toml: Option<&str>,
    report: &mut BootstrapReport,
) -> Result<(), String> {
    for template in embedded_templates() {
        if template.relative_path == "config.toml" {
            let body = if let Some(s) =
                interactive_merged_config_toml.filter(|_| options.has_overrides())
            {
                s.to_string()
            } else if options.has_overrides() {
                build_config_toml(options)?
            } else {
                CONFIG_TEMPLATE.to_string()
            };
            write_if_missing_string(&layout.root, template.relative_path, &body, report)?;
        } else {
            write_if_missing_string(
                &layout.root,
                template.relative_path,
                template.contents,
                report,
            )?;
        }
    }

    write_embedded_synthetic_skill_tree(&layout.root, report)?;
    write_embedded_kernel_porting_tree(&layout.root, report)?;
    write_embedded_autotrainess_tree(&layout.root, report)?;

    let overlay_ref = workspace_ml_engineer_overlay_reference();
    write_if_missing_string(
        &layout.root,
        "workspace/ML_ENGINEER_OVERLAY.md",
        &overlay_ref,
        report,
    )?;

    Ok(())
}

/// Human-readable copy of the embedded ML overlay (same bytes as `crate::ml_engineer::HARNESS_OVERLAY`).
fn workspace_ml_engineer_overlay_reference() -> String {
    format!(
        "# ML engineer overlay (reference copy)\n\n\
This file is created by **`isanagent onboard`**. It mirrors the policy text that the binary \
appends to the system prompt when **`[harness.ml_engineer] enabled = true`** in `config.toml`. \
Editing this file does **not** change runtime behavior (the live text is embedded in the \
`isanagent` build). For workspace-specific ML rules, add or edit **`ML_POLICY.md`** in this \
directory (merged by `compile_system_prompt`).\n\n---\n\n{}",
        HARNESS_OVERLAY
    )
}

const SYNTHETIC_SKILL_REL_PREFIX: &str = "workspace/skills/synthetic-dataset-with-afterimage";
const KERNEL_PORTING_SKILL_REL_PREFIX: &str = "workspace/skills/kernel-porting";
const AUTOTRAINESS_SKILL_REL_PREFIX: &str = "workspace/skills/autotrainess";
const KERNEL_AGENT_PROMPTS_REL_PREFIX: &str = "workspace/.agents/prompts";
const KERNEL_REFERENCE_REL_PREFIX: &str = "workspace/kernels/reference";
const KERNEL_BENCHMARKS_REL_PREFIX: &str = "workspace/benchmarks";

fn write_embedded_synthetic_skill_tree(
    root: &Path,
    report: &mut BootstrapReport,
) -> Result<(), String> {
    write_embedded_dir_recursive(
        &ONBOARD_SYNTHETIC_SKILL_DIR,
        root,
        SYNTHETIC_SKILL_REL_PREFIX,
        Path::new(""),
        report,
    )
}

fn write_embedded_kernel_porting_tree(
    root: &Path,
    report: &mut BootstrapReport,
) -> Result<(), String> {
    write_embedded_dir_recursive(
        &ONBOARD_KERNEL_PORTING_SKILL_DIR,
        root,
        KERNEL_PORTING_SKILL_REL_PREFIX,
        Path::new(""),
        report,
    )?;
    write_embedded_dir_recursive(
        &ONBOARD_KERNEL_AGENT_PROMPTS_DIR,
        root,
        KERNEL_AGENT_PROMPTS_REL_PREFIX,
        Path::new(""),
        report,
    )?;
    write_embedded_dir_recursive(
        &ONBOARD_KERNEL_REFERENCE_DIR,
        root,
        KERNEL_REFERENCE_REL_PREFIX,
        Path::new(""),
        report,
    )?;
    write_embedded_dir_recursive(
        &ONBOARD_KERNEL_BENCHMARKS_DIR,
        root,
        KERNEL_BENCHMARKS_REL_PREFIX,
        Path::new(""),
        report,
    )?;
    // Symlink-style copy: gpu_to_jax plan accessible at .agents/kernel-porting/
    let plan_src = root.join(format!(
        "{}/gpu_to_jax_plan.json",
        KERNEL_PORTING_SKILL_REL_PREFIX
    ));
    let plan_dest = root.join("workspace/.agents/kernel-porting/gpu_to_jax_plan.json");
    if plan_src.exists() {
        if let Some(parent) = plan_dest.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if !plan_dest.exists() && fs::copy(&plan_src, &plan_dest).is_ok() {
            report.created.push(PathBuf::from(
                "workspace/.agents/kernel-porting/gpu_to_jax_plan.json",
            ));
        }
    }
    let schema_src = root.join(format!(
        "{}/map_elites.schema.json",
        KERNEL_PORTING_SKILL_REL_PREFIX
    ));
    let schema_dest = root.join("workspace/.agents/kernel-porting/map_elites.schema.json");
    if schema_src.exists() {
        if let Some(parent) = schema_dest.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if !schema_dest.exists() && fs::copy(&schema_src, &schema_dest).is_ok() {
            report.created.push(PathBuf::from(
                "workspace/.agents/kernel-porting/map_elites.schema.json",
            ));
        }
    }
    Ok(())
}

fn write_embedded_autotrainess_tree(
    root: &Path,
    report: &mut BootstrapReport,
) -> Result<(), String> {
    write_embedded_dir_recursive(
        &ONBOARD_AUTOTRAINESS_SKILL_DIR,
        root,
        AUTOTRAINESS_SKILL_REL_PREFIX,
        Path::new(""),
        report,
    )?;
    // Convenience copy: iteration plan accessible at .agents/autotrainess/
    let plan_src = root.join(format!(
        "{}/iteration_plan.json",
        AUTOTRAINESS_SKILL_REL_PREFIX
    ));
    let plan_dest = root.join("workspace/.agents/autotrainess/iteration_plan.json");
    if plan_src.exists() {
        if let Some(parent) = plan_dest.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if !plan_dest.exists() && fs::copy(&plan_src, &plan_dest).is_ok() {
            report.created.push(PathBuf::from(
                "workspace/.agents/autotrainess/iteration_plan.json",
            ));
        }
    }
    Ok(())
}

fn write_embedded_dir_recursive(
    dir: &Dir<'_>,
    root: &Path,
    skill_dest_dir: &str,
    rel_inside: &Path,
    report: &mut BootstrapReport,
) -> Result<(), String> {
    for file in dir.files() {
        let rel = if rel_inside.as_os_str().is_empty() {
            file.path().to_path_buf()
        } else {
            rel_inside.join(file.path())
        };
        let dest_rel = Path::new(skill_dest_dir).join(rel);
        let dest_rel_str = dest_rel.to_string_lossy().replace('\\', "/");
        write_if_missing_bytes(root, &dest_rel_str, file.contents(), report)?;
    }
    for sub in dir.dirs() {
        let next = if rel_inside.as_os_str().is_empty() {
            sub.path().to_path_buf()
        } else {
            rel_inside.join(sub.path())
        };
        write_embedded_dir_recursive(sub, root, skill_dest_dir, &next, report)?;
    }
    Ok(())
}

fn write_if_missing_bytes(
    root: &Path,
    relative_path: &str,
    contents: &[u8],
    report: &mut BootstrapReport,
) -> Result<(), String> {
    let rel = PathBuf::from(relative_path);
    let dest = root.join(&rel);

    if dest.exists() {
        report.skipped.push(rel);
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Failed to create parent directory for {}: {}",
                dest.display(),
                e
            )
        })?;
    }

    fs::write(&dest, contents).map_err(|e| format!("Failed to write {}: {}", dest.display(), e))?;
    report.created.push(rel);
    Ok(())
}

fn write_if_missing_string(
    root: &Path,
    relative_path: &str,
    contents: &str,
    report: &mut BootstrapReport,
) -> Result<(), String> {
    write_if_missing_bytes(root, relative_path, contents.as_bytes(), report)
}

fn validate_generated_files(layout: &WorkspaceLayout) -> Result<(), String> {
    let config_path = layout.root.join("config.toml");
    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read generated config.toml: {}", e))?;
    toml::from_str::<AppConfig>(&config_str)
        .map_err(|e| format!("Generated config.toml is invalid: {}", e))?;

    for relative_path in [
        "workspace/AGENTS.md",
        "workspace/USER.md",
        "workspace/SOUL.md",
        "workspace/ML_ENGINEER_OVERLAY.md",
    ] {
        let path = layout.root.join(relative_path);
        fs::read_to_string(&path).map_err(|e| {
            format!(
                "Required onboarding file is missing, not a file, or not readable UTF-8 ({}): {}",
                path.display(),
                e
            )
        })?;
    }

    let registry = SkillRegistry::new(layout.skills_dir.clone());
    let skill_names = registry.get_skill_names();
    for required in [
        "cron",
        "skill-creator",
        "execution-research",
        "jupyter-heavy-output",
        "scientific-python-debugging",
        "ml-execution-preflight",
        "literature-to-recipe",
        "oom-recovery-playbook",
        "colab-cli",
        "synthetic-dataset-with-afterimage",
    ] {
        if !skill_names.iter().any(|skill| skill == required) {
            return Err(format!(
                "Generated skill '{}' was not loadable from {}",
                required,
                layout.skills_dir.display()
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_config_toml_merge_api_and_terminal() {
        let o = OnboardOptions {
            api_enabled: Some(true),
            api_port: Some(9090),
            terminal_enable: Some(false),
            ..Default::default()
        };
        let s = build_config_toml(&o).expect("toml");
        let cfg: AppConfig = toml::from_str(&s).expect("parse back");
        let api = cfg.api.as_ref().expect("api section");
        assert_eq!(api.enabled, Some(true));
        assert_eq!(api.port, 9090);
        assert!(!cfg.terminal_enabled());
    }

    #[test]
    fn onboard_options_default_has_no_overrides() {
        assert!(!OnboardOptions::default().has_overrides());
    }

    #[test]
    fn workspace_ml_engineer_overlay_reference_matches_embedded_overlay() {
        let s = workspace_ml_engineer_overlay_reference();
        assert!(
            s.contains("ML engineer harness"),
            "reference should include the embedded overlay body"
        );
        assert!(
            s.contains("reference copy"),
            "preamble should explain this is not the live prompt source"
        );
    }

    #[test]
    fn build_config_toml_merge_preserves_ml_engineer_section() {
        let o = OnboardOptions {
            api_enabled: Some(true),
            ..Default::default()
        };
        let s = build_config_toml(&o).expect("toml");
        assert!(
            s.contains("ml_engineer") && s.contains("enabled"),
            "merged config should still carry harness.ml_engineer from template: {}",
            s
        );
    }

    #[test]
    fn build_interactive_config_toml_preserves_template_comments() {
        let o = OnboardOptions {
            provider_name: Some("openai_compatible".to_string()),
            provider_model: Some("test-model".to_string()),
            provider_api_key_env: Some("GEMINI_API_KEY".to_string()),
            provider_base_url: Some("https://example.com/v1/chat/completions".to_string()),
            ..Default::default()
        };
        let s = build_interactive_config_toml(&o).expect("toml_edit merge");
        assert!(
            s.contains("# Local stdin/stdout chat"),
            "expected terminal section comment from template"
        );
        assert!(
            s.contains("# Optional: route web_search"),
            "expected jina section comment from template"
        );
        assert!(
            s.contains("# Execution: docs/execution-user-guide.md"),
            "expected harness execution pointer comment from template"
        );
        assert!(
            s.contains("# cancel_children_on_parent_cancel"),
            "expected subagents optional key as comment"
        );
        assert!(
            s.contains("default_provider = \"local\""),
            "expected execution default_provider from template"
        );
        assert!(
            s.contains("[harness.execution.jupyter]"),
            "expected jupyter subsection from template"
        );
        assert!(
            s.contains("[harness.ml_engineer]") || s.contains("harness.ml_engineer"),
            "expected ml_engineer section preserved from template"
        );
        assert!(
            s.contains("[harness.background_jobs]") || s.contains("harness.background_jobs"),
            "expected background_jobs section preserved from template"
        );
        assert!(
            s.contains("[harness.notifications]") || s.contains("harness.notifications"),
            "expected notifications section preserved from template"
        );
        assert!(
            !s.contains("allow_path_outside_sandbox = false"),
            "allow_path_outside_sandbox should not appear as an active key in template output"
        );
        assert!(s.contains("test-model"));
    }

    #[test]
    fn build_interactive_config_toml_known_provider_drops_base_url() {
        let o = OnboardOptions {
            provider_name: Some("gemini".to_string()),
            provider_model: Some("gemini-2.5-flash".to_string()),
            provider_api_key_env: Some("GEMINI_API_KEY".to_string()),
            provider_base_url: None,
            ..Default::default()
        };
        let s = build_interactive_config_toml(&o).expect("toml_edit merge");
        let doc: toml_edit::DocumentMut = s.parse().expect("parse merged toml");
        let provider = doc["provider"]
            .as_table()
            .expect("[provider] table preserved");
        assert_eq!(
            provider["provider_name"].as_str(),
            Some("gemini"),
            "provider_name should be persisted verbatim"
        );
        assert!(
            !provider.contains_key("base_url"),
            "base_url should be elided for known providers; got: {s}"
        );
    }

    #[test]
    fn build_interactive_config_toml_openai_compatible_keeps_base_url() {
        let o = OnboardOptions {
            provider_name: Some("openai_compatible".to_string()),
            provider_model: Some("custom-model".to_string()),
            provider_api_key_env: Some("MY_KEY".to_string()),
            provider_base_url: Some("https://my-relay.example/v1/chat/completions".to_string()),
            ..Default::default()
        };
        let s = build_interactive_config_toml(&o).expect("toml_edit merge");
        let doc: toml_edit::DocumentMut = s.parse().expect("parse merged toml");
        let provider = doc["provider"].as_table().expect("[provider]");
        assert_eq!(
            provider["provider_name"].as_str(),
            Some("openai_compatible")
        );
        assert_eq!(
            provider["base_url"].as_str(),
            Some("https://my-relay.example/v1/chat/completions"),
        );
    }
}
