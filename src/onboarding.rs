use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{AppConfig, SlackMode};
use crate::skills::SkillRegistry;
use crate::workspace::{ensure_workspace_layout, WorkspaceLayout};
use clap::Args;
use toml_edit::{value, DocumentMut};

const CONFIG_TEMPLATE: &str = include_str!("../assets/onboarding/config.toml");
const AGENTS_TEMPLATE: &str = include_str!("../assets/onboarding/AGENTS.md");
const USER_TEMPLATE: &str = include_str!("../assets/onboarding/USER.md");
const SOUL_TEMPLATE: &str = include_str!("../assets/onboarding/SOUL.md");
const CRON_SKILL_TEMPLATE: &str = include_str!("../assets/onboarding/skills/cron/SKILL.md");
const SKILL_CREATOR_TEMPLATE: &str =
    include_str!("../assets/onboarding/skills/skill-creator/SKILL.md");

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

    /// Sets `[terminal] enable` (stdin/stdout chat).
    #[arg(long, help_heading = "Terminal channel")]
    pub terminal_enable: Option<bool>,

    #[arg(long, help_heading = "Provider")]
    pub provider_model: Option<String>,
    #[arg(long, help_heading = "Provider")]
    pub provider_api_key_env: Option<String>,
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
    pub harness_execution_enabled: Option<bool>,
}

impl OnboardOptions {
    /// True if any override is set (config must be emitted via merge + serialize).
    pub fn has_overrides(&self) -> bool {
        self.restrict_to_workspace.is_some()
            || self.max_iterations.is_some()
            || self.max_tool_output_chars.is_some()
            || self.max_web_tool_output_chars.is_some()
            || self.terminal_enable.is_some()
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
            || self.harness_execution_enabled.is_some()
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
        cfg.terminal.get_or_insert_with(Default::default).enable = Some(v);
    }

    if let Some(p) = cfg.provider.as_mut() {
        if let Some(ref m) = opts.provider_model {
            p.model_name = m.clone();
        }
        if let Some(ref e) = opts.provider_api_key_env {
            p.api_key_env = e.clone();
        }
        if let Some(ref u) = opts.provider_base_url {
            p.base_url = u.clone();
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
        || opts.harness_execution_enabled.is_some()
    {
        let h = cfg.harness.get_or_insert_with(Default::default);
        if let Some(v) = opts.harness_git_worktree_enabled {
            h.git_worktree.get_or_insert_with(Default::default).enabled = Some(v);
        }
        if let Some(v) = opts.harness_subagents_enabled {
            h.subagents.get_or_insert_with(Default::default).enabled = Some(v);
        }
        if let Some(v) = opts.harness_execution_enabled {
            h.execution.get_or_insert_with(Default::default).enabled = Some(v);
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
        doc["terminal"]["enable"] = value(v);
    }

    if let Some(ref m) = options.provider_model {
        doc["provider"]["model_name"] = value(m.as_str());
    }
    if let Some(ref e) = options.provider_api_key_env {
        doc["provider"]["api_key_env"] = value(e.as_str());
    }
    if let Some(ref u) = options.provider_base_url {
        doc["provider"]["base_url"] = value(u.as_str());
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
    if let Some(v) = options.harness_execution_enabled {
        doc["harness"]["execution"]["enabled"] = value(v);
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
    static TEMPLATES: [TemplateFile; 6] = [
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

    Ok(())
}

fn write_if_missing_string(
    root: &Path,
    relative_path: &str,
    contents: &str,
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
    for required in ["cron", "skill-creator"] {
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
    fn build_interactive_config_toml_preserves_template_comments() {
        let o = OnboardOptions {
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
            s.contains("# When default_provider = \"jupyter\""),
            "expected harness jupyter comment from template"
        );
        assert!(
            s.contains("# cancel_children_on_parent_cancel"),
            "expected subagents optional key as comment"
        );
        assert!(
            s.contains("# default_provider = \"local\""),
            "expected execution default_provider as comment"
        );
        assert!(
            s.contains("# max_wall_secs = 300"),
            "expected execution max_wall_secs as comment"
        );
        assert!(
            !s.contains("allow_path_outside_sandbox = false"),
            "allow_path_outside_sandbox should not appear as an active key in template output"
        );
        assert!(s.contains("test-model"));
    }
}
