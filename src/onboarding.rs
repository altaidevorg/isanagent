use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{AppConfig, SlackMode};
use crate::skills::SkillRegistry;
use crate::workspace::{ensure_workspace_layout, WorkspaceLayout};
use clap::Args;

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
/// When any field is set, the written `config.toml` is produced by parse → merge → serialize
/// (embedded template comments are omitted). With all fields unset, the embedded template
/// file is copied verbatim (comments preserved).
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

/// Bootstrap a workspace at `root`, optionally overriding embedded `config.toml` values.
pub fn onboard_workspace(root: &Path, options: &OnboardOptions) -> Result<BootstrapReport, String> {
    let layout = ensure_workspace_layout(root)?;
    let report_root = fs::canonicalize(&layout.root).unwrap_or_else(|_| layout.root.clone());
    let mut report = BootstrapReport::new(report_root);

    write_all_templates(&layout, options, &mut report)?;
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
    report: &mut BootstrapReport,
) -> Result<(), String> {
    for template in embedded_templates() {
        if template.relative_path == "config.toml" {
            let body = if options.has_overrides() {
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
        let mut o = OnboardOptions::default();
        o.api_enabled = Some(true);
        o.api_port = Some(9090);
        o.terminal_enable = Some(false);
        let s = build_config_toml(&o).expect("toml");
        let cfg: AppConfig = toml::from_str(&s).expect("parse back");
        let api = cfg.api.as_ref().expect("api section");
        assert_eq!(api.enabled, Some(true));
        assert_eq!(api.port, 9090);
        assert_eq!(cfg.terminal_enabled(), false);
    }

    #[test]
    fn onboard_options_default_has_no_overrides() {
        assert!(!OnboardOptions::default().has_overrides());
    }
}
