use std::fs;
use std::path::{Path, PathBuf};

use crate::config::AppConfig;
use crate::skills::SkillRegistry;
use crate::workspace::{ensure_workspace_layout, WorkspaceLayout};

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

pub fn onboard_workspace(root: &Path) -> Result<BootstrapReport, String> {
    let layout = ensure_workspace_layout(root)?;
    let report_root = fs::canonicalize(&layout.root).unwrap_or_else(|_| layout.root.clone());
    let mut report = BootstrapReport::new(report_root);

    write_all_templates(&layout, &mut report)?;
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
    report: &mut BootstrapReport,
) -> Result<(), String> {
    for template in embedded_templates() {
        write_if_missing(
            &layout.root,
            template.relative_path,
            template.contents,
            report,
        )?;
    }

    Ok(())
}

fn write_if_missing(
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
        let metadata = fs::metadata(&path).map_err(|e| {
            format!(
                "Required onboarding file is missing or inaccessible ({}): {}",
                path.display(),
                e
            )
        })?;
        if !metadata.is_file() {
            return Err(format!(
                "Required onboarding file is not a regular file: {}",
                path.display()
            ));
        }
        fs::read_to_string(&path).map_err(|e| {
            format!(
                "Required onboarding file is not readable UTF-8 ({}): {}",
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
