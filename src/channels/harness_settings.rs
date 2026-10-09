//! Harness toggles edited from the web settings screen.
//!
//! Values are read from and written to `config.toml`. The running process keeps
//! the harness it built at startup, so a successful write always requires a restart.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use toml_edit::{value, DocumentMut};

use crate::config::{AppConfig, ShellPolicyMode};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HarnessSettings {
    pub restrict_to_workspace: bool,
    pub shell_mode: String,
    pub execution_enabled: bool,
    pub subagents_enabled: bool,
    pub builtin_tools_enabled: bool,
    pub ml_engineer_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessSettingsError {
    Io(String),
    Parse(String),
    InvalidShellMode,
}

impl HarnessSettings {
    pub fn from_app(config: &AppConfig) -> Self {
        let shell_mode = match config.resolved_shell_policy().interactive_mode {
            ShellPolicyMode::Ask => "ask",
            ShellPolicyMode::Deny => "deny",
            ShellPolicyMode::Allow => "allow",
        };
        Self {
            restrict_to_workspace: config.restrict_to_workspace.unwrap_or(true),
            shell_mode: shell_mode.to_string(),
            execution_enabled: config.execution_harness_enabled(),
            subagents_enabled: config.subagent_harness_enabled(),
            builtin_tools_enabled: config.builtin_tools_enabled(),
            ml_engineer_enabled: config.ml_engineer_harness_enabled(),
        }
    }

    pub fn defaults() -> Self {
        Self::from_app(&AppConfig::default())
    }
}

pub fn read_harness_settings(path: &Path) -> Result<HarnessSettings, HarnessSettingsError> {
    if !path.is_file() {
        return Ok(HarnessSettings::defaults());
    }
    let raw = fs::read_to_string(path).map_err(|error| {
        HarnessSettingsError::Io(format!("Failed to read {}: {error}", path.display()))
    })?;
    let config: AppConfig = toml::from_str(&raw).map_err(|error| {
        HarnessSettingsError::Parse(format!("Failed to parse {}: {error}", path.display()))
    })?;
    Ok(HarnessSettings::from_app(&config))
}

pub fn write_harness_settings(
    path: &Path,
    patch: &HarnessSettings,
) -> Result<HarnessSettings, HarnessSettingsError> {
    let shell_mode = normalize_shell_mode(&patch.shell_mode)?;
    let raw = if path.is_file() {
        fs::read_to_string(path).map_err(|error| {
            HarnessSettingsError::Io(format!("Failed to read {}: {error}", path.display()))
        })?
    } else {
        String::new()
    };
    let mut doc: DocumentMut = if raw.trim().is_empty() {
        DocumentMut::new()
    } else {
        raw.parse().map_err(|error| {
            HarnessSettingsError::Parse(format!("Failed to parse {}: {error}", path.display()))
        })?
    };

    doc["restrict_to_workspace"] = value(patch.restrict_to_workspace);
    doc["harness"]["shell_policy"]["mode"] = value(shell_mode);
    doc["harness"]["execution"]["enabled"] = value(patch.execution_enabled);
    doc["harness"]["subagents"]["enabled"] = value(patch.subagents_enabled);
    doc["harness"]["builtin_tools"]["enabled"] = value(patch.builtin_tools_enabled);
    doc["harness"]["ml_engineer"]["enabled"] = value(patch.ml_engineer_enabled);

    let updated = doc.to_string();
    let config: AppConfig = toml::from_str(&updated).map_err(|error| {
        HarnessSettingsError::Parse(format!(
            "Updated config would be invalid {}: {error}",
            path.display()
        ))
    })?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| {
                HarnessSettingsError::Io(format!("Failed to create {}: {error}", parent.display()))
            })?;
        }
    }
    fs::write(path, &updated).map_err(|error| {
        HarnessSettingsError::Io(format!("Failed to write {}: {error}", path.display()))
    })?;
    Ok(HarnessSettings::from_app(&config))
}

fn normalize_shell_mode(raw: &str) -> Result<&'static str, HarnessSettingsError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "ask" => Ok("ask"),
        "deny" => Ok("deny"),
        "allow" => Ok("allow"),
        _ => Err(HarnessSettingsError::InvalidShellMode),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> HarnessSettings {
        HarnessSettings {
            restrict_to_workspace: false,
            shell_mode: "deny".to_string(),
            execution_enabled: true,
            subagents_enabled: true,
            builtin_tools_enabled: false,
            ml_engineer_enabled: true,
        }
    }

    #[test]
    fn write_preserves_comments_and_unrelated_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"# workspace boundary
restrict_to_workspace = true

[api]
enabled = true
port = 8099

[harness.shell_policy]
# interactive shell
mode = "ask"
unattended_default = "allow"

[harness.execution]
# off until a provider is configured
enabled = false
"#,
        )
        .expect("seed config");

        let saved = write_harness_settings(&path, &sample()).expect("write");
        assert_eq!(saved, sample());
        let text = fs::read_to_string(&path).expect("read back");
        assert!(text.contains("# workspace boundary"));
        assert!(text.contains("# interactive shell"));
        assert!(text.contains("# off until a provider is configured"));
        assert!(text.contains("unattended_default = \"allow\""));
        assert!(text.contains("port = 8099"));
        let reread = read_harness_settings(&path).expect("reread");
        assert_eq!(reread, sample());
    }

    #[test]
    fn invalid_shell_mode_does_not_touch_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, "restrict_to_workspace = true\n").expect("seed");
        let mut patch = sample();
        patch.shell_mode = "sometimes".to_string();
        let error = write_harness_settings(&path, &patch).unwrap_err();
        assert_eq!(error, HarnessSettingsError::InvalidShellMode);
        let text = fs::read_to_string(&path).expect("unchanged");
        assert_eq!(text, "restrict_to_workspace = true\n");
    }

    #[test]
    fn missing_file_reads_as_process_defaults() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = read_harness_settings(&dir.path().join("config.toml")).expect("defaults");
        assert_eq!(settings, HarnessSettings::defaults());
        assert!(settings.restrict_to_workspace);
        assert_eq!(settings.shell_mode, "ask");
        assert!(!settings.execution_enabled);
        assert!(settings.builtin_tools_enabled);
        assert!(!settings.ml_engineer_enabled);
    }
}
