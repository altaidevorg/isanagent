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
    /// `[harness.permissions].inside_workspace`: file tools on paths inside the sandbox
    /// (`allow` / `ask` / `deny`). Always set in responses; omit in a write to keep the file's value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inside_workspace: Option<String>,
    /// `[harness.permissions].outside_workspace`: file tools on paths outside the sandbox.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outside_workspace: Option<String>,
    /// Legacy view of `outside_workspace` (`true` = outside access is not free, i.e. `ask` or
    /// `deny`). Responses report it for older clients; a write uses it only when
    /// `outside_workspace` is absent (`true` → `ask`, `false` → `allow`, matching how the legacy
    /// config key is read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restrict_to_workspace: Option<bool>,
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
    InvalidPermissionMode,
}

impl HarnessSettings {
    pub fn from_app(config: &AppConfig) -> Self {
        let policy = config.resolved_shell_policy();
        let outside = policy.permissions.outside_workspace;
        Self {
            inside_workspace: Some(policy.permissions.inside_workspace.as_str().to_string()),
            outside_workspace: Some(outside.as_str().to_string()),
            restrict_to_workspace: Some(outside != ShellPolicyMode::Allow),
            shell_mode: policy.interactive_mode.as_str().to_string(),
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
    let inside = patch
        .inside_workspace
        .as_deref()
        .map(normalize_permission_mode)
        .transpose()?;
    let outside = match patch.outside_workspace.as_deref() {
        Some(raw) => Some(normalize_permission_mode(raw)?),
        None => patch
            .restrict_to_workspace
            .map(|restrict| if restrict { "ask" } else { "allow" }),
    };
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

    if let Some(inside) = inside {
        doc["harness"]["permissions"]["inside_workspace"] = value(inside);
    }
    if let Some(outside) = outside {
        doc["harness"]["permissions"]["outside_workspace"] = value(outside);
        // The new key wins over the legacy one anyway; drop it so the file has one source.
        doc.remove("restrict_to_workspace");
    }
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
    ShellPolicyMode::parse(raw)
        .map(ShellPolicyMode::as_str)
        .ok_or(HarnessSettingsError::InvalidShellMode)
}

fn normalize_permission_mode(raw: &str) -> Result<&'static str, HarnessSettingsError> {
    ShellPolicyMode::parse(raw)
        .map(ShellPolicyMode::as_str)
        .ok_or(HarnessSettingsError::InvalidPermissionMode)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> HarnessSettings {
        HarnessSettings {
            inside_workspace: Some("ask".to_string()),
            outside_workspace: Some("allow".to_string()),
            restrict_to_workspace: Some(false),
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
        assert!(!text.contains("restrict_to_workspace"), "{text}");
        assert!(text.contains("inside_workspace = \"ask\""), "{text}");
        assert!(text.contains("outside_workspace = \"allow\""), "{text}");
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
        assert_eq!(settings.inside_workspace.as_deref(), Some("allow"));
        assert_eq!(settings.outside_workspace.as_deref(), Some("ask"));
        assert_eq!(settings.restrict_to_workspace, Some(true));
        assert_eq!(settings.shell_mode, "ask");
        assert!(!settings.execution_enabled);
        assert!(settings.builtin_tools_enabled);
        assert!(!settings.ml_engineer_enabled);
    }

    #[test]
    fn legacy_restrict_flag_reads_and_writes_through_the_new_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, "restrict_to_workspace = true\n").expect("seed");
        let read = read_harness_settings(&path).expect("read");
        assert_eq!(read.outside_workspace.as_deref(), Some("ask"));
        assert_eq!(read.restrict_to_workspace, Some(true));

        // An older client that only knows the boolean.
        let mut patch = read.clone();
        patch.inside_workspace = None;
        patch.outside_workspace = None;
        patch.restrict_to_workspace = Some(false);
        let saved = write_harness_settings(&path, &patch).expect("write");
        assert_eq!(saved.outside_workspace.as_deref(), Some("allow"));
        let text = fs::read_to_string(&path).expect("read back");
        assert!(!text.contains("restrict_to_workspace"), "{text}");
        patch.restrict_to_workspace = Some(true);
        let saved = write_harness_settings(&path, &patch).expect("write");
        assert_eq!(saved.outside_workspace.as_deref(), Some("ask"));
    }

    #[test]
    fn invalid_permission_mode_does_not_touch_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, "restrict_to_workspace = true\n").expect("seed");
        let mut patch = sample();
        patch.outside_workspace = Some("sometimes".to_string());
        let error = write_harness_settings(&path, &patch).unwrap_err();
        assert_eq!(error, HarnessSettingsError::InvalidPermissionMode);
        assert_eq!(
            fs::read_to_string(&path).expect("unchanged"),
            "restrict_to_workspace = true\n"
        );
    }
}
