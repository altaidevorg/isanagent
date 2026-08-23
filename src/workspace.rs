use crate::config::AppConfig;
use log::{info, warn};
use shellexpand;
use std::fs;
use std::path::{Path, PathBuf};
use toml;

#[derive(Clone, Debug)]
pub struct WorkspaceLayout {
    pub root: PathBuf,
    pub sandbox_dir: PathBuf,
    pub skills_dir: PathBuf,
}

pub fn resolve_workspace_root(path_override: Option<&str>) -> PathBuf {
    let path_str = path_override.unwrap_or("~/.isanagent");
    PathBuf::from(shellexpand::tilde(path_str).to_string())
}

pub fn ensure_workspace_layout(root: &Path) -> Result<WorkspaceLayout, String> {
    if !root.exists() {
        info!("Creating workspace directory at {root:?}");
    }
    fs::create_dir_all(root).map_err(|e| format!("Failed to create workspace dir: {e}"))?;

    let system_dir = root.join(".system_generated");
    fs::create_dir_all(&system_dir)
        .map_err(|e| format!("Failed to create .system_generated dir: {e}"))?;

    let sandbox_dir = root.join("workspace");
    fs::create_dir_all(&sandbox_dir).map_err(|e| format!("Failed to create sandbox dir: {e}"))?;

    let skills_dir = sandbox_dir.join("skills");
    fs::create_dir_all(&skills_dir).map_err(|e| format!("Failed to create skills dir: {e}"))?;

    Ok(WorkspaceLayout {
        root: root.to_path_buf(),
        sandbox_dir,
        skills_dir,
    })
}

/// Represents the isanagent workspace, serving as the single source of truth
/// for the agent's identity, memory, and skills.
#[derive(Clone, Debug)]
pub struct IsanagentWorkspace {
    pub dir: std::path::PathBuf,
    pub sandbox_dir: std::path::PathBuf,
    pub skills_dir: std::path::PathBuf,
    pub config: AppConfig,
}

impl IsanagentWorkspace {
    /// Initializes a new workspace at the given path.
    /// If no path is provided, it defaults to `~/.isanagent`.
    pub fn new(path_override: Option<&str>, config_override: Option<&str>) -> Result<Self, String> {
        Self::new_with_sandbox(path_override, config_override, None)
    }

    /// Initializes an IsanAgent state workspace with an optional, distinct
    /// project sandbox. Embedders use this to keep durable state separate from
    /// the project files that agent tools may read and edit.
    pub fn new_with_sandbox(
        path_override: Option<&str>,
        config_override: Option<&str>,
        sandbox_override: Option<&Path>,
    ) -> Result<Self, String> {
        let target_dir = resolve_workspace_root(path_override);
        let layout = ensure_workspace_layout(&target_dir)?;
        let sandbox_dir = match sandbox_override {
            Some(path) => {
                if !path.is_dir() {
                    return Err(format!(
                        "Configured sandbox is not a directory: {}",
                        path.display()
                    ));
                }
                path.canonicalize()
                    .map_err(|error| format!("Failed to resolve sandbox directory: {error}"))?
            }
            None => layout.sandbox_dir,
        };

        auto_load_env_files(&target_dir, &sandbox_dir);

        // 3. Load config.toml if it exists
        let config_path = config_override
            .map(|s| PathBuf::from(shellexpand::tilde(s).to_string()))
            .unwrap_or_else(|| target_dir.join("config.toml"));

        let config = if config_path.exists() {
            let toml_str = fs::read_to_string(&config_path)
                .map_err(|e| format!("Failed to read config.toml: {e}"))?;
            toml::from_str(&toml_str).map_err(|e| format!("Failed to parse config.toml: {e}"))?
        } else {
            AppConfig::default()
        };

        Ok(Self {
            dir: layout.root,
            sandbox_dir,
            skills_dir: layout.skills_dir,
            config,
        })
    }

    pub fn db_path(&self) -> PathBuf {
        self.dir.join(".system_generated").join("agent_memory.db")
    }

    pub fn skills_path(&self) -> PathBuf {
        self.skills_dir.clone()
    }

    /// Reads an optional markdown file from the workspace sandbox (e.g., AGENTS.md, USER.md, SOUL.md).
    fn read_md_file(&self, filename: &str) -> Option<String> {
        let file_path = self.sandbox_dir.join(filename);
        if file_path.exists() {
            match fs::read_to_string(&file_path) {
                Ok(content) => {
                    info!("Loaded {filename}");
                    Some(content)
                }
                Err(e) => {
                    warn!("Failed to read {filename}: {e}");
                    None
                }
            }
        } else {
            None
        }
    }

    /// Compiles the base system prompt bypassing the identity files.
    pub fn compile_system_prompt(&self) -> String {
        let mut prompt_parts = Vec::new();

        if let Some(agent) = self.read_md_file("AGENTS.md") {
            prompt_parts.push(format!("--- AGENT INSTRUCTIONS ---\n{agent}\n"));
        }
        if let Some(soul) = self.read_md_file("SOUL.md") {
            prompt_parts.push(format!("--- AGENT PERSONA (SOUL) ---\n{soul}\n"));
        }
        if let Some(user) = self.read_md_file("USER.md") {
            prompt_parts.push(format!("--- USER PROFILE ---\n{user}\n"));
        }
        if let Some(memory) = self.read_md_file("MEMORY.md") {
            prompt_parts.push(format!("--- LONG TERM MEMORY ---\n{memory}\n"));
        }
        // ML_POLICY.md belongs to the `[harness.ml_engineer]` feature: only merge it when that
        // gate is enabled, matching the embedded HARNESS_OVERLAY injection in host.rs.
        if self.config.ml_engineer_harness_enabled() {
            if let Some(ml) = self.read_md_file("ML_POLICY.md") {
                prompt_parts.push(format!(
                    "--- ML / TRAINING POLICY (ML_POLICY.md) ---\n{ml}\n"
                ));
            }
        }

        if prompt_parts.is_empty() {
            "You are isanagent, a helpful AI assistant.".to_string()
        } else {
            prompt_parts.join("\n")
        }
    }
}

/// Load non-empty environment variables from a `.env` file into `std::env` if not already set.
pub fn load_env_file_if_exists(path: &Path) {
    if !path.is_file() {
        return;
    }
    if let Ok(content) = fs::read_to_string(path) {
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let line_to_parse = if let Some(stripped) = trimmed.strip_prefix("export ") {
                stripped.trim()
            } else {
                trimmed
            };
            if let Some((key, val)) = line_to_parse.split_once('=') {
                let key = key.trim();
                let mut val = val.trim();
                if ((val.starts_with('"') && val.ends_with('"'))
                    || (val.starts_with('\'') && val.ends_with('\'')))
                    && val.len() >= 2
                {
                    val = &val[1..val.len() - 1];
                }
                if !key.is_empty() && std::env::var(key).is_err() {
                    std::env::set_var(key, val);
                }
            }
        }
    }
}

/// Automatically inspect current directory, workspace directory, sandbox directory, and user home
/// directory for `.env` and `.env.local` files, loading any missing keys into process environment.
pub fn auto_load_env_files(workspace_dir: &Path, sandbox_dir: &Path) {
    if let Ok(cwd) = std::env::current_dir() {
        load_env_file_if_exists(&cwd.join(".env"));
        load_env_file_if_exists(&cwd.join(".env.local"));
    }
    load_env_file_if_exists(&sandbox_dir.join(".env"));
    load_env_file_if_exists(&sandbox_dir.join(".env.local"));
    load_env_file_if_exists(&workspace_dir.join(".env"));
    load_env_file_if_exists(&workspace_dir.join(".env.local"));
    if let Some(home) = dirs_home_dir() {
        load_env_file_if_exists(&home.join(".env"));
        load_env_file_if_exists(&home.join(".env.local"));
    }
}

fn dirs_home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_override_keeps_state_and_project_roots_distinct() {
        let state = tempfile::tempdir().expect("state directory");
        let project = tempfile::tempdir().expect("project directory");

        let workspace = IsanagentWorkspace::new_with_sandbox(
            Some(state.path().to_str().expect("utf-8 state path")),
            None,
            Some(project.path()),
        )
        .expect("workspace should initialize");

        assert_eq!(workspace.dir, state.path());
        assert_eq!(
            workspace.sandbox_dir,
            project.path().canonicalize().unwrap()
        );
        assert_eq!(
            workspace.skills_path(),
            state.path().join("workspace/skills")
        );
    }
}
