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
        info!("Creating workspace directory at {:?}", root);
    }
    fs::create_dir_all(root).map_err(|e| format!("Failed to create workspace dir: {}", e))?;

    let system_dir = root.join(".system_generated");
    fs::create_dir_all(&system_dir)
        .map_err(|e| format!("Failed to create .system_generated dir: {}", e))?;

    let sandbox_dir = root.join("workspace");
    fs::create_dir_all(&sandbox_dir).map_err(|e| format!("Failed to create sandbox dir: {}", e))?;

    let skills_dir = sandbox_dir.join("skills");
    fs::create_dir_all(&skills_dir).map_err(|e| format!("Failed to create skills dir: {}", e))?;

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

        // 3. Load config.toml if it exists
        let config_path = config_override
            .map(|s| PathBuf::from(shellexpand::tilde(s).to_string()))
            .unwrap_or_else(|| target_dir.join("config.toml"));

        let config = if config_path.exists() {
            let toml_str = fs::read_to_string(&config_path)
                .map_err(|e| format!("Failed to read config.toml: {}", e))?;
            toml::from_str(&toml_str).map_err(|e| format!("Failed to parse config.toml: {}", e))?
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
                    info!("Loaded {}", filename);
                    Some(content)
                }
                Err(e) => {
                    warn!("Failed to read {}: {}", filename, e);
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
            prompt_parts.push(format!("--- AGENT INSTRUCTIONS ---\n{}\n", agent));
        }
        if let Some(soul) = self.read_md_file("SOUL.md") {
            prompt_parts.push(format!("--- AGENT PERSONA (SOUL) ---\n{}\n", soul));
        }
        if let Some(user) = self.read_md_file("USER.md") {
            prompt_parts.push(format!("--- USER PROFILE ---\n{}\n", user));
        }
        if let Some(memory) = self.read_md_file("MEMORY.md") {
            prompt_parts.push(format!("--- LONG TERM MEMORY ---\n{}\n", memory));
        }
        if let Some(ml) = self.read_md_file("ML_POLICY.md") {
            prompt_parts.push(format!(
                "--- ML / TRAINING POLICY (ML_POLICY.md) ---\n{}\n",
                ml
            ));
        }

        if prompt_parts.is_empty() {
            "You are isanagent, a helpful AI assistant.".to_string()
        } else {
            prompt_parts.join("\n")
        }
    }
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
