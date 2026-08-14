//! Plugin and Harness Pack engine — hierarchical discovery, declarative manifests,
//! and runtime integration for skills, subagents, rules, hooks, and MCP servers.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::agent::registry::AgentRegistry;
use crate::skills::SkillRegistry;

/// Manifest metadata for a Plugin or Harness Pack (`plugin.json` or `pack.toml`).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, Default)]
pub struct PluginManifest {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub authors: Option<Vec<String>>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub min_isanagent_version: Option<String>,
    #[serde(default)]
    pub overlay_prompt: Option<String>,
    #[serde(default)]
    pub overlay_file: Option<String>,
}

/// A resolved Plugin / Harness Pack with its discovered component directories.
#[derive(Clone, Debug)]
pub struct Plugin {
    pub name: String,
    pub dir: PathBuf,
    pub manifest: PluginManifest,
    pub agents_dir: Option<PathBuf>,
    pub skills_dir: Option<PathBuf>,
    pub rules_dir: Option<PathBuf>,
    pub hooks_path: Option<PathBuf>,
    pub mcp_config_path: Option<PathBuf>,
}

impl Plugin {
    /// Loads a plugin from a directory containing `plugin.json` or `pack.toml`
    /// (or inferred from an existing `skills/` / `agents/` structure).
    pub fn load_from_dir(dir: &Path) -> Option<Self> {
        if !dir.is_dir() {
            return None;
        }

        let dir_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed-plugin")
            .to_string();

        let plugin_json = dir.join("plugin.json");
        let pack_toml = dir.join("pack.toml");

        let manifest = if plugin_json.is_file() {
            let content = std::fs::read_to_string(&plugin_json).ok()?;
            let mut m: PluginManifest = serde_json::from_str(&content).ok()?;
            if m.name.is_empty() {
                m.name = dir_name.clone();
            }
            m
        } else if pack_toml.is_file() {
            let content = std::fs::read_to_string(&pack_toml).ok()?;
            let mut m: PluginManifest = toml::from_str(&content).ok()?;
            if m.name.is_empty() {
                m.name = dir_name.clone();
            }
            m
        } else {
            // Infer manifest if standard component directories exist
            let has_agents = dir.join("agents").is_dir();
            let has_skills = dir.join("skills").is_dir();
            let has_rules = dir.join("rules").is_dir();
            let has_mcp = dir.join("mcp_config.json").is_file();
            let has_hooks = dir.join("hooks.json").is_file();

            if !has_agents && !has_skills && !has_rules && !has_mcp && !has_hooks {
                return None;
            }

            PluginManifest {
                name: dir_name.clone(),
                description: Some(format!("Discovered plugin '{dir_name}'")),
                ..Default::default()
            }
        };

        let agents_dir = dir.join("agents");
        let skills_dir = dir.join("skills");
        let rules_dir = dir.join("rules");
        let hooks_path = dir.join("hooks.json");
        let mcp_config_path = dir.join("mcp_config.json");

        Some(Self {
            name: manifest.name.clone(),
            dir: dir.to_path_buf(),
            manifest,
            agents_dir: agents_dir.is_dir().then_some(agents_dir),
            skills_dir: skills_dir.is_dir().then_some(skills_dir),
            rules_dir: rules_dir.is_dir().then_some(rules_dir),
            hooks_path: hooks_path.is_file().then_some(hooks_path),
            mcp_config_path: mcp_config_path.is_file().then_some(mcp_config_path),
        })
    }
}

/// Registry holding all discovered plugins.
#[derive(Clone, Debug, Default)]
pub struct PluginRegistry {
    plugins: HashMap<String, Plugin>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Discovers plugins hierarchically from global and workspace customization roots:
    /// 1. Global discovery: `~/.isanagent/plugins/` & `~/.isanagent/packs/`
    /// 2. Workspace discovery: `<workspace>/.agents/plugins/`, `<workspace>/.isanagent/plugins/`
    pub fn discover(workspace_root: &Path, global_root: Option<&Path>) -> Self {
        let mut registry = Self::new();

        // 1. Scan global root
        if let Some(global) = global_root {
            registry.scan_directory(&global.join("plugins"));
            registry.scan_directory(&global.join("packs"));
        }

        // 2. Scan workspace root (overrides global plugins on name collision)
        registry.scan_directory(&workspace_root.join(".agents").join("plugins"));
        registry.scan_directory(&workspace_root.join(".agents").join("packs"));
        registry.scan_directory(&workspace_root.join(".isanagent").join("plugins"));
        registry.scan_directory(&workspace_root.join(".isanagent").join("packs"));
        registry.scan_directory(&workspace_root.join("plugins"));

        registry
    }

    /// Scans a container directory where each immediate subdirectory may be a plugin.
    pub fn scan_directory(&mut self, container_dir: &Path) {
        if !container_dir.is_dir() {
            return;
        }

        let Ok(entries) = std::fs::read_dir(container_dir) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(plugin) = Plugin::load_from_dir(&path) {
                    self.plugins.insert(plugin.name.clone(), plugin);
                }
            }
        }
    }

    pub fn register(&mut self, plugin: Plugin) {
        self.plugins.insert(plugin.name.clone(), plugin);
    }

    pub fn get(&self, name: &str) -> Option<&Plugin> {
        self.plugins.get(name)
    }

    pub fn list(&self) -> Vec<&Plugin> {
        let mut list: Vec<&Plugin> = self.plugins.values().collect();
        list.sort_by_key(|p| &p.name);
        list
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Ingests all plugin subagents into the provided `AgentRegistry`.
    pub fn populate_agent_registry(&self, agent_registry: &mut AgentRegistry) {
        for plugin in self.plugins.values() {
            if let Some(ref agents_dir) = plugin.agents_dir {
                agent_registry.load_from_directory(agents_dir);
            }
        }
    }

    /// Ingests all plugin skills into the provided `SkillRegistry`.
    pub fn populate_skill_registry(&self, skill_registry: &mut SkillRegistry) {
        for plugin in self.plugins.values() {
            if let Some(ref skills_dir) = plugin.skills_dir {
                skill_registry.load_from_directory(skills_dir);
            }
        }
    }

    /// Compiles all active plugin overlay prompts into a merged system prompt section.
    pub fn compile_overlay_prompts(&self) -> String {
        let mut section = String::new();
        for plugin in self.list() {
            if let Some(ref prompt) = plugin.manifest.overlay_prompt {
                let trimmed = prompt.trim();
                if !trimmed.is_empty() {
                    section.push_str(&format!(
                        "\n\n--- Plugin: {} ---\n{}\n",
                        plugin.name, trimmed
                    ));
                }
            } else if let Some(ref file) = plugin.manifest.overlay_file {
                let path = plugin.dir.join(file);
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        section.push_str(&format!(
                            "\n\n--- Plugin: {} ---\n{}\n",
                            plugin.name, trimmed
                        ));
                    }
                }
            }
        }
        section
    }

    /// Clones and installs a remote plugin Git repository into `target_dir/<plugin_name>`.
    pub async fn install_from_repo(
        target_plugins_dir: &Path,
        repo_url: &str,
        custom_name: Option<&str>,
    ) -> Result<Plugin, String> {
        let url = if !repo_url.starts_with("http://")
            && !repo_url.starts_with("https://")
            && !repo_url.starts_with("git@")
        {
            format!("https://github.com/{repo_url}")
        } else {
            repo_url.to_string()
        };

        let leaf = url
            .trim_end_matches('/')
            .split('/')
            .next_back()
            .unwrap_or("plugin")
            .trim_end_matches(".git");
        let inferred_name = leaf.strip_prefix("pack-").unwrap_or(leaf);

        let target_name = custom_name.unwrap_or(inferred_name).trim();
        if target_name.is_empty()
            || target_name.contains('/')
            || target_name.contains('\\')
            || target_name == "."
            || target_name == ".."
        {
            return Err(format!(
                "Invalid plugin name '{target_name}': must not contain path separators or parent directory references"
            ));
        }
        let destination = target_plugins_dir.join(target_name);

        if destination.exists() {
            return Err(format!(
                "Plugin destination already exists: {}",
                destination.display()
            ));
        }

        std::fs::create_dir_all(target_plugins_dir)
            .map_err(|e| format!("Failed to create plugins directory: {e}"))?;

        let output = tokio::process::Command::new("git")
            .arg("clone")
            .arg("--depth")
            .arg("1")
            .arg(&url)
            .arg(&destination)
            .output()
            .await
            .map_err(|e| format!("Failed to execute git clone: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git clone failed: {stderr}"));
        }

        Plugin::load_from_dir(&destination).ok_or_else(|| {
            format!(
                "Installed repository at {} is not a valid plugin",
                destination.display()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_plugin_with_manifest_and_subdirectories() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let plugin_dir = temp_dir.path().join("ml-kit");
        std::fs::create_dir_all(&plugin_dir).expect("mkdir");

        let manifest = r#"{
            "name": "ml-kit",
            "version": "1.0.0",
            "description": "Machine Learning toolkit"
        }"#;
        std::fs::write(plugin_dir.join("plugin.json"), manifest).expect("write manifest");

        let agents_dir = plugin_dir.join("agents");
        std::fs::create_dir_all(&agents_dir).expect("mkdir agents");
        let skills_dir = plugin_dir.join("skills");
        std::fs::create_dir_all(&skills_dir).expect("mkdir skills");

        let plugin = Plugin::load_from_dir(&plugin_dir).expect("loaded plugin");
        assert_eq!(plugin.name, "ml-kit");
        assert_eq!(plugin.manifest.version.as_deref(), Some("1.0.0"));
        assert!(plugin.agents_dir.is_some());
        assert!(plugin.skills_dir.is_some());
        assert!(plugin.rules_dir.is_none());
    }

    #[test]
    fn discovers_plugins_in_workspace() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let plugins_root = temp_dir.path().join(".agents").join("plugins");
        let pack1 = plugins_root.join("pack-one");
        let pack2 = plugins_root.join("pack-two");
        std::fs::create_dir_all(&pack1).expect("mkdir");
        std::fs::create_dir_all(&pack2).expect("mkdir");

        std::fs::write(
            pack1.join("plugin.json"),
            r#"{"name": "pack-one", "description": "first"}"#,
        )
        .expect("write");
        std::fs::write(
            pack2.join("pack.toml"),
            r#"name = "pack-two"
description = "second""#,
        )
        .expect("write");

        let registry = PluginRegistry::discover(temp_dir.path(), None);
        assert_eq!(registry.len(), 2);
        assert!(registry.get("pack-one").is_some());
        assert!(registry.get("pack-two").is_some());
    }
}
