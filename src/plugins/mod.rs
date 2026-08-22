//! Agent Plugins 1.0 Engine
//!
//! Conforms to the open Agent Plugins 1.0 Specification (`agent-plugins.org`).
//! Provides hierarchical discovery, standard manifest loading (`plugin.json`),
//! component integration (`skills/`, `mcp.json`), and reverse-domain extension
//! namespaces (`dev.altai.isanagent`) for declarative subagents, rules, hooks, and prompt overlays.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::agent::registry::AgentRegistry;
use crate::skills::SkillRegistry;
use crate::tools::ToolRegistry;

/// Canonical Agent Plugins 1.0 JSON Schema URL.
pub const AGENT_PLUGINS_SCHEMA_URL: &str =
    "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json";

/// Primary reverse-domain extension namespace for isanagent.
pub const ISANAGENT_EXTENSION_NAMESPACE: &str = "dev.altai.isanagent";

/// Recognized extension namespaces scanned for client-specific components.
pub const RECOGNIZED_EXTENSION_NAMESPACES: &[&str] = &[
    "dev.altai.isanagent",
    "dev.altai",
    "com.google.antigravity",
    "com.github.copilot",
];

/// Manifest metadata for an Agent Plugin (`plugin.json`).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, Default)]
pub struct PluginManifest {
    #[serde(rename = "$schema", default)]
    pub schema: Option<String>,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub author: Option<serde_json::Value>,
    #[serde(default)]
    pub homepage: Option<String>,
    #[serde(default)]
    pub repository: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub keywords: Option<Vec<String>>,
    #[serde(default)]
    pub extensions: HashMap<String, serde_json::Value>,

    // Compatibility fields
    #[serde(default)]
    pub overlay_prompt: Option<String>,
    #[serde(default)]
    pub overlay_file: Option<String>,
    #[serde(default)]
    pub min_isanagent_version: Option<String>,
}

impl PluginManifest {
    /// Extracts the prompt overlay from `extensions["dev.altai.isanagent"]` or legacy fields.
    pub fn get_overlay_prompt(&self) -> Option<String> {
        for ns in RECOGNIZED_EXTENSION_NAMESPACES {
            if let Some(ext) = self.extensions.get(*ns) {
                if let Some(p) = ext.get("overlay_prompt").and_then(|v| v.as_str()) {
                    let trimmed = p.trim().to_string();
                    if !trimmed.is_empty() {
                        return Some(trimmed);
                    }
                }
            }
        }
        self.overlay_prompt
            .as_deref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Extracts the overlay prompt file path from `extensions["dev.altai.isanagent"]` or legacy fields.
    pub fn get_overlay_file(&self) -> Option<String> {
        for ns in RECOGNIZED_EXTENSION_NAMESPACES {
            if let Some(ext) = self.extensions.get(*ns) {
                if let Some(f) = ext.get("overlay_file").and_then(|v| v.as_str()) {
                    let trimmed = f.trim().to_string();
                    if !trimmed.is_empty() {
                        return Some(trimmed);
                    }
                }
            }
        }
        self.overlay_file
            .as_deref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }
}

/// A resolved Agent Plugin with its discovered component directories.
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
    /// Loads an Agent Plugin from a directory conforming to Agent Plugins 1.0
    /// (or inferred from an existing directory layout).
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
            if let Err(err) = validate_plugin_name(&m.name) {
                log::warn!(
                    "Plugin manifest at {} has invalid name: {err}",
                    plugin_json.display()
                );
                return None;
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
            let has_agents = dir.join("agents").is_dir()
                || RECOGNIZED_EXTENSION_NAMESPACES
                    .iter()
                    .any(|ns| dir.join(ns).join("agents").is_dir());
            let has_skills = dir.join("skills").is_dir();
            let has_rules = dir.join("rules").is_dir()
                || RECOGNIZED_EXTENSION_NAMESPACES
                    .iter()
                    .any(|ns| dir.join(ns).join("rules").is_dir());
            let has_mcp = dir.join("mcp.json").is_file() || dir.join("mcp_config.json").is_file();
            let has_hooks = dir.join("hooks.json").is_file()
                || RECOGNIZED_EXTENSION_NAMESPACES
                    .iter()
                    .any(|ns| dir.join(ns).join("hooks.json").is_file());

            if !has_agents && !has_skills && !has_rules && !has_mcp && !has_hooks {
                return None;
            }

            PluginManifest {
                name: dir_name.clone(),
                description: Some(format!("Discovered plugin '{dir_name}'")),
                ..Default::default()
            }
        };

        // 1. Discover skills/ (standard Agent Plugins 1.0 location)
        let skills_dir = {
            let root_skills = dir.join("skills");
            if root_skills.is_dir() {
                Some(root_skills)
            } else {
                let mut found = None;
                for ns in RECOGNIZED_EXTENSION_NAMESPACES {
                    let ns_skills = dir.join(ns).join("skills");
                    if ns_skills.is_dir() {
                        found = Some(ns_skills);
                        break;
                    }
                }
                found
            }
        };

        // 2. Discover MCP servers: mcp.json (standard) or mcp_config.json (legacy)
        let mcp_config_path = {
            let root_mcp = dir.join("mcp.json");
            let root_legacy = dir.join("mcp_config.json");
            if root_mcp.is_file() {
                Some(root_mcp)
            } else if root_legacy.is_file() {
                Some(root_legacy)
            } else {
                let mut found = None;
                for ns in RECOGNIZED_EXTENSION_NAMESPACES {
                    let ns_mcp = dir.join(ns).join("mcp.json");
                    if ns_mcp.is_file() {
                        found = Some(ns_mcp);
                        break;
                    }
                }
                found
            }
        };

        // 3. Discover declarative subagents (dev.altai.isanagent/agents/ or agents/)
        let agents_dir = {
            let mut found = None;
            for ns in RECOGNIZED_EXTENSION_NAMESPACES {
                let ns_agents = dir.join(ns).join("agents");
                if ns_agents.is_dir() {
                    found = Some(ns_agents);
                    break;
                }
            }
            if found.is_none() {
                let root_agents = dir.join("agents");
                if root_agents.is_dir() {
                    found = Some(root_agents);
                }
            }
            found
        };

        // 4. Discover rules (dev.altai.isanagent/rules/ or rules/)
        let rules_dir = {
            let mut found = None;
            for ns in RECOGNIZED_EXTENSION_NAMESPACES {
                let ns_rules = dir.join(ns).join("rules");
                if ns_rules.is_dir() {
                    found = Some(ns_rules);
                    break;
                }
            }
            if found.is_none() {
                let root_rules = dir.join("rules");
                if root_rules.is_dir() {
                    found = Some(root_rules);
                }
            }
            found
        };

        // 5. Discover hooks (dev.altai.isanagent/hooks.json or hooks.json)
        let hooks_path = {
            let mut found = None;
            for ns in RECOGNIZED_EXTENSION_NAMESPACES {
                let ns_hooks = dir.join(ns).join("hooks.json");
                if ns_hooks.is_file() {
                    found = Some(ns_hooks);
                    break;
                }
            }
            if found.is_none() {
                let root_hooks = dir.join("hooks.json");
                if root_hooks.is_file() {
                    found = Some(root_hooks);
                }
            }
            found
        };

        Some(Self {
            name: manifest.name.clone(),
            dir: dir.to_path_buf(),
            manifest,
            agents_dir,
            skills_dir,
            rules_dir,
            hooks_path,
            mcp_config_path,
        })
    }
}

/// Registry holding all discovered Agent Plugins.
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

    /// Discovers Agent Plugins hierarchically from global and workspace customization roots:
    /// 1. Global discovery: `~/.agent-plugins/` & `~/.isanagent/plugins/`
    /// 2. Workspace discovery: `<workspace>/.agents/plugins/`, `<workspace>/.isanagent/plugins/`, `<workspace>/plugins/`
    pub fn discover(workspace_root: &Path, global_root: Option<&Path>) -> Self {
        let mut registry = Self::new();

        // 1. Scan user home standard Agent Plugins root (~/.agent-plugins/)
        if let Some(user_home) = dirs_next_home_dir() {
            registry.scan_directory(&user_home.join(".agent-plugins"));
        }

        // 2. Scan global root (~/.isanagent/plugins/)
        if let Some(global) = global_root {
            registry.scan_directory(&global.join("plugins"));
            registry.scan_directory(&global.join("packs"));
        }

        // 3. Scan workspace root (overrides global plugins on name collision)
        registry.scan_directory(&workspace_root.join(".agents").join("plugins"));
        registry.scan_directory(&workspace_root.join(".agents").join("packs"));
        registry.scan_directory(&workspace_root.join(".isanagent").join("plugins"));
        registry.scan_directory(&workspace_root.join(".isanagent").join("packs"));
        registry.scan_directory(&workspace_root.join("plugins"));
        registry.scan_directory(&workspace_root.join("assets").join("plugins"));

        registry
    }

    /// Scans a container directory where each immediate subdirectory may be an Agent Plugin.
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
            if let Some(prompt) = plugin.manifest.get_overlay_prompt() {
                let trimmed = prompt.trim();
                if !trimmed.is_empty() {
                    section.push_str(&format!(
                        "\n\n--- Plugin: {} ---\n{}\n",
                        plugin.name, trimmed
                    ));
                }
            } else if let Some(file) = plugin.manifest.get_overlay_file() {
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

    /// Ingests and initializes all plugin MCP tools into the provided `ToolRegistry`.
    pub async fn populate_tool_registry(&self, tool_registry: &mut ToolRegistry, uv_binary: &str) {
        for plugin in self.plugins.values() {
            if let Some(ref mcp_path) = plugin.mcp_config_path {
                let Ok(content) = std::fs::read_to_string(mcp_path) else {
                    continue;
                };
                let Ok(config) = serde_json::from_str::<crate::tools::mcp::McpConfigFile>(&content)
                else {
                    log::warn!("Failed to parse MCP config at {}", mcp_path.display());
                    continue;
                };

                for (server_name, entry) in config.mcp_servers {
                    log::info!(
                        "Launching MCP server '{server_name}' for plugin '{}'",
                        plugin.name
                    );
                    match crate::tools::mcp::McpClient::launch(&plugin.dir, &entry, uv_binary).await
                    {
                        Ok(client) => {
                            let client_arc = std::sync::Arc::new(client);
                            match client_arc.list_tools().await {
                                Ok(tools) => {
                                    for tool_def in tools {
                                        log::info!(
                                            "Registering MCP tool '{}' from plugin '{}'",
                                            tool_def.name,
                                            plugin.name
                                        );
                                        let proxy = crate::tools::mcp::McpProxyTool::new(
                                            tool_def,
                                            client_arc.clone(),
                                        );
                                        tool_registry.register(Box::new(proxy));
                                    }
                                }
                                Err(e) => {
                                    log::warn!(
                                        "Failed to list tools for MCP server '{server_name}' in plugin '{}': {e}",
                                        plugin.name
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            log::warn!(
                                "Failed to launch MCP server '{server_name}' for plugin '{}': {e}",
                                plugin.name
                            );
                        }
                    }
                }
            }
        }
    }

    /// Clones and installs a remote Agent Plugin Git repository into `target_dir/<plugin_name>`.
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
        let inferred_name = leaf
            .strip_prefix("pack-")
            .or_else(|| leaf.strip_prefix("plugin-"))
            .unwrap_or(leaf);

        let target_name = custom_name.unwrap_or(inferred_name).trim();
        validate_plugin_name(target_name)?;

        let destination = target_plugins_dir.join(target_name);

        if destination.exists() {
            return Err(format!(
                "Plugin destination already exists: {}",
                destination.display()
            ));
        }

        std::fs::create_dir_all(target_plugins_dir)
            .map_err(|e| format!("Failed to create plugins directory: {e}"))?;

        // Audit R5: stage the clone in a temporary sibling directory and only move it
        // into place after the plugin validates. A failed/interrupted install can no
        // longer leave a half-cloned tree that discovery would pick up as a broken
        // plugin. Rename within the same directory/volume is atomic on all platforms.
        let staging = target_plugins_dir.join(format!(
            ".{}.tmp-{}-{}",
            target_name,
            std::process::id(),
            chrono::Utc::now().timestamp_millis()
        ));
        let _ = std::fs::remove_dir_all(&staging);

        let output = tokio::process::Command::new("git")
            .arg("clone")
            .arg("--depth")
            .arg("1")
            .arg(&url)
            .arg(&staging)
            .output()
            .await
            .map_err(|e| format!("Failed to execute git clone: {e}"));

        // Any failure from here on must not leave staging behind.
        let output = match output {
            Ok(o) => o,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(e);
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = std::fs::remove_dir_all(&staging);
            return Err(format!("git clone failed: {stderr}"));
        }

        let plugin = match Plugin::load_from_dir(&staging) {
            Some(p) => p,
            None => {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(format!(
                    "Cloned repository at {} is not a valid plugin",
                    staging.display()
                ));
            }
        };

        std::fs::rename(&staging, &destination).map_err(|e| {
            let _ = std::fs::remove_dir_all(&staging);
            format!("Failed to move staged plugin into place: {e}")
        })?;

        Ok(plugin)
    }
}

/// Validates a plugin name against the Agent Plugins 1.0 Specification (§5.5).
/// Name must be 1–64 characters, using only lowercase alphanumeric characters, hyphens, and periods,
/// starting and ending with an alphanumeric character, and without consecutive separators (`--`, `..`, `.-`, `-.`).
pub fn validate_plugin_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Plugin name cannot be empty".to_string());
    }
    if name.len() > 64 {
        return Err(format!(
            "Plugin name '{name}' exceeds maximum length of 64 characters (has {})",
            name.len()
        ));
    }

    let bytes = name.as_bytes();
    let first = bytes[0];
    let last = bytes[bytes.len() - 1];

    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return Err(format!(
            "Plugin name '{name}' must start with a lowercase alphanumeric character"
        ));
    }
    if !last.is_ascii_lowercase() && !last.is_ascii_digit() {
        return Err(format!(
            "Plugin name '{name}' must end with a lowercase alphanumeric character"
        ));
    }

    let mut prev = '\0';
    for ch in name.chars() {
        if !ch.is_ascii_lowercase() && !ch.is_ascii_digit() && ch != '-' && ch != '.' {
            return Err(format!(
                "Plugin name '{name}' contains invalid character '{ch}': only lowercase alphanumeric, '-' and '.' are allowed"
            ));
        }
        if (ch == '-' || ch == '.') && (prev == '-' || prev == '.') {
            return Err(format!(
                "Plugin name '{name}' must not contain consecutive separators ('{prev}{ch}')"
            ));
        }
        prev = ch;
    }

    Ok(())
}

/// Resolves a plugin-relative path ensuring containment within the plugin root (§4.1).
pub fn resolve_contained_path(plugin_root: &Path, rel_path: &str) -> Result<PathBuf, String> {
    let raw = Path::new(rel_path);
    if raw.is_absolute() {
        return Err(format!(
            "Absolute path '{rel_path}' is not allowed as a plugin-relative path"
        ));
    }

    let mut out = plugin_root.to_path_buf();
    for comp in raw.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if out == plugin_root {
                    return Err(format!(
                        "Path '{rel_path}' escapes plugin root '{}'",
                        plugin_root.display()
                    ));
                }
                out.pop();
            }
            std::path::Component::Normal(name) => out.push(name),
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(format!(
                    "Path '{rel_path}' contains forbidden root or prefix component"
                ));
            }
        }
    }

    // If target exists on disk, check symlink resolution doesn't escape plugin_root
    if out.exists() && plugin_root.exists() {
        if let (Ok(canon_out), Ok(canon_root)) = (out.canonicalize(), plugin_root.canonicalize()) {
            if !canon_out.starts_with(&canon_root) {
                return Err(format!(
                    "Path '{}' canonicalizes to '{}' outside plugin root '{}'",
                    rel_path,
                    canon_out.display(),
                    canon_root.display()
                ));
            }
        }
    }

    Ok(out)
}

fn dirs_next_home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                let drive = std::env::var("HOMEDRIVE").ok()?;
                let path = std::env::var("HOMEPATH").ok()?;
                Some(PathBuf::from(format!("{drive}{path}")))
            })
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_agent_plugins_1_0_spec_manifest() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let plugin_dir = temp_dir.path().join("ml-kit");
        std::fs::create_dir_all(&plugin_dir).expect("mkdir");

        let manifest = r#"{
            "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
            "name": "ml-kit",
            "version": "1.0.0",
            "description": "Machine Learning toolkit",
            "author": {
                "name": "Altai Dev",
                "url": "https://altai.dev"
            },
            "license": "Apache-2.0",
            "keywords": ["ml", "pytorch"],
            "extensions": {
                "dev.altai.isanagent": {
                    "overlay_prompt": "Always use float32 precision for benchmark baselines."
                }
            }
        }"#;
        std::fs::write(plugin_dir.join("plugin.json"), manifest).expect("write manifest");

        let ns_agents_dir = plugin_dir.join("dev.altai.isanagent").join("agents");
        std::fs::create_dir_all(&ns_agents_dir).expect("mkdir ns agents");
        let skills_dir = plugin_dir.join("skills");
        std::fs::create_dir_all(&skills_dir).expect("mkdir skills");
        std::fs::write(plugin_dir.join("mcp.json"), r#"{"mcpServers":{}}"#)
            .expect("write mcp.json");

        let plugin = Plugin::load_from_dir(&plugin_dir).expect("loaded plugin");
        assert_eq!(plugin.name, "ml-kit");
        assert_eq!(plugin.manifest.version.as_deref(), Some("1.0.0"));
        assert_eq!(
            plugin.manifest.get_overlay_prompt().as_deref(),
            Some("Always use float32 precision for benchmark baselines.")
        );
        assert!(plugin.agents_dir.is_some());
        assert!(plugin.skills_dir.is_some());
        assert!(plugin.mcp_config_path.is_some());
        assert!(plugin.rules_dir.is_none());
    }

    #[test]
    fn discovers_plugins_in_workspace() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let plugins_root = temp_dir.path().join(".agents").join("plugins");
        let plugin1 = plugins_root.join("plugin-one");
        let plugin2 = plugins_root.join("plugin-two");
        std::fs::create_dir_all(&plugin1).expect("mkdir");
        std::fs::create_dir_all(&plugin2).expect("mkdir");

        std::fs::write(
            plugin1.join("plugin.json"),
            r#"{"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "plugin-one", "description": "first"}"#,
        )
        .expect("write");
        std::fs::write(
            plugin2.join("plugin.json"),
            r#"{"name": "plugin-two", "description": "second"}"#,
        )
        .expect("write");

        let registry = PluginRegistry::discover(temp_dir.path(), None);
        assert_eq!(registry.len(), 2);
        assert!(registry.get("plugin-one").is_some());
        assert!(registry.get("plugin-two").is_some());
    }

    #[test]
    fn validates_spec_conforming_plugin_names() {
        assert!(validate_plugin_name("kernel-porting").is_ok());
        assert!(validate_plugin_name("autotrainess").is_ok());
        assert!(validate_plugin_name("tool.v1").is_ok());
        assert!(validate_plugin_name("a").is_ok());
        assert!(validate_plugin_name("0").is_ok());

        assert!(validate_plugin_name("").is_err());
        assert!(validate_plugin_name("-abc").is_err());
        assert!(validate_plugin_name("abc-").is_err());
        assert!(validate_plugin_name(".abc").is_err());
        assert!(validate_plugin_name("abc.").is_err());
        assert!(validate_plugin_name("a--b").is_err());
        assert!(validate_plugin_name("a..b").is_err());
        assert!(validate_plugin_name("a.-b").is_err());
        assert!(validate_plugin_name("a-.b").is_err());
        assert!(validate_plugin_name("ML-Kit").is_err());
        assert!(validate_plugin_name("a/b").is_err());
        assert!(validate_plugin_name("a b").is_err());
    }

    #[test]
    fn enforces_path_containment() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let plugin_dir = temp_dir.path().join("my-plugin");
        std::fs::create_dir_all(&plugin_dir).expect("mkdir");

        let valid = resolve_contained_path(&plugin_dir, "./skills/test").expect("valid path");
        assert!(valid.starts_with(&plugin_dir));

        let escaping = resolve_contained_path(&plugin_dir, "../outside");
        assert!(escaping.is_err());
    }
}
