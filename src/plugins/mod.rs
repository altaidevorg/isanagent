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

/// Host settings for [`PluginRegistry::populate_tool_registry`].
#[derive(Clone, Debug)]
pub struct McpHostOptions {
    pub uv_binary: String,
    /// The agent sandbox: `${SANDBOX_DIR}`, `ISANAGENT_SANDBOX_DIR` and `roots/list`.
    pub sandbox_dir: Option<PathBuf>,
    pub harness: crate::config::McpHarnessConfig,
    pub defer_mode: crate::config::McpDeferMode,
    pub defer_threshold_chars: usize,
    /// False when nothing could load a deferred tool (no `search_tools`): never defer.
    pub defer_supported: bool,
}

impl McpHostOptions {
    pub fn from_config(
        config: &crate::config::AppConfig,
        sandbox_dir: Option<PathBuf>,
        defer_supported: bool,
    ) -> Self {
        let harness = config.mcp_harness();
        Self {
            uv_binary: config.execution_uv_binary(),
            sandbox_dir,
            defer_threshold_chars: harness
                .defer_threshold_chars
                .unwrap_or(crate::config::DEFAULT_MCP_DEFER_THRESHOLD_CHARS),
            harness,
            defer_mode: config.mcp_defer_mode(),
            defer_supported,
        }
    }
}

/// Registry holding all discovered Agent Plugins.
#[derive(Clone, Debug, Default)]
pub struct PluginRegistry {
    /// Name-ordered so everything derived from plugins (tools, prompts) is the same on every
    /// start.
    plugins: std::collections::BTreeMap<String, Plugin>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: std::collections::BTreeMap::new(),
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

    /// Launches every plugin MCP server (concurrently, each within its startup timeout) and
    /// registers its tools as `{server}__{tool}` (see [`crate::tools::mcp::namespaced_tool_name`]).
    ///
    /// Registration follows plugin name, then server name, so the provider `tools` array is the
    /// same on every start. A server that fails or times out is logged and skipped; it never
    /// blocks the others. Tool filters (`includeTools` / `excludeTools`) and deferral follow
    /// `[harness.mcp]` (see [`McpHostOptions`]).
    pub async fn populate_tool_registry(
        &self,
        tool_registry: &mut ToolRegistry,
        opts: &McpHostOptions,
        mcp_apps: &crate::tools::mcp::McpAppBridge,
    ) {
        use crate::tools::mcp::{McpClient, McpConfigFile, McpProxyTool};

        struct Job {
            plugin: String,
            server: String,
            entry: crate::tools::mcp::McpServerEntry,
            launch: crate::tools::mcp::McpLaunchOptions,
            defer: Option<bool>,
        }

        let harness = &opts.harness;
        let mut jobs: Vec<Job> = Vec::new();
        let mut seen_servers: HashMap<String, String> = HashMap::new();
        for plugin in self.plugins.values() {
            let Some(ref mcp_path) = plugin.mcp_config_path else {
                continue;
            };
            let Ok(content) = std::fs::read_to_string(mcp_path) else {
                continue;
            };
            let config = match serde_json::from_str::<McpConfigFile>(&content) {
                Ok(config) => config,
                Err(e) => {
                    log::warn!("Failed to parse MCP config at {}: {e}", mcp_path.display());
                    continue;
                }
            };
            for (server_name, mut entry) in config.mcp_servers {
                let over = harness
                    .servers
                    .get(&server_name)
                    .cloned()
                    .unwrap_or_default();
                if over.enabled == Some(false) {
                    log::info!("MCP server '{server_name}' (plugin '{}') disabled in [harness.mcp.servers]", plugin.name);
                    continue;
                }
                if let Some(previous) = seen_servers.get(&server_name) {
                    log::warn!(
                        "MCP server '{server_name}' in plugin '{}' has the same name as one in plugin '{previous}'; skipping it (tool names would collide). Rename one of the servers.",
                        plugin.name
                    );
                    continue;
                }
                if entry.server_type != "stdio" {
                    log::warn!(
                        "MCP server '{server_name}' (plugin '{}') has type '{}'; only stdio is supported",
                        plugin.name,
                        entry.server_type
                    );
                    continue;
                }
                // Claim the name only for a server that will actually be launched, so a skipped
                // entry never blocks a usable server of the same name in another plugin.
                seen_servers.insert(server_name.clone(), plugin.name.clone());

                if over.include_tools.is_some() {
                    entry.include_tools = over.include_tools.clone();
                }
                entry
                    .exclude_tools
                    .extend(over.exclude_tools.iter().cloned());
                let startup = over
                    .startup_timeout_secs
                    .or(entry.startup_timeout_secs)
                    .or(harness.startup_timeout_secs)
                    .unwrap_or(crate::tools::mcp::DEFAULT_MCP_STARTUP_TIMEOUT_SECS)
                    .max(1);
                let tool_timeout = over
                    .tool_timeout_secs
                    .or(entry.tool_timeout_secs)
                    .or(harness.tool_timeout_secs)
                    .unwrap_or(crate::tools::mcp::DEFAULT_MCP_TOOL_TIMEOUT_SECS)
                    .max(1);
                let mut launch = crate::tools::mcp::McpLaunchOptions::new(
                    server_name.clone(),
                    &plugin.dir,
                    &opts.uv_binary,
                );
                launch.expansion.sandbox_dir = opts.sandbox_dir.clone();
                launch.expansion.env_passthrough = harness
                    .env_passthrough
                    .iter()
                    .chain(over.env_passthrough.iter())
                    .cloned()
                    .collect();
                launch.startup_timeout = std::time::Duration::from_secs(startup);
                launch.tool_timeout = std::time::Duration::from_secs(tool_timeout);
                let defer = over.defer_tools.or(entry.defer_tools);
                jobs.push(Job {
                    plugin: plugin.name.clone(),
                    server: server_name,
                    entry,
                    launch,
                    defer,
                });
            }
        }
        if jobs.is_empty() {
            return;
        }

        // Launch concurrently: one slow server (first `uv run` installing packages) does not
        // hold up the others, and each is bounded by its own startup timeout.
        let launched = futures::future::join_all(jobs.iter().map(|job| async move {
            log::info!(
                "Launching MCP server '{}' for plugin '{}'",
                job.server,
                job.plugin
            );
            let client = McpClient::launch(&job.entry, &job.launch).await?;
            let tools = client.list_tools().await?;
            Ok::<_, String>((std::sync::Arc::new(client), tools))
        }))
        .await;

        // Collect the tools that pass the filters, then decide deferral on the total size.
        struct Ready {
            server: String,
            client: std::sync::Arc<McpClient>,
            tools: Vec<crate::tools::mcp::McpToolDefinition>,
            defer: Option<bool>,
        }
        let mut ready: Vec<Ready> = Vec::new();
        for (job, result) in jobs.iter().zip(launched) {
            match result {
                Ok((client, tools)) => {
                    mcp_apps.retain_client(client.clone());
                    let total = tools.len();
                    let kept: Vec<_> = tools
                        .into_iter()
                        .filter(|def| {
                            if !crate::tools::mcp::tool_visible_to_model(def.meta.as_ref()) {
                                log::info!(
                                    "Skipping app-only MCP tool '{}' from server '{}'",
                                    def.name,
                                    job.server
                                );
                                return false;
                            }
                            crate::tools::mcp::tool_passes_filters(
                                &def.name,
                                job.entry.include_tools.as_deref(),
                                &job.entry.exclude_tools,
                            )
                        })
                        .collect();
                    if kept.len() < total {
                        log::info!(
                            "MCP server '{}': registering {} of {} tools (filters / app-only)",
                            job.server,
                            kept.len(),
                            total
                        );
                    }
                    ready.push(Ready {
                        server: job.server.clone(),
                        client,
                        tools: kept,
                        defer: job.defer,
                    });
                }
                Err(e) => log::warn!(
                    "MCP server '{}' (plugin '{}') is unavailable: {e}",
                    job.server,
                    job.plugin
                ),
            }
        }

        let schema_chars: usize = ready
            .iter()
            .flat_map(|r| r.tools.iter())
            .map(|def| {
                def.name.len()
                    + def.description.as_deref().map_or(0, str::len)
                    + def
                        .input_schema
                        .as_ref()
                        .map_or(0, |schema| schema.to_string().len())
            })
            .sum();
        let auto_defer = match opts.defer_mode {
            crate::config::McpDeferMode::Always => true,
            crate::config::McpDeferMode::Never => false,
            crate::config::McpDeferMode::Auto => schema_chars > opts.defer_threshold_chars,
        };
        if !opts.defer_supported && (auto_defer || ready.iter().any(|r| r.defer == Some(true))) {
            log::warn!(
                "MCP tool schemas would be deferred, but search_tools is unavailable ([harness.builtin_tools] enabled = false); sending every schema instead"
            );
        }

        for r in ready {
            let defer = opts.defer_supported && r.defer.unwrap_or(auto_defer);
            let mut registered = 0usize;
            for def in r.tools {
                let proxy = McpProxyTool::new(def, r.client.clone(), mcp_apps.clone());
                let ok = if defer {
                    tool_registry.register_deferred(Box::new(proxy), &r.server)
                } else {
                    tool_registry.register(Box::new(proxy))
                };
                if ok {
                    registered += 1;
                }
            }
            log::info!(
                "MCP server '{}': {registered} tools registered{}",
                r.server,
                if defer { " (deferred)" } else { "" }
            );
        }
    }

    /// Installs an Agent Plugin into `target_dir/<plugin_name>` from a local directory
    /// or a Git repository (`owner/repo`, `https://…`, `git@…`). An existing local
    /// directory wins over the `owner/repo` shorthand; a full URL always means Git.
    pub async fn install(
        target_plugins_dir: &Path,
        source: &str,
        custom_name: Option<&str>,
    ) -> Result<Plugin, String> {
        let local_dir = if Path::new(source).is_dir() {
            Some(
                std::fs::canonicalize(source)
                    .map_err(|e| format!("Failed to resolve plugin directory {source}: {e}"))?,
            )
        } else {
            None
        };
        let url = if local_dir.is_some()
            || source.starts_with("http://")
            || source.starts_with("https://")
            || source.starts_with("git@")
        {
            source.to_string()
        } else {
            format!("https://github.com/{source}")
        };

        // The canonical path also names `.` after the directory it stands for.
        let local_leaf = local_dir
            .as_deref()
            .and_then(Path::file_name)
            .map(|name| name.to_string_lossy().into_owned());
        let leaf = local_leaf.as_deref().unwrap_or_else(|| {
            url.trim_end_matches('/')
                .split('/')
                .next_back()
                .unwrap_or("plugin")
                .trim_end_matches(".git")
        });
        let inferred_name = leaf
            .strip_prefix("pack-")
            .or_else(|| leaf.strip_prefix("plugin-"))
            .unwrap_or(leaf);

        let target_name = custom_name.unwrap_or(inferred_name).trim();
        validate_plugin_name(target_name)?;

        let destination = target_plugins_dir.join(target_name);

        // Copying a directory into a folder inside it would copy the copy forever,
        // e.g. `isanagent plugin install .` from the workspace root.
        if let Some(dir) = &local_dir {
            if resolve_existing_prefix(target_plugins_dir).starts_with(dir) {
                return Err(format!(
                    "Cannot install {} into {}, which is inside it",
                    dir.display(),
                    target_plugins_dir.display()
                ));
            }
        }

        if destination.exists() {
            return Err(format!(
                "Plugin destination already exists: {}",
                destination.display()
            ));
        }

        std::fs::create_dir_all(target_plugins_dir)
            .map_err(|e| format!("Failed to create plugins directory: {e}"))?;

        // Audit R5: stage the copy or clone in a temporary sibling directory and only
        // move it into place after the plugin validates. A failed/interrupted install
        // can no longer leave a half-copied tree that discovery would pick up as a
        // broken plugin. Rename within the same directory/volume is atomic on all platforms.
        let staging = target_plugins_dir.join(format!(
            ".{}.tmp-{}-{}",
            target_name,
            std::process::id(),
            chrono::Utc::now().timestamp_millis()
        ));
        let _ = std::fs::remove_dir_all(&staging);

        let fetched = match &local_dir {
            Some(dir) => crate::skills::copy_dir_recursive(dir, &staging),
            None => clone_repo(&url, &staging).await,
        };
        // Any failure from here on must not leave staging behind.
        if let Err(e) = fetched {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e);
        }

        let plugin = match Plugin::load_from_dir(&staging) {
            Some(p) => p,
            None => {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(format!("{source} is not a valid plugin"));
            }
        };

        std::fs::rename(&staging, &destination).map_err(|e| {
            let _ = std::fs::remove_dir_all(&staging);
            format!("Failed to move staged plugin into place: {e}")
        })?;

        Ok(plugin)
    }
}

/// `path` with the part that exists resolved like `canonicalize` (symlinks such as
/// macOS `/tmp` → `/private/tmp` included), so a folder that is not created yet
/// can still be compared with a canonical path.
fn resolve_existing_prefix(path: &Path) -> PathBuf {
    let mut missing = Vec::new();
    let mut current = path;
    loop {
        if let Ok(real) = std::fs::canonicalize(current) {
            return missing.iter().rev().fold(real, |p, part| p.join(part));
        }
        match (current.parent(), current.file_name()) {
            (Some(parent), Some(name)) => {
                missing.push(name.to_os_string());
                current = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// Shallow `git clone` of `url` into `dest`.
async fn clone_repo(url: &str, dest: &Path) -> Result<(), String> {
    let output = tokio::process::Command::new("git")
        .arg("clone")
        .arg("--depth")
        .arg("1")
        .arg(url)
        .arg(dest)
        .output()
        .await
        .map_err(|e| format!("Failed to execute git clone: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git clone failed: {stderr}"));
    }
    Ok(())
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

    fn write_fake_mcp_plugin(container: &Path, plugin: &str, servers: serde_json::Value) {
        let dir = container.join(plugin);
        std::fs::create_dir_all(&dir).expect("mkdir plugin");
        std::fs::write(
            dir.join("plugin.json"),
            format!(r#"{{"name": "{plugin}", "version": "0.1.0"}}"#),
        )
        .expect("manifest");
        std::fs::write(
            dir.join("mcp.json"),
            serde_json::json!({ "mcpServers": servers }).to_string(),
        )
        .expect("mcp.json");
    }

    fn fake_server_entry(extra: serde_json::Value) -> serde_json::Value {
        let script =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_mcp_server.py");
        let mut entry = serde_json::json!({
            "command": "python3",
            "args": ["-u", script.to_string_lossy()],
        });
        if let (Some(obj), Some(extra)) = (entry.as_object_mut(), extra.as_object()) {
            for (k, v) in extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        entry
    }

    fn host_options(defer_mode: crate::config::McpDeferMode) -> McpHostOptions {
        McpHostOptions {
            uv_binary: "uv".to_string(),
            sandbox_dir: None,
            harness: crate::config::McpHarnessConfig::default(),
            defer_mode,
            defer_threshold_chars: crate::config::DEFAULT_MCP_DEFER_THRESHOLD_CHARS,
            defer_supported: true,
        }
    }

    #[tokio::test]
    async fn mcp_tools_are_namespaced_filtered_and_isolated_from_broken_servers() {
        if which::which("python3").is_err() {
            return;
        }
        let root = tempfile::tempdir().expect("tempdir");
        write_fake_mcp_plugin(
            root.path(),
            "chem-plugin",
            serde_json::json!({
                "chem": fake_server_entry(serde_json::json!({"excludeTools": ["crash", "slow"]})),
                "broken": {"command": "definitely-not-a-real-binary-xyz"},
            }),
        );
        // A second plugin reusing the server name `chem` is skipped (names would collide).
        write_fake_mcp_plugin(
            root.path(),
            "other-plugin",
            serde_json::json!({ "chem": fake_server_entry(serde_json::json!({})) }),
        );
        let mut registry = PluginRegistry::new();
        registry.scan_directory(root.path());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(crate::tools::builtin::ReadFileTool {
            workspace_dir: root.path().to_path_buf(),
            restrict_to_workspace: true,
        }));
        let apps = crate::tools::mcp::McpAppBridge::new();
        registry
            .populate_tool_registry(
                &mut tools,
                &host_options(crate::config::McpDeferMode::Never),
                &apps,
            )
            .await;

        let mut names = tools.get_tool_names();
        names.sort();
        assert_eq!(
            names,
            vec![
                "chem__cancelled",
                "chem__echo",
                "chem__env",
                "chem__roots",
                "read_file"
            ]
        );
        let out = tools
            .execute_tool("chem__echo", serde_json::json!({"text": "routed"}))
            .await
            .expect("namespaced call reaches the original tool");
        assert_eq!(out, "routed");
    }

    #[tokio::test]
    async fn a_skipped_server_does_not_claim_its_name() {
        if which::which("python3").is_err() {
            return;
        }
        let root = tempfile::tempdir().expect("tempdir");
        // `a-plugin` sorts first: its unsupported `sse` entry must not block `b-plugin`'s `chem`.
        write_fake_mcp_plugin(
            root.path(),
            "a-plugin",
            serde_json::json!({ "chem": {"type": "sse", "command": "unused"} }),
        );
        write_fake_mcp_plugin(
            root.path(),
            "b-plugin",
            serde_json::json!({ "chem": fake_server_entry(serde_json::json!({})) }),
        );
        let mut registry = PluginRegistry::new();
        registry.scan_directory(root.path());
        let mut tools = ToolRegistry::new();
        let apps = crate::tools::mcp::McpAppBridge::new();
        registry
            .populate_tool_registry(
                &mut tools,
                &host_options(crate::config::McpDeferMode::Never),
                &apps,
            )
            .await;
        assert!(tools.get_tool("chem__echo").is_some());
    }

    #[tokio::test]
    async fn per_server_defer_keeps_schemas_out_until_activated() {
        if which::which("python3").is_err() {
            return;
        }
        let root = tempfile::tempdir().expect("tempdir");
        write_fake_mcp_plugin(
            root.path(),
            "chem-plugin",
            serde_json::json!({ "chem": fake_server_entry(serde_json::json!({"deferTools": true})) }),
        );
        let mut registry = PluginRegistry::new();
        registry.scan_directory(root.path());
        let mut tools = ToolRegistry::new();
        let apps = crate::tools::mcp::McpAppBridge::new();
        registry
            .populate_tool_registry(
                &mut tools,
                &host_options(crate::config::McpDeferMode::Never),
                &apps,
            )
            .await;
        assert_eq!(tools.get_tool_names().len(), 6);
        assert!(tools
            .list_tools_for_session(None, false, Some("s"))
            .is_empty());
        tools
            .deferred_handle()
            .activate_matching("s", &["chem__e*".to_string()]);
        assert_eq!(
            tools.list_tools_for_session(None, false, Some("s")).len(),
            2
        );
    }

    #[tokio::test]
    async fn installs_from_a_local_directory() {
        let source_root = tempfile::tempdir().expect("tempdir");
        let source = source_root.path().join("plugin-ml-kit");
        std::fs::create_dir_all(source.join("skills")).expect("mkdir");
        std::fs::write(
            source.join("plugin.json"),
            r#"{"name": "ml-kit", "version": "1.2.0"}"#,
        )
        .expect("write manifest");
        let source_arg = source.display().to_string();
        let target = tempfile::tempdir().expect("tempdir");
        let plugins = target.path().join("plugins");
        let entries = || std::fs::read_dir(&plugins).expect("read plugins").count();

        let plugin = PluginRegistry::install(&plugins, &source_arg, None)
            .await
            .expect("install");
        assert_eq!(plugin.manifest.version.as_deref(), Some("1.2.0"));
        // Named like a repo install: the `plugin-` prefix is dropped.
        assert!(plugins.join("ml-kit").join("plugin.json").is_file());
        assert!(source.join("plugin.json").is_file(), "source left in place");
        assert_eq!(entries(), 1, "no staging directory left behind");

        assert!(PluginRegistry::install(&plugins, &source_arg, None)
            .await
            .is_err());
        let not_a_plugin = source_root.path().join("notes");
        std::fs::create_dir_all(&not_a_plugin).expect("mkdir");
        let err = PluginRegistry::install(&plugins, &not_a_plugin.display().to_string(), None)
            .await
            .expect_err("not a plugin");
        assert!(err.contains("is not a valid plugin"), "{err}");
        assert_eq!(entries(), 1);

        let inside = source.join(".agents").join("plugins");
        let err = PluginRegistry::install(&inside, &source_arg, Some("copy"))
            .await
            .expect_err("install into itself");
        assert!(err.contains("which is inside it"), "{err}");
        assert!(!inside.exists(), "a refused install creates nothing");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_install_keeps_symlinks() {
        // e.g. a virtualenv's `lib64 -> lib`: a link to a directory cannot be copied as a file.
        let source_root = tempfile::tempdir().expect("tempdir");
        let source = source_root.path().join("venv-kit");
        std::fs::create_dir_all(source.join("skills")).expect("mkdir");
        std::fs::create_dir_all(source.join(".venv").join("lib")).expect("mkdir");
        std::os::unix::fs::symlink("lib", source.join(".venv").join("lib64")).expect("symlink");
        let target = tempfile::tempdir().expect("tempdir");

        PluginRegistry::install(target.path(), &source.display().to_string(), None)
            .await
            .expect("install");
        let link = target.path().join("venv-kit").join(".venv").join("lib64");
        assert_eq!(
            std::fs::read_link(link).expect("still a link"),
            Path::new("lib")
        );
    }

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
