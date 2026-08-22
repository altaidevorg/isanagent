//! Agent registry — loads named agent definitions and builds per-agent providers.

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::{AgentDefinition, AgentMode};

/// A resolved agent definition with loaded system prompt text.
#[derive(Clone, Debug)]
pub struct AgentManifest {
    pub name: String,
    pub description: String,
    pub mode: AgentMode,
    pub system_prompt: String,
    pub allowed_tools: Option<Vec<String>>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_iterations: Option<usize>,
    pub hidden: bool,
    pub color: Option<String>,
}

impl AgentManifest {
    pub fn allowed_tools_set(&self) -> Option<Vec<String>> {
        self.allowed_tools.clone()
    }
}

/// Frontmatter metadata parsed from `AGENT.md` or `AGENT.markdown` files.
#[derive(Clone, Debug, serde::Deserialize, Default)]
pub struct AgentFrontmatter {
    pub name: Option<String>,
    pub description: Option<String>,
    pub mode: Option<AgentMode>,
    pub allowed_tools: Option<Vec<String>>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_iterations: Option<usize>,
    pub hidden: Option<bool>,
    pub color: Option<String>,
}

/// Parses an `AGENT.md` file extracting YAML frontmatter and using the Markdown body as system prompt.
pub fn parse_agent_md(path: &std::path::Path) -> Option<AgentManifest> {
    let content = std::fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return None;
    }

    let default_name = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .filter(|&n| n != "agents" && n != ".")
        .or_else(|| {
            path.file_stem()
                .and_then(|n| n.to_str())
                .filter(|&n| n != "AGENT" && n != "agent")
        })
        .unwrap_or("unnamed-agent")
        .to_string();

    let first_non_empty = lines.iter().position(|l| !l.trim().is_empty())?;

    if lines[first_non_empty].trim() != "---" {
        // No frontmatter, treat entire file as system prompt
        return Some(AgentManifest {
            name: default_name,
            description: String::new(),
            mode: AgentMode::Subagent,
            system_prompt: content.trim().to_string(),
            allowed_tools: None,
            model: None,
            temperature: None,
            max_iterations: None,
            hidden: false,
            color: None,
        });
    }

    let mut end_idx = 0;
    for (i, &line) in lines.iter().enumerate().skip(first_non_empty + 1) {
        if line.trim() == "---" {
            end_idx = i;
            break;
        }
    }

    if end_idx == 0 {
        return None;
    }

    let frontmatter_str = lines[first_non_empty + 1..end_idx].join("\n");
    let body_str = lines[end_idx + 1..].join("\n").trim().to_string();

    let meta: AgentFrontmatter = serde_yaml::from_str(&frontmatter_str).ok()?;

    let name = meta.name.unwrap_or(default_name);
    let description = meta.description.unwrap_or_default();
    let mode = meta.mode.unwrap_or(AgentMode::Subagent);

    Some(AgentManifest {
        name,
        description,
        mode,
        system_prompt: body_str,
        allowed_tools: meta.allowed_tools,
        model: meta.model,
        temperature: meta.temperature,
        max_iterations: meta.max_iterations,
        hidden: meta.hidden.unwrap_or(false),
        color: meta.color,
    })
}

/// Holds all loaded named agents and provides spawn-time lookups.
#[derive(Clone, Debug, Default)]
pub struct AgentRegistry {
    agents: HashMap<String, Arc<AgentManifest>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    pub fn register(&mut self, manifest: AgentManifest) {
        self.agents
            .insert(manifest.name.clone(), Arc::new(manifest));
    }

    /// Load all `AGENT.md` subagent declarations from an agents directory tree.
    /// Traverses both direct files (`agents/<name>.md`) and directories (`agents/<name>/AGENT.md`).
    pub fn load_from_directory(&mut self, agents_dir: &std::path::Path) {
        if !agents_dir.is_dir() {
            return;
        }
        let Ok(entries) = std::fs::read_dir(agents_dir) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let agent_md = path.join("AGENT.md");
                if agent_md.is_file() {
                    if let Some(manifest) = parse_agent_md(&agent_md) {
                        self.register(manifest);
                    }
                } else {
                    let agent_md_lower = path.join("agent.md");
                    if agent_md_lower.is_file() {
                        if let Some(manifest) = parse_agent_md(&agent_md_lower) {
                            self.register(manifest);
                        }
                    }
                }
            } else if path.is_file() {
                let file_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                if (file_name.ends_with(".md") || file_name.ends_with(".markdown"))
                    && file_name != "README.md"
                {
                    if let Some(manifest) = parse_agent_md(&path) {
                        self.register(manifest);
                    }
                }
            }
        }
    }

    /// Merges another `AgentRegistry` into this one, overwriting duplicates.
    pub fn merge(&mut self, other: AgentRegistry) {
        for (name, manifest) in other.agents {
            self.agents.insert(name, manifest);
        }
    }

    pub fn from_definitions(
        definitions: &HashMap<String, AgentDefinition>,
        sandbox_dir: &std::path::Path,
    ) -> Self {
        let agents: HashMap<String, Arc<AgentManifest>> = definitions
            .iter()
            .map(|(name, def)| {
                let system_prompt = resolve_system_prompt(def, sandbox_dir);
                (
                    name.clone(),
                    Arc::new(AgentManifest {
                        name: name.clone(),
                        description: def.description.clone(),
                        mode: def.mode.clone(),
                        system_prompt,
                        allowed_tools: def.allowed_tools.clone(),
                        model: def.model.clone(),
                        temperature: def.temperature,
                        max_iterations: def.max_iterations,
                        hidden: def.hidden,
                        color: def.color.clone(),
                    }),
                )
            })
            .collect();
        Self { agents }
    }

    pub fn get(&self, name: &str) -> Option<&Arc<AgentManifest>> {
        self.agents.get(name)
    }

    pub fn list(&self) -> Vec<&Arc<AgentManifest>> {
        let mut names: Vec<&String> = self.agents.keys().collect();
        names.sort();
        names
            .into_iter()
            .filter_map(|n| self.agents.get(n))
            .collect()
    }

    pub fn list_visible(&self) -> Vec<&Arc<AgentManifest>> {
        self.list().into_iter().filter(|m| !m.hidden).collect()
    }

    pub fn compile_agent_prompt_section(&self) -> String {
        let visible = self.list_visible();
        if visible.is_empty() {
            return String::new();
        }
        let mut s = String::from("\n\n## Available Specialized Agents\n\n");
        s.push_str("You are a coordinator. Delegate work to specialized agents using the `subagent_spawn` tool. ");
        s.push_str(
            "Use `agent_list` to refresh your knowledge of available agents at any time.\n\n",
        );
        for m in &visible {
            let tools_summary = if m.mode == AgentMode::SembleScout {
                "local Semble code search only (no model, arbitrary shell, or project writes)"
                    .to_string()
            } else {
                match &m.allowed_tools {
                    None => "inherits harness allowlist".to_string(),
                    Some(v) if v.is_empty() => "read-only, no tools".to_string(),
                    Some(v) => v.join(", "),
                }
            };
            let iter_hint = match m.max_iterations {
                Some(n) => format!(", max {n} iterations"),
                None => String::new(),
            };
            s.push_str(&format!(
                "- **{}**: {}. Tools: {}{}\n",
                m.name, m.description, tools_summary, iter_hint
            ));
        }
        s.push_str("\nGuidelines:\n- For research: delegate to a research-capable agent.\n");
        s.push_str("- For exploring an unfamiliar local codebase, delegate to Semble Scout before broad grep or full-file reads.\n");
        s.push_str("- For code changes: delegate to a coder agent.\n");
        s.push_str("- For review: delegate to a read-only review agent.\n");
        s.push_str("- Use `wait=false` for parallel work; `wait=true` when result is needed.\n");
        s.push_str("- Check agent status with `task_list`; audit completed work with `task_history_list`.\n");
        s
    }

    pub fn len(&self) -> usize {
        self.agents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

fn resolve_system_prompt(def: &AgentDefinition, sandbox_dir: &std::path::Path) -> String {
    if let Some(ref inline) = def.system_prompt {
        let trimmed = inline.trim();
        if !trimmed.is_empty() {
            if let Some(path) = trimmed
                .strip_prefix("{file:")
                .and_then(|s| s.strip_suffix('}'))
            {
                if let Some(resolved) = crate::utils::resolve_path(sandbox_dir, path.trim()) {
                    if let Ok(content) = std::fs::read_to_string(&resolved) {
                        return content;
                    }
                    log::warn!("Agent system_prompt file not found: {}", resolved.display());
                } else {
                    log::warn!(
                        "Agent system_prompt file path is outside sandbox or does not exist: {}",
                        path.trim()
                    );
                }
            }
            return trimmed.to_string();
        }
    }
    if let Some(ref file_path) = def.system_prompt_file {
        if let Some(resolved) = crate::utils::resolve_path(sandbox_dir, file_path.trim()) {
            if let Ok(content) = std::fs::read_to_string(&resolved) {
                return content;
            }
            log::warn!("Agent system_prompt_file not found: {}", resolved.display());
        } else {
            log::warn!(
                "Agent system_prompt_file path is outside sandbox or does not exist: {}",
                file_path.trim()
            );
        }
    }
    if def.description.is_empty() {
        "You are a specialized sub-agent. Complete the assigned task thoroughly.".to_string()
    } else {
        format!(
            "You are a specialized sub-agent: {}. Complete the assigned task thoroughly.",
            def.description
        )
    }
}

/// Built-in named-agent defaults. `ml_domain_enabled` mirrors the config gate of the
/// same name (audit X4): when false, the `researcher` allowlist omits arXiv tools so
/// `agent_list` never advertises tools that are not registered.
pub fn default_agent_definitions(ml_domain_enabled: bool) -> HashMap<String, AgentDefinition> {
    let mut map = HashMap::new();
    map.insert(
        "semble-scout".to_string(),
        AgentDefinition {
            description: "Find the most relevant local code snippets for a natural-language or symbol query using Semble".to_string(),
            mode: AgentMode::SembleScout,
            system_prompt: None,
            system_prompt_file: None,
            allowed_tools: Some(vec![]),
            model: None,
            temperature: None,
            max_iterations: None,
            hidden: false,
            color: Some("#8B5CF6".into()),
        },
    );
    map.insert(
        "researcher".to_string(),
        AgentDefinition {
            description: "Research topics, find papers, and gather context from web and files"
                .to_string(),
            mode: AgentMode::Subagent,
            system_prompt: None,
            system_prompt_file: None,
            allowed_tools: Some({
                let mut tools = vec![
                    "web_search".to_string(),
                    "web_fetch".to_string(),
                    "read_file".to_string(),
                    "search_text".to_string(),
                    "glob_files".to_string(),
                    "list_dir".to_string(),
                    "search_memory".to_string(),
                    "fetch_memory_by_date".to_string(),
                    "exec_status".to_string(),
                    "exec_send".to_string(),
                    "execution_job_status".to_string(),
                    "execution_job_result".to_string(),
                    "execution_artifact_list".to_string(),
                    "todo_write".to_string(),
                    "recall_tool_result".to_string(),
                ];
                // Audit X4: arXiv tools only when the ML domain gate is on.
                if ml_domain_enabled {
                    tools.push("arxiv_search".to_string());
                    tools.push("arxiv_fetch".to_string());
                }
                tools
            }),
            model: None,
            temperature: Some(0.1),
            max_iterations: Some(15),
            hidden: false,
            color: Some("#2196F3".into()),
        },
    );
    map.insert(
        "coder".to_string(),
        AgentDefinition {
            description: "Implement code changes, write tests, refactor, and fix bugs".to_string(),
            mode: AgentMode::Subagent,
            system_prompt: None,
            system_prompt_file: None,
            allowed_tools: None,
            model: None,
            temperature: Some(0.2),
            max_iterations: Some(30),
            hidden: false,
            color: Some("#4CAF50".into()),
        },
    );
    map.insert(
        "evaluator".to_string(),
        AgentDefinition {
            description: "Review code, find bugs, assess quality, and audit changes".to_string(),
            mode: AgentMode::Subagent,
            system_prompt: None,
            system_prompt_file: None,
            allowed_tools: Some(vec![
                "read_file".into(),
                "search_text".into(),
                "glob_files".into(),
                "list_dir".into(),
                "web_search".into(),
                "web_fetch".into(),
                "exec_status".into(),
                "exec_send".into(),
                "execution_job_status".into(),
                "execution_job_result".into(),
                "execution_artifact_list".into(),
                "todo_write".into(),
                "recall_tool_result".into(),
            ]),
            model: None,
            temperature: Some(0.0),
            max_iterations: Some(12),
            hidden: false,
            color: Some("#FF9800".into()),
        },
    );
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Default for AgentDefinition {
        fn default() -> Self {
            Self {
                description: String::new(),
                mode: AgentMode::Subagent,
                system_prompt: None,
                system_prompt_file: None,
                allowed_tools: None,
                model: None,
                temperature: None,
                max_iterations: None,
                hidden: false,
                color: None,
            }
        }
    }

    #[test]
    fn registry_from_definitions_resolves_inline_prompt() {
        let mut defs = HashMap::new();
        defs.insert(
            "test".into(),
            AgentDefinition {
                description: "tester".into(),
                system_prompt: Some("Be testy.".into()),
                ..Default::default()
            },
        );
        let reg = AgentRegistry::from_definitions(&defs, std::path::Path::new("."));
        let m = reg.get("test").expect("present");
        assert_eq!(m.system_prompt, "Be testy.");
    }

    #[test]
    fn default_agents_have_expected_roles() {
        let defs = default_agent_definitions(false);
        assert_eq!(defs.len(), 4);
        assert!(matches!(
            defs.get("semble-scout").map(|agent| agent.mode.clone()),
            Some(AgentMode::SembleScout)
        ));
        assert!(defs.contains_key("researcher"));
        assert!(defs.contains_key("coder"));
        assert!(defs.contains_key("evaluator"));
    }

    #[test]
    fn researcher_allowlist_gates_ml_tools_on_flag() {
        // Audit X4: arXiv tool names appear in the built-in researcher allowlist
        // only when the ML domain gate is enabled.
        let off = default_agent_definitions(false);
        let on = default_agent_definitions(true);
        let researcher_off = off.get("researcher").expect("researcher");
        let researcher_on = on.get("researcher").expect("researcher");
        let allowed_off = researcher_off.allowed_tools.as_ref().expect("tools");
        let allowed_on = researcher_on.allowed_tools.as_ref().expect("tools");
        assert!(!allowed_off.iter().any(|t| t.starts_with("arxiv_")));
        assert!(allowed_on.contains(&"arxiv_search".to_string()));
        assert!(allowed_on.contains(&"arxiv_fetch".to_string()));
        assert_eq!(allowed_off.len() + 2, allowed_on.len());
    }

    #[test]
    fn compile_agent_prompt_includes_visible_agents() {
        let mut defs = HashMap::new();
        defs.insert(
            "visible".into(),
            AgentDefinition {
                description: "seen".into(),
                allowed_tools: Some(vec!["read_file".into()]),
                ..Default::default()
            },
        );
        defs.insert(
            "hidden".into(),
            AgentDefinition {
                description: "secret".into(),
                hidden: true,
                ..Default::default()
            },
        );
        let reg = AgentRegistry::from_definitions(&defs, std::path::Path::new("."));
        let prompt = reg.compile_agent_prompt_section();
        assert!(prompt.contains("visible"));
        assert!(prompt.contains("seen"));
        assert!(!prompt.contains("secret"));
    }

    #[test]
    fn parse_agent_md_extracts_frontmatter_and_prompt() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let agent_file = temp_dir.path().join("AGENT.md");
        let content = r#"---
name: custom_analyst
description: Expert in system analysis
model: gpt-4o
temperature: 0.15
max_iterations: 25
allowed_tools:
  - read_file
  - search_text
---

# Analysis Expert
You are a principal system analyst. Follow strict verification.
"#;
        std::fs::write(&agent_file, content).expect("write");

        let manifest = parse_agent_md(&agent_file).expect("parsed manifest");
        assert_eq!(manifest.name, "custom_analyst");
        assert_eq!(manifest.description, "Expert in system analysis");
        assert_eq!(manifest.model.as_deref(), Some("gpt-4o"));
        assert_eq!(manifest.temperature, Some(0.15));
        assert_eq!(manifest.max_iterations, Some(25));
        assert_eq!(
            manifest.allowed_tools,
            Some(vec!["read_file".into(), "search_text".into()])
        );
        assert_eq!(
            manifest.system_prompt,
            "# Analysis Expert\nYou are a principal system analyst. Follow strict verification."
        );
    }

    #[test]
    fn load_from_directory_scans_nested_agent_directories() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let agent_dir = temp_dir.path().join("researcher");
        std::fs::create_dir_all(&agent_dir).expect("mkdir");
        let agent_file = agent_dir.join("AGENT.md");
        let content = r#"---
description: Deep web researcher
temperature: 0.0
---

Always cite sources.
"#;
        std::fs::write(&agent_file, content).expect("write");

        let mut reg = AgentRegistry::new();
        reg.load_from_directory(temp_dir.path());

        let m = reg.get("researcher").expect("found researcher agent");
        assert_eq!(m.name, "researcher");
        assert_eq!(m.description, "Deep web researcher");
        assert_eq!(m.system_prompt, "Always cite sources.");
    }
}
