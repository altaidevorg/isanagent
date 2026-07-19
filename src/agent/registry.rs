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

/// Holds all loaded named agents and provides spawn-time lookups.
#[derive(Clone, Debug, Default)]
pub struct AgentRegistry {
    agents: HashMap<String, Arc<AgentManifest>>,
}

impl AgentRegistry {
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
                Some(n) => format!(", max {} iterations", n),
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

pub fn default_agent_definitions() -> HashMap<String, AgentDefinition> {
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
            allowed_tools: Some(vec![
                "web_search".into(),
                "web_fetch".into(),
                "arxiv_search".into(),
                "arxiv_fetch".into(),
                "read_file".into(),
                "search_text".into(),
                "glob_files".into(),
                "list_dir".into(),
                "search_memory".into(),
                "fetch_memory_by_date".into(),
            ]),
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
        let defs = default_agent_definitions();
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
}
