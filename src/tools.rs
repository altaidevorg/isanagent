use crate::traits::Tool;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub mod builtin;
pub mod workflow;

/// Score `(name, description)` entries for a free-text `query`. Higher is better.
pub fn search_tool_index(
    entries: &[(String, String)],
    query: &str,
    limit: usize,
) -> Vec<(String, usize)> {
    let q = query.trim().to_lowercase();
    if q.is_empty() || limit == 0 {
        return Vec::new();
    }

    let mut scored: Vec<(String, usize)> = entries
        .iter()
        .map(|(name, desc)| {
            let name_l = name.to_lowercase();
            let desc_l = desc.to_lowercase();
            let mut score: usize = 0;
            if name_l.contains(&q) {
                score += 120;
            }
            if desc_l.contains(&q) {
                score += 60;
            }
            for token in q.split_whitespace().filter(|t| !t.is_empty()) {
                if name_l.contains(token) {
                    score += 45;
                } else if desc_l.contains(token) {
                    score += 18;
                }
            }
            (name.clone(), score)
        })
        .filter(|(_, s)| *s > 0)
        .collect();

    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(limit);
    scored
}

/// A registry that holds available tools for the agent.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    /// Registration order (name, description) for `search_tools` and stable listing.
    catalog: Arc<RwLock<Vec<(String, String)>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            catalog: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Clone the shared catalog handle (for registering [`workflow::ToolSearchTool`]).
    pub fn catalog_handle(&self) -> Arc<RwLock<Vec<(String, String)>>> {
        Arc::clone(&self.catalog)
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        let desc = tool.description().to_string();
        self.catalog
            .write()
            .expect("catalog write")
            .push((name.clone(), desc));
        self.tools.insert(name, tool);
    }

    pub fn get_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn get_tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn list_tools(&self) -> Vec<Value> {
        let cat = self.catalog.read().expect("catalog read");
        cat.iter()
            .filter_map(|(name, _)| {
                self.tools.get(name).map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name(),
                            "description": t.description(),
                            "parameters": t.parameters(),
                        }
                    })
                })
            })
            .collect()
    }

    pub async fn execute_tool(&self, name: &str, args: Value) -> Result<String, String> {
        if let Some(tool) = self.get_tool(name) {
            tool.execute(args).await
        } else {
            Err(format!("Tool '{}' not found", name))
        }
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod registry_tests {
    use super::{Tool, ToolRegistry};
    use async_trait::async_trait;
    use serde_json::Value;

    struct NamedTool {
        n: &'static str,
    }

    #[async_trait]
    impl Tool for NamedTool {
        fn name(&self) -> &str {
            self.n
        }

        fn description(&self) -> &str {
            "d"
        }

        fn parameters(&self) -> Value {
            serde_json::json!({})
        }

        async fn execute(&self, _: Value) -> Result<String, String> {
            Ok(String::new())
        }
    }

    #[test]
    fn list_tools_follows_registration_order() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(NamedTool { n: "first_tool" }));
        r.register(Box::new(NamedTool { n: "second_tool" }));
        let listed = r.list_tools();
        let names: Vec<String> = listed
            .iter()
            .map(|v| v["function"]["name"].as_str().expect("name").to_string())
            .collect();
        assert_eq!(names, vec!["first_tool", "second_tool"]);
    }
}

#[cfg(test)]
mod tool_index_tests {
    use super::search_tool_index;

    #[test]
    fn search_prefers_name_over_description() {
        let entries = vec![
            ("alpha".to_string(), "does beta things".to_string()),
            ("gamma".to_string(), "alpha keyword only here".to_string()),
        ];
        let hits = search_tool_index(&entries, "alpha", 10);
        assert_eq!(hits[0].0, "alpha");
    }

    #[test]
    fn search_multi_token() {
        let entries = vec![
            ("glob_files".to_string(), "Find paths by glob.".to_string()),
            ("read_file".to_string(), "Read file contents.".to_string()),
        ];
        let hits = search_tool_index(&entries, "glob path", 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, "glob_files");
    }

    #[test]
    fn search_respects_limit() {
        let entries: Vec<_> = (0..20)
            .map(|i| (format!("tool_{}", i), "searchable token xyz".to_string()))
            .collect();
        let hits = search_tool_index(&entries, "xyz", 3);
        assert_eq!(hits.len(), 3);
    }
}
