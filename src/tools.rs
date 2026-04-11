use crate::traits::Tool;
use serde_json::Value;
use std::collections::HashMap;

pub mod builtin;

/// A registry that holds available tools for the agent.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn get_tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn list_tools(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.parameters(),
                    }
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
