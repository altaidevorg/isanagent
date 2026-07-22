use crate::traits::{MutationPreview, Tool, ToolErrorCode, ToolResult};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

pub mod autotrainess;
pub mod builtin;
pub mod compact;
pub mod execution;
pub mod isanagent_ignore;
pub mod kernel_porting;
pub mod ml_domain;
pub mod recall;
pub mod workflow;

/// Convert one legacy `Result<String, String>` into the typed contract. This is
/// the only text-classification fallback in the executor: natively typed tool
/// results bypass it completely and therefore cannot be overwritten by prose.
fn normalize_legacy_tool_result(tool_name: &str, mut result: ToolResult) -> ToolResult {
    if !result.is_legacy() {
        return result;
    }
    result.mark_normalized();
    if result.is_error() {
        return result;
    }
    if !crate::utils::tool_output_signals_failure(tool_name, &result.content) {
        return result;
    }

    let code = if matches!(tool_name, "exec" | "python_run") {
        ToolErrorCode::NonZeroExit
    } else {
        ToolErrorCode::LegacyReportedFailure
    };
    let message = match code {
        ToolErrorCode::NonZeroExit => format!("{tool_name} exited with a non-zero status"),
        _ => format!("{tool_name} reported a failure"),
    };
    ToolResult::error_with_content(code, message, result.content)
}

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

    /// Execute through the typed central boundary. Typed tool implementations
    /// are authoritative; only legacy results pass through the compatibility
    /// classifier above.
    pub async fn execute_tool_result(&self, name: &str, args: Value) -> ToolResult {
        match self.get_tool(name) {
            Some(tool) => normalize_legacy_tool_result(
                name,
                tool.execute_with_approved_mutation_typed(args, None).await,
            ),
            None => ToolResult::error(ToolErrorCode::NotFound, format!("Tool '{name}' not found")),
        }
    }

    /// Ask a tool for its file-mutation preview without executing it.
    pub async fn preview_mutation_scoped(
        &self,
        name: &str,
        args: &Value,
        allowlist: Option<&HashSet<String>>,
        is_subagent: bool,
    ) -> Result<Option<MutationPreview>, String> {
        self.authorize_scoped(name, allowlist, is_subagent)?;
        match self.get_tool(name) {
            Some(tool) => tool.preview_mutation(args).await,
            None => Err(format!("Tool '{}' not found", name)),
        }
    }

    /// Execute with a preview previously approved by the user.
    pub async fn execute_tool_scoped_with_approved_mutation(
        &self,
        name: &str,
        args: Value,
        approved_preview: Option<&MutationPreview>,
        allowlist: Option<&HashSet<String>>,
        is_subagent: bool,
    ) -> Result<String, String> {
        self.authorize_scoped(name, allowlist, is_subagent)?;
        match self.get_tool(name) {
            Some(tool) => {
                tool.execute_with_approved_mutation(args, approved_preview)
                    .await
            }
            None => Err(format!("Tool '{}' not found", name)),
        }
    }

    /// Typed mutation execution used by the reasoning loop.
    pub async fn execute_tool_scoped_with_approved_mutation_result(
        &self,
        name: &str,
        args: Value,
        approved_preview: Option<&MutationPreview>,
        allowlist: Option<&HashSet<String>>,
        is_subagent: bool,
    ) -> ToolResult {
        if let Err(error) = self.authorize_scoped(name, allowlist, is_subagent) {
            return ToolResult::error(ToolErrorCode::NotAllowed, error);
        }
        match self.get_tool(name) {
            Some(tool) => normalize_legacy_tool_result(
                name,
                tool.execute_with_approved_mutation_typed(args, approved_preview)
                    .await,
            ),
            None => ToolResult::error(ToolErrorCode::NotFound, format!("Tool '{name}' not found")),
        }
    }

    fn authorize_scoped(
        &self,
        name: &str,
        allowlist: Option<&HashSet<String>>,
        is_subagent: bool,
    ) -> Result<(), String> {
        if is_subagent && Self::is_subagent_restricted_tool(name) {
            return Err(format!(
                "Tool '{}' is not available inside a sub-agent run",
                name
            ));
        }
        if let Some(set) = allowlist {
            if !set.is_empty() && !set.contains(name) {
                return Err(format!(
                    "Tool '{}' is not allowed for this sub-agent (allowlist)",
                    name
                ));
            }
        }
        Ok(())
    }

    /// Tools that must not run inside a sub-agent loop (prevents unbounded recursion).
    pub fn is_subagent_restricted_tool(name: &str) -> bool {
        matches!(name, "subagent_spawn" | "subagent_plan_execute")
    }

    /// Read-only or side-effect-free tools safe to run concurrently (same assistant turn).
    pub fn is_parallel_safe_tool(name: &str) -> bool {
        matches!(
            name,
            "read_file"
                | "glob_files"
                | "list_dir"
                | "search_text"
                | "web_search"
                | "web_fetch"
                | "search_memory"
                | "fetch_memory_by_date"
                | "search_tools"
                | "load_skill_instructions"
                | "arxiv_search"
                | "arxiv_fetch"
                | "hf_hub_file_fetch"
                | "execution_env_info"
                | "task_history_list"
        )
    }

    /// Tool list for provider calls when `is_subagent` is true and/or an allowlist applies.
    pub fn list_tools_scoped(
        &self,
        allowlist: Option<&HashSet<String>>,
        is_subagent: bool,
    ) -> Vec<Value> {
        let cat = self.catalog.read().expect("catalog read");
        cat.iter()
            .filter_map(|(name, _)| {
                if is_subagent && Self::is_subagent_restricted_tool(name) {
                    return None;
                }
                if let Some(set) = allowlist {
                    if !set.is_empty() && !set.contains(name) {
                        return None;
                    }
                }
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

    /// Execute with sub-agent allowlist and nested-tool restrictions.
    pub async fn execute_tool_scoped(
        &self,
        name: &str,
        args: Value,
        allowlist: Option<&HashSet<String>>,
        is_subagent: bool,
    ) -> Result<String, String> {
        self.authorize_scoped(name, allowlist, is_subagent)?;
        self.execute_tool(name, args).await
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

#[cfg(test)]
mod scoped_tools_tests {
    use super::{Tool, ToolRegistry};
    use async_trait::async_trait;
    use serde_json::Value;
    use std::collections::HashSet;

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

    #[tokio::test]
    async fn execute_scoped_denies_nested_spawn_in_subagent() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(NamedTool {
            n: "subagent_spawn",
        }));
        let err = r
            .execute_tool_scoped("subagent_spawn", Value::Null, None, true)
            .await
            .unwrap_err();
        assert!(err.contains("not available"));
    }

    #[tokio::test]
    async fn system_initiated_ask_user_bypasses_subagent_allowlist() {
        // Regression for PR #62 review feedback: system-initiated approval prompts
        // (shell + edit policy gates) call ask_user with allowlist=None so a
        // restricted sub-agent allowlist (e.g. {write_file} / {exec}) can still
        // surface the approval dialog.
        let mut r = ToolRegistry::new();
        r.register(Box::new(NamedTool { n: "ask_user" }));
        r.register(Box::new(NamedTool { n: "write_file" }));

        // A sub-agent allowlisted to {write_file} (no ask_user) blocks a direct
        // ask_user call — this is the failure the None-allowlist bypass avoids.
        let allow: HashSet<String> = ["write_file".to_string()].into_iter().collect();
        let err = r
            .execute_tool_scoped("ask_user", Value::Null, Some(&allow), true)
            .await
            .unwrap_err();
        assert!(err.contains("not allowed"), "{err}");

        // With allowlist=None the system-initiated ask_user succeeds.
        r.execute_tool_scoped("ask_user", Value::Null, None, true)
            .await
            .expect("system-initiated ask_user must bypass the allowlist");
    }

    #[test]
    fn list_scoped_filters_allowlist_and_nested_tools() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(NamedTool { n: "read_file" }));
        r.register(Box::new(NamedTool {
            n: "subagent_spawn",
        }));
        let allow: HashSet<String> = ["read_file".to_string()].into_iter().collect();
        let listed = r.list_tools_scoped(Some(&allow), true);
        let names: Vec<_> = listed
            .iter()
            .map(|v| v["function"]["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["read_file"]);
    }
}

#[cfg(test)]
mod typed_result_tests {
    use super::{Tool, ToolRegistry};
    use crate::traits::{ToolErrorCode, ToolResult, ToolResultStatus};
    use async_trait::async_trait;
    use serde_json::Value;

    struct LegacyTool {
        name: &'static str,
        output: &'static str,
    }

    #[async_trait]
    impl Tool for LegacyTool {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "legacy test tool"
        }

        fn parameters(&self) -> Value {
            serde_json::json!({})
        }

        async fn execute(&self, _args: Value) -> Result<String, String> {
            Ok(self.output.to_string())
        }
    }

    struct NativeTypedTool;

    #[async_trait]
    impl Tool for NativeTypedTool {
        fn name(&self) -> &str {
            "typed"
        }

        fn description(&self) -> &str {
            "typed test tool"
        }

        fn parameters(&self) -> Value {
            serde_json::json!({})
        }

        async fn execute(&self, _args: Value) -> Result<String, String> {
            Ok("legacy path must not decide status".to_string())
        }

        async fn execute_with_approved_mutation_typed(
            &self,
            _args: Value,
            _approved_preview: Option<&crate::traits::MutationPreview>,
        ) -> ToolResult {
            ToolResult::error_with_content(
                ToolErrorCode::PolicyDenied,
                "denied by typed policy",
                "request was denied without a legacy error marker",
            )
        }
    }

    struct NativeTypedSuccessTool;

    #[async_trait]
    impl Tool for NativeTypedSuccessTool {
        fn name(&self) -> &str {
            "typed_success"
        }

        fn description(&self) -> &str {
            "typed success test tool"
        }

        fn parameters(&self) -> Value {
            serde_json::json!({})
        }

        async fn execute(&self, _args: Value) -> Result<String, String> {
            Err("legacy path must not decide status".to_string())
        }

        async fn execute_with_approved_mutation_typed(
            &self,
            _args: Value,
            _approved_preview: Option<&crate::traits::MutationPreview>,
        ) -> ToolResult {
            ToolResult::success("Error: this is literal successful content")
        }
    }

    #[tokio::test]
    async fn typed_result_is_authoritative_and_serializes_stable_codes() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(NativeTypedTool));

        let result = registry.execute_tool_result("typed", Value::Null).await;
        assert_eq!(result.status, ToolResultStatus::Error);
        assert_eq!(result.error_code(), Some(ToolErrorCode::PolicyDenied));
        let wire = serde_json::to_value(&result).expect("serialize typed result");
        assert_eq!(wire["status"], "error");
        assert_eq!(wire["error"]["code"], "policy_denied");
    }

    #[tokio::test]
    async fn scoped_dispatch_does_not_bypass_native_typed_result() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(NativeTypedTool));

        let result = registry
            .execute_tool_scoped_with_approved_mutation_result(
                "typed",
                Value::Null,
                None,
                None,
                false,
            )
            .await;

        assert_eq!(result.error_code(), Some(ToolErrorCode::PolicyDenied));
        assert_eq!(
            result.content,
            "request was denied without a legacy error marker"
        );
    }

    #[tokio::test]
    async fn native_typed_success_is_not_reclassified_from_its_content() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(NativeTypedSuccessTool));

        let result = registry
            .execute_tool_result("typed_success", Value::Null)
            .await;

        assert_eq!(result.status, ToolResultStatus::Success);
        assert_eq!(result.error, None);
        assert_eq!(result.content, "Error: this is literal successful content");
    }

    #[tokio::test]
    async fn legacy_mapping_is_scoped_and_cannot_false_flag_file_content() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(LegacyTool {
            name: "exec",
            output: "build failed\nExit code: 7",
        }));
        registry.register(Box::new(LegacyTool {
            name: "read_file",
            output: "Error: this is saved file content",
        }));

        let failed = registry.execute_tool_result("exec", Value::Null).await;
        assert_eq!(failed.error_code(), Some(ToolErrorCode::NonZeroExit));
        assert_eq!(failed.content, "build failed\nExit code: 7");

        let content = registry.execute_tool_result("read_file", Value::Null).await;
        assert_eq!(content.status, ToolResultStatus::Success);
        assert_eq!(content.error, None);
    }

    #[tokio::test]
    async fn registry_failures_have_typed_root_causes() {
        let registry = ToolRegistry::new();
        let result = registry.execute_tool_result("missing", Value::Null).await;
        assert_eq!(result.error_code(), Some(ToolErrorCode::NotFound));
        assert!(result.content.contains("missing"));
    }
}
