//! Capability → tool / branch matrix (documentation + light helpers for Phase 2+).

use super::capabilities::ProviderCapabilities;

/// Markdown table for operators and prompt authors. Keep in sync with harness tools when they land.
pub const PREFLIGHT_MARKDOWN: &str = r#"# Execution preflight (capabilities → agent surface)

| Conceptual tool / branch | Required capability (provider or session) |
|--------------------------|-----------------------------------------------|
| `execution_session_create` | Provider registered and execution harness not disabled in config (Phase 2). |
| `execution_run` | `supports_persistent_sessions` **or** ephemeral one-shot policy documented per provider. |
| `execution_run_background` | Same session/run contract as `execution_run`; returns a process-local `job_id` immediately. |
| `execution_job_status` / `execution_job_result` / `execution_job_list` | Jobs registry (always available when execution harness is enabled). |
| `execution_job_cancel` | `supports_interrupt` (same as `execution_cancel` for the job’s session). |
| `execution_cancel` | `supports_interrupt` |
| `execution_package_install` (optional) | `supports_package_install` |
| `execution_ssh_exec` (optional) | `supports_remote_shell` + provider implements `SshRemoteShell` |
| `execution_env_info` / GPU summary (optional) | Provider-specific; gate on session or extension trait |

**Preflight rule:** the executor MUST reject calls with [`crate::execution::ExecutionError::Unsupported`] before hitting the network or subprocess when the capability matrix says the operation is unavailable—even if the model requests it.

**Context injection:** inject a short JSON summary of [`ProviderCapabilities`] / session snapshot so the model avoids planning impossible steps.
"#;

/// Returns stable tags for optional tools that are allowed given static capabilities.
/// Phase 2 tools will register only when the matching tags are non-empty / conditions hold.
pub fn allowed_optional_tool_tags(cap: &ProviderCapabilities) -> Vec<&'static str> {
    let mut tags = Vec::new();
    if cap.supports_interrupt {
        tags.push("execution_cancel");
    }
    if cap.supports_package_install {
        tags.push("execution_package_install");
    }
    if cap.supports_remote_shell {
        tags.push("execution_ssh_exec");
    }
    if cap.jupyter_kernel {
        tags.push("jupyter_kernel");
    }
    tags
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{NetworkPolicy, ProviderCapabilities};

    #[test]
    fn optional_tags_track_flags() {
        let mut cap = ProviderCapabilities::minimal("local");
        assert!(allowed_optional_tool_tags(&cap).is_empty());

        cap.supports_interrupt = true;
        assert_eq!(allowed_optional_tool_tags(&cap), vec!["execution_cancel"]);

        cap.supports_package_install = true;
        cap.supports_remote_shell = true;
        cap.jupyter_kernel = true;
        cap.network_policy = NetworkPolicy::Full;
        let t = allowed_optional_tool_tags(&cap);
        assert!(t.contains(&"execution_cancel"));
        assert!(t.contains(&"execution_package_install"));
        assert!(t.contains(&"execution_ssh_exec"));
        assert!(t.contains(&"jupyter_kernel"));
    }
}
