use std::path::Path;
use std::sync::Arc;

use crate::bus::TelemetryEvent;
use crate::config::HarnessConfig;
use crate::plugins::PluginRegistry;

use super::observation::{
    observation_params_from_config, start_observation_hooks, HookObservationMeta,
    ObservationHooksHandle,
};
use super::plugin_hooks::load_plugin_hook_sets;
use super::steering::{
    build_steering_engine_with, HookAuditRecord, HookAuditSink, HookSessionInfo,
};

/// Shared observation + steering runtimes for one process (optional each).
#[derive(Clone)]
pub struct ToolCallHookContext {
    pub observation: Option<Arc<super::ObservationHooksHandle>>,
    pub steering: Option<Arc<super::SteeringHooksEngine>>,
}

/// Forwards steering-hook audit records to the observation pipeline as `HookDecision` events.
struct ObservationAuditSink(Arc<ObservationHooksHandle>);

impl HookAuditSink for ObservationAuditSink {
    fn record(&self, session: &HookSessionInfo<'_>, r: HookAuditRecord) {
        self.0.try_emit(
            TelemetryEvent::HookDecision {
                chat_id: session.chat_id.to_string(),
                channel: session.channel.to_string(),
                hook_event: r.hook_event.to_string(),
                source: r.source,
                command: r.command,
                tool_name: r.tool_name,
                decision: r.decision.to_string(),
                reason: r.reason,
                changed: r.changed.into_iter().map(str::to_string).collect(),
            },
            HookObservationMeta {
                channel: session.channel,
                chat_id: session.chat_id,
                thread_id: session.thread_id,
                is_subagent: session.is_subagent,
                metadata: session.metadata,
            },
        );
    }
}

impl ToolCallHookContext {
    /// Returns `Some` when at least one hook subsystem is active. Plugin `hooks.json` files are
    /// loaded only for plugins listed in `[harness.hooks.steering] plugin_hooks`.
    pub fn from_harness_config(
        workspace_dir: &Path,
        sandbox_dir: &Path,
        harness: &HarnessConfig,
        plugins: Option<&PluginRegistry>,
    ) -> Option<Arc<Self>> {
        let hooks = harness.hooks.as_ref();

        let obs_cfg = hooks.and_then(|h| h.observation.as_ref());
        let observation = match obs_cfg.filter(|o| o.enabled.unwrap_or(false)) {
            Some(o) => start_observation_hooks(observation_params_from_config(
                workspace_dir,
                o.jsonl_path.as_deref(),
                o.webhook_url.as_deref(),
                o.webhook_hmac_secret.as_deref(),
                o.metadata_keys.clone().unwrap_or_default(),
                o.queue_capacity,
            ))
            .map(Arc::new),
            None => None,
        };

        let steer_cfg = hooks.and_then(|h| h.steering.as_ref());
        let steering = match steer_cfg.filter(|s| s.enabled.unwrap_or(false)) {
            Some(s) => {
                let plugin_sets = match (plugins, s.plugin_hooks.as_deref()) {
                    (Some(reg), Some(allow)) if !allow.is_empty() => {
                        load_plugin_hook_sets(reg, allow)
                    }
                    (Some(reg), _) => {
                        // Still report plugins whose hooks are being ignored.
                        load_plugin_hook_sets(reg, &[])
                    }
                    (None, _) => Vec::new(),
                };
                let audit = observation.as_ref().map(|o| {
                    Arc::new(ObservationAuditSink(Arc::clone(o))) as Arc<dyn HookAuditSink>
                });
                match build_steering_engine_with(
                    s,
                    workspace_dir.to_path_buf(),
                    sandbox_dir.to_path_buf(),
                    &plugin_sets,
                    audit,
                ) {
                    Ok(e) => Some(e),
                    Err(e) => {
                        log::error!("hooks steering init failed: {e}");
                        None
                    }
                }
            }
            None => {
                if let Some(reg) = plugins {
                    let with_hooks: Vec<&str> = reg
                        .list()
                        .into_iter()
                        .filter(|p| p.hooks_path.is_some())
                        .map(|p| p.name.as_str())
                        .collect();
                    if !with_hooks.is_empty() {
                        log::info!(
                            "hooks: plugins {with_hooks:?} ship hooks, but [harness.hooks.steering] is not enabled; skipped"
                        );
                    }
                }
                None
            }
        };

        if observation.is_none() && steering.is_none() {
            return None;
        }

        Some(Arc::new(Self {
            observation,
            steering,
        }))
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::hooks::{run_pre_tool_hooks, PreToolOutcome};
    use serde_json::json;
    use std::collections::HashMap;

    fn harness(toml_src: &str) -> HarnessConfig {
        toml::from_str(toml_src).expect("harness toml")
    }

    /// A workspace with one plugin `guard` whose Claude-style hooks.json runs a script from the
    /// plugin root (found through `$CLAUDE_PLUGIN_ROOT`) that rewrites exec commands.
    fn workspace_with_plugin() -> std::path::PathBuf {
        let ws = std::env::temp_dir().join(format!("isanagent_plughooks_{}", uuid::Uuid::new_v4()));
        let plugin = ws.join(".agents/plugins/guard");
        let ns = plugin.join("dev.altai.isanagent");
        std::fs::create_dir_all(&ns).unwrap();
        std::fs::create_dir_all(ws.join("workspace")).unwrap();
        std::fs::write(
            plugin.join("plugin.json"),
            r#"{"name":"guard","version":"0.1.0"}"#,
        )
        .unwrap();
        std::fs::write(
            ns.join("hooks.json"),
            r#"{"hooks":{"PreToolUse":[{"matcher":"^exec$","hooks":[
                {"type":"command","command":"sh \"${CLAUDE_PLUGIN_ROOT}/rewrite.sh\"","timeout":5}]}]}}"#,
        )
        .unwrap();
        std::fs::write(
            plugin.join("rewrite.sh"),
            "cat >/dev/null; printf '%s' '{\"hookSpecificOutput\":{\"updatedInput\":{\"command\":\"echo from-plugin\"}}}'\n",
        )
        .unwrap();
        ws
    }

    async fn run_exec(ctx: &ToolCallHookContext) -> PreToolOutcome {
        let meta = HashMap::new();
        run_pre_tool_hooks(
            ctx.steering.as_ref().unwrap(),
            "exec",
            None,
            json!({"command": "ls"}),
            HookSessionInfo {
                channel: "test",
                chat_id: "c1",
                thread_id: None,
                metadata: &meta,
                is_subagent: false,
            },
        )
        .await
    }

    #[tokio::test]
    async fn allowed_plugin_hooks_run_and_are_audited() {
        let ws = workspace_with_plugin();
        let registry = PluginRegistry::discover(&ws, None);
        let h = harness(
            r#"
            [hooks.observation]
            enabled = true
            jsonl_path = "audit.jsonl"
            [hooks.steering]
            enabled = true
            plugin_hooks = ["guard"]
            "#,
        );
        let ctx = ToolCallHookContext::from_harness_config(
            &ws,
            &ws.join("workspace"),
            &h,
            Some(&registry),
        )
        .unwrap();
        let st = ctx.steering.as_ref().unwrap();
        assert_eq!(st.pre_tool.len(), 1);
        assert_eq!(&*st.pre_tool[0].source, "plugin:guard");

        match run_exec(&ctx).await {
            PreToolOutcome::Proceed(d) => {
                assert_eq!(d.args, json!({"command": "echo from-plugin"}))
            }
            other => panic!("{other:?}"),
        }

        // The rewrite is reported to the observation journal as a HookDecision.
        let path = ws.join("audit.jsonl");
        let mut text = String::new();
        for _ in 0..50 {
            text = std::fs::read_to_string(&path).unwrap_or_default();
            if !text.is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let line: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        let ev = &line["telemetry"]["HookDecision"];
        assert_eq!(ev["source"], "plugin:guard", "{line}");
        assert_eq!(ev["decision"], "modify");
        assert_eq!(ev["changed"], json!(["args"]));
        assert_eq!(ev["tool_name"], "exec");
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    async fn plugin_hooks_need_an_explicit_allowlist() {
        let ws = workspace_with_plugin();
        let registry = PluginRegistry::discover(&ws, None);
        for allow in ["", "plugin_hooks = []", "plugin_hooks = [\"other\"]"] {
            let h = harness(&format!("[hooks.steering]\nenabled = true\n{allow}\n"));
            let ctx = ToolCallHookContext::from_harness_config(
                &ws,
                &ws.join("workspace"),
                &h,
                Some(&registry),
            )
            .unwrap();
            assert!(
                ctx.steering.as_ref().unwrap().pre_tool.is_empty(),
                "{allow}"
            );
        }
        // Steering disabled: nothing at all, even with the plugin allowed.
        let h = harness("[hooks.steering]\nenabled = false\nplugin_hooks = [\"*\"]\n");
        assert!(ToolCallHookContext::from_harness_config(
            &ws,
            &ws.join("workspace"),
            &h,
            Some(&registry)
        )
        .is_none());
        let _ = std::fs::remove_dir_all(&ws);
    }
}
