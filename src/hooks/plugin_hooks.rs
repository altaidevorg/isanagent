//! Agent Plugin `hooks.json` → steering hooks.
//!
//! Two shapes are accepted (optionally wrapped in a top-level `"hooks"` object):
//!
//! * **Claude Code style** — event → list of `{matcher, hooks: [{type: "command", command, timeout}]}`
//!   groups, with `timeout` in seconds:
//!   `{"hooks": {"PreToolUse": [{"matcher": "exec", "hooks": [{"type": "command", "command": "${CLAUDE_PLUGIN_ROOT}/check.sh"}]}]}}`
//! * **isanagent style** — event → flat list of `{matcher, command, timeout_ms, on_error}`
//!   entries, the same fields as `[[harness.hooks.steering.<event>]]`.
//!
//! Event names: `pre_tool` / `PreToolUse`, `post_tool` / `PostToolUse`, `user_prompt` /
//! `UserPromptSubmit`, `post_model` / `PostModel`. Other events are ignored with a warning.
//! Matchers are regexes over isanagent tool names (`exec`, `write_file`, …), not Claude Code's.

use std::path::Path;

use serde_json::Value;

use super::steering::PluginHookSet;
use crate::config::{HarnessHookCommandConfig, HarnessHooksSteeringConfig};
use crate::plugins::PluginRegistry;

/// Result of parsing one `hooks.json`.
#[derive(Debug, Default)]
pub struct ParsedPluginHooks {
    pub events: HarnessHooksSteeringConfig,
    /// Non-fatal problems (unsupported events / hook types, entries without a command).
    pub warnings: Vec<String>,
}

fn event_slot<'a>(
    events: &'a mut HarnessHooksSteeringConfig,
    name: &str,
) -> Option<&'a mut Vec<HarnessHookCommandConfig>> {
    let slot = match name {
        "pre_tool" | "PreToolUse" | "preToolUse" => &mut events.pre_tool,
        "post_tool" | "PostToolUse" | "postToolUse" => &mut events.post_tool,
        "user_prompt" | "UserPromptSubmit" | "userPromptSubmit" => &mut events.user_prompt,
        "post_model" | "PostModel" | "postModel" => &mut events.post_model,
        _ => return None,
    };
    Some(slot.get_or_insert_with(Vec::new))
}

fn str_field(v: &Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// `timeout_ms` wins; otherwise Claude-style `timeout` in seconds.
fn timeout_ms(v: &Value) -> Option<u64> {
    v.get("timeout_ms").and_then(Value::as_u64).or_else(|| {
        v.get("timeout")
            .and_then(Value::as_f64)
            .filter(|s| *s > 0.0)
            .map(|s| (s * 1000.0) as u64)
    })
}

fn command_entry(
    v: &Value,
    group_matcher: Option<String>,
    group_on_error: Option<String>,
    warnings: &mut Vec<String>,
    event: &str,
) -> Option<HarnessHookCommandConfig> {
    if let Some(kind) = str_field(v, "type") {
        if kind != "command" {
            warnings.push(format!(
                "{event}: hook type {kind:?} is not supported (only \"command\"); skipped"
            ));
            return None;
        }
    }
    let Some(command) = str_field(v, "command") else {
        warnings.push(format!("{event}: hook entry without a command; skipped"));
        return None;
    };
    Some(HarnessHookCommandConfig {
        matcher: str_field(v, "matcher").or(group_matcher),
        command,
        timeout_ms: timeout_ms(v),
        // Plugin hooks run in the agent sandbox like config hooks; `cwd` is not configurable.
        cwd: None,
        on_error: str_field(v, "on_error").or(group_on_error),
    })
}

/// Parse a plugin `hooks.json` document.
pub fn parse_plugin_hooks_json(text: &str) -> Result<ParsedPluginHooks, String> {
    let doc: Value = serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let root = match doc.get("hooks") {
        Some(inner @ Value::Object(_)) => inner,
        _ => &doc,
    };
    let Some(map) = root.as_object() else {
        return Err("hooks.json must be a JSON object".to_string());
    };
    let mut out = ParsedPluginHooks::default();
    for (event, entries) in map {
        if event == "$schema" || event == "description" {
            continue;
        }
        let Some(list) = entries.as_array() else {
            out.warnings
                .push(format!("{event}: expected a list of hook entries; skipped"));
            continue;
        };
        let mut parsed = Vec::new();
        for entry in list {
            match entry.get("hooks").and_then(Value::as_array) {
                // Claude Code group: {matcher, hooks: [...]}
                Some(inner) => {
                    let group_matcher = str_field(entry, "matcher");
                    let group_on_error = str_field(entry, "on_error");
                    for h in inner {
                        parsed.extend(command_entry(
                            h,
                            group_matcher.clone(),
                            group_on_error.clone(),
                            &mut out.warnings,
                            event,
                        ));
                    }
                }
                None => parsed.extend(command_entry(entry, None, None, &mut out.warnings, event)),
            }
        }
        match event_slot(&mut out.events, event) {
            Some(slot) => slot.extend(parsed),
            None => out.warnings.push(format!(
                "event {event:?} is not supported by isanagent; its hooks are ignored"
            )),
        }
    }
    Ok(out)
}

fn allowed(allow: &[String], plugin: &str) -> bool {
    allow.iter().any(|a| {
        let a = a.trim();
        a == "*" || a == plugin
    })
}

/// Load `hooks.json` from every discovered plugin that `allow` (from
/// `[harness.hooks.steering] plugin_hooks`) permits. Plugins with hooks that are not allowed are
/// logged so operators know what is being skipped.
pub fn load_plugin_hook_sets(registry: &PluginRegistry, allow: &[String]) -> Vec<PluginHookSet> {
    let mut sets = Vec::new();
    for plugin in registry.list() {
        let Some(path) = plugin.hooks_path.as_deref() else {
            continue;
        };
        if !allowed(allow, &plugin.name) {
            log::info!(
                "hooks: plugin '{}' ships hooks ({}) but is not listed in [harness.hooks.steering] plugin_hooks; skipped",
                plugin.name,
                path.display()
            );
            continue;
        }
        match load_one(path) {
            Ok(parsed) => {
                for w in &parsed.warnings {
                    log::warn!("hooks: plugin '{}': {w}", plugin.name);
                }
                let root = plugin
                    .dir
                    .canonicalize()
                    .unwrap_or_else(|_| plugin.dir.clone());
                log::info!(
                    "hooks: loaded plugin '{}' hooks (pre_tool={}, post_tool={}, user_prompt={}, post_model={})",
                    plugin.name,
                    count(&parsed.events.pre_tool),
                    count(&parsed.events.post_tool),
                    count(&parsed.events.user_prompt),
                    count(&parsed.events.post_model),
                );
                sets.push(PluginHookSet {
                    plugin: plugin.name.clone(),
                    root,
                    events: parsed.events,
                });
            }
            Err(e) => log::warn!(
                "hooks: plugin '{}': cannot load {}: {e}",
                plugin.name,
                path.display()
            ),
        }
    }
    sets
}

fn count(v: &Option<Vec<HarnessHookCommandConfig>>) -> usize {
    v.as_ref().map_or(0, Vec::len)
}

fn load_one(path: &Path) -> Result<ParsedPluginHooks, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse_plugin_hooks_json(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_claude_code_style() {
        let p = parse_plugin_hooks_json(
            r#"{"description":"x","hooks":{
                "PreToolUse":[{"matcher":"exec|write_file","hooks":[
                    {"type":"command","command":"${CLAUDE_PLUGIN_ROOT}/pre.sh","timeout":10},
                    {"type":"prompt","prompt":"nope"}]}],
                "UserPromptSubmit":[{"hooks":[{"type":"command","command":"u.sh"}]}],
                "Stop":[{"hooks":[{"type":"command","command":"s.sh"}]}]}}"#,
        )
        .unwrap();
        let pre = p.events.pre_tool.unwrap();
        assert_eq!(pre.len(), 1);
        assert_eq!(pre[0].matcher.as_deref(), Some("exec|write_file"));
        assert_eq!(pre[0].command, "${CLAUDE_PLUGIN_ROOT}/pre.sh");
        assert_eq!(pre[0].timeout_ms, Some(10_000));
        assert_eq!(p.events.user_prompt.unwrap().len(), 1);
        assert!(
            p.warnings.iter().any(|w| w.contains("\"prompt\"")),
            "{:?}",
            p.warnings
        );
        assert!(
            p.warnings.iter().any(|w| w.contains("\"Stop\"")),
            "{:?}",
            p.warnings
        );
    }

    #[test]
    fn parses_native_style() {
        let p = parse_plugin_hooks_json(
            r#"{"post_model":[{"command":"m.sh","on_error":"block","timeout_ms":2500}],
                "post_tool":[{"matcher":"^read_file$","command":"r.sh"},{"matcher":"x"}]}"#,
        )
        .unwrap();
        let pm = p.events.post_model.unwrap();
        assert_eq!(pm[0].on_error.as_deref(), Some("block"));
        assert_eq!(pm[0].timeout_ms, Some(2500));
        assert_eq!(p.events.post_tool.unwrap().len(), 1);
        assert!(p.warnings.iter().any(|w| w.contains("without a command")));
    }

    #[test]
    fn rejects_non_objects() {
        assert!(parse_plugin_hooks_json("[]").is_err());
        assert!(parse_plugin_hooks_json("{").is_err());
    }

    #[test]
    fn allowlist() {
        assert!(allowed(&["*".into()], "a"));
        assert!(allowed(&["a".into(), "b".into()], "b"));
        assert!(!allowed(&[], "a"));
        assert!(!allowed(&["ab".into()], "a"));
    }
}
