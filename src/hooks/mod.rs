//! Lifecycle hooks: async observation sinks (JSONL / webhook) and optional synchronous
//! command hooks for policy and context (see `docs/hooks.md`).

mod context;
mod observation;
mod plugin_hooks;
mod steering;

pub use context::ToolCallHookContext;
pub use observation::{
    observation_params_from_config, start_observation_hooks, HookObservationMeta,
    ObservationHooksHandle,
};
pub use plugin_hooks::{load_plugin_hook_sets, parse_plugin_hooks_json, ParsedPluginHooks};
pub use steering::{
    build_steering_engine, build_steering_engine_with, run_post_model_hooks, run_post_tool_hooks,
    run_pre_tool_hooks, run_user_prompt_hooks, HookAuditRecord, HookAuditSink, HookSessionInfo,
    HookToolCall, PluginHookSet, PostModelDecision, PostModelOutcome, PostToolDecision,
    PreToolDecision, PreToolOutcome, SteeringHooksEngine, UserPromptDecision,
    UserPromptHookOutcome,
};
