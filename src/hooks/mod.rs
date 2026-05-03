//! Lifecycle hooks: async observation sinks (JSONL / webhook) and optional synchronous
//! command hooks for policy and context (see `docs/hooks.md`).

mod context;
mod observation;
mod steering;

pub use context::ToolCallHookContext;
pub use observation::{
    observation_params_from_config, start_observation_hooks, HookObservationMeta,
    ObservationHooksHandle,
};
pub use steering::{
    build_steering_engine, run_post_tool_hooks, run_pre_tool_hooks, run_user_prompt_hooks,
    HookSessionInfo, PreToolOutcome, SteeringHooksEngine, UserPromptHookOutcome,
};
