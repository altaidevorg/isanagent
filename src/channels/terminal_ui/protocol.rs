//! Metadata keys for terminal pseudo-outbounds (tool notices, thoughts, errors).

pub const ISANAGENT_AGENT_THOUGHT: &str = "isanagent_agent_thought";
/// User-visible failure in the transcript (reasoning loop, provider, etc.).
pub const ISANAGENT_TERMINAL_ERROR: &str = "isanagent_terminal_error";
/// Live `execution_run` stream chunk (Jupyter iopub) for the Ratatui execution panel.
pub const ISANAGENT_EXECUTION_STREAM: &str = "isanagent_execution_stream";
pub const METADATA_EXECUTION_SESSION_ID: &str = "execution_session_id";
pub const METADATA_EXECUTION_RUN_ID: &str = "execution_run_id";
/// One-line completion / failure notice for `execution_run_background` (Ratatui execution strip).
pub const ISANAGENT_EXECUTION_JOB: &str = "isanagent_execution_job";
/// One-line "started" notice when a background execution job is registered (Ratatui multi-job strip).
pub const ISANAGENT_EXECUTION_JOB_STARTED: &str = "isanagent_execution_job_started";
pub const METADATA_EXECUTION_JOB_ID: &str = "execution_job_id";
pub const METADATA_EXECUTION_JOB_STATUS: &str = "execution_job_status";
/// Originating tool name for a background execution job (e.g.
/// `execution_run_background`, `execution_run` after auto-promote).
pub const METADATA_EXECUTION_JOB_TOOL_NAME: &str = "execution_job_tool_name";
/// Short human-facing line for Ratatui execution strip (optional on stream/job notices).
pub const METADATA_EXECUTION_DESCRIPTION: &str = "execution_description";
/// Ephemeral tool status line (updates active tool strip only; no transcript cell).
pub const ISANAGENT_TOOL_PROGRESS: &str = "isanagent_tool_progress";
/// Tool name for Ratatui tool rail / transcript previews (paired with `isanagent_tool_notify`).
pub const METADATA_TOOL_NAME: &str = "isanagent_tool_name";
/// Short args preview (no raw JSON dump) for tool call notices.
pub const METADATA_TOOL_CALL_PREVIEW: &str = "isanagent_tool_call_preview";
/// Short result summary for tool result / fail notices.
pub const METADATA_TOOL_RESULT_PREVIEW: &str = "isanagent_tool_result_preview";
/// Auxiliary full result size in characters (used as suffix, not primary display text).
pub const METADATA_TOOL_RESULT_CHAR_COUNT: &str = "isanagent_tool_result_char_count";
/// LLM-supplied stable id correlating a tool-call notice with its result/fail notice.
/// Used by the terminal UI to upsert the same Cell::ToolNotice in place (pending → done/failed)
/// instead of appending two separate cells per invocation.
pub const METADATA_TOOL_CALL_ID: &str = "isanagent_tool_call_id";
/// Set on `ISANAGENT_TERMINAL_ERROR` outbounds emitted after exhausted LLM retries. When the
/// terminal UI sees this, it activates a `/retry` banner that re-injects the last user inbound.
pub const ISANAGENT_LLM_RETRY_AVAILABLE: &str = "isanagent_llm_retry_available";
/// One-line notice when a sub-agent task is spawned (Ratatui agent-tasks strip).
pub const ISANAGENT_SUBAGENT_TASK_STARTED: &str = "isanagent_subagent_task_started";
/// One-line notice when a sub-agent task finishes (completed / failed / cancelled).
pub const ISANAGENT_SUBAGENT_TASK_FINISHED: &str = "isanagent_subagent_task_finished";
pub const METADATA_SUBAGENT_TASK_ID: &str = "subagent_task_id";
pub const METADATA_SUBAGENT_CHILD_CHAT_ID: &str = "subagent_child_chat_id";
pub const METADATA_SUBAGENT_AGENT_NAME: &str = "subagent_agent_name";
pub const METADATA_SUBAGENT_DISPLAY_NAME: &str = "subagent_display_name";
pub const METADATA_SUBAGENT_STATUS: &str = "subagent_status";
/// One-line notice when a generic background job is registered (Ratatui multi-job strip).
pub const ISANAGENT_BACKGROUND_JOB_STARTED: &str = "isanagent_background_job_started";
/// One-line notice when a generic background job finishes.
pub const ISANAGENT_BACKGROUND_JOB_FINISHED: &str = "isanagent_background_job_finished";
pub const METADATA_BACKGROUND_JOB_STATUS: &str = "background_job_status";
pub const METADATA_BACKGROUND_JOB_DESCRIPTION: &str = "background_job_description";
pub const METADATA_BACKGROUND_JOB_TOOL_NAME: &str = "background_job_tool_name";
