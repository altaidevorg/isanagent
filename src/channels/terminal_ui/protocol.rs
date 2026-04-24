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
pub const METADATA_EXECUTION_JOB_ID: &str = "execution_job_id";
pub const METADATA_EXECUTION_JOB_STATUS: &str = "execution_job_status";
