//! Execution plane: Phase 0 contracts (errors, run specs, capabilities, provider trait),
//! Phase 1 local subprocess provider ([`local::LocalExecutionProvider`]), Phase 3
//! Jupyter kernel provider ([`jupyter::JupyterExecutionProvider`]), and Phase 4
//! SSH exec provider ([`ssh::SshExecutionProvider`]).
//!
//! ## Design: object-safe core + capability traits
//!
//! - [`ExecutionProvider`] is `async_trait`-object-safe (`Arc<dyn ExecutionProvider>`).
//! - Optional features use additional traits ([`SshRemoteShell`], [`PackageOperations`]) on concrete
//!   types, plus [`ProviderCapabilities`] for preflight and prompt injection.
//! - See [`preflight::PREFLIGHT_MARKDOWN`] for the capability → tool mapping.

mod artifacts;
mod auto_promote;
mod capabilities;
mod error;
mod execution_jobs;
mod harness;
mod ids;
mod inflight;
mod jupyter;
mod jupyter_notebook;
mod local;
mod post_run;
mod preflight;
mod provider;
mod repl_framing;
mod run;
mod run_events;
mod run_history;
mod ssh;

pub use artifacts::{
    artifact_run_rel_dir, sanitize_session_segment, ArtifactLimits, ARTIFACT_ROOT_DIR,
};
pub use auto_promote::{run_with_auto_promote, AutoPromoteOutcome, PromoteReason};
pub use capabilities::{
    NetworkPolicy, ProviderCapabilities, ProviderCapabilitiesSnapshot, SessionCapabilities,
};
pub use error::ExecutionError;
pub use execution_jobs::{
    job_status_str, AdoptInflightRequest, ArbitraryWork, CancelOutcome, ExecutionJobManager,
    ExecutionJobRecord, SpawnArbitraryRequest, SpawnBackgroundRunRequest,
};
pub use harness::{build_execution_harness, ExecutionHarness};
pub use ids::SessionId;
pub use inflight::{InflightGuard, InflightSyncRegistry};
pub use jupyter::{JupyterExecutionProvider, JupyterExecutionProviderConfig};
pub use local::{
    build_python_host_command, install_uv_best_effort, parse_uv_pip_list_and_diff,
    uv_binary_available, uv_managed_env_python, uv_requirements_status, LocalExecMode,
    LocalExecutionConfig, LocalExecutionProvider, LocalPythonRuntime,
};
pub use post_run::{persist_successful_execution_run, PersistSuccessfulExecutionRunParams};
pub use preflight::{allowed_optional_tool_tags, PREFLIGHT_MARKDOWN};
pub use provider::{ExecutionProvider, PackageOperations, SshRemoteShell};
pub use repl_framing::{
    repl_round_trip, string_from_utf8_lossy_trim_cap, truncate_utf8_str_cap, MAX_REPL_SOURCE_BYTES,
    PYTHON_REPL_BOOTSTRAP,
};
pub use run::{
    CwdPolicy, RunAttachmentRef, RunResult, RunSpec, SessionCreateRequest, SessionHandle,
};
pub use run_events::RunEvent;
pub use run_history::{run_history_dir, write_run_journal, RunJournal, RunJournalParams};
pub use ssh::{
    resolve_ssh_run_cwd, validate_remote_workdir, SshExecutionProvider, SshExecutionProviderConfig,
};
