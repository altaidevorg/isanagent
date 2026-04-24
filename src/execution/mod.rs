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
mod capabilities;
mod error;
mod execution_jobs;
mod harness;
mod ids;
mod jupyter;
mod jupyter_notebook;
mod local;
mod post_run;
mod preflight;
mod provider;
mod run;
mod run_events;
mod run_history;
mod ssh;

pub use artifacts::{
    artifact_run_rel_dir, sanitize_session_segment, ArtifactLimits, ARTIFACT_ROOT_DIR,
};
pub use capabilities::{
    NetworkPolicy, ProviderCapabilities, ProviderCapabilitiesSnapshot, SessionCapabilities,
};
pub use error::ExecutionError;
pub use execution_jobs::{
    job_status_str, ExecutionJobManager, ExecutionJobRecord, SpawnBackgroundRunRequest,
};
pub use harness::{build_execution_harness, ExecutionHarness};
pub use ids::SessionId;
pub use jupyter::{JupyterExecutionProvider, JupyterExecutionProviderConfig};
pub use local::{
    build_python_host_command, LocalExecMode, LocalExecutionConfig, LocalExecutionProvider,
};
pub use post_run::{persist_successful_execution_run, PersistSuccessfulExecutionRunParams};
pub use preflight::{allowed_optional_tool_tags, PREFLIGHT_MARKDOWN};
pub use provider::{ExecutionProvider, PackageOperations, SshRemoteShell};
pub use run::{
    CwdPolicy, RunAttachmentRef, RunResult, RunSpec, SessionCreateRequest, SessionHandle,
};
pub use run_events::RunEvent;
pub use run_history::{write_run_journal, RunJournalParams};
pub use ssh::{validate_remote_workdir, SshExecutionProvider, SshExecutionProviderConfig};
