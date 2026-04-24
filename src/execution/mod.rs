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
mod harness;
mod ids;
mod jupyter;
mod local;
mod preflight;
mod provider;
mod run;
mod ssh;

pub use artifacts::{
    artifact_run_rel_dir, sanitize_session_segment, ArtifactLimits, ARTIFACT_ROOT_DIR,
};
pub use capabilities::{
    NetworkPolicy, ProviderCapabilities, ProviderCapabilitiesSnapshot, SessionCapabilities,
};
pub use error::ExecutionError;
pub use harness::{build_execution_harness, ExecutionHarness};
pub use ids::SessionId;
pub use jupyter::{JupyterExecutionProvider, JupyterExecutionProviderConfig};
pub use local::{LocalExecMode, LocalExecutionConfig, LocalExecutionProvider};
pub use preflight::{allowed_optional_tool_tags, PREFLIGHT_MARKDOWN};
pub use provider::{ExecutionProvider, PackageOperations, SshRemoteShell};
pub use run::{
    CwdPolicy, RunAttachmentRef, RunResult, RunSpec, SessionCreateRequest, SessionHandle,
};
pub use ssh::{validate_remote_workdir, SshExecutionProvider, SshExecutionProviderConfig};
