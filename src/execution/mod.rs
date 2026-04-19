//! Execution plane contracts (Phase 0): errors, run specs, capability snapshots, provider trait.
//!
//! Runtime implementations (local subprocess, Jupyter, SSH, …) live in future phases.
//!
//! ## Design: object-safe core + capability traits
//!
//! - [`ExecutionProvider`] is `async_trait`-object-safe (`Arc<dyn ExecutionProvider>`).
//! - Optional features use additional traits ([`SshRemoteShell`], [`PackageOperations`]) on concrete
//!   types, plus [`ProviderCapabilities`] for preflight and prompt injection.
//! - See [`preflight::PREFLIGHT_MARKDOWN`] for the capability → tool mapping.

mod capabilities;
mod error;
mod ids;
mod preflight;
mod provider;
mod run;

pub use capabilities::{
    NetworkPolicy, ProviderCapabilities, ProviderCapabilitiesSnapshot, SessionCapabilities,
};
pub use error::ExecutionError;
pub use ids::SessionId;
pub use preflight::{allowed_optional_tool_tags, PREFLIGHT_MARKDOWN};
pub use provider::{ExecutionProvider, PackageOperations, SshRemoteShell};
pub use run::{
    CwdPolicy, RunAttachmentRef, RunResult, RunSpec, SessionCreateRequest, SessionHandle,
};
