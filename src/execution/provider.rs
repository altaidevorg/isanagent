//! `ExecutionProvider` and optional capability traits.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde_json::Value;

use super::capabilities::ProviderCapabilities;
use super::error::ExecutionError;
use super::ids::SessionId;
use super::run::{RunResult, RunSpec, SessionCreateRequest, SessionHandle};

/// Core execution surface: every backend implements this. Use `async_trait` so the trait is
/// object-safe as `dyn ExecutionProvider` for registries and actors.
///
/// ## Object safety and extensions
///
/// Optional features (SSH shell, package installs beyond `run`, GPU queries) are **not** methods
/// on this trait. Use separate [`SshRemoteShell`], [`PackageOperations`], etc., implemented by the
/// concrete provider type; the executor resolves them via `provider_id`, [`ProviderCapabilities`]
/// preflight, or `downcast_rs` / enum dispatch where static typing is available.
#[async_trait]
pub trait ExecutionProvider: Send + Sync {
    fn provider_id(&self) -> &str;

    fn capabilities(&self) -> ProviderCapabilities;

    /// Optional metadata for durable run journals (kernel id, notebook path, etc.). Default: none.
    fn session_journal_extensions(
        &self,
        _session_id: &SessionId,
    ) -> Option<BTreeMap<String, Value>> {
        None
    }

    async fn create_session(
        &self,
        req: SessionCreateRequest,
    ) -> Result<SessionHandle, ExecutionError>;

    async fn close_session(&self, session_id: &SessionId) -> Result<(), ExecutionError>;

    async fn run(&self, session_id: &SessionId, spec: RunSpec)
        -> Result<RunResult, ExecutionError>;

    /// Best-effort interrupt (kernel interrupt, SIGINT, remote equivalent).
    async fn cancel(&self, session_id: &SessionId) -> Result<(), ExecutionError>;
}

/// Non-universal: interactive or batch remote shell over SSH (Phase 4 providers).
#[async_trait]
pub trait SshRemoteShell: Send + Sync {
    /// Run a non-interactive remote command in the session’s remote context.
    async fn exec_remote_argv(
        &self,
        session_id: &SessionId,
        argv: &[String],
    ) -> Result<RunResult, ExecutionError>;
}

/// Optional package / environment mutations (guarded by config in later phases).
#[async_trait]
pub trait PackageOperations: Send + Sync {
    async fn install_packages(
        &self,
        session_id: &SessionId,
        spec: &str,
    ) -> Result<RunResult, ExecutionError>;
}
