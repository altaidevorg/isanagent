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
/// Optional features (SSH shell, package installs beyond `run`, GPU queries) are **not**
/// methods on this trait; introduce focused traits implemented by concrete providers only
/// when such features actually ship, and gate them via [`ProviderCapabilities`] preflight.
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
