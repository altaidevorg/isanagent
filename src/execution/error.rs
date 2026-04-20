//! Error taxonomy for the execution plane (stable across providers).

use std::io;

use thiserror::Error;

/// Provider- or executor-level failure. Prefer structured variants over opaque strings.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ExecutionError {
    #[error("unsupported operation '{operation}': {reason}")]
    Unsupported { operation: String, reason: String },

    #[error("invalid or unknown session: {0}")]
    InvalidSession(String),

    #[error("execution timed out after {timeout_secs} seconds")]
    Timeout { timeout_secs: u64 },

    #[error("limit exceeded for {resource}: {details}")]
    LimitExceeded { resource: String, details: String },

    #[error("execution was cancelled")]
    Cancelled,

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("serialization error: {0}")]
    Serialization(String),
}

impl ExecutionError {
    pub fn unsupported(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Unsupported {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    pub fn limit_exceeded(resource: impl Into<String>, details: impl Into<String>) -> Self {
        Self::LimitExceeded {
            resource: resource.into(),
            details: details.into(),
        }
    }
}

impl From<io::Error> for ExecutionError {
    fn from(e: io::Error) -> Self {
        ExecutionError::Provider(e.to_string())
    }
}

impl From<serde_json::Error> for ExecutionError {
    fn from(e: serde_json::Error) -> Self {
        ExecutionError::Serialization(e.to_string())
    }
}
