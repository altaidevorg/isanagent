//! Audit X9: provider failover machinery, split out of the former
//! `agent/mod.rs` god-file.
//!
//! Contents: [`FallbackProviderSpec`] config plumbing, run-scoped provider
//! snapshots ([`RunProviderContext`]), transient-retry + fallback chat
//! ([`chat_with_retry`] / [`try_fallbacks`]), and the user-facing LLM
//! failure banner. Items consumed outside this module stay reachable via
//! re-imports in the parent facade, so historical `crate::agent::*` paths
//! resolve unchanged.

use std::collections::HashMap;

use crate::bus::{BusMessage, LogEvent, OutboundMessage};
use crate::logging::LoggerHandle;
use crate::traits::Provider;

/// A configured alternate LLM provider to fail over to when the primary's retries are exhausted.
/// Holds everything [`crate::provider::create_provider`] needs; resolved once at startup from the
/// `[providers.*]` config.
#[derive(Clone)]
pub struct FallbackProviderSpec {
    pub provider_name: String,
    pub base_url: String,
    pub api_key: String,
    pub model_name: String,
}

// Manual `Debug` so a stray `{:?}` can never dump the API key into a log.
impl std::fmt::Debug for FallbackProviderSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FallbackProviderSpec")
            .field("provider_name", &self.provider_name)
            .field("base_url", &self.base_url)
            .field("api_key", &"[redacted]")
            .field("model_name", &self.model_name)
            .finish()
    }
}

/// Filter `candidates` to genuine fallbacks: drop any whose (provider, base_url, model) matches the
/// active primary, so the primary is never retried as its own fallback. Matching the full identity
/// (not just provider+model) correctly excludes the primary even when it came from the `[provider]`
/// default block rather than the `[providers.*]` map a candidate was built from.
pub fn build_fallback_specs(
    primary_provider: &str,
    primary_base_url: &str,
    primary_model: &str,
    candidates: Vec<FallbackProviderSpec>,
) -> Vec<FallbackProviderSpec> {
    candidates
        .into_iter()
        .filter(|c| {
            // Normalize before comparing so the primary isn't accidentally retried as its own
            // fallback: trailing slashes on base URLs are insignificant, and provider/model names
            // are matched case-insensitively (e.g. `https://api.openai.com/v1/` vs `.../v1`, or
            // `OpenAI` vs `openai`).
            let norm_c_url = c.base_url.trim_end_matches('/');
            let norm_p_url = primary_base_url.trim_end_matches('/');
            !(c.provider_name.eq_ignore_ascii_case(primary_provider)
                && norm_c_url == norm_p_url
                && c.model_name.eq_ignore_ascii_case(primary_model))
        })
        .collect()
}

/// Provider object and the exact credentials that created it. Keeping them behind one lock makes a
/// model switch atomic: a run can never snapshot the old provider with the new credential identity
/// (or vice versa).
pub(crate) struct ActiveProviderConfig {
    pub(crate) provider: Box<dyn Provider>,
    pub(crate) credentials: crate::provider::ProviderCredentials,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProviderRunIdentity {
    pub(crate) provider_name: String,
    pub(crate) model_name: String,
    pub(crate) secret_identity: String,
}

/// Immutable provider/fallback ownership for one accepted run. This value is also stored with a
/// queued inbound, so a later `/model` switch cannot rewrite already-admitted work.
#[derive(Clone)]
pub(crate) struct RunProviderContext {
    pub(crate) provider: Box<dyn Provider>,
    pub(crate) fallback_providers: Vec<FallbackProviderSpec>,
    pub(crate) identity: ProviderRunIdentity,
}

impl RunProviderContext {
    pub(crate) fn snapshot(
        active: &ActiveProviderConfig,
        candidates: &[FallbackProviderSpec],
    ) -> Self {
        let credentials = &active.credentials;
        let fallback_providers = if credentials.is_usable() {
            build_fallback_specs(
                &credentials.provider_name,
                &credentials.base_url,
                &credentials.model_name,
                candidates.to_vec(),
            )
        } else {
            Vec::new()
        };
        Self {
            provider: dyn_clone::clone_box(&*active.provider),
            fallback_providers,
            identity: ProviderRunIdentity {
                provider_name: credentials.provider_name.clone(),
                model_name: credentials.model_name.clone(),
                secret_identity: provider_secret_identity(&credentials.api_key),
            },
        }
    }
}

fn provider_secret_identity(api_key: &str) -> String {
    if api_key.is_empty() {
        return "none".to_string();
    }
    use sha2::Digest;
    let digest = sha2::Sha256::digest(api_key.as_bytes());
    format!("sha256:{}", &hex::encode(digest)[..16])
}

/// Result of attempting the configured fallback providers.
pub(crate) enum FallbackOutcome {
    Ok(crate::utils::LLMResponse),
    Cancelled,
    Exhausted,
}

/// Borrowed logging identity for the failover loop (bundled to keep the arg count in check).
pub(crate) struct FailoverLogCtx<'a> {
    pub(crate) logger_tx: &'a LoggerHandle,
    pub(crate) name: &'a str,
    pub(crate) chat_id: &'a str,
}

/// Try each fallback provider **once**, returning the first successful response. `build` constructs
/// a provider from a spec — real code passes [`crate::provider::create_provider`]; tests inject a
/// mock builder, keeping this loop fully testable without network. Cancellation preempts a
/// fallback chat.
pub(crate) async fn try_fallbacks<F>(
    fallbacks: &[FallbackProviderSpec],
    build: F,
    context: &[crate::utils::ChatMessage],
    tools_payload: &Option<serde_json::Value>,
    cancel_token: &tokio_util::sync::CancellationToken,
    log: FailoverLogCtx<'_>,
) -> FallbackOutcome
where
    F: Fn(&FallbackProviderSpec) -> Box<dyn crate::traits::Provider>,
{
    for spec in fallbacks {
        let _ = log.logger_tx.send(BusMessage::Log(
            LogEvent::warn(
                log.name,
                &format!(
                    "Primary LLM exhausted; failing over to provider={} model={}",
                    spec.provider_name, spec.model_name
                ),
            )
            .with_chat_id(log.chat_id),
        ));
        let provider = build(spec);
        let res = tokio::select! {
            r = provider.chat(context, tools_payload.clone()) => r,
            _ = cancel_token.cancelled() => return FallbackOutcome::Cancelled,
        };
        match res {
            Ok(resp) => {
                let _ = log.logger_tx.send(BusMessage::Log(
                    LogEvent::info(
                        log.name,
                        &format!(
                            "Fallback succeeded: provider={} model={}",
                            spec.provider_name, spec.model_name
                        ),
                    )
                    .with_chat_id(log.chat_id),
                ));
                return FallbackOutcome::Ok(resp);
            }
            Err(e) => {
                let _ = log.logger_tx.send(BusMessage::Log(
                    LogEvent::warn(
                        log.name,
                        &format!("Fallback provider={} failed: {}", spec.provider_name, e),
                    )
                    .with_chat_id(log.chat_id),
                ));
            }
        }
    }
    FallbackOutcome::Exhausted
}

/// Outcome of a `provider.chat` invocation that may be retried for transient errors.
pub(crate) enum ChatRetryOutcome {
    Ok {
        response: crate::utils::LLMResponse,
        retries: u32,
    },
    /// Cancellation token fired during a chat or sleep; caller exits the reasoning loop.
    Cancelled,
    /// Retries exhausted; final user-facing error string. The caller is expected to surface
    /// an LLM-failed banner.
    Failed(String),
    /// PR-4: provider rejected the request because the input exceeded its context
    /// window. Not retried — bouncing the same payload guarantees the same failure.
    /// The reasoning loop is expected to (eventually, PR-4.1) emergency-compact
    /// and retry once.
    ContextOverflow {
        tokens_attempted: u32,
        max: Option<u32>,
    },
}

/// Wrap `provider.chat` with a small retry loop for transient errors (network/5xx/429).
/// Up to 3 total attempts with exponential backoff (1s/2s/4s); the cancel token preempts
/// both the chat and the sleep.
pub(crate) async fn chat_with_retry(
    provider: &dyn crate::traits::Provider,
    context: &[crate::utils::ChatMessage],
    tools_payload: Option<serde_json::Value>,
    fallback_providers: &[FallbackProviderSpec],
    cancel_token: &tokio_util::sync::CancellationToken,
    log_ctx: FailoverLogCtx<'_>,
) -> ChatRetryOutcome {
    let FailoverLogCtx {
        logger_tx,
        name,
        chat_id,
    } = log_ctx;
    const MAX_ATTEMPTS: u32 = 3;
    const BACKOFF_BASE_MS: u64 = 1000;
    let mut last_err: Option<crate::utils::LLMError> = None;
    for attempt in 0..MAX_ATTEMPTS {
        let res = tokio::select! {
            r = provider.chat(context, tools_payload.clone()) => r,
            _ = cancel_token.cancelled() => {
                let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
                    name,
                    "Reasoning loop cancelled during LLM call.",
                ).with_chat_id(chat_id)));
                return ChatRetryOutcome::Cancelled;
            }
        };
        match res {
            Ok(resp) => {
                return ChatRetryOutcome::Ok {
                    response: resp,
                    retries: attempt,
                }
            }
            Err(crate::utils::LLMError::ContextOverflow {
                tokens_attempted,
                max,
            }) => {
                // PR-4: short-circuit — retrying the identical payload guarantees
                // the same overflow. Caller decides whether to compact and retry.
                return ChatRetryOutcome::ContextOverflow {
                    tokens_attempted,
                    max,
                };
            }
            Err(e) => {
                let transient = e.is_transient();
                let is_last = attempt + 1 >= MAX_ATTEMPTS;
                if !transient || is_last {
                    last_err = Some(e);
                    break;
                }
                let backoff_ms = BACKOFF_BASE_MS * (1u64 << attempt);
                let _ = logger_tx.send(BusMessage::Log(
                    LogEvent::warn(
                        name,
                        &format!(
                            "LLM call failed (attempt {}/{}): {}. Retrying in {}ms.",
                            attempt + 1,
                            MAX_ATTEMPTS,
                            e,
                            backoff_ms
                        ),
                    )
                    .with_chat_id(chat_id),
                ));
                last_err = Some(e);
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)) => {}
                    _ = cancel_token.cancelled() => {
                        return ChatRetryOutcome::Cancelled;
                    }
                }
            }
        }
    }
    // Primary exhausted. Before surfacing a failure, try each configured fallback provider once, so
    // a transient outage / key rotation / model deprecation on the primary doesn't drop a long
    // unattended turn. The primary stays the active provider — failover is per-call.
    match try_fallbacks(
        fallback_providers,
        |s| {
            crate::provider::create_provider(
                &s.provider_name,
                &s.base_url,
                &s.api_key,
                &s.model_name,
            )
        },
        context,
        &tools_payload,
        cancel_token,
        FailoverLogCtx {
            logger_tx,
            name,
            chat_id,
        },
    )
    .await
    {
        FallbackOutcome::Ok(resp) => {
            return ChatRetryOutcome::Ok {
                response: resp,
                retries: MAX_ATTEMPTS.saturating_sub(1),
            }
        }
        FallbackOutcome::Cancelled => return ChatRetryOutcome::Cancelled,
        FallbackOutcome::Exhausted => {}
    }

    ChatRetryOutcome::Failed(
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown LLM error".to_string()),
    )
}

/// Build the user-facing banner for an LLM failure.
///
/// Only the terminal channel exposes the `/retry` command and its accompanying
/// metadata flag. Other clients receive a channel-neutral recovery hint and use
/// the typed lifecycle outcome to decide whether to offer retry controls.
pub(crate) fn build_llm_failed_banner(
    channel: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    error: &str,
    retryable: bool,
) -> OutboundMessage {
    let content = if channel == "terminal" && retryable {
        format!(
            "LLM call failed after 3 attempts: {error}\nPress /retry to try again or /cancel to abandon."
        )
    } else if retryable {
        format!(
            "LLM call failed after provider retries were exhausted: {error}\nThis run can be retried from the client."
        )
    } else {
        format!("LLM call failed: {error}")
    };
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    if channel == "terminal" {
        metadata.insert(
            crate::protocol::ISANAGENT_TERMINAL_ERROR.to_string(),
            serde_json::json!(true),
        );
        if retryable {
            metadata.insert(
                crate::protocol::ISANAGENT_LLM_RETRY_AVAILABLE.to_string(),
                serde_json::json!(true),
            );
        }
    }
    OutboundMessage {
        channel: channel.to_string(),
        chat_id: chat_id.to_string(),
        thread_id: thread_id.map(|s| s.to_string()),
        content,
        metadata,
    }
}
