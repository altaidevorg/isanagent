//! Headless one-shot channel used by embedding hosts such as ALTAI CLI.
//!
//! The channel injects a single inbound prompt, captures the final outbound
//! assistant message, and rejects interactive clarification/approval waits so
//! non-TTY callers never hang or silently approve.

use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot, watch};

use crate::bus::{BusMessage, InboundMessage, OutboundMessage, RunLifecycleEvent, RunOutcome};
use crate::channels::Channel;
use crate::utils::ContentPart;

pub const ONESHOT_CHANNEL_NAME: &str = "altai-cli";

/// Terminal result of a one-shot host run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OneshotResult {
    pub chat_id: String,
    pub run_id: Option<String>,
    pub outcome: OneshotOutcome,
    pub final_text: Option<String>,
}

/// Why a one-shot run stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OneshotOutcome {
    Completed,
    Failed(String),
    Cancelled,
    TimedOut,
    ApprovalRequired { detail: String },
    ClarificationRequired { detail: String },
}

impl OneshotOutcome {
    pub fn from_run_outcome(outcome: &RunOutcome) -> Self {
        match outcome {
            RunOutcome::Completed => Self::Completed,
            RunOutcome::Cancelled => Self::Cancelled,
            RunOutcome::Failed { failure, .. } => Self::Failed(format!("{failure:?}")),
            RunOutcome::Stuck { reason } => Self::Failed(format!("stuck:{reason:?}")),
            RunOutcome::BudgetExhausted { budget } => {
                Self::Failed(format!("budget_exhausted:{budget:?}"))
            }
        }
    }
}

#[derive(Debug, Default)]
struct OneshotState {
    final_text: Option<String>,
    run_id: Option<String>,
    finished: bool,
}

/// Captures one-shot progress and completes when the run terminates or an
/// interactive approval/clarification would otherwise block a non-TTY host.
pub struct OneshotChannel {
    chat_id: String,
    prompt: String,
    attachments: Vec<ContentPart>,
    files: Vec<PathBuf>,
    state: Arc<Mutex<OneshotState>>,
    result_tx: Mutex<Option<oneshot::Sender<OneshotResult>>>,
    shutdown_tx: mpsc::UnboundedSender<()>,
    observe_tx: Option<mpsc::UnboundedSender<BusMessage>>,
    started: watch::Sender<bool>,
}

impl OneshotChannel {
    pub fn new(
        chat_id: String,
        prompt: String,
        files: Vec<PathBuf>,
        attachments: Vec<ContentPart>,
        result_tx: oneshot::Sender<OneshotResult>,
        shutdown_tx: mpsc::UnboundedSender<()>,
        observe_tx: Option<mpsc::UnboundedSender<BusMessage>>,
    ) -> Self {
        let (started, _) = watch::channel(false);
        Self {
            chat_id,
            prompt,
            attachments,
            files,
            state: Arc::new(Mutex::new(OneshotState::default())),
            result_tx: Mutex::new(Some(result_tx)),
            shutdown_tx,
            observe_tx,
            started,
        }
    }

    fn observe(&self, msg: BusMessage) {
        if let Some(tx) = &self.observe_tx {
            let _ = tx.send(msg);
        }
    }

    fn complete(&self, outcome: OneshotOutcome) {
        let mut state = self.state.lock().expect("oneshot state");
        if state.finished {
            return;
        }
        state.finished = true;
        let result = OneshotResult {
            chat_id: self.chat_id.clone(),
            run_id: state.run_id.clone(),
            outcome,
            final_text: state.final_text.clone(),
        };
        drop(state);
        if let Some(tx) = self.result_tx.lock().expect("oneshot result").take() {
            let _ = tx.send(result);
        }
        let _ = self.shutdown_tx.send(());
    }

    /// Observe host bus traffic that is not delivered through [`Channel::send`].
    pub fn observe_bus_message(&self, msg: &BusMessage) {
        self.observe(msg.clone());
        match msg {
            BusMessage::RunLifecycle(RunLifecycleEvent::Started { run_id, chat_id })
                if chat_id == &self.chat_id =>
            {
                let mut state = self.state.lock().expect("oneshot state");
                state.run_id = Some(run_id.clone());
            }
            BusMessage::RunLifecycle(RunLifecycleEvent::Terminated {
                chat_id,
                run_id,
                outcome,
            }) if chat_id == &self.chat_id =>
            {
                {
                    let mut state = self.state.lock().expect("oneshot state");
                    state.run_id = Some(run_id.clone());
                }
                self.complete(OneshotOutcome::from_run_outcome(outcome));
            }
            BusMessage::Telemetry(crate::bus::TelemetryEvent::ShellPolicyDecision {
                chat_id,
                decision,
                command_preview,
                ..
            }) if chat_id == &self.chat_id && decision == "approval_requested" =>
            {
                self.complete(OneshotOutcome::ApprovalRequired {
                    detail: command_preview.clone(),
                });
            }
            BusMessage::Outbound(out)
                if out.channel == ONESHOT_CHANNEL_NAME && out.chat_id == self.chat_id =>
            {
                // Captured via Channel::send as well; keep observe path consistent.
            }
            _ => {}
        }
    }

    fn attachment_note(files: &[PathBuf]) -> String {
        if files.is_empty() {
            return String::new();
        }
        let list = files
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("\n\n[Attached files: {list}]")
    }
}

#[async_trait]
impl Channel for OneshotChannel {
    fn name(&self) -> &str {
        ONESHOT_CHANNEL_NAME
    }

    async fn start(&self, bus_tx: mpsc::Sender<BusMessage>) -> Result<(), String> {
        let mut content = self.prompt.clone();
        // Prefer real multimodal attachments; only keep a path note when loading failed.
        if self.attachments.is_empty() {
            content.push_str(&Self::attachment_note(&self.files));
        }

        let inbound = InboundMessage {
            channel: ONESHOT_CHANNEL_NAME.to_string(),
            sender_id: "altai-cli".to_string(),
            chat_id: self.chat_id.clone(),
            thread_id: None,
            content,
            attachments: self.attachments.clone(),
            metadata: HashMap::new(),
        };
        self.observe(BusMessage::Inbound(inbound.clone()));
        bus_tx
            .send(BusMessage::Inbound(inbound))
            .await
            .map_err(|error| format!("failed to enqueue oneshot prompt: {error}"))?;
        let _ = self.started.send(true);
        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        self.observe(BusMessage::Outbound(msg.clone()));

        if msg
            .metadata
            .get(crate::clarification::METADATA_CLARIFICATION)
            .and_then(|value| value.as_bool())
            == Some(true)
            || msg
                .metadata
                .get(crate::bus::METADATA_CLARIFICATION_TICKET_ID)
                .is_some()
        {
            if let Some(edit) = msg.metadata.get("edit_diff") {
                let file = edit
                    .get("file")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(unknown)");
                let truncated = edit
                    .get("truncated")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let diff = edit.get("diff").and_then(|v| v.as_str()).unwrap_or("");
                let mut detail = format!("edit approval required for {file}");
                if truncated {
                    detail.push_str(" [diff truncated]");
                }
                if !diff.is_empty() {
                    detail.push('\n');
                    detail.push_str(diff);
                } else if !msg.content.is_empty() {
                    detail.push('\n');
                    detail.push_str(&msg.content);
                }
                self.complete(OneshotOutcome::ApprovalRequired { detail });
            } else {
                self.complete(OneshotOutcome::ClarificationRequired {
                    detail: msg.content.clone(),
                });
            }
            return Ok(());
        }

        if msg
            .metadata
            .get("isanagent_notification")
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            return Ok(());
        }

        {
            let mut state = self.state.lock().expect("oneshot state");
            state.final_text = Some(msg.content);
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_completed_run_outcome() {
        assert_eq!(
            OneshotOutcome::from_run_outcome(&RunOutcome::Completed),
            OneshotOutcome::Completed
        );
        assert_eq!(
            OneshotOutcome::from_run_outcome(&RunOutcome::Cancelled),
            OneshotOutcome::Cancelled
        );
    }
}
