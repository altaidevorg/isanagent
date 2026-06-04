use crate::bus::{BusMessage, LogEvent, ReflectionKind, TelemetryEvent};
use crate::config::MemoryConfig;
use crate::logging::LoggerHandle;
use crate::memory::{MemoryMessage, SharedReply};
use crate::traits::Provider;
use crate::utils::ChatMessage;
use crate::NodeHandle;
use std::fs;
use std::path::PathBuf;
use tokio::sync::watch;
use tokio::time::{sleep, Duration};

pub struct ReflectionEngine {
    memory_node: NodeHandle<MemoryMessage>,
    workspace_dir: PathBuf,
    provider: Box<dyn Provider>,
    config: MemoryConfig,
    logger_tx: LoggerHandle,
    shutdown_rx: watch::Receiver<bool>,
}

impl ReflectionEngine {
    pub fn new(
        memory_node: NodeHandle<MemoryMessage>,
        workspace_dir: PathBuf,
        provider: Box<dyn Provider>,
        config: MemoryConfig,
        logger_tx: LoggerHandle,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            memory_node,
            workspace_dir,
            provider,
            config,
            logger_tx,
            shutdown_rx,
        }
    }

    pub fn start(self) -> tokio::task::JoinHandle<()> {
        let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info(
            "ReflectionEngine",
            "ReflectionEngine starting...",
        )));
        tokio::spawn(async move {
            self.run_loop().await;
        })
    }

    async fn run_loop(mut self) {
        let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info(
            "ReflectionEngine",
            "ReflectionEngine run loop started.",
        )));
        loop {
            tokio::select! {
                _ = sleep(Duration::from_secs(60)) => {}
                changed = self.shutdown_rx.changed() => {
                    if changed.is_ok() && *self.shutdown_rx.borrow() {
                        let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info("ReflectionEngine", "ReflectionEngine stopping...")));
                        break;
                    }
                }
            }

            if !self.config.enabled.unwrap_or(false) {
                continue;
            }

            if let Err(e) = self.run_short_term_reflection().await {
                let _ = self.logger_tx.send(BusMessage::Log(LogEvent::error(
                    "ReflectionEngine",
                    &format!("Short-term reflection failed: {}", e),
                )));
            }

            if let Err(e) = self.run_long_term_reflection().await {
                let _ = self.logger_tx.send(BusMessage::Log(LogEvent::error(
                    "ReflectionEngine",
                    &format!("Long-term reflection failed: {}", e),
                )));
            }
        }
    }

    async fn run_short_term_reflection(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.memory_node
            .send_packet(MemoryMessage::GetThreadsNeedingReflection {
                threshold_mins: self.config.short_term_threshold_mins.unwrap_or(3),
                reply: SharedReply::new(tx),
            })
            .await
            .map_err(|e| e.to_string())?;

        let session_ids = rx.await??;

        for session_id in session_ids {
            let (tx, rx) = tokio::sync::oneshot::channel();
            self.memory_node
                .send_packet(MemoryMessage::GetMessagesSinceReflection {
                    thread_id: session_id.clone(),
                    reply: SharedReply::new(tx),
                })
                .await
                .map_err(|e| e.to_string())?;

            let (new_messages, _last_msg_id) = rx.await??;

            if new_messages.is_empty() {
                continue;
            }

            let reflection_started = std::time::Instant::now();
            let _ = self
                .logger_tx
                .send(BusMessage::Telemetry(TelemetryEvent::ReflectionStarted {
                    chat_id: Some(session_id.clone()),
                    kind: ReflectionKind::ShortTerm,
                    inputs_consumed: new_messages.len().min(u32::MAX as usize) as u32,
                }));

            let _ = self.logger_tx.send(BusMessage::Log(
                LogEvent::debug(
                    "ReflectionEngine",
                    &format!(
                        "Thread {} reached short-term reflection threshold (idle)",
                        session_id
                    ),
                )
                .with_chat_id(&session_id),
            ));
            // Trigger summary
            let mut transcript = String::new();
            for (_, msg) in &new_messages {
                let body = msg
                    .content
                    .as_ref()
                    .map(|c| c.text_content())
                    .unwrap_or_default();
                let tools = msg.tool_calls.as_ref().map_or(String::new(), |tcs| {
                    let names: Vec<&str> = tcs.iter().map(|tc| tc.function.name.as_str()).collect();
                    format!(" [tool_calls: {}]", names.join(", "))
                });
                transcript.push_str(&format!("{}: {}{}\n\n", msg.role, body, tools));
            }

            // PR-2.1: short-term reflection now produces the same 8-slot sectional
            // JSON as the in-loop auto-compaction (PR-2). Markdown render goes into
            // the legacy `summary` column for backward-compat readers; the JSON
            // form is persisted via `WriteSectionsJson`.
            let prompt = crate::agent::compaction::build_sectional_prompt(None, &transcript, None);

            let context = vec![ChatMessage::user(&prompt)];
            match self.provider.chat(&context, None).await {
                Ok(response) => {
                    let text = response.content;
                    // Use robust JSON extractor
                    if let Some(val) = crate::utils::extract_json_from_llm_response(&text) {
                        let sections = crate::agent::compaction::SummarySections::from_json(&val);
                        let summary_md = sections.to_markdown();
                        let output_bytes = summary_md.len().min(u32::MAX as usize) as u32;
                        let sections_json =
                            serde_json::to_string(&sections).unwrap_or_else(|_| "{}".to_string());

                        let (tx, rx) = tokio::sync::oneshot::channel();
                        self.memory_node
                            .send_packet(MemoryMessage::AddSummary {
                                thread_id: session_id.clone(),
                                summary: summary_md,
                                key_info: String::new(),
                                knowledge_gaps: String::new(),
                                reply: SharedReply::new(tx),
                            })
                            .await
                            .map_err(|e| e.to_string())?;
                        rx.await??;

                        // Fire-and-forget the structured JSON into the
                        // `sections_json` column. Failures are non-fatal — the
                        // row still has the rendered Markdown.
                        let (tx, rx) = tokio::sync::oneshot::channel();
                        self.memory_node
                            .send_packet(MemoryMessage::WriteSectionsJson {
                                thread_id: session_id.clone(),
                                sections_json,
                                reply: SharedReply::new(tx),
                            })
                            .await
                            .map_err(|e| e.to_string())?;
                        rx.await??;

                        let highest_id = new_messages.last().unwrap().0;
                        let (tx, rx) = tokio::sync::oneshot::channel();
                        self.memory_node
                            .send_packet(MemoryMessage::UpdateThreadMetadata {
                                thread_id: session_id.clone(),
                                last_reflection_msg_id: Some(highest_id),
                                reply: SharedReply::new(tx),
                            })
                            .await
                            .map_err(|e| e.to_string())?;
                        rx.await??;

                        let _ = self.logger_tx.send(BusMessage::Telemetry(
                            TelemetryEvent::ReflectionCompleted {
                                chat_id: Some(session_id.clone()),
                                kind: ReflectionKind::ShortTerm,
                                output_bytes,
                                wall_ms: reflection_started
                                    .elapsed()
                                    .as_millis()
                                    .min(u64::MAX as u128)
                                    as u64,
                            },
                        ));

                        let _ = self.logger_tx.send(BusMessage::Log(
                            LogEvent::info(
                                "ReflectionEngine",
                                &format!("Generated short-term summary for thread {}", session_id),
                            )
                            .with_chat_id(&session_id),
                        ));
                    }
                }
                Err(e) => {
                    let _ = self.logger_tx.send(BusMessage::Log(
                        LogEvent::error(
                            "ReflectionEngine",
                            &format!("Failed to call provider for reflection: {}", e),
                        )
                        .with_chat_id(&session_id),
                    ));
                }
            }
        }
        Ok(())
    }

    async fn run_long_term_reflection(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let memory_md_path = self.workspace_dir.join("MEMORY.md");

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.memory_node
            .send_packet(MemoryMessage::GetLongTermReflectionState {
                threshold: self.config.long_term_threshold_summaries.unwrap_or(5),
                reply: SharedReply::new(tx),
            })
            .await
            .map_err(|e| e.to_string())?;

        let (should_run, summaries_content, max_id) = rx.await??;

        if !should_run || summaries_content.is_empty() {
            return Ok(());
        }

        let reflection_started = std::time::Instant::now();
        // Long-term aggregation has no single chat_id (it consolidates across all threads).
        // `inputs_consumed` uses the line count of `summaries_content` as a rough proxy for
        // the number of summary records consumed — the underlying SQL function returns a
        // concatenated blob rather than a count.
        let inputs_consumed_estimate =
            summaries_content.lines().count().min(u32::MAX as usize) as u32;
        let _ = self
            .logger_tx
            .send(BusMessage::Telemetry(TelemetryEvent::ReflectionStarted {
                chat_id: None,
                kind: ReflectionKind::LongTerm,
                inputs_consumed: inputs_consumed_estimate,
            }));

        let current_memory = if memory_md_path.exists() {
            fs::read_to_string(&memory_md_path).unwrap_or_default()
        } else {
            "No memory currently.".to_string()
        };

        let prompt = format!(
            "You are consolidating the agent's long-term memory. Below is the current MEMORY.md content, and a list of recent conversation summaries.\n\
            Rewrite the long-term memory to incorporate any new facts, user preferences, project context, or relationships found in the summaries.\n\
            Maintain a structured, organized markdown document.\n\n\
            CURRENT MEMORY:\n{}\n\nRECENT SUMMARIES:\n{}",
            current_memory, summaries_content
        );

        let context = vec![ChatMessage::user(&prompt)];
        if let Ok(response) = self.provider.chat(&context, None).await {
            let mut answer = response.content;
            if let Some(start) = answer.find("```markdown") {
                if let Some(end) = answer[start + 11..].find("```") {
                    answer = answer[start + 11..start + 11 + end].to_string();
                }
            }
            let trimmed = answer.trim();
            let output_bytes = trimmed.len().min(u32::MAX as usize) as u32;
            fs::write(&memory_md_path, trimmed)?;

            let (tx, rx) = tokio::sync::oneshot::channel();
            self.memory_node
                .send_packet(MemoryMessage::SetLongTermReflectionState {
                    max_id,
                    reply: SharedReply::new(tx),
                })
                .await
                .map_err(|e| e.to_string())?;
            rx.await??;

            let _ =
                self.logger_tx
                    .send(BusMessage::Telemetry(TelemetryEvent::ReflectionCompleted {
                        chat_id: None,
                        kind: ReflectionKind::LongTerm,
                        output_bytes,
                        wall_ms: reflection_started
                            .elapsed()
                            .as_millis()
                            .min(u64::MAX as u128) as u64,
                    }));

            let _ = self.logger_tx.send(BusMessage::Log(LogEvent::info(
                "ReflectionEngine",
                "Generated long-term memory update",
            )));
        }

        Ok(())
    }
}
