use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    mpsc::{sync_channel, Receiver, SyncSender},
    OnceLock,
};

use async_trait::async_trait;
use log::{Level, LevelFilter, Log, Metadata, Record, SetLoggerError};
use regex::Regex;
use serde_json::json;
use tokio::time::Duration;

use crate::bus::{BusMessage, LogEvent, LogLevel, LoggerControlMessage, TelemetryEvent};
use crate::{ActorError, ActorLogic};

pub const LOGGER_QUEUE_CAPACITY: usize = 4096;

#[derive(Clone)]
pub struct LoggerHandle {
    sender: SyncSender<BusMessage>,
}

impl LoggerHandle {
    pub fn send(&self, msg: BusMessage) -> Result<(), String> {
        // Use try_send to avoid blocking the caller if the logger is backed up.
        // For system logs, it's better to drop a log than to dead-lock the entire agent.
        self.sender
            .try_send(msg)
            .map_err(|e| format!("logger error: {}", e))
    }
}

pub fn create_logger_channel(capacity: usize) -> (LoggerHandle, Receiver<BusMessage>) {
    let (sender, receiver) = sync_channel(capacity);
    (LoggerHandle { sender }, receiver)
}

static LOGGER_SENDER: OnceLock<LoggerHandle> = OnceLock::new();
static ACTOR_RUNTIME_LOGGER: ActorRuntimeLogger = ActorRuntimeLogger;

struct ActorRuntimeLogger;

impl Log for ActorRuntimeLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        should_capture(metadata.target(), metadata.level())
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let Some(sender) = LOGGER_SENDER.get() else {
            return;
        };

        let message = sanitize_message(&record.args().to_string());
        if message.contains("LoggingActor") || message.contains("Supervisor(LoggingActor)") {
            return;
        }

        let thread = std::thread::current();
        let mut event = LogEvent::new(
            map_level(record.level()),
            record.module_path().unwrap_or(record.target()),
            &message,
        )
        .with_target(record.target());

        if let Some(file) = record.file() {
            event = event.with_location(file, record.line());
        }

        event = event.with_metadata(json!({
            "module_path": record.module_path(),
            "file": record.file(),
            "line": record.line(),
            "thread_name": thread.name(),
            "thread_id": format!("{:?}", thread.id()),
        }));

        let _ = sender.send(BusMessage::Log(event));
    }

    fn flush(&self) {}
}

pub fn init_runtime_logger(sender: LoggerHandle) -> Result<(), SetLoggerError> {
    let _ = LOGGER_SENDER.set(sender);
    log::set_logger(&ACTOR_RUNTIME_LOGGER)?;
    log::set_max_level(LevelFilter::Trace);
    Ok(())
}

fn should_capture(target: &str, level: Level) -> bool {
    let is_internal = target == "isanagent"
        || target.starts_with("isanagent")
        || matches!(
            target,
            "Altbot"
                | "TerminalChannel"
                | "SlackChannel"
                | "ApiChannel"
                | "EmailChannel"
                | "ReflectionEngine"
                | "DailyBriefingCron"
                | "BusMessage"
                | "Telemetry"
        );

    if is_internal {
        return true;
    }

    level <= Level::Warn
}

fn map_level(level: Level) -> LogLevel {
    match level {
        Level::Trace => LogLevel::Trace,
        Level::Debug => LogLevel::Debug,
        Level::Info => LogLevel::Info,
        Level::Warn => LogLevel::Warn,
        Level::Error => LogLevel::Error,
    }
}

fn sanitize_message(message: &str) -> String {
    static BEARER_RE: OnceLock<Regex> = OnceLock::new();
    static GEMINI_KEY_RE: OnceLock<Regex> = OnceLock::new();
    static EMAIL_RE: OnceLock<Regex> = OnceLock::new();

    let bearer_re = BEARER_RE
        .get_or_init(|| Regex::new(r"Bearer\s+[A-Za-z0-9._\-]+").expect("valid bearer regex"));
    let gemini_key_re = GEMINI_KEY_RE
        .get_or_init(|| Regex::new(r"AIza[0-9A-Za-z\-_]{20,}").expect("valid api key regex"));
    let email_re = EMAIL_RE.get_or_init(|| {
        Regex::new(r"\b([A-Za-z0-9._%+\-])([A-Za-z0-9._%+\-]*?)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")
            .expect("valid email regex")
    });

    let redacted = bearer_re.replace_all(message, "Bearer [REDACTED]");
    let redacted = gemini_key_re.replace_all(&redacted, "[REDACTED_API_KEY]");
    email_re.replace_all(&redacted, "$1***@$3").into_owned()
}

/// The sole component responsible for writing workspace log files.
pub struct LoggingActor {
    conversation_writer: BufWriter<File>,
    runtime_writer: BufWriter<File>,
}

struct LoggingFallbackActor {
    init_error: String,
    reported: bool,
}

impl LoggingFallbackActor {
    fn new(init_error: String) -> Self {
        Self {
            init_error,
            reported: false,
        }
    }
}

pub fn create_logging_actor_or_fallback(workspace_dir: PathBuf) -> Box<dyn ActorLogic<BusMessage>> {
    match LoggingActor::new(workspace_dir) {
        Ok(actor) => Box::new(actor),
        Err(err) => Box::new(LoggingFallbackActor::new(err)),
    }
}

impl LoggingActor {
    pub fn new(workspace_dir: PathBuf) -> Result<Self, String> {
        let logs_dir = workspace_dir.join(".system_generated").join("logs");
        std::fs::create_dir_all(&logs_dir)
            .map_err(|e| format!("Failed to create logs directory: {}", e))?;

        Ok(Self {
            conversation_writer: open_writer(&logs_dir.join("conversation.jsonl"))?,
            runtime_writer: open_writer(&logs_dir.join("runtime.log"))?,
        })
    }

    fn write_conversation(&mut self, packet: &BusMessage) -> Result<(), ActorError> {
        let json_line = match packet {
            BusMessage::Inbound(inv) => serde_json::to_string(inv),
            BusMessage::Outbound(out) => serde_json::to_string(out),
            BusMessage::Telemetry(tel) => serde_json::to_string(tel),
            BusMessage::Log(_) => return Ok(()),
            BusMessage::LoggerControl(_) => return Ok(()),
            BusMessage::Cancel(_) => return Ok(()),
            BusMessage::PromoteSyncToBackground(_) => return Ok(()),
            BusMessage::SetTerminalSessionChat { .. } => return Ok(()),
            BusMessage::SwitchModel { .. } => return Ok(()),
            // PR-5: a manual trigger is internal control flow; the resulting
            // `CompactionTriggered` / `CompactionCompleted` telemetry pair already
            // shows up in the conversation log via the `Telemetry(_)` arm above.
            BusMessage::TriggerCompaction { .. } => return Ok(()),
        }
        .map_err(|e| ActorError::from(format!("Failed to serialize conversation event: {}", e)))?;

        writeln!(self.conversation_writer, "{}", json_line)
            .map_err(|e| ActorError::from(format!("Failed to write conversation log: {}", e)))
    }

    fn write_runtime_event(&mut self, event: &LogEvent) -> Result<(), ActorError> {
        writeln!(self.runtime_writer, "{}", event.format_line())
            .map_err(|e| ActorError::from(format!("Failed to write runtime log: {}", e)))
    }

    fn write_shadow_runtime_event(&mut self, packet: &BusMessage) -> Result<(), ActorError> {
        let event = match packet {
            BusMessage::Inbound(msg) => LogEvent::info(
                "BusMessage",
                &format!(
                    "Inbound received on channel={} sender={} content_len={}",
                    msg.channel,
                    msg.sender_id,
                    msg.content.len()
                ),
            )
            .with_chat_id(&msg.chat_id)
            .with_metadata(json!({
                "direction": "inbound",
                "thread_id": msg.thread_id,
                "metadata": msg.metadata,
            })),
            BusMessage::Outbound(msg) => LogEvent::info(
                "BusMessage",
                &format!(
                    "Outbound sent on channel={} content_len={}",
                    msg.channel,
                    msg.content.len()
                ),
            )
            .with_chat_id(&msg.chat_id)
            .with_metadata(json!({
                "direction": "outbound",
                "thread_id": msg.thread_id,
                "metadata": msg.metadata,
            })),
            BusMessage::Telemetry(telemetry) => telemetry_to_log_event(telemetry),
            BusMessage::Log(_) => return Ok(()),
            BusMessage::LoggerControl(_) => return Ok(()),
            BusMessage::Cancel(chat_id) => LogEvent::info(
                "BusMessage",
                &format!("Cancel reasoning loop for chat_id={}", chat_id),
            )
            .with_chat_id(chat_id),
            BusMessage::PromoteSyncToBackground(chat_id) => LogEvent::info(
                "BusMessage",
                &format!("PromoteSyncToBackground requested for chat_id={}", chat_id),
            )
            .with_chat_id(chat_id),
            BusMessage::SetTerminalSessionChat { chat_id } => LogEvent::info(
                "BusMessage",
                &format!("SetTerminalSessionChat chat_id={}", chat_id),
            )
            .with_chat_id(chat_id),
            BusMessage::SwitchModel {
                provider_name,
                model_name,
                ..
            } => LogEvent::info(
                "BusMessage",
                &format!(
                    "SwitchModel provider={} model={}",
                    provider_name, model_name
                ),
            ),
            BusMessage::TriggerCompaction {
                session_key,
                focus_instructions,
                trigger,
            } => LogEvent::info(
                "BusMessage",
                &format!(
                    "TriggerCompaction session_key={} trigger={:?} focus={}",
                    session_key,
                    trigger,
                    focus_instructions
                        .as_deref()
                        .map(|s| if s.is_empty() { "-" } else { s })
                        .unwrap_or("-"),
                ),
            ),
        };

        self.write_runtime_event(&event)
    }

    fn flush_all(&mut self) -> Result<(), ActorError> {
        self.conversation_writer
            .flush()
            .map_err(|e| ActorError::from(format!("Failed to flush conversation log: {}", e)))?;
        self.runtime_writer
            .flush()
            .map_err(|e| ActorError::from(format!("Failed to flush runtime log: {}", e)))?;
        Ok(())
    }
}

impl Drop for LoggingActor {
    fn drop(&mut self) {
        let _ = self.flush_all();
    }
}

fn open_writer(path: &Path) -> Result<BufWriter<File>, String> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
    Ok(BufWriter::new(file))
}

fn telemetry_to_log_event(telemetry: &TelemetryEvent) -> LogEvent {
    match telemetry {
        TelemetryEvent::ToolCall {
            chat_id,
            channel,
            tool_name,
            args,
            tool_call_id,
            background_job_id,
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "ToolCall channel={} tool={} args_len={} id={} bg={}",
                channel,
                tool_name,
                args.len(),
                tool_call_id.as_deref().unwrap_or("-"),
                background_job_id.as_deref().unwrap_or("-"),
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::ToolResult {
            chat_id,
            channel,
            tool_name,
            result,
            is_error: _,
            tool_call_id,
            background_job_id,
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "ToolResult channel={} tool={} result_len={} id={} bg={}",
                channel,
                tool_name,
                result.len(),
                tool_call_id.as_deref().unwrap_or("-"),
                background_job_id.as_deref().unwrap_or("-"),
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::AgentThought { chat_id, thought, background_job_id } => LogEvent::debug(
            "Telemetry",
            &format!("AgentThought thought_len={} bg={}", thought.len(), background_job_id.as_deref().unwrap_or("-")),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::AgentUsage {
            chat_id,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cache_read_tokens,
            cache_creation_tokens,
            background_job_id,
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "AgentUsage model={} prompt={} completion={} total={} cache_read={} cache_create={} bg={}",
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                background_job_id.as_deref().unwrap_or("-")
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::CronTrigger { job_id, message } => LogEvent::info(
            "Telemetry",
            &format!(
                "CronTrigger job_id={} message_len={}",
                job_id,
                message.len()
            ),
        ),
        TelemetryEvent::ToolCallStarted {
            chat_id,
            tool_name,
            args,
            ..
        } => LogEvent::debug(
            "Telemetry",
            &format!("ToolCallStarted tool={} args_len={}", tool_name, args.len()),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::ToolCallFinished {
            chat_id,
            tool_name,
            result,
            ..
        } => LogEvent::debug(
            "Telemetry",
            &format!(
                "ToolCallFinished tool={} result_len={}",
                tool_name,
                result.len()
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::ToolProgress {
            chat_id,
            channel,
            tool_name,
            tool_call_id,
            message,
            background_job_id,
        } => LogEvent::debug(
            "Telemetry",
            &format!(
                "ToolProgress channel={} tool={} id={} bg={} msg_len={}",
                channel,
                tool_name,
                tool_call_id.as_deref().unwrap_or("-"),
                background_job_id.as_deref().unwrap_or("-"),
                message.len()
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::ExecutionRunFinished {
            chat_id,
            channel,
            provider_id,
            session_id,
            exit_code,
            duration_ms,
            stdout_len,
            stderr_len,
            artifact_count,
            git_head,
            description,
            ..
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "ExecutionRunFinished channel={} provider={} session={} exit={:?} ms={} out={}/{} artifacts={} git_head={} desc={}",
                channel,
                provider_id,
                session_id,
                exit_code,
                duration_ms,
                stdout_len,
                stderr_len,
                artifact_count,
                git_head.as_deref().unwrap_or("-"),
                description.as_deref().unwrap_or("-")
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::ExecutionJobFinished {
            chat_id,
            channel,
            job_id,
            session_id,
            provider_id,
            status,
            duration_ms,
            exit_code,
            stdout_len,
            stderr_len,
            artifact_count,
            description,
            ..
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "ExecutionJobFinished channel={} job={} session={} provider={} status={} exit={:?} ms={} out={}/{} artifacts={} desc={}",
                channel,
                job_id,
                session_id,
                provider_id,
                status,
                exit_code,
                duration_ms,
                stdout_len,
                stderr_len,
                artifact_count,
                description.as_deref().unwrap_or("-")
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::SubagentSpawned {
            parent_chat_id,
            child_chat_id,
            task_id,
            display_name,
            ..
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "SubagentSpawned parent={} child={} task={} name={}",
                parent_chat_id,
                child_chat_id,
                task_id,
                display_name.as_deref().unwrap_or("-")
            ),
        )
        .with_chat_id(parent_chat_id.as_str()),
        TelemetryEvent::SubagentFinished {
            parent_chat_id,
            child_chat_id,
            task_id,
            status,
            ..
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "SubagentFinished parent={} child={} task={} status={}",
                parent_chat_id, child_chat_id, task_id, status
            ),
        )
        .with_chat_id(parent_chat_id.as_str()),
        TelemetryEvent::ShellPolicyDecision {
            chat_id,
            channel,
            mode,
            decision,
            command_preview,
            ..
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "ShellPolicyDecision channel={} mode={} decision={} command={}",
                channel, mode, decision, command_preview
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::ShellGrepLikeDetected {
            chat_id,
            channel,
            command_preview,
            ..
        } => LogEvent::warn(
            "Telemetry",
            &format!(
                "ShellGrepLikeDetected channel={} command={}",
                channel, command_preview
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::ResearchDepthNudge {
            chat_id,
            channel,
            reason,
            ..
        } => LogEvent::info(
            "Telemetry",
            &format!("ResearchDepthNudge channel={} reason={}", channel, reason),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::BackgroundJobUpdated {
            job_id,
            chat_id,
            channel,
            state,
            kind,
            detail,
            ..
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "BackgroundJobUpdated channel={} job={} kind={} state={} detail={}",
                channel,
                job_id,
                kind,
                state,
                detail.as_deref().unwrap_or("-")
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::NotificationCreated {
            notification_id,
            chat_id,
            channel,
            kind,
            title,
            ..
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "NotificationCreated channel={} id={} kind={} title={}",
                channel, notification_id, kind, title
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::NotificationUpdated {
            notification_id,
            chat_id,
            channel,
            state,
            ..
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "NotificationUpdated channel={} id={} state={}",
                channel, notification_id, state
            ),
        )
        .with_chat_id(chat_id),
        // --- Phase 0 (compaction overhaul) telemetry ---
        TelemetryEvent::CompactionTriggered {
            chat_id,
            reason,
            tokens_before,
            turns_before,
            tokens_after_preprocess,
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "CompactionTriggered reason={:?} tokens_before={} turns_before={} tokens_after_preprocess={}",
                reason, tokens_before, turns_before, tokens_after_preprocess
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::CompactionCompleted {
            chat_id,
            tokens_before,
            tokens_after,
            wall_ms,
            summary_bytes,
            section_completeness,
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "CompactionCompleted tokens={}→{} wall_ms={} summary_bytes={} completeness={:.2}",
                tokens_before, tokens_after, wall_ms, summary_bytes, section_completeness
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::CompactionFailed {
            chat_id,
            reason,
            tokens_at_failure,
        } => LogEvent::warn(
            "Telemetry",
            &format!(
                "CompactionFailed reason={} tokens_at_failure={}",
                reason, tokens_at_failure
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::ReflectionStarted {
            chat_id,
            kind,
            inputs_consumed,
        } => {
            let ev = LogEvent::info(
                "Telemetry",
                &format!(
                    "ReflectionStarted kind={:?} inputs_consumed={}",
                    kind, inputs_consumed
                ),
            );
            match chat_id {
                Some(id) => ev.with_chat_id(id),
                None => ev,
            }
        }
        TelemetryEvent::ReflectionCompleted {
            chat_id,
            kind,
            output_bytes,
            wall_ms,
        } => {
            let ev = LogEvent::info(
                "Telemetry",
                &format!(
                    "ReflectionCompleted kind={:?} output_bytes={} wall_ms={}",
                    kind, output_bytes, wall_ms
                ),
            );
            match chat_id {
                Some(id) => ev.with_chat_id(id),
                None => ev,
            }
        }
        TelemetryEvent::ToolResultRefetch {
            chat_id,
            tool_call_id,
        } => LogEvent::info(
            "Telemetry",
            &format!("ToolResultRefetch tool_call_id={}", tool_call_id),
        )
        .with_chat_id(chat_id),
    }
}

#[async_trait]
impl ActorLogic<BusMessage> for LoggingActor {
    fn name(&self) -> String {
        "LoggingActor".to_string()
    }

    fn tick_interval(&self) -> Option<Duration> {
        Some(Duration::from_secs(1))
    }

    async fn on_tick(&mut self) -> Result<Option<(String, BusMessage)>, ActorError> {
        self.flush_all()?;
        Ok(None)
    }

    async fn process(
        &mut self,
        packet: BusMessage,
    ) -> Result<Option<(String, BusMessage)>, ActorError> {
        match &packet {
            BusMessage::LoggerControl(LoggerControlMessage::Flush) => {
                self.flush_all()?;
                return Ok(Some((
                    "logger_control".to_string(),
                    BusMessage::LoggerControl(LoggerControlMessage::Flushed),
                )));
            }
            BusMessage::LoggerControl(LoggerControlMessage::Flushed) => return Ok(None),
            BusMessage::Log(event) => self.write_runtime_event(event)?,
            _ => {
                self.write_conversation(&packet)?;
                self.write_shadow_runtime_event(&packet)?;
            }
        }

        Ok(None)
    }
}

#[async_trait]
impl ActorLogic<BusMessage> for LoggingFallbackActor {
    fn name(&self) -> String {
        "LoggingFallbackActor".to_string()
    }

    async fn process(
        &mut self,
        packet: BusMessage,
    ) -> Result<Option<(String, BusMessage)>, ActorError> {
        if !self.reported {
            eprintln!(
                "Logging fallback actor active; runtime logs are disabled: {}",
                self.init_error
            );
            self.reported = true;
        }

        match packet {
            BusMessage::LoggerControl(LoggerControlMessage::Flush) => Ok(Some((
                "logger_control".to_string(),
                BusMessage::LoggerControl(LoggerControlMessage::Flushed),
            ))),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{sanitize_message, should_capture};
    use log::Level;

    #[test]
    fn sanitize_message_masks_secrets_and_emails() {
        let message =
            "Bearer abc.def ghi AIzaSyDummyKeyValue012345678901234 myawesomesecret umut@altai.dev";
        let sanitized = sanitize_message(message);
        assert!(sanitized.contains("Bearer [REDACTED]"));
        assert!(sanitized.contains("[REDACTED_API_KEY]"));
        assert!(sanitized.contains("u***@altai.dev"));
        // Arbitrary words are not redacted; only Bearer tokens, Gemini-style keys, and emails.
        assert!(sanitized.contains("myawesomesecret"));
    }

    #[test]
    fn should_capture_filters_external_noise() {
        assert!(should_capture("isanagent::utils", Level::Debug));
        assert!(should_capture("SlackChannel", Level::Info));
        assert!(!should_capture(
            "hyper_util::client::legacy::pool",
            Level::Info
        ));
        assert!(should_capture(
            "hyper_util::client::legacy::pool",
            Level::Warn
        ));
    }
}
