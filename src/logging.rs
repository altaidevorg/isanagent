use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{
    mpsc::{sync_channel, Receiver, SyncSender},
    OnceLock,
};

use async_trait::async_trait;
use log::{Level, LevelFilter, Log, Metadata, Record, SetLoggerError};
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::time::Duration;

use crate::bus::{BusMessage, LogEvent, LogLevel, LoggerControlMessage, TelemetryEvent};
use crate::config::{AppConfig, EffectiveLoggingConfig};
use crate::log_rotation::{RotatingLineWriter, WriteLineOutcome};
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
    match log::set_logger(&ACTOR_RUNTIME_LOGGER) {
        Ok(()) => {
            log::set_max_level(LevelFilter::Trace);
            Ok(())
        }
        // A prior host/test in this process may have already installed the logger.
        Err(_) => {
            log::set_max_level(LevelFilter::Trace);
            Ok(())
        }
    }
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
    conversation_writer: Option<RotatingLineWriter>,
    runtime_writer: Option<RotatingLineWriter>,
    logs_dir: PathBuf,
    max_total_bytes: u64,
    failure_reported: bool,
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
        let logging_config = load_logging_config(&workspace_dir);
        Self::new_with_config(workspace_dir, logging_config)
    }

    pub fn new_with_config(
        workspace_dir: PathBuf,
        logging_config: EffectiveLoggingConfig,
    ) -> Result<Self, String> {
        let logs_dir = workspace_dir.join(".system_generated").join("logs");
        fs::create_dir_all(&logs_dir)
            .map_err(|e| format!("Failed to create logs directory: {}", e))?;

        let mut actor = Self {
            conversation_writer: None,
            runtime_writer: None,
            logs_dir: logs_dir.clone(),
            max_total_bytes: logging_config.max_total_bytes,
            failure_reported: false,
        };
        if !logging_config.enabled {
            return Ok(actor);
        }

        actor.conversation_writer = Some(
            RotatingLineWriter::new(
                logs_dir.join("conversation.jsonl"),
                logging_config.conversation_max_bytes,
                logging_config.retained_generations,
            )
            .map_err(|e| format!("Failed to open conversation log: {}", e))?,
        );
        actor.runtime_writer = Some(
            RotatingLineWriter::new(
                logs_dir.join("runtime.log"),
                logging_config.runtime_max_bytes,
                logging_config.retained_generations,
            )
            .map_err(|e| format!("Failed to open runtime log: {}", e))?,
        );
        actor
            .enforce_total_log_cap()
            .map_err(|e| format!("Failed to enforce diagnostic log cap: {}", e))?;
        Ok(actor)
    }

    fn write_conversation(&mut self, packet: &BusMessage) {
        // Serialize to a `Value` first so secrets can be redacted before this analytical journal
        // hits disk — tool results / inbound text routinely carry keys (an `env` dump, an echoed
        // `$OPENAI_API_KEY`). `conversation.jsonl` is write-only telemetry (agent memory lives in
        // SQLite), so redaction here never affects what the agent can recall or use.
        let serialized = match packet {
            BusMessage::Inbound(inv) => serde_json::to_value(inv),
            BusMessage::Outbound(out) => serde_json::to_value(out),
            BusMessage::Telemetry(tel) => serde_json::to_value(tel),
            BusMessage::RunLifecycle(event) => serde_json::to_value(event),
            BusMessage::Log(_) => return,
            BusMessage::LoggerControl(_) => return,
            BusMessage::Cancel(_) => return,
            BusMessage::CancelRun { .. } => return,
            BusMessage::Steer { .. } => return,
            BusMessage::PromoteSyncToBackground(_) => return,
            BusMessage::SetTerminalSessionChat { .. } => return,
            BusMessage::SwitchModel { .. } => return,
            // PR-5: a manual trigger is internal control flow; the resulting
            // `CompactionTriggered` / `CompactionCompleted` telemetry pair already
            // shows up in the conversation log via the `Telemetry(_)` arm above.
            BusMessage::TriggerCompaction { .. } => return,
            BusMessage::InstallSkill { .. } => return,
        };
        let mut value = match serialized {
            Ok(value) => value,
            Err(error) => {
                self.disable_after_failure("serialize conversation event", error.to_string());
                return;
            }
        };

        crate::redact::shared().redact_json(&mut value);
        let json_line = match serde_json::to_string(&value) {
            Ok(line) => line,
            Err(error) => {
                self.disable_after_failure("encode conversation event", error.to_string());
                return;
            }
        };
        let event_kind = conversation_event_kind(packet);
        let result = match self.conversation_writer.as_mut() {
            Some(writer) => write_conversation_record(writer, &json_line, event_kind),
            None => return,
        };
        if let Err(error) = result {
            self.disable_after_failure("write conversation log", error);
            return;
        }
        self.enforce_total_or_disable();
    }

    fn write_runtime_event(&mut self, event: &LogEvent) {
        let result = match self.runtime_writer.as_mut() {
            Some(writer) => write_runtime_record(writer, &event.format_line()),
            None => return,
        };
        if let Err(error) = result {
            self.disable_after_failure("write runtime log", error);
            return;
        }
        self.enforce_total_or_disable();
    }

    fn write_shadow_runtime_event(&mut self, packet: &BusMessage) {
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
            BusMessage::RunLifecycle(event) => {
                LogEvent::info("BusMessage", &format!("RunLifecycle event={event:?}"))
            }
            BusMessage::Log(_) => return,
            BusMessage::LoggerControl(_) => return,
            BusMessage::Cancel(chat_id) => LogEvent::info(
                "BusMessage",
                &format!("Cancel reasoning loop for chat_id={}", chat_id),
            )
            .with_chat_id(chat_id),
            BusMessage::CancelRun { chat_id, run_id } => LogEvent::info(
                "BusMessage",
                &format!(
                    "Cancel reasoning loop for chat_id={} run_id={}",
                    chat_id, run_id
                ),
            )
            .with_chat_id(chat_id),
            BusMessage::Steer {
                chat_id, run_id, ..
            } => LogEvent::info(
                "BusMessage",
                &format!("Steer active run for chat_id={} run_id={}", chat_id, run_id),
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
            BusMessage::InstallSkill {
                repo_url,
                skill_name,
            } => LogEvent::info(
                "BusMessage",
                &format!(
                    "InstallSkill requested for repo={} specific={}",
                    repo_url,
                    skill_name.as_deref().unwrap_or("-")
                ),
            ),
        };

        self.write_runtime_event(&event);
    }

    fn flush_all(&mut self) {
        let conversation_result = self
            .conversation_writer
            .as_mut()
            .map(RotatingLineWriter::flush)
            .transpose();
        if let Err(error) = conversation_result {
            self.disable_after_failure("flush conversation log", error.to_string());
            return;
        }
        let runtime_result = self
            .runtime_writer
            .as_mut()
            .map(RotatingLineWriter::flush)
            .transpose();
        if let Err(error) = runtime_result {
            self.disable_after_failure("flush runtime log", error.to_string());
            return;
        }
        self.enforce_total_or_disable();
    }

    fn enforce_total_or_disable(&mut self) {
        if let Err(error) = self.enforce_total_log_cap() {
            self.disable_after_failure("enforce diagnostic log cap", error.to_string());
        }
    }

    fn enforce_total_log_cap(&self) -> io::Result<()> {
        let mut total_bytes = 0u64;
        let mut rotated = Vec::new();
        for entry in fs::read_dir(&self.logs_dir)? {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if !metadata.is_file() || !recognized_log_file(&path) {
                continue;
            }
            let bytes = self.active_log_bytes(&path).unwrap_or(metadata.len());
            total_bytes = total_bytes.saturating_add(bytes);
            if is_rotated_log_file(&path) {
                rotated.push((
                    metadata
                        .modified()
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                    path,
                    bytes,
                ));
            }
        }

        rotated.sort_by_key(|(modified, path, _)| (*modified, path.clone()));
        for (_, path, bytes) in rotated {
            if total_bytes <= self.max_total_bytes {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                total_bytes = total_bytes.saturating_sub(bytes);
            }
        }
        Ok(())
    }

    fn active_log_bytes(&self, path: &Path) -> Option<u64> {
        let name = path.file_name()?.to_str()?;
        match name {
            "conversation.jsonl" => self
                .conversation_writer
                .as_ref()
                .map(RotatingLineWriter::current_bytes),
            "runtime.log" => self
                .runtime_writer
                .as_ref()
                .map(RotatingLineWriter::current_bytes),
            _ => None,
        }
    }

    fn disable_after_failure(&mut self, operation: &str, error: String) {
        self.conversation_writer = None;
        self.runtime_writer = None;
        if !self.failure_reported {
            self.failure_reported = true;
            eprintln!(
                "IsanAgent diagnostic logging disabled after {}: {}",
                operation,
                sanitize_message(&error)
            );
        }
    }
}

impl Drop for LoggingActor {
    fn drop(&mut self) {
        self.flush_all();
    }
}

fn load_logging_config(workspace_dir: &Path) -> EffectiveLoggingConfig {
    let config_path = workspace_dir.join("config.toml");
    fs::read_to_string(config_path)
        .ok()
        .and_then(|contents| toml::from_str::<AppConfig>(&contents).ok())
        .map(|config| config.effective_logging_config())
        .unwrap_or_else(|| AppConfig::default().effective_logging_config())
}

fn conversation_event_kind(packet: &BusMessage) -> &'static str {
    match packet {
        BusMessage::Inbound(_) => "inbound",
        BusMessage::Outbound(_) => "outbound",
        BusMessage::Telemetry(_) => "telemetry",
        BusMessage::RunLifecycle(_) => "run_lifecycle",
        _ => "internal",
    }
}

fn write_conversation_record(
    writer: &mut RotatingLineWriter,
    json_line: &str,
    event_kind: &str,
) -> Result<(), String> {
    match writer
        .write_line(json_line)
        .map_err(|error| error.to_string())?
    {
        WriteLineOutcome::Written => Ok(()),
        WriteLineOutcome::RecordTooLarge {
            record_bytes,
            max_record_bytes: _,
        } => {
            let digest = format!("{:x}", Sha256::digest(json_line.as_bytes()));
            let replacement = json!({
                "type": "truncated_log_record",
                "original_event_type": event_kind,
                "original_bytes": record_bytes,
                "sha256": digest,
            })
            .to_string();
            match writer
                .write_line(&replacement)
                .map_err(|error| error.to_string())?
            {
                WriteLineOutcome::Written => Ok(()),
                WriteLineOutcome::RecordTooLarge { .. } => Err(
                    "conversation log record limit is too small for a truncation marker"
                        .to_string(),
                ),
            }
        }
    }
}

fn write_runtime_record(writer: &mut RotatingLineWriter, line: &str) -> Result<(), String> {
    match writer.write_line(line).map_err(|error| error.to_string())? {
        WriteLineOutcome::Written => Ok(()),
        WriteLineOutcome::RecordTooLarge {
            record_bytes,
            max_record_bytes,
        } => {
            let replacement = truncate_runtime_line(line, record_bytes, max_record_bytes);
            match writer
                .write_line(&replacement)
                .map_err(|error| error.to_string())?
            {
                WriteLineOutcome::Written => Ok(()),
                WriteLineOutcome::RecordTooLarge { .. } => {
                    Err("runtime log record limit is too small for a truncation marker".to_string())
                }
            }
        }
    }
}

fn truncate_runtime_line(line: &str, original_bytes: u64, max_record_bytes: u64) -> String {
    let max_record_bytes = max_record_bytes as usize;
    let marker = format!(" [truncated original_bytes={original_bytes}]");
    if marker.len() >= max_record_bytes {
        return "~".repeat(max_record_bytes);
    }
    let prefix_limit = max_record_bytes - marker.len();
    format!("{}{}", truncate_utf8(line, prefix_limit), marker)
}

fn truncate_utf8(input: &str, max_bytes: usize) -> &str {
    if input.len() <= max_bytes {
        return input;
    }
    let mut end = max_bytes;
    while end > 0 && !input.is_char_boundary(end) {
        end -= 1;
    }
    &input[..end]
}

fn log_generation(path: &Path, base: &str) -> Option<usize> {
    let name = path.file_name()?.to_str()?;
    name.strip_prefix(&format!("{base}."))?
        .parse::<usize>()
        .ok()
        .filter(|generation| *generation > 0)
}

fn recognized_log_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    matches!(name, "conversation.jsonl" | "runtime.log")
        || log_generation(path, "conversation.jsonl").is_some()
        || log_generation(path, "runtime.log").is_some()
}

fn is_rotated_log_file(path: &Path) -> bool {
    log_generation(path, "conversation.jsonl").is_some()
        || log_generation(path, "runtime.log").is_some()
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
            is_error,
            tool_call_id,
            background_job_id,
        } => LogEvent::info(
            "Telemetry",
            &format!(
                "ToolResult channel={} tool={} result_len={} is_error={} id={} bg={}",
                channel,
                tool_name,
                result.len(),
                is_error,
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
            tool_call_id,
            ..
        } => LogEvent::debug(
            "Telemetry",
            &format!(
                "ToolCallStarted tool={} args_len={} id={}",
                tool_name,
                args.len(),
                tool_call_id.as_deref().unwrap_or("-")
            ),
        )
        .with_chat_id(chat_id),
        TelemetryEvent::ToolCallFinished {
            chat_id,
            tool_name,
            result,
            is_error,
            tool_call_id,
            ..
        } => LogEvent::debug(
            "Telemetry",
            &format!(
                "ToolCallFinished tool={} result_len={} is_error={} id={}",
                tool_name,
                result.len(),
                is_error,
                tool_call_id.as_deref().unwrap_or("-")
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
        self.flush_all();
        Ok(None)
    }

    async fn process(
        &mut self,
        packet: BusMessage,
    ) -> Result<Option<(String, BusMessage)>, ActorError> {
        match &packet {
            BusMessage::LoggerControl(LoggerControlMessage::Flush) => {
                self.flush_all();
                return Ok(Some((
                    "logger_control".to_string(),
                    BusMessage::LoggerControl(LoggerControlMessage::Flushed),
                )));
            }
            BusMessage::LoggerControl(LoggerControlMessage::Flushed) => return Ok(None),
            BusMessage::Log(event) => self.write_runtime_event(event),
            _ => {
                self.write_conversation(&packet);
                self.write_shadow_runtime_event(&packet);
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
    use super::{
        recognized_log_file, sanitize_message, should_capture, telemetry_to_log_event, LoggingActor,
    };
    use crate::bus::{BusMessage, InboundMessage, LogEvent, TelemetryEvent};
    use crate::config::EffectiveLoggingConfig;
    use log::Level;
    use std::collections::HashMap;
    use std::fs;
    use tempfile::tempdir;

    fn logging_config(max_total_bytes: u64) -> EffectiveLoggingConfig {
        EffectiveLoggingConfig {
            enabled: true,
            conversation_max_bytes: 256,
            runtime_max_bytes: 256,
            retained_generations: 2,
            max_total_bytes,
        }
    }

    fn inbound(content: String) -> BusMessage {
        BusMessage::Inbound(InboundMessage {
            channel: "test".to_string(),
            sender_id: "sender".to_string(),
            chat_id: "chat".to_string(),
            thread_id: None,
            content,
            attachments: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    #[test]
    fn tool_result_diagnostic_uses_typed_error_status() {
        for (is_error, result) in [(false, "error: harmless text"), (true, "all good")] {
            let event = telemetry_to_log_event(&TelemetryEvent::ToolResult {
                chat_id: "chat".to_string(),
                channel: "api".to_string(),
                tool_name: "exec".to_string(),
                result: result.to_string(),
                is_error,
                tool_call_id: Some("call-7".to_string()),
                background_job_id: None,
            });

            assert!(event.message.contains(&format!("is_error={is_error}")));
            assert!(event.message.contains("id=call-7"));
        }
    }

    #[test]
    fn conversation_rotation_keeps_jsonl_valid_and_redacted() {
        let workspace = tempdir().expect("tempdir");
        let mut actor = LoggingActor::new_with_config(
            workspace.path().to_path_buf(),
            logging_config(90 * 1024 * 1024),
        )
        .expect("logger");
        let secret = "Bearer abc.def-0123456789";

        for _ in 0..4 {
            actor.write_conversation(&inbound(format!("{secret} {}", "x".repeat(80))));
        }
        actor.write_conversation(&inbound("x".repeat(4096)));
        actor.flush_all();

        let logs_dir = workspace.path().join(".system_generated/logs");
        for entry in fs::read_dir(logs_dir).expect("read logs") {
            let path = entry.expect("entry").path();
            if !recognized_log_file(&path) {
                continue;
            }
            let contents = fs::read_to_string(&path).expect("read log");
            assert!(!contents.contains(secret));
            for line in contents.lines() {
                serde_json::from_str::<serde_json::Value>(line).expect("valid JSONL record");
            }
        }
    }

    #[test]
    fn total_log_cap_removes_only_rotated_logs() {
        let workspace = tempdir().expect("tempdir");
        let mut actor =
            LoggingActor::new_with_config(workspace.path().to_path_buf(), logging_config(512))
                .expect("logger");

        for index in 0..8 {
            actor.write_conversation(&inbound(format!("message-{index}-{}", "x".repeat(80))));
            actor.write_runtime_event(&LogEvent::info("test", &"y".repeat(180)));
        }
        actor.flush_all();

        let logs_dir = workspace.path().join(".system_generated/logs");
        let files = fs::read_dir(logs_dir)
            .expect("read logs")
            .map(|entry| entry.expect("entry"))
            .filter(|entry| recognized_log_file(&entry.path()))
            .map(|entry| {
                (
                    entry.file_name().to_string_lossy().to_string(),
                    entry.metadata().expect("metadata").len(),
                )
            })
            .collect::<Vec<_>>();
        let total: u64 = files.iter().map(|(_, bytes)| *bytes).sum();
        assert!(
            total <= 512,
            "diagnostic logs total {total} bytes: {files:?}"
        );
    }

    #[test]
    fn oversized_runtime_record_is_marked_without_breaking_logging() {
        let workspace = tempdir().expect("tempdir");
        let mut actor = LoggingActor::new_with_config(
            workspace.path().to_path_buf(),
            logging_config(90 * 1024 * 1024),
        )
        .expect("logger");

        actor.write_runtime_event(&LogEvent::info("test", &"x".repeat(4096)));
        actor.flush_all();

        let runtime_log = workspace.path().join(".system_generated/logs/runtime.log");
        let contents = fs::read_to_string(runtime_log).expect("read runtime log");
        assert!(contents.contains("truncated original_bytes="));
    }

    #[test]
    fn workspace_logging_config_can_disable_file_logging() {
        let workspace = tempdir().expect("tempdir");
        fs::write(
            workspace.path().join("config.toml"),
            "[logging]\nenabled = false\n",
        )
        .expect("write config");
        let mut actor = LoggingActor::new(workspace.path().to_path_buf()).expect("logger");

        actor.write_conversation(&inbound("do not write".to_string()));
        actor.flush_all();

        assert!(!workspace
            .path()
            .join(".system_generated/logs/conversation.jsonl")
            .exists());
    }

    #[cfg(unix)]
    #[test]
    fn created_log_files_are_user_only() {
        use std::os::unix::fs::PermissionsExt;

        let workspace = tempdir().expect("tempdir");
        let _actor = LoggingActor::new_with_config(
            workspace.path().to_path_buf(),
            logging_config(90 * 1024 * 1024),
        )
        .expect("logger");
        let conversation = workspace
            .path()
            .join(".system_generated/logs/conversation.jsonl");
        let mode = fs::metadata(conversation)
            .expect("metadata")
            .permissions()
            .mode();

        assert_eq!(mode & 0o077, 0);
    }

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
