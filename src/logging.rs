use std::path::PathBuf;
use std::fs::OpenOptions;
use std::io::Write;
use async_trait::async_trait;
use log::error;

use crate::{ActorLogic, ActorError};
use crate::bus::BusMessage;

#[derive(Clone)]
pub struct WorkspaceLoggingActor {
    log_file_path: PathBuf,
}

impl WorkspaceLoggingActor {
    pub fn new(workspace_dir: PathBuf) -> Self {
        let logs_dir = workspace_dir.join(".system_generated").join("logs");
        if !logs_dir.exists() {
            let _ = std::fs::create_dir_all(&logs_dir);
        }
        
        Self {
            log_file_path: logs_dir.join("conversation.jsonl"),
        }
    }
}

#[async_trait]
impl ActorLogic<BusMessage> for WorkspaceLoggingActor {
    fn name(&self) -> String {
        "WorkspaceLogger".to_string()
    }

    async fn process(&mut self, packet: BusMessage) -> Result<Option<(String, BusMessage)>, ActorError> {
        let json_line = match &packet {
            BusMessage::Inbound(inv) => {
                serde_json::to_string(inv)
            }
            BusMessage::Outbound(out) => {
                serde_json::to_string(out)
            }
            BusMessage::Telemetry(tel) => {
                serde_json::to_string(tel)
            }
        }.unwrap_or_else(|_| "{}".to_string());

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file_path)
            .map_err(|e| {
                error!("WorkspaceLogger I/O error: {}", e);
                ActorError::from(format!("Failed to open log file: {}", e))
            })?;

        writeln!(file, "{}", json_line).map_err(|e| {
            error!("WorkspaceLogger write error: {}", e);
            ActorError::from(format!("Failed to write to log file: {}", e))
        })?;

        // Pass it through untouched on a "next" route if anything wants to chain it
        Ok(Some(("next".to_string(), packet)))
    }
}
