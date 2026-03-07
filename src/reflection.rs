use std::path::PathBuf;
use tokio::time::{sleep, Duration};
use log::{info, error, debug};
use crate::config::MemoryConfig;
use crate::traits::Provider;
use crate::utils::ChatMessage;
use crate::NodeHandle;
use crate::memory::{MemoryMessage, SharedReply};
use std::fs;

pub struct ReflectionEngine {
    memory_node: NodeHandle<MemoryMessage>,
    workspace_dir: PathBuf,
    provider: Box<dyn Provider>,
    config: MemoryConfig,
}

impl ReflectionEngine {
    pub fn new(memory_node: NodeHandle<MemoryMessage>, workspace_dir: PathBuf, provider: Box<dyn Provider>, config: MemoryConfig) -> Self {
        Self { memory_node, workspace_dir, provider, config }
    }

    pub fn start(self) {
        tokio::spawn(async move {
            self.run_loop().await;
        });
    }

    async fn run_loop(self) {
        info!("ReflectionEngine started.");
        loop {
            // Wake up every 1 minute to check for short-term and long-term reflections
            sleep(Duration::from_secs(60)).await;

            if !self.config.enabled.unwrap_or(false) {
                continue;
            }

            if let Err(e) = self.run_short_term_reflection().await {
                error!("Short-term reflection failed: {}", e);
            }

            if let Err(e) = self.run_long_term_reflection().await {
                error!("Long-term reflection failed: {}", e);
            }
        }
    }

    async fn run_short_term_reflection(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.memory_node.send_packet(MemoryMessage::GetSessionsNeedingReflection {
            threshold_mins: self.config.short_term_threshold_mins.unwrap_or(3),
            reply: SharedReply::new(tx),
        }).await.map_err(|e| e.to_string())?;
        
        let session_ids = rx.await??;

        for session_id in session_ids {
            let (tx, rx) = tokio::sync::oneshot::channel();
            self.memory_node.send_packet(MemoryMessage::GetMessagesSinceReflection {
                session_id: session_id.clone(),
                reply: SharedReply::new(tx),
            }).await.map_err(|e| e.to_string())?;

            let (new_messages, _last_msg_id) = rx.await??;

            if new_messages.is_empty() {
                continue;
            }

            debug!("Session {} reached short-term reflection threshold (idle)", session_id);
            // Trigger summary
            let mut transcript = String::new();
            for (_, role, content) in &new_messages {
                transcript.push_str(&format!("{}: {}\n\n", role, content));
            }

                let prompt = format!(
                    "Summarize the following conversation. Extract key information, facts and any potential knowledge gaps.\n\
                    Format your response EXACTLY as a JSON object with these keys: \"summary\", \"key_info\", \"knowledge_gaps\".\n\n\
                    Conversation:\n{}", transcript
                );

                let context = vec![ChatMessage::user(&prompt)];
                match self.provider.chat(&context).await {
                    Ok(response) => {
                        let text = response.content;
                        // Use robust JSON extractor
                        if let Some(val) = crate::utils::extract_json_from_llm_response(&text) {
                            let summary = val.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let key_info = val.get("key_info").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let knowledge_gaps = val.get("knowledge_gaps").and_then(|v| v.as_str()).unwrap_or("").to_string();

                                    let (tx, rx) = tokio::sync::oneshot::channel();
                                    self.memory_node.send_packet(MemoryMessage::AddSummary {
                                        session_id: session_id.clone(),
                                        summary,
                                        key_info,
                                        knowledge_gaps,
                                        reply: SharedReply::new(tx),
                                    }).await.map_err(|e| e.to_string())?;
                                    rx.await??;

                                    let highest_id = new_messages.last().unwrap().0;
                                    let (tx, rx) = tokio::sync::oneshot::channel();
                                    self.memory_node.send_packet(MemoryMessage::UpdateSessionMetadata {
                                        session_id: session_id.clone(),
                                        last_reflection_msg_id: Some(highest_id),
                                        reply: SharedReply::new(tx),
                                    }).await.map_err(|e| e.to_string())?;
                                    rx.await??;
                                    
                                    info!("Generated short-term summary for session {}", session_id);
                                }
                    }
                    Err(e) => error!("Failed to call provider for reflection: {}", e),
                }
        }
        Ok(())
    }

    async fn run_long_term_reflection(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let memory_md_path = self.workspace_dir.join("MEMORY.md");
        
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.memory_node.send_packet(MemoryMessage::GetLongTermReflectionState {
            threshold: self.config.long_term_threshold_summaries.unwrap_or(5),
            reply: SharedReply::new(tx),
        }).await.map_err(|e| e.to_string())?;
        
        let (should_run, summaries_content, max_id) = rx.await??;

        if !should_run || summaries_content.is_empty() {
             return Ok(());
        }

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
        if let Ok(response) = self.provider.chat(&context).await {
            let mut answer = response.content;
            if let Some(start) = answer.find("```markdown") {
                if let Some(end) = answer[start+11..].find("```") {
                    answer = answer[start+11..start+11+end].to_string();
                }
            }
            fs::write(&memory_md_path, answer.trim())?;
            
            let (tx, rx) = tokio::sync::oneshot::channel();
            self.memory_node.send_packet(MemoryMessage::SetLongTermReflectionState {
                max_id,
                reply: SharedReply::new(tx),
            }).await.map_err(|e| e.to_string())?;
            rx.await??;
            
            info!("Generated long-term memory update");
        }
        
        Ok(())
    }
}
