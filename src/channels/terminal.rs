use async_trait::async_trait;
use crate::channels::Channel;
use crate::bus::{InboundMessage, OutboundMessage};
use tokio::sync::mpsc::Sender;
use std::io::{self, Write};
use log::error;
use colored::Colorize;
use crossterm::{cursor, terminal::{Clear, ClearType}, execute};

/// A Channel implementation that reads from standard input and writes to standard output.
pub struct TerminalChannel {
    chat_id: String,
}

impl TerminalChannel {
    pub fn new(chat_id: &str) -> Self {
        Self {
            chat_id: chat_id.to_string(),
        }
    }
}

#[async_trait]
impl Channel for TerminalChannel {
    fn name(&self) -> &str {
        "terminal"
    }

    async fn start(&self, inbound_tx: Sender<InboundMessage>) -> Result<(), String> {
        let channel_name = self.name().to_string();
        let mut chat_id = self.chat_id.clone();
        
        tokio::task::spawn_blocking(move || {
            let stdin = io::stdin();
            let mut input = String::new();

            loop {
                // Not the most robust CLI, but good enough for testing.
                print!("{}", "> ".bold().green());
                let _ = io::stdout().flush();
                input.clear();
                
                // blocking wait on stdin
                match stdin.read_line(&mut input) {
                    Ok(_) => {
                        let text = input.trim();
                        if text.is_empty() {
                            continue;
                        }

                        // Handle terminal-specific slash commands
                        if text.starts_with('/') {
                            if text.eq_ignore_ascii_case("/exit") || text.eq_ignore_ascii_case("/quit") {
                                println!("{}", "Safely shutting down Advanced Agent-RS System...".yellow());
                                std::process::exit(0);
                            }
                            if text.eq_ignore_ascii_case("/new") {
                                chat_id = uuid::Uuid::new_v4().to_string();
                                println!("{}", "Created a fresh new session!".green());
                                continue;
                            }
                            println!("{}", "Unknown slash command. Try /exit to quit, or /new to start fresh.".red());
                            continue;
                        }

                        // Handle legacy exit variants for user convenience
                        if text.eq_ignore_ascii_case("exit") || text.eq_ignore_ascii_case("quit") {
                            println!("{}", "Safely shutting down Advanced Agent-RS System...".yellow());
                            std::process::exit(0);
                        }

                        let msg = InboundMessage {
                            channel: channel_name.clone(),
                            sender_id: "local_user".to_string(),
                            chat_id: chat_id.clone(),
                            thread_id: None,
                            content: text.to_string(),
                            metadata: Default::default(),
                        };

                        if let Err(e) = inbound_tx.blocking_send(msg) {
                            error!("Terminal channel failed to send InboundMessage: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Failed to read from stdin: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        // Just let the stdin loop die organically or kill process
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        let mut stdout = io::stdout();
        // Clear the current line (which might have "> " and some partial user typing)
        let _ = execute!(
            stdout,
            cursor::MoveToColumn(0),
            Clear(ClearType::CurrentLine)
        );

        let header = "[Agent]:".cyan().bold();
        let content = msg.content.green();
        
        println!("{} {}", header, content);
        
        // Reprint the prompt marker 
        print!("{}", "> ".bold().green());
        let _ = stdout.flush();
        Ok(())
    }
}
