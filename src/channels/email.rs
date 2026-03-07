use async_trait::async_trait;
use crate::channels::Channel;
use crate::bus::{InboundMessage, OutboundMessage};
use crate::config::EmailConfig;
use tokio::sync::mpsc::Sender;
use log::{info, error};
use std::time::Duration;
use lettre::{Message, SmtpTransport, Transport, transport::smtp::authentication::Credentials};
use native_tls::TlsConnector;

pub struct EmailChannel {
    config: EmailConfig,
}

impl EmailChannel {
    pub fn new(config: EmailConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Channel for EmailChannel {
    fn name(&self) -> &str {
        "email"
    }

    async fn start(&self, inbound_tx: Sender<InboundMessage>) -> Result<(), String> {
        info!("Starting Email channel...");
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                let cfg = config.clone();
                let tx = inbound_tx.clone();

                let res = tokio::task::spawn_blocking(move || -> Result<(), String> {
                    let tls = TlsConnector::builder().build().map_err(|e| e.to_string())?;
                    let client = imap::connect(
                        (&cfg.imap_host as &str, cfg.imap_port),
                        &cfg.imap_host,
                        &tls,
                    ).map_err(|e| e.to_string())?;

                    let mut session = client
                        .login(&cfg.imap_username, &cfg.imap_password)
                        .map_err(|e| e.0.to_string())?;

                    session.select("INBOX").map_err(|e| e.to_string())?;
                    
                    // Permanent streaming connection
                    loop {
                        log::info!("Searching for UNSEEN emails...");
                        let messages = session.search("UNSEEN").map_err(|e| e.to_string())?;
                        
                        log::info!("Found {} UNSEEN messages", messages.len());
                        for seq in messages {
                            log::info!("Fetching message {}", seq);
                            if let Ok(fetches) = session.fetch(seq.to_string(), "(ENVELOPE BODY[TEXT])") {
                                for m in fetches.iter() {
                                    let mut sender = String::from("unknown@example.com");
                                    let mut subject = String::new();
                                    
                                    if let Some(envelope) = m.envelope() {
                                        if let Some(subject_bytes) = envelope.subject {
                                            subject = String::from_utf8_lossy(subject_bytes).to_string();
                                        }
                                        if let Some(froms) = envelope.from.as_ref() {
                                            if let Some(from) = froms.first() {
                                                if let (Some(mailbox), Some(host)) = (from.mailbox, from.host) {
                                                    sender = format!("{}@{}", 
                                                        String::from_utf8_lossy(mailbox), 
                                                        String::from_utf8_lossy(host)
                                                    );
                                                }
                                            }
                                        }
                                    }

                                    let mut content = String::new();
                                    if let Some(body) = m.text() {
                                        content = String::from_utf8_lossy(body).trim().to_string();
                                    }

                                    let thread_id = subject;
                                    let chat_id = sender.clone();

                                    let inbound = InboundMessage {
                                        channel: "email".to_string(),
                                        sender_id: sender.clone(),
                                        chat_id,
                                        thread_id: Some(thread_id),
                                        content,
                                        metadata: std::collections::HashMap::new(),
                                    };

                                    if let Err(e) = tx.blocking_send(inbound) {
                                        error!("Failed to route email to agent bus: {}", e);
                                    }
                                }
                            }
                            // Mark as read
                            let _ = session.store(format!("{}", seq), "+FLAGS (\\Seen)");
                        }
                        
                        log::info!("Entering IMAP IDLE state...");
                        {
                            let idle = session.idle().map_err(|e| e.to_string())?;
                            idle.wait_keepalive().map_err(|e| format!("IDLE failed: {}", e))?;
                        }
                        log::info!("Woke up from IMAP IDLE state!");
                    }
                }).await;

                if let Err(e) = res {
                    error!("IMAP panic: {}", e);
                } else if let Ok(Err(e)) = res {
                    error!("IMAP Error: {}. Reconnecting in 15 seconds.", e);
                }

                tokio::time::sleep(Duration::from_secs(15)).await;
            }
        });

        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        info!("Stopping Email channel...");
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        let config = self.config.clone();
        
        let to_address = msg.chat_id.clone();
        let subject = msg.thread_id.unwrap_or_else(|| "Re: Message from Altbot".to_string());
        
        tokio::task::spawn_blocking(move || -> Result<(), String> {
            let email = Message::builder()
                .from(config.email_address.parse().map_err(|e| format!("Invalid from address: {}", e))?)
                .to(to_address.parse().map_err(|e| format!("Invalid to address: {}", e))?)
                .subject(if subject.starts_with("Re:") { subject.clone() } else { format!("Re: {}", subject) })
                .body(msg.content.clone())
                .map_err(|e| e.to_string())?;

            let creds = Credentials::new(config.imap_username.clone(), config.imap_password.clone());

            let mailer = SmtpTransport::relay(&config.smtp_host)
                .map_err(|e| format!("Invalid SMTP host: {}", e))?
                .port(config.smtp_port)
                .credentials(creds)
                .build();

            mailer.send(&email).map_err(|e| e.to_string())?;
            
            info!("Successfully sent reply email to {}", to_address);
            Ok(())
        })
        .await
        .map_err(|e| format!("SMTP task panicked: {}", e))?
    }
}
