use crate::bus::{BusMessage, InboundMessage, OutboundMessage};
use crate::channels::Channel;
use crate::config::EmailConfig;
use crate::logging::LoggerHandle;
use async_trait::async_trait;
use imap::ClientBuilder;
use lettre::{transport::smtp::authentication::Credentials, Message, SmtpTransport, Transport};
use log::{error, info};
use std::sync::Mutex;
use std::time::Duration;
use tokio::sync::{mpsc::Sender, watch};

pub struct EmailChannel {
    config: EmailConfig,
    logger_tx: LoggerHandle,
    shutdown_tx: watch::Sender<bool>,
    task_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl EmailChannel {
    pub fn new(config: EmailConfig, logger_tx: LoggerHandle) -> Self {
        let (shutdown_tx, _) = watch::channel(false);
        Self {
            config,
            logger_tx,
            shutdown_tx,
            task_handle: Mutex::new(None),
        }
    }
}

#[async_trait]
impl Channel for EmailChannel {
    fn name(&self) -> &str {
        "email"
    }

    async fn start(&self, bus_tx: Sender<BusMessage>) -> Result<(), String> {
        let _ = self
            .logger_tx
            .send(crate::bus::BusMessage::Log(crate::bus::LogEvent::info(
                "EmailChannel",
                "Starting Email channel...",
            )));
        let config = self.config.clone();
        let logger_tx = self.logger_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        let handle = tokio::spawn(async move {
            let _ = logger_tx.send(crate::bus::BusMessage::Log(crate::bus::LogEvent::info(
                "EmailChannel",
                "Started email background listener task.",
            )));

            loop {
                if *shutdown_rx.borrow() {
                    break;
                }

                let cfg = config.clone();
                let tx = bus_tx.clone();
                let res = tokio::task::spawn_blocking(move || poll_inbox_once(cfg, tx)).await;

                if let Err(e) = res {
                    error!("IMAP panic: {}", e);
                } else if let Ok(Err(e)) = res {
                    error!("IMAP Error: {}. Reconnecting in 15 seconds.", e);
                }

                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(15)) => {}
                    changed = shutdown_rx.changed() => {
                        if changed.is_ok() && *shutdown_rx.borrow() {
                            break;
                        }
                    }
                }
            }

            let _ = logger_tx.send(crate::bus::BusMessage::Log(crate::bus::LogEvent::info(
                "EmailChannel",
                "Email channel stopped.",
            )));
        });

        *self.task_handle.lock().unwrap() = Some(handle);
        Ok(())
    }

    async fn stop(&self) -> Result<(), String> {
        info!("Stopping Email channel...");
        let _ = self.shutdown_tx.send(true);
        let handle = self.task_handle.lock().unwrap().take();
        if let Some(handle) = handle {
            let _ = handle.await;
        }
        Ok(())
    }

    async fn send(&self, msg: OutboundMessage) -> Result<(), String> {
        let config = self.config.clone();

        let to_address = msg.chat_id.clone();
        let subject = msg
            .thread_id
            .unwrap_or_else(|| "Re: Message from Altbot".to_string());

        tokio::task::spawn_blocking(move || -> Result<(), String> {
            let email = Message::builder()
                .from(
                    config
                        .email_address
                        .parse()
                        .map_err(|e| format!("Invalid from address: {}", e))?,
                )
                .to(to_address
                    .parse()
                    .map_err(|e| format!("Invalid to address: {}", e))?)
                .subject(if subject.starts_with("Re:") {
                    subject.clone()
                } else {
                    format!("Re: {}", subject)
                })
                .body(msg.content.clone())
                .map_err(|e| e.to_string())?;

            let creds =
                Credentials::new(config.imap_username.clone(), config.imap_password.clone());

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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn poll_inbox_once(config: EmailConfig, bus_tx: Sender<BusMessage>) -> Result<(), String> {
    let client = ClientBuilder::new(&config.imap_host, config.imap_port)
        .connect()
        .map_err(|e| e.to_string())?;

    let mut session = client
        .login(&config.imap_username, &config.imap_password)
        .map_err(|e| e.0.to_string())?;

    session.select("INBOX").map_err(|e| e.to_string())?;
    info!("Searching for UNSEEN emails...");
    let messages = session.search("UNSEEN").map_err(|e| e.to_string())?;
    info!("Found {} UNSEEN messages", messages.len());

    for seq in messages {
        info!("Fetching message {}", seq);
        if let Ok(fetches) = session.fetch(seq.to_string(), "(ENVELOPE BODY[TEXT])") {
            for m in fetches.iter() {
                let mut sender = String::from("unknown@example.com");
                let mut subject = String::new();

                if let Some(envelope) = m.envelope() {
                    if let Some(subject_bytes) = &envelope.subject {
                        subject = String::from_utf8_lossy(subject_bytes.as_ref()).to_string();
                    }
                    if let Some(froms) = envelope.from.as_ref() {
                        if let Some(from) = froms.first() {
                            if let (Some(mailbox), Some(host)) = (&from.mailbox, &from.host) {
                                sender = format!(
                                    "{}@{}",
                                    String::from_utf8_lossy(mailbox.as_ref()),
                                    String::from_utf8_lossy(host.as_ref())
                                );
                            }
                        }
                    }
                }

                let mut content = String::new();
                if let Some(body) = m.text() {
                    content = String::from_utf8_lossy(body).trim().to_string();
                }

                let inbound = InboundMessage {
                    channel: "email".to_string(),
                    sender_id: sender.clone(),
                    chat_id: sender,
                    thread_id: Some(subject),
                    content,
                    attachments: Vec::new(),
                    metadata: std::collections::HashMap::new(),
                };

                if let Err(e) = bus_tx.blocking_send(BusMessage::Inbound(inbound)) {
                    error!("Failed to route email to agent bus: {}", e);
                }
            }
        }
        let _ = session.store(format!("{}", seq), "+FLAGS (\\Seen)");
    }

    let _ = session.logout();
    Ok(())
}
