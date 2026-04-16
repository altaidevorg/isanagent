use crate::bus::{BusMessage, InboundMessage, LogEvent, OutboundMessage};
use crate::channels::Channel;
use crate::logging::LoggerHandle;
use crate::utils::{resolve_path, ContentPart, ImageUrl};
use async_trait::async_trait;
use colored::Colorize;
use crossterm::{
    cursor, execute,
    terminal::{Clear, ClearType},
};
use log::{error, info};
use serde_json::json;
use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use tokio::sync::mpsc::Sender;

const ISANAGENT_TOOL_NOTIFY: &str = "isanagent_tool_notify";
const ISANAGENT_TOOL_PHASE: &str = "isanagent_tool_phase";

fn truncate_display(s: &str, max_chars: usize) -> String {
    let t = s.trim();
    let n = t.chars().count();
    if n <= max_chars {
        return t.to_string();
    }
    let shortened: String = t.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{shortened}…")
}

fn summarize_tool_result_for_terminal(result: &str) -> String {
    let t = result.trim();
    if t.is_empty() {
        return "(empty output)".to_string();
    }
    if t.starts_with("Error:") {
        let line = t.lines().next().unwrap_or(t);
        return truncate_display(line, 160);
    }
    if t.chars().count() <= 120 {
        return t.to_string();
    }
    format!("{} chars", t.chars().count())
}

/// Live terminal line when a tool is invoked (mirrors telemetry, user-visible).
pub fn build_tool_call_terminal_notice(
    chat_id: &str,
    tool_name: &str,
    args: &str,
) -> OutboundMessage {
    let detail = truncate_display(args, 220);
    let content = if detail.is_empty() {
        tool_name.to_string()
    } else {
        format!("{tool_name} {detail}")
    };
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_TOOL_NOTIFY.to_string(), json!(true));
    metadata.insert(ISANAGENT_TOOL_PHASE.to_string(), json!("call"));
    OutboundMessage {
        channel: "terminal".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content,
        metadata,
    }
}

/// Live terminal line when a tool finishes (short summary; avoids flooding the TTY).
pub fn build_tool_result_terminal_notice(
    chat_id: &str,
    tool_name: &str,
    result: &str,
) -> OutboundMessage {
    let summary = summarize_tool_result_for_terminal(result);
    let content = format!("{tool_name} → {summary}");
    let mut metadata = HashMap::new();
    metadata.insert(ISANAGENT_TOOL_NOTIFY.to_string(), json!(true));
    metadata.insert(ISANAGENT_TOOL_PHASE.to_string(), json!("result"));
    OutboundMessage {
        channel: "terminal".to_string(),
        chat_id: chat_id.to_string(),
        thread_id: None,
        content,
        metadata,
    }
}

/// A Channel implementation that reads from standard input and writes to standard output.
pub struct TerminalChannel {
    chat_id: String,
    logger_tx: LoggerHandle,
    shutdown_tx: tokio::sync::mpsc::UnboundedSender<()>,
    /// All user-supplied `@<filepath>` references are resolved relative to this
    /// directory.  Paths that escape the sandbox boundary are silently rejected.
    sandbox_dir: PathBuf,
}

impl TerminalChannel {
    pub fn new(
        chat_id: &str,
        logger_tx: LoggerHandle,
        shutdown_tx: tokio::sync::mpsc::UnboundedSender<()>,
        sandbox_dir: PathBuf,
    ) -> Self {
        Self {
            chat_id: chat_id.to_string(),
            logger_tx,
            shutdown_tx,
            sandbox_dir,
        }
    }
}

/// Detects the MIME type of an image file from its extension.
/// Returns `None` for unsupported or non-image extensions.
fn image_mime_from_extension(path: &std::path::Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => Some("image/jpeg"),
        Some("png") => Some("image/png"),
        Some("gif") => Some("image/gif"),
        Some("webp") => Some("image/webp"),
        _ => None,
    }
}

/// Parses a terminal input string for `@<filepath>` references.
///
/// Each `@<path>` token is removed from the returned text and the referenced
/// file is read from disk, base64-encoded, and returned as an
/// `ContentPart::ImageUrl` attachment using a data URI.
///
/// Paths are resolved against `sandbox_dir`; any path that escapes the sandbox
/// boundary (via `../`, absolute references outside the sandbox, etc.) is
/// silently skipped.  Unsupported file types and unreadable files are also
/// silently skipped (a warning is printed to stderr instead of aborting the
/// whole message).
fn parse_terminal_attachments(
    input: &str,
    sandbox_dir: &std::path::Path,
) -> (String, Vec<ContentPart>) {
    use base64::Engine as _;
    let engine = base64::engine::general_purpose::STANDARD;

    let mut clean_parts: Vec<&str> = Vec::new();
    let mut attachments: Vec<ContentPart> = Vec::new();
    let mut last_end = 0;

    // Find all `@<path>` tokens.  A path token starts with `@` and ends at
    // the next whitespace character or end-of-string.
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < input.len() {
        if bytes[i] == b'@' {
            // Capture everything before this token as plain text
            clean_parts.push(&input[last_end..i]);

            // Advance past `@`
            let path_start = i + 1;
            let mut path_end = path_start;
            while path_end < input.len() && !bytes[path_end].is_ascii_whitespace() {
                path_end += 1;
            }

            let raw_path = &input[path_start..path_end];
            // Only expand `~` (home directory shorthand). Intentionally do NOT
            // use shellexpand::full() to avoid unintended environment variable
            // expansion (e.g. `@$HOME/.ssh/id_rsa`).
            let expanded = shellexpand::tilde(raw_path).into_owned();

            // Resolve and sandbox-check the path.
            let expanded_path = std::path::Path::new(&expanded);
            let path_exists = expanded_path.exists() || sandbox_dir.join(expanded_path).exists();
            match resolve_path(sandbox_dir, &expanded) {
                None if path_exists => {
                    eprintln!("Warning: @<path> is outside the sandbox boundary, skipping.");
                }
                None => {
                    eprintln!("Warning: @<path> does not exist or is not accessible, skipping.");
                }
                Some(file_path) => match image_mime_from_extension(&file_path) {
                    None => {
                        eprintln!("Warning: @<path> is not a supported image type (jpeg/png/gif/webp), skipping.");
                    }
                    Some(mime) => match std::fs::read(&file_path) {
                        Err(_) => {
                            eprintln!("Warning: could not read @<path>, skipping.");
                        }
                        Ok(bytes) => {
                            let data_uri =
                                format!("data:{};base64,{}", mime, engine.encode(&bytes));
                            attachments.push(ContentPart::ImageUrl {
                                image_url: ImageUrl {
                                    url: data_uri,
                                    detail: None,
                                },
                            });
                        }
                    },
                },
            }

            last_end = path_end;
            i = path_end;
        } else {
            i += 1;
        }
    }

    // Append any trailing plain text
    clean_parts.push(&input[last_end..]);

    let clean_text = clean_parts.join("").trim().to_string();
    (clean_text, attachments)
}

#[async_trait]
impl Channel for TerminalChannel {
    fn name(&self) -> &str {
        "terminal"
    }

    async fn start(&self, bus_tx: Sender<BusMessage>) -> Result<(), String> {
        let channel_name = self.name().to_string();
        let mut chat_id = self.chat_id.clone();
        let logger_tx = self.logger_tx.clone();
        let shutdown_tx = self.shutdown_tx.clone();
        let sandbox_dir = self.sandbox_dir.clone();

        let _ = logger_tx.send(BusMessage::Log(LogEvent::info(
            "TerminalChannel",
            "Starting Terminal channel...",
        )));

        tokio::task::spawn_blocking(move || {
            let stdin = io::stdin();
            let stdin_is_tty = stdin.is_terminal();
            let mut input = String::new();

            loop {
                // Not the most robust CLI, but good enough for testing.
                print!("{}", "> ".bold().green());
                let _ = io::stdout().flush();
                input.clear();

                // blocking wait on stdin
                match stdin.read_line(&mut input) {
                    Ok(0) => {
                        // `Ok(0)` is EOF with no bytes read. If we treated it like an empty line we
                        // would `continue` forever when stdin is closed/non-interactive (e.g. Docker
                        // without `-i`).
                        let msg = if !stdin_is_tty {
                            "Terminal channel: exiting stdin loop after EOF while stdin is not a TTY \
(non-interactive). Stops repeated empty reads; for API-only runs prefer `[terminal] enable = false` in config.toml."
                        } else {
                            "Terminal channel: stdin EOF; closing terminal input loop."
                        };
                        let _ =
                            logger_tx.send(BusMessage::Log(LogEvent::info("TerminalChannel", msg)));
                        info!("{}", msg);
                        break;
                    }
                    Ok(_) => {
                        let text = input.trim();
                        if text.is_empty() {
                            continue;
                        }

                        // Handle terminal-specific slash commands
                        if text.starts_with('/') {
                            if text.eq_ignore_ascii_case("/exit")
                                || text.eq_ignore_ascii_case("/quit")
                            {
                                println!(
                                    "{}",
                                    "Safely shutting down Advanced isanagent System...".yellow()
                                );
                                let _ = shutdown_tx.send(());
                                break;
                            }
                            if text.eq_ignore_ascii_case("/new") {
                                chat_id = uuid::Uuid::new_v4().to_string();
                                println!("{}", "Created a fresh new session!".green());
                                continue;
                            }
                            println!(
                                "{}",
                                "Unknown slash command. Try /exit to quit, or /new to start fresh."
                                    .red()
                            );
                            continue;
                        }

                        // Handle legacy exit variants for user convenience
                        if text.eq_ignore_ascii_case("exit") || text.eq_ignore_ascii_case("quit") {
                            println!(
                                "{}",
                                "Safely shutting down Advanced isanagent System...".yellow()
                            );
                            let _ = shutdown_tx.send(());
                            break;
                        }

                        // Parse @filepath references into multimodal attachments
                        let (clean_text, attachments) =
                            parse_terminal_attachments(text, &sandbox_dir);

                        let msg = InboundMessage {
                            channel: channel_name.clone(),
                            sender_id: "local_user".to_string(),
                            chat_id: chat_id.clone(),
                            thread_id: None,
                            content: clean_text,
                            attachments,
                            metadata: Default::default(),
                        };

                        if let Err(e) = bus_tx.blocking_send(BusMessage::Inbound(msg)) {
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

        let tool_notify = msg
            .metadata
            .get(ISANAGENT_TOOL_NOTIFY)
            .and_then(|v| v.as_bool())
            == Some(true);
        let phase = msg
            .metadata
            .get(ISANAGENT_TOOL_PHASE)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if tool_notify {
            match phase {
                "call" => println!("{} {}", "[Tool]".yellow().bold(), msg.content),
                "result" => println!("{} {}", "[Tool done]".yellow().bold(), msg.content.green()),
                _ => println!("{} {}", "[Tool]".yellow().bold(), msg.content),
            }
        } else {
            println!("{} {}", "[Agent]:".cyan().bold(), msg.content.green());
        }

        // Reprint the prompt marker (flush so tool/agent lines appear before the next await).
        print!("{}", "> ".bold().green());
        let _ = stdout.flush();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::parse_terminal_attachments;

    #[test]
    fn no_at_references_returns_text_unchanged() {
        let sandbox = std::env::temp_dir();
        let (text, attachments) = parse_terminal_attachments("hello world", &sandbox);
        assert_eq!(text, "hello world");
        assert!(attachments.is_empty());
    }

    #[test]
    fn unsupported_extension_is_skipped() {
        let sandbox = std::env::temp_dir();
        // Create the file inside the sandbox so the only reason it's skipped is
        // the unsupported extension, not a missing-file / sandbox rejection.
        let path = sandbox.join("isanagent_test_skip.txt");
        std::fs::write(&path, b"hello").ok();
        let input = format!("show @{} please", path.display());
        let (text, attachments) = parse_terminal_attachments(&input, &sandbox);
        assert!(!text.contains('@'));
        assert!(attachments.is_empty(), "non-image files must be skipped");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_skipped() {
        let sandbox = std::env::temp_dir();
        // Build a path that is inside the sandbox but does not exist.
        let path = sandbox.join("isanagent_nonexistent_image.png");
        // Ensure it really doesn't exist.
        let _ = std::fs::remove_file(&path);
        let input = format!("see @{} thanks", path.display());
        let (text, attachments) = parse_terminal_attachments(&input, &sandbox);
        assert!(!text.contains('@'));
        assert!(
            attachments.is_empty(),
            "missing file must be skipped gracefully"
        );
    }

    #[test]
    fn path_outside_sandbox_is_rejected() {
        use std::io::Write as _;
        // Create a sandbox subdirectory so we can reference a file *outside* it.
        let tmp = std::env::temp_dir();
        let sandbox = tmp.join("isanagent_sandbox_test");
        std::fs::create_dir_all(&sandbox).expect("create sandbox");

        // Put a real image file *outside* the sandbox (in tmp directly).
        let outside = tmp.join("isanagent_outside_test.png");
        let png_bytes: &[u8] = &[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
        let mut f = std::fs::File::create(&outside).expect("create outside png");
        f.write_all(png_bytes).expect("write png");
        drop(f);

        let input = format!("describe @{} please", outside.display());
        let (_text, attachments) = parse_terminal_attachments(&input, &sandbox);
        assert!(
            attachments.is_empty(),
            "file outside sandbox must be rejected"
        );

        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir_all(&sandbox);
    }

    #[test]
    fn existing_image_is_attached_as_data_uri() {
        use std::io::Write;
        // Use a temporary sandbox directory and write the image inside it.
        let sandbox = std::env::temp_dir().join("isanagent_sandbox_image_test");
        std::fs::create_dir_all(&sandbox).expect("create sandbox");
        let tmp = sandbox.join("isanagent_terminal_test.png");
        // Minimal 1x1 PNG bytes
        let png_bytes: &[u8] = &[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG signature
            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, // IHDR chunk length + type
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // width=1, height=1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // bit depth=8, color type=2
            0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, // IHDR CRC + IDAT chunk
            0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00, // IDAT data
            0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc, // IDAT data cont.
            0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, // IDAT CRC + IEND
            0x44, 0xae, 0x42, 0x60, 0x82, // IEND data
        ];
        let mut f = std::fs::File::create(&tmp).expect("create temp png");
        f.write_all(png_bytes).expect("write png");
        drop(f);

        let input = format!("describe this image @{} please", tmp.display());
        let (text, attachments) = parse_terminal_attachments(&input, &sandbox);

        // The @path token is removed from the text
        assert!(!text.contains('@'));
        assert_eq!(attachments.len(), 1, "should have one image attachment");

        if let crate::utils::ContentPart::ImageUrl { image_url } = &attachments[0] {
            assert!(
                image_url.url.starts_with("data:image/png;base64,"),
                "expected a PNG data URI, got: {}",
                &image_url.url[..40.min(image_url.url.len())]
            );
        } else {
            panic!("expected ContentPart::ImageUrl");
        }

        let _ = std::fs::remove_dir_all(&sandbox);
    }
}
