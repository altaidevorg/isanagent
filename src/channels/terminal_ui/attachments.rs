//! `@path` image attachment parsing for terminal input (Ratatui compose line).

use crate::utils::{resolve_path, ContentPart, ImageUrl};
use base64::Engine as _;

/// Detects the MIME type of an image file from its extension.
pub(crate) fn image_mime_from_extension(path: &std::path::Path) -> Option<&'static str> {
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
/// Each `@<path>` token that resolves to a supported image file (png/jpeg/gif/webp)
/// inside the sandbox is consumed: removed from the returned text, base64-encoded,
/// and returned as a `ContentPart::ImageUrl` attachment using a data URI.
///
/// `@<token>` references that do NOT resolve to an image (missing files, non-image
/// extensions, outside sandbox) are preserved in the text so they can serve as
/// agent mentions or other syntax.
pub(crate) fn parse_terminal_attachments(
    input: &str,
    sandbox_dir: &std::path::Path,
) -> (String, Vec<ContentPart>) {
    let engine = base64::engine::general_purpose::STANDARD;

    let mut clean_parts: Vec<&str> = Vec::new();
    let mut attachments: Vec<ContentPart> = Vec::new();
    let mut last_end = 0;

    let bytes = input.as_bytes();
    let mut i = 0;
    while i < input.len() {
        if bytes[i] == b'@' {
            let path_start = i + 1;
            let mut path_end = path_start;
            while path_end < input.len() && !bytes[path_end].is_ascii_whitespace() {
                path_end += 1;
            }

            let raw_path = &input[path_start..path_end];
            let expanded = shellexpand::tilde(raw_path).into_owned();

            let expanded_path = std::path::Path::new(&expanded);
            let path_exists = expanded_path.exists() || sandbox_dir.join(expanded_path).exists();

            // Only strip @token from text when it resolves to a supported image file.
            // Unresolvable tokens (missing files, non-image types, outside sandbox) are
            // preserved as regular text so that @agent mentions and similar syntax survive.
            let mut consumed = false;
            match resolve_path(sandbox_dir, &expanded) {
                None if path_exists => {
                    eprintln!("Warning: @<path> is outside the sandbox boundary, skipping.");
                }
                None => {
                    // File doesn't exist — leave @token as text (may be an agent mention).
                }
                Some(file_path) => match image_mime_from_extension(&file_path) {
                    None => {
                        // Not a supported image type — leave @token as text.
                    }
                    Some(mime) => match std::fs::read(&file_path) {
                        Err(_) => {
                            eprintln!("Warning: could not read @<path>, skipping.");
                        }
                        Ok(file_bytes) => {
                            let data_uri =
                                format!("data:{};base64,{}", mime, engine.encode(&file_bytes));
                            attachments.push(ContentPart::ImageUrl {
                                image_url: ImageUrl {
                                    url: data_uri,
                                    detail: None,
                                },
                            });
                            consumed = true;
                        }
                    },
                },
            }

            if consumed {
                clean_parts.push(&input[last_end..i]);
                last_end = path_end;
                i = path_end;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    clean_parts.push(&input[last_end..]);

    let clean_text = clean_parts.join("").trim().to_string();
    (clean_text, attachments)
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
        let path = sandbox.join("isanagent_test_skip.txt");
        std::fs::write(&path, b"hello").ok();
        let input = format!("show @{} please", path.display());
        let (text, attachments) = parse_terminal_attachments(&input, &sandbox);
        assert!(
            text.contains('@'),
            "non-image @reference must stay in text (may be an agent mention)"
        );
        assert!(attachments.is_empty(), "non-image files must be skipped");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_skipped() {
        let sandbox = std::env::temp_dir();
        let path = sandbox.join("isanagent_nonexistent_image.png");
        let _ = std::fs::remove_file(&path);
        let input = format!("see @{} thanks", path.display());
        let (text, attachments) = parse_terminal_attachments(&input, &sandbox);
        assert!(
            text.contains('@'),
            "unresolved @reference must stay in text (may be an agent mention)"
        );
        assert!(
            attachments.is_empty(),
            "missing file must be skipped gracefully"
        );
    }

    #[test]
    fn path_outside_sandbox_is_rejected() {
        use std::io::Write as _;
        let tmp = std::env::temp_dir();
        let sandbox = tmp.join("isanagent_sandbox_test");
        std::fs::create_dir_all(&sandbox).expect("create sandbox");

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
        let sandbox = std::env::temp_dir().join("isanagent_sandbox_image_test");
        std::fs::create_dir_all(&sandbox).expect("create sandbox");
        let tmp = sandbox.join("isanagent_terminal_test.png");
        let png_bytes: &[u8] = &[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc,
            0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        let mut f = std::fs::File::create(&tmp).expect("create temp png");
        f.write_all(png_bytes).expect("write png");
        drop(f);

        let input = format!("describe this image @{} please", tmp.display());
        let (text, attachments) = parse_terminal_attachments(&input, &sandbox);

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
