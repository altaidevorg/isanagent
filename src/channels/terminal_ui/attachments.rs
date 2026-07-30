//! `@path` attachment parsing for terminal input (Ratatui compose + line mode).

use crate::utils::{resolve_path, ContentPart, Document, ImageUrl};
use base64::Engine as _;
use std::path::{Path, PathBuf};

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

fn is_probably_text_extension(path: &Path) -> bool {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .as_deref()
    {
        None => true, // extensionless (Dockerfile, Makefile, LICENSE, …)
        Some(ext) => matches!(
            ext,
            "rs" | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "json"
                | "toml"
                | "yaml"
                | "yml"
                | "md"
                | "txt"
                | "css"
                | "html"
                | "py"
                | "go"
                | "java"
                | "kt"
                | "swift"
                | "c"
                | "h"
                | "cpp"
                | "hpp"
                | "cs"
                | "rb"
                | "php"
                | "sh"
                | "bash"
                | "zsh"
                | "sql"
                | "graphql"
                | "xml"
                | "svg"
                | "env"
                | "ini"
                | "cfg"
                | "conf"
                | "lock"
                | "gitignore"
                | "dockerfile"
                | "makefile"
                | "cmake"
                | "gradle"
                | "properties"
        ),
    }
}

fn is_pdf(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

const MAX_TEXT_ATTACH_CHARS: usize = 200_000;

/// Load a sandbox-scoped file into a multimodal content part.
pub fn load_sandbox_file_attachment(
    sandbox_dir: &Path,
    path: &Path,
) -> Result<ContentPart, String> {
    let expanded = shellexpand::tilde(&path.display().to_string()).into_owned();
    let resolved = resolve_path(sandbox_dir, &expanded)
        .or_else(|| fuzzy_resolve_in_sandbox(sandbox_dir, &expanded))
        .ok_or_else(|| {
            format!(
                "could not resolve attachment path inside workspace: {}",
                path.display()
            )
        })?;

    if let Some(mime) = image_mime_from_extension(&resolved) {
        let bytes = std::fs::read(&resolved)
            .map_err(|error| format!("could not read {}: {error}", resolved.display()))?;
        let engine = base64::engine::general_purpose::STANDARD;
        let data_uri = format!("data:{mime};base64,{}", engine.encode(&bytes));
        return Ok(ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: data_uri,
                detail: None,
            },
        });
    }

    if is_pdf(&resolved) {
        let bytes = std::fs::read(&resolved)
            .map_err(|error| format!("could not read {}: {error}", resolved.display()))?;
        let engine = base64::engine::general_purpose::STANDARD;
        let name = resolved
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string);
        return Ok(ContentPart::Document {
            document: Document {
                data: engine.encode(&bytes),
                media_type: "application/pdf".into(),
                name,
            },
        });
    }

    if !is_probably_text_extension(&resolved) {
        return Err(format!(
            "unsupported attachment type for {}: use text, image, or PDF",
            resolved.display()
        ));
    }

    let mut text = std::fs::read_to_string(&resolved)
        .map_err(|error| format!("could not read {}: {error}", resolved.display()))?;
    let truncated = text.chars().count() > MAX_TEXT_ATTACH_CHARS;
    if truncated {
        text = text.chars().take(MAX_TEXT_ATTACH_CHARS).collect();
        text.push_str("\n… [truncated]");
    }
    let sandbox_canon =
        std::fs::canonicalize(sandbox_dir).unwrap_or_else(|_| sandbox_dir.to_path_buf());
    let resolved_canon = std::fs::canonicalize(&resolved).unwrap_or(resolved.clone());
    let rel = resolved_canon
        .strip_prefix(&sandbox_canon)
        .unwrap_or(resolved_canon.as_path());
    let label = rel.display();
    let body = format!("<context-file path=\"{label}\">\n{text}\n</context-file>");
    Ok(ContentPart::Text { text: body })
}

/// Load many host `--file` paths into attachments; returns warnings for failures.
pub fn load_host_file_attachments(
    sandbox_dir: &Path,
    files: &[PathBuf],
) -> (Vec<ContentPart>, Vec<String>) {
    let mut attachments = Vec::new();
    let mut warnings = Vec::new();
    for path in files {
        match load_sandbox_file_attachment(sandbox_dir, path) {
            Ok(part) => attachments.push(part),
            Err(error) => warnings.push(error),
        }
    }
    (attachments, warnings)
}

/// Fuzzy-resolve a partial path / basename under the sandbox (depth-limited walk).
pub fn fuzzy_resolve_in_sandbox(sandbox_dir: &Path, query: &str) -> Option<PathBuf> {
    let query = query.trim().trim_start_matches("./");
    if query.is_empty() {
        return None;
    }
    let query_lower = query.to_ascii_lowercase();
    let mut matches: Vec<PathBuf> = Vec::new();
    let mut stack = vec![(sandbox_dir.to_path_buf(), 0u32)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > 6 || matches.len() >= 32 {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if name == ".git" || name == "node_modules" || name == "target" || name == ".isanagent"
            {
                continue;
            }
            if path.is_dir() {
                stack.push((path, depth + 1));
                continue;
            }
            let rel = path
                .strip_prefix(sandbox_dir)
                .unwrap_or(path.as_path())
                .to_string_lossy()
                .to_ascii_lowercase();
            if name == query_lower
                || rel == query_lower
                || rel.ends_with(&format!("/{query_lower}"))
                || rel.contains(&query_lower)
            {
                matches.push(path);
            }
        }
    }
    if matches.len() == 1 {
        return matches.pop();
    }
    // Prefer exact basename match when multiple fuzzy hits.
    let exact: Vec<_> = matches
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.eq_ignore_ascii_case(query))
        })
        .cloned()
        .collect();
    if exact.len() == 1 {
        return exact.into_iter().next();
    }
    None
}

/// Parses a terminal input string for `@<filepath>` references.
///
/// Supported attachments: images, PDFs, and common text source files.
/// Unresolvable `@token`s are preserved as literal text (agent mentions).
pub fn parse_terminal_attachments(
    input: &str,
    sandbox_dir: &std::path::Path,
) -> (String, Vec<ContentPart>) {
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

            let mut consumed = false;
            match load_sandbox_file_attachment(sandbox_dir, Path::new(&expanded)) {
                Ok(part) => {
                    attachments.push(part);
                    consumed = true;
                }
                Err(error) => {
                    // Keep @token when it looks like an agent mention (no slash / no extension).
                    if expanded.contains('/') || expanded.contains('.') {
                        eprintln!("Warning: {error}");
                    }
                }
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
    use super::*;
    use std::fs;

    #[test]
    fn no_at_references_returns_text_unchanged() {
        let sandbox = std::env::temp_dir();
        let (text, attachments) = parse_terminal_attachments("hello world", &sandbox);
        assert_eq!(text, "hello world");
        assert!(attachments.is_empty());
    }

    #[test]
    fn loads_text_file_as_context_part() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("note.md");
        fs::write(&path, "# hello\n").expect("write");
        let part = load_sandbox_file_attachment(dir.path(), Path::new("note.md")).expect("load");
        match part {
            ContentPart::Text { text } => {
                assert!(text.contains("path=\"note.md\""));
                assert!(text.contains("# hello"));
            }
            other => panic!("expected text part, got {other:?}"),
        }
    }

    #[test]
    fn fuzzy_resolves_unique_basename() {
        let dir = tempfile::tempdir().expect("tempdir");
        let nested = dir.path().join("src");
        fs::create_dir_all(&nested).expect("mkdir");
        fs::write(nested.join("unique_m4_file.rs"), "fn main() {}\n").expect("write");
        let found = fuzzy_resolve_in_sandbox(dir.path(), "unique_m4_file.rs").expect("fuzzy");
        assert!(found.ends_with("unique_m4_file.rs"));
    }

    #[test]
    fn at_text_reference_is_consumed() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("README.md"), "docs\n").expect("write");
        let (text, attachments) =
            parse_terminal_attachments("summarize @README.md please", dir.path());
        assert_eq!(text, "summarize  please");
        assert_eq!(attachments.len(), 1);
    }
}
