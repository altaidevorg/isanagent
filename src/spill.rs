//! Spill storage for large tool outputs.
//!
//! Preserves complete tool outputs (large logs, dataset dumps, compiler outputs)
//! in session-scoped storage while returning bounded head/tail previews with opaque
//! locators to the LLM.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

/// Spill storage manager for session-scoped large tool outputs.
#[derive(Clone, Debug)]
pub struct SpillStore {
    pub root_dir: PathBuf,
}

impl SpillStore {
    pub fn new(workspace_dir: &Path) -> Self {
        Self {
            root_dir: workspace_dir.join(".system_generated").join("spill"),
        }
    }

    /// Checks whether output exceeds `max_inline_chars`. If it does, persists
    /// the full output and returns a structured head/tail preview along with the `spill_id`.
    pub fn maybe_spill(
        &self,
        session_id: &str,
        tool_name: &str,
        content: &str,
        max_inline_chars: usize,
    ) -> (String, Option<String>) {
        if content.len() <= max_inline_chars {
            return (content.to_string(), None);
        }

        let clean_session: String = session_id
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let clean_session = if clean_session.is_empty() {
            "default".to_string()
        } else {
            clean_session
        };

        let clean_tool_name: String = tool_name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let clean_tool_name = if clean_tool_name.is_empty() {
            "tool".to_string()
        } else {
            clean_tool_name
        };

        let digest = Sha256::digest(content.as_bytes());
        let spill_id = format!("spill_{}_{}", clean_tool_name, hex::encode(&digest[..6]));

        let session_dir = self.root_dir.join(&clean_session);
        if let Err(e) = fs::create_dir_all(&session_dir) {
            log::warn!("Failed to create spill dir {}: {e}", session_dir.display());
            return (content.to_string(), None);
        }

        let file_path = session_dir.join(format!("{spill_id}.log"));
        if let Err(e) = fs::write(&file_path, content) {
            log::warn!("Failed to write spill file {}: {e}", file_path.display());
            return (content.to_string(), None);
        }

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let total_bytes = content.len();

        let preview = if total_lines > 120 {
            let head = lines[..60].join("\n");
            let tail = lines[total_lines - 60..].join("\n");
            let omitted = total_lines - 120;
            format!(
                "[Spill ID: {spill_id}] (Output exceeded limit: {total_bytes} bytes, {total_lines} lines)\n\
                Full output saved to spill storage. Retrieve specific line ranges with `recall_tool_result` using spill_id '{spill_id}'.\n\n\
                --- [Lines 1–60] ---\n\
                {head}\n\n\
                ... [{omitted} lines omitted] ...\n\n\
                --- [Lines {}–{total_lines}] ---\n\
                {tail}",
                total_lines - 59
            )
        } else {
            let mut end = max_inline_chars;
            while !content.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            format!(
                "[Spill ID: {spill_id}] (Output exceeded limit: {total_bytes} bytes)\n\
                Full output saved to spill storage. Retrieve slices with `recall_tool_result` using spill_id '{spill_id}'.\n\n\
                {}\n... (truncated)",
                &content[..end]
            )
        };

        (preview, Some(spill_id))
    }

    /// Retrieves a line slice from a saved spill file.
    pub fn read_spill_slice(
        &self,
        session_id: &str,
        spill_id: &str,
        start_line: usize,
        line_count: usize,
    ) -> Result<String, String> {
        let clean_session: String = session_id
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let clean_session = if clean_session.is_empty() {
            "default".to_string()
        } else {
            clean_session
        };

        if spill_id.contains('/') || spill_id.contains('\\') || spill_id.contains("..") {
            return Err("Invalid spill ID: contains path traversal".to_string());
        }

        let file_path = self
            .root_dir
            .join(&clean_session)
            .join(format!("{spill_id}.log"));

        if !file_path.is_file() {
            // Also try fallback search across all sessions in case the session changed
            let mut found = None;
            if let Ok(entries) = fs::read_dir(&self.root_dir) {
                for entry in entries.flatten() {
                    let candidate = entry.path().join(format!("{spill_id}.log"));
                    if candidate.is_file() {
                        found = Some(candidate);
                        break;
                    }
                }
            }

            match found {
                Some(p) => return Self::read_slice_from_file(&p, start_line, line_count),
                None => {
                    return Err(format!(
                        "Spill record '{spill_id}' not found at {}",
                        file_path.display()
                    ))
                }
            }
        }

        Self::read_slice_from_file(&file_path, start_line, line_count)
    }

    fn read_slice_from_file(
        path: &Path,
        start_line: usize,
        line_count: usize,
    ) -> Result<String, String> {
        let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();

        if total == 0 {
            return Ok(String::new());
        }

        let start = start_line.max(1);
        if start > total {
            return Ok(format!(
                "[Spill has {total} lines; start_line {start} is beyond end of file]"
            ));
        }

        let count = line_count.clamp(1, 500);
        let end = (start + count - 1).min(total);

        let slice = lines[start - 1..end].join("\n");
        Ok(format!("--- Lines {start}–{end} of {total} ---\n{slice}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_content_does_not_spill() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = SpillStore::new(temp.path());

        let (out, spill_id) = store.maybe_spill("test-chat", "test_tool", "short content", 1000);
        assert_eq!(out, "short content");
        assert!(spill_id.is_none());
    }

    #[test]
    fn large_content_spills_and_retrieves_slice() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = SpillStore::new(temp.path());

        let mut lines = Vec::new();
        for i in 1..=200 {
            lines.push(format!(
                "Line #{i}: this is some long content to exceed limits"
            ));
        }
        let content = lines.join("\n");

        let (preview, spill_id) = store.maybe_spill("test-chat", "exec", &content, 500);
        assert!(spill_id.is_some());
        let id = spill_id.unwrap();
        assert!(preview.contains(&id));
        assert!(preview.contains("Lines 1–60"));
        assert!(preview.contains("80 lines omitted"));

        // Retrieve slice
        let slice = store
            .read_spill_slice("test-chat", &id, 65, 10)
            .expect("slice");
        assert!(slice.contains("Line #65"));
        assert!(slice.contains("Line #74"));
        assert!(!slice.contains("Line #100"));
    }

    #[test]
    fn rejects_path_traversal_spill_ids() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = SpillStore::new(temp.path());

        let res = store.read_spill_slice("test-chat", "../../../etc/passwd", 1, 10);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Invalid spill ID"));
    }

    #[test]
    fn sanitizes_session_id_with_special_characters() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = SpillStore::new(temp.path());

        let (preview, spill_id) = store.maybe_spill(
            "chat/../../hack:1",
            "my::tool",
            "a\n".repeat(200).as_str(),
            10,
        );
        assert!(spill_id.is_some());
        let id = spill_id.unwrap();
        assert!(preview.contains(&id));

        let slice = store
            .read_spill_slice("chat/../../hack:1", &id, 1, 5)
            .expect("slice");
        assert!(slice.contains("Lines 1–5"));
    }
}
