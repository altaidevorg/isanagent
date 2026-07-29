//! Approval / edit-diff helpers for the Ratatui terminal UI.

use ratatui::style::Modifier;
use ratatui::text::{Line, Span};

use super::theme::Theme;

/// Workspace-relative edit preview carried on clarification metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditDiffPayload {
    pub file: String,
    pub diff: String,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffLineKind {
    Add,
    Del,
    Hunk,
    Meta,
    Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffLine {
    pub kind: DiffLineKind,
    pub gutter: &'static str,
    pub text: String,
}

/// Parse unified-diff text into styled line descriptors (mirrors desktop EditApprovalCard).
pub fn parse_diff_lines(diff: &str) -> Vec<DiffLine> {
    diff.lines()
        .map(|raw| {
            let (kind, gutter, text) = if raw.starts_with("+++") || raw.starts_with("---") {
                (DiffLineKind::Meta, " ", raw.to_string())
            } else if raw.starts_with("@@") {
                (DiffLineKind::Hunk, " ", raw.to_string())
            } else if let Some(rest) = raw.strip_prefix('+') {
                (DiffLineKind::Add, "+", rest.to_string())
            } else if let Some(rest) = raw.strip_prefix('-') {
                (DiffLineKind::Del, "-", rest.to_string())
            } else if let Some(rest) = raw.strip_prefix(' ') {
                (DiffLineKind::Context, " ", rest.to_string())
            } else {
                (DiffLineKind::Context, " ", raw.to_string())
            };
            DiffLine { kind, gutter, text }
        })
        .collect()
}

pub fn diff_lines_to_spans(diff: &str, max_lines: usize) -> Vec<Line<'static>> {
    let parsed = parse_diff_lines(diff);
    let truncated_input = parsed.len() > max_lines;
    let mut out = Vec::new();
    for line in parsed.into_iter().take(max_lines) {
        let style = match line.kind {
            DiffLineKind::Add => Theme::tool_done(),
            DiffLineKind::Del => Theme::error(),
            DiffLineKind::Hunk => Theme::clarification(),
            DiffLineKind::Meta => Theme::dim(),
            DiffLineKind::Context => Theme::text(),
        };
        out.push(Line::from(vec![
            Span::styled(line.gutter.to_string(), Theme::dim()),
            Span::styled(line.text, style),
        ]));
    }
    if truncated_input {
        out.push(Line::from(Span::styled(
            "… [diff truncated for display]",
            Theme::dim().add_modifier(Modifier::ITALIC),
        )));
    }
    out
}

/// Four canonical approval choices shown in overlays / line mode.
pub const APPROVAL_CHOICES: &[&str] = &["approve", "deny", "always", "abort"];

/// Map a hotkey / digit to a canonical approval reply.
pub fn approval_hotkey_reply(ch: char) -> Option<&'static str> {
    match ch {
        'y' | 'Y' | '1' => Some("approve"),
        'n' | 'N' | '2' => Some("deny"),
        'a' | 'A' | '3' => Some("always"),
        'x' | 'X' | '4' | '\u{1b}' => Some("abort"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_unified_diff_kinds() {
        let lines = parse_diff_lines(
            "--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new\n context\n",
        );
        assert_eq!(lines[0].kind, DiffLineKind::Meta);
        assert_eq!(lines[2].kind, DiffLineKind::Hunk);
        assert_eq!(lines[3].kind, DiffLineKind::Del);
        assert_eq!(lines[4].kind, DiffLineKind::Add);
        assert_eq!(lines[5].kind, DiffLineKind::Context);
    }

    #[test]
    fn hotkeys_map_four_way_choices() {
        assert_eq!(approval_hotkey_reply('y'), Some("approve"));
        assert_eq!(approval_hotkey_reply('3'), Some("always"));
        assert_eq!(approval_hotkey_reply('x'), Some("abort"));
    }
}
