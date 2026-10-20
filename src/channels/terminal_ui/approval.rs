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

/// True when `choices` is exactly the shell / edit approval set (in order), so the
/// single-key approval hotkeys apply. Custom `ask_user` questions return false.
pub fn is_approval_choice_set(choices: &[String]) -> bool {
    choices.len() == APPROVAL_CHOICES.len()
        && choices
            .iter()
            .zip(APPROVAL_CHOICES)
            .all(|(c, a)| c.trim().eq_ignore_ascii_case(a))
}

/// Reply for a single key pressed on an empty compose line while a clarification is pending.
/// Approval prompts keep the y/n/a/x + 1-4 hotkeys; other questions only map digit `N` to
/// option `N` of their own list. `None` means the key should be typed normally.
pub fn clarification_hotkey_reply(choices: &[String], ch: char) -> Option<String> {
    if is_approval_choice_set(choices) {
        return approval_hotkey_reply(ch).map(str::to_string);
    }
    let n = ch.to_digit(10)? as usize;
    n.checked_sub(1).and_then(|i| choices.get(i)).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_unified_diff_kinds() {
        let lines = parse_diff_lines("--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new\n context\n");
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

    fn owned(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn approval_prompts_keep_letter_hotkeys() {
        let c = owned(APPROVAL_CHOICES);
        assert!(is_approval_choice_set(&c));
        assert_eq!(clarification_hotkey_reply(&c, 'n').as_deref(), Some("deny"));
        assert_eq!(
            clarification_hotkey_reply(&c, '3').as_deref(),
            Some("always")
        );
        assert_eq!(clarification_hotkey_reply(&c, 'q'), None);
    }

    #[test]
    fn custom_questions_map_digits_to_their_own_options() {
        let c = owned(&["red", "green", "blue"]);
        assert!(!is_approval_choice_set(&c));
        assert_eq!(
            clarification_hotkey_reply(&c, '2').as_deref(),
            Some("green")
        );
        assert_eq!(clarification_hotkey_reply(&c, '0'), None);
        assert_eq!(clarification_hotkey_reply(&c, '4'), None);
        // Letters are free text, never an approval reply.
        assert_eq!(clarification_hotkey_reply(&c, 'n'), None);
        assert_eq!(clarification_hotkey_reply(&c, 'y'), None);
        assert_eq!(clarification_hotkey_reply(&[], '1'), None);
    }
}
