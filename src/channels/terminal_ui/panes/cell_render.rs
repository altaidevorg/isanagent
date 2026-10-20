//! Cell → line rendering shared by transcript and tool history.

use ratatui::prelude::*;
use ratatui::style::Modifier;

use crate::channels::terminal_ui::markdown;
use crate::channels::terminal_ui::{Cell, Theme, ToolNoticePhase};

/// Greedy wrap by display width (`unicode_width`); preserves explicit newlines.
pub fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width < 4 {
        return vec![text.to_string()];
    }
    let mut lines: Vec<String> = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut col = 0usize;
        for ch in paragraph.chars() {
            let w = unicode_width::UnicodeWidthChar::width(ch)
                .unwrap_or(0)
                .max(1);
            if col + w > width && !line.is_empty() {
                lines.push(std::mem::take(&mut line));
                col = 0;
            }
            if col == 0 && ch.is_whitespace() {
                continue;
            }
            line.push(ch);
            col += w;
        }
        if !line.is_empty() || paragraph.is_empty() {
            lines.push(line);
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

pub(crate) fn cell_block_lines(cell: &Cell, inner_width: usize) -> Vec<Line<'static>> {
    let w = inner_width.max(8);
    match cell {
        Cell::User { text } => {
            let mut v = vec![Line::from(vec![
                Span::styled(" ● ", Theme::user_prefix()),
                Span::styled("you", Theme::user_prefix()),
            ])];
            for ln in wrap_text(text, w.saturating_sub(2)) {
                v.push(Line::from(Span::styled(ln, Theme::text())));
            }
            v.push(Line::from(""));
            v
        }
        Cell::Assistant { markdown } => {
            let mut v = vec![Line::from(vec![
                Span::styled(" ◆ ", Theme::assistant_bullet()),
                Span::styled("agent", Theme::dim()),
            ])];
            v.extend(markdown::assistant_markdown_lines(
                markdown,
                w.saturating_sub(2),
            ));
            v.push(Line::from(""));
            v
        }
        Cell::Thinking { text } => {
            let mut v = vec![Line::from(Span::styled(" … thought", Theme::thinking()))];
            for ln in wrap_text(text, w) {
                v.push(Line::from(Span::styled(ln, Theme::thinking())));
            }
            v.push(Line::from(""));
            v
        }
        Cell::ToolNotice {
            phase,
            content,
            tool_call_id: _,
        } => {
            let label_style = match phase {
                ToolNoticePhase::Pending => Theme::tool_pending().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Call => Theme::tool_call().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Result => Theme::tool_done().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Failed => Theme::error().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Other => Theme::tool_call().add_modifier(Modifier::BOLD),
            };
            let label = match phase {
                ToolNoticePhase::Pending => "tool",
                ToolNoticePhase::Call => "tool",
                ToolNoticePhase::Result => "done",
                ToolNoticePhase::Failed => "fail",
                ToolNoticePhase::Other => "tool",
            };
            let mut v = vec![Line::from(vec![
                Span::styled(" ⚡ ", label_style),
                Span::styled(label, label_style),
            ])];
            let body_style = match phase {
                ToolNoticePhase::Pending => Theme::tool_pending(),
                ToolNoticePhase::Failed => Theme::error(),
                _ => Theme::text(),
            };
            for ln in wrap_text(content, w.saturating_sub(2)) {
                v.push(Line::from(Span::styled(ln, body_style)));
            }
            v.push(Line::from(""));
            v
        }
        Cell::Clarification {
            text,
            choices,
            edit_diff,
        } => {
            let inner = w.saturating_sub(2).max(8);
            let is_approval = crate::channels::terminal_ui::is_approval_choice_set(choices);
            let title = if edit_diff.is_some() {
                "edit approval"
            } else if is_approval {
                "approval"
            } else {
                "question"
            };
            let mut v = vec![Line::from(vec![
                Span::styled(" ? ", Theme::clarification()),
                Span::styled(title, Theme::clarification().add_modifier(Modifier::BOLD)),
            ])];
            if let Some(diff) = edit_diff {
                v.push(Line::from(vec![
                    Span::styled(" file ", Theme::dim()),
                    Span::styled(diff.file.clone(), Theme::active()),
                ]));
                if diff.truncated {
                    v.push(Line::from(Span::styled(" [truncated]", Theme::tool_call())));
                }
                v.extend(crate::channels::terminal_ui::approval::diff_lines_to_spans(
                    &diff.diff, 40,
                ));
            }
            for ln in wrap_text(text, inner) {
                v.push(Line::from(Span::styled(ln, Theme::clarification())));
            }
            if !choices.is_empty() {
                v.push(Line::from(Span::styled(
                    clarification_hint(choices, is_approval),
                    Theme::dim(),
                )));
                let indent = "   ";
                for (i, choice) in choices.iter().enumerate() {
                    let n = i + 1;
                    let head = format!("{n}. ");
                    let first = format!("{head}{choice}");
                    let lines = wrap_text(&first, inner);
                    for (li, seg) in lines.iter().enumerate() {
                        let line = if li == 0 {
                            seg.clone()
                        } else {
                            format!("{indent}{seg}")
                        };
                        v.push(Line::from(Span::styled(line, Theme::clarification())));
                    }
                }
            }
            v.push(Line::from(""));
            v
        }
        Cell::System { message } => {
            let mut v = vec![Line::from(Span::styled(" — system —", Theme::dim()))];
            for ln in wrap_text(message, w) {
                v.push(Line::from(Span::styled(ln, Theme::dim())));
            }
            v.push(Line::from(""));
            v
        }
        Cell::Error { message } => {
            let mut v = vec![Line::from(vec![
                Span::styled(" ! ", Theme::error()),
                Span::styled("error", Theme::error().add_modifier(Modifier::BOLD)),
            ])];
            for ln in wrap_text(message, w.saturating_sub(2)) {
                v.push(Line::from(Span::styled(ln, Theme::error())));
            }
            v.push(Line::from(""));
            v
        }
    }
}

/// Key hint above a clarification's numbered option list.
fn clarification_hint(choices: &[String], is_approval: bool) -> String {
    if is_approval {
        return "y/1 approve · n/2 deny · a/3 always · x/4 abort  (or type a reply)".into();
    }
    match choices.len().min(9) {
        0 => "type a reply".into(),
        1 => "press 1 to choose it, or type a reply".into(),
        n => format!("press 1-{n} to choose, or type a reply"),
    }
}

pub(crate) fn flatten_cells_to_lines(cells: &[Cell], inner_width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for cell in cells {
        lines.extend(cell_block_lines(cell, inner_width));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(cell: &Cell) -> Vec<String> {
        cell_block_lines(cell, 80)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn custom_question_shows_its_own_options() {
        let cell = Cell::Clarification {
            text: "Pick a color".into(),
            choices: vec!["red".into(), "green".into()],
            edit_diff: None,
        };
        let lines = text_of(&cell);
        assert!(lines[0].contains("question"), "{lines:?}");
        assert!(!lines.iter().any(|l| l.contains("approve")), "{lines:?}");
        assert!(lines.iter().any(|l| l == "1. red"), "{lines:?}");
        assert!(lines.iter().any(|l| l == "2. green"), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("press 1-2")), "{lines:?}");
    }

    #[test]
    fn approval_prompt_keeps_hotkey_hint() {
        let cell = Cell::Clarification {
            text: "Run `rm -rf build`?".into(),
            choices: crate::channels::terminal_ui::APPROVAL_CHOICES
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            edit_diff: None,
        };
        let lines = text_of(&cell);
        assert!(lines[0].contains("approval"), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("n/2 deny")), "{lines:?}");
    }

    #[test]
    fn free_text_question_lists_no_options() {
        let cell = Cell::Clarification {
            text: "What is the project name?".into(),
            choices: Vec::new(),
            edit_diff: None,
        };
        let lines = text_of(&cell);
        assert!(lines[0].contains("question"), "{lines:?}");
        assert!(!lines.iter().any(|l| l.starts_with("1. ")), "{lines:?}");
    }
}
