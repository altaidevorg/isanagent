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
        Cell::Clarification { text, choices } => {
            let inner = w.saturating_sub(2).max(8);
            let mut v = vec![Line::from(vec![
                Span::styled(" ? ", Theme::clarification()),
                Span::styled("question", Theme::clarification()),
            ])];
            for ln in wrap_text(text, inner) {
                v.push(Line::from(Span::styled(ln, Theme::clarification())));
            }
            if !choices.is_empty() {
                v.push(Line::from(Span::styled(
                    "Reply with a number (1–n) or the exact option text.",
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

pub(crate) fn flatten_cells_to_lines(cells: &[Cell], inner_width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for cell in cells {
        lines.extend(cell_block_lines(cell, inner_width));
    }
    lines
}
