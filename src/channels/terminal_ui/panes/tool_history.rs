use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::style::Modifier;
use ratatui::widgets::{Block, Borders, Paragraph};

use super::cell_render::wrap_text;
use crate::channels::terminal_ui::{Theme, ToolNoticePhase, ToolRailEntry};

pub fn tool_history_paragraph(
    entries: &[ToolRailEntry],
    area: Rect,
    scroll_from_bottom: u16,
) -> (Paragraph<'static>, u16) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();
    if entries.is_empty() {
        lines.push(Line::from(Span::styled(
            "No tool calls yet. Invoke the agent with tools enabled to see activity here.",
            Theme::dim(),
        )));
    } else {
        for e in entries {
            let label_style = match e.phase {
                ToolNoticePhase::Pending => Theme::tool_pending().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Call => Theme::tool_call().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Result => Theme::tool_done().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Failed => Theme::error().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Other => Theme::tool_call().add_modifier(Modifier::BOLD),
            };
            let label = match e.phase {
                ToolNoticePhase::Pending => "wait",
                ToolNoticePhase::Call => "call",
                ToolNoticePhase::Result => "done",
                ToolNoticePhase::Failed => "fail",
                ToolNoticePhase::Other => "tool",
            };
            let body_w = inner_w.max(8).saturating_sub(8);
            let wrapped = wrap_text(&e.summary, body_w);
            for (i, seg) in wrapped.into_iter().enumerate() {
                if i == 0 {
                    lines.push(Line::from(vec![
                        Span::styled(format!(" {label} "), label_style),
                        Span::styled(seg, Theme::text()),
                    ]));
                } else {
                    lines.push(Line::from(Span::styled(
                        format!("       {seg}"),
                        Theme::dim(),
                    )));
                }
            }
        }
    }
    let visible = area.height.saturating_sub(2) as usize;
    let total = lines.len();
    let max_scroll = total.saturating_sub(visible).min(u16::MAX as usize) as u16;
    let start = total
        .saturating_sub(visible)
        .saturating_sub(scroll_from_bottom as usize);
    let take = visible.max(1);
    let slice: Vec<Line<'static>> = lines.into_iter().skip(start).take(take).collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" tool activity ", Theme::tool_call()))
        .border_style(Theme::dim());
    (Paragraph::new(Text::from(slice)).block(block), max_scroll)
}
