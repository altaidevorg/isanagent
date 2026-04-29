use ratatui::layout::Rect;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::channels::terminal_ui::text_format::{format_last_activity, truncate_chars_display};
use crate::channels::terminal_ui::App;
use crate::channels::terminal_ui::Theme;

/// Display columns reserved for last-activity time, ` | `, and list selection mark before the preview.
const SESSION_LIST_PREFIX_WIDTH: usize = 24;
/// Minimum display columns for the truncated preview when the pane is very narrow.
const MIN_PREVIEW_WIDTH: usize = 12;

/// Scrollable list of past root terminal sessions (memory).
pub fn conversations_list_paragraph(app: &App, area: Rect) -> (Paragraph<'static>, usize) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let visible = area.height.saturating_sub(2) as usize;
    let n = app.conversations_items.len();
    let max_scroll = if n == 0 {
        0usize
    } else {
        n.saturating_sub(visible.max(1))
    };
    let st = app.conversations_list_scroll_top.min(max_scroll);
    let mut slice: Vec<Line<'static>> = Vec::new();
    if let Some(err) = app.conversations_error.as_deref() {
        slice.push(Line::from(Span::styled(
            format!("Could not load past sessions: {err}"),
            Theme::error(),
        )));
    } else if app.conversations_items.is_empty() {
        slice.push(Line::from(Span::styled(
            "No saved terminal sessions yet. Send a message, use Tab to switch panes, or F5 to refresh.",
            Theme::dim(),
        )));
    } else {
        for (row, item) in app
            .conversations_items
            .iter()
            .enumerate()
            .skip(st)
            .take(visible.max(1))
        {
            let sel = app.conversations_selected_idx == Some(row);
            let mark = if sel { "› " } else { "  " };
            let ts = format_last_activity(item.last_activity_ms);
            let line = format!(
                "{} | {}",
                ts,
                truncate_chars_display(
                    &item.preview,
                    inner_w
                        .saturating_sub(SESSION_LIST_PREFIX_WIDTH)
                        .max(MIN_PREVIEW_WIDTH),
                )
            );
            let style = if sel {
                Theme::tool_call()
            } else {
                Theme::text()
            };
            slice.push(Line::from(vec![
                Span::styled(mark, Theme::tool_done()),
                Span::styled(line, style),
            ]));
        }
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " past sessions (terminal) ",
            Theme::tool_done(),
        ))
        .border_style(Theme::dim());
    (Paragraph::new(Text::from(slice)).block(block), max_scroll)
}

pub fn conversations_ensure_list_shows_selection(app: &mut App, list_inner_height: usize) {
    let v = list_inner_height.max(1);
    let Some(sel) = app.conversations_selected_idx else {
        return;
    };
    if sel < app.conversations_list_scroll_top {
        app.conversations_list_scroll_top = sel;
    }
    if sel >= app.conversations_list_scroll_top.saturating_add(v) {
        app.conversations_list_scroll_top = sel.saturating_sub(v.saturating_sub(1));
    }
}
