use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use super::cell_render::wrap_text;
use crate::channels::terminal_ui::execution_browser;
use crate::channels::terminal_ui::syntect_highlight;
use crate::channels::terminal_ui::text_format::truncate_chars_display;
use crate::channels::terminal_ui::{App, Theme};
use crate::execution::RunJournal;

fn execution_run_list_line(
    item: &execution_browser::ExecutionRunListItem,
    inner_w: usize,
) -> String {
    let ex = item
        .exit_code
        .map(|c| format!("exit {c}"))
        .unwrap_or_else(|| "exit ?".to_string());
    let desc = item
        .description
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("—");
    let ts_short: String = item.ts.chars().take(26).collect();
    let raw = format!("{ts_short}  {ex}  {desc}");
    truncate_chars_display(&raw, inner_w.max(12))
}

fn execution_code_lines(source: &str, inner_w: usize) -> Vec<Line<'static>> {
    syntect_highlight::highlight_source_wrapped(source, inner_w.max(8))
}

fn execution_output_lines(j: &RunJournal, inner_w: usize) -> Vec<Line<'static>> {
    let w = inner_w.max(8);
    let mut v: Vec<Line<'static>> = Vec::new();
    v.push(Line::from(vec![Span::styled(
        "— stdout —",
        Theme::tool_call(),
    )]));
    for ln in wrap_text(&j.stdout, w) {
        v.push(Line::from(Span::styled(ln, Theme::text())));
    }
    v.push(Line::from(vec![Span::styled("— stderr —", Theme::error())]));
    for ln in wrap_text(&j.stderr, w) {
        v.push(Line::from(Span::styled(ln, Theme::error())));
    }
    v
}

pub fn executions_list_paragraph(app: &App, area: Rect) -> (Paragraph<'static>, usize) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let visible = area.height.saturating_sub(2) as usize;
    let max_scroll = if app.executions_runs.is_empty() {
        0usize
    } else {
        app.executions_runs.len().saturating_sub(visible.max(1))
    };
    let st = app.executions_list_scroll_top.min(max_scroll);
    let mut slice: Vec<Line<'static>> = Vec::new();
    if let Some(err) = app.executions_runs_error.as_deref() {
        slice.push(Line::from(Span::styled(
            format!("Could not load execution_runs.jsonl: {err}"),
            Theme::error(),
        )));
    } else if app.executions_runs.is_empty() {
        slice.push(Line::from(Span::styled(
            "No execution runs recorded for this terminal thread yet.",
            Theme::dim(),
        )));
        slice.push(Line::from(Span::styled(
            "Runs append after each successful execution_run (workspace .system_generated/).",
            Theme::dim(),
        )));
    } else {
        for (row, item) in app
            .executions_runs
            .iter()
            .enumerate()
            .skip(st)
            .take(visible.max(1))
        {
            let sel = app.executions_selected_idx == Some(row);
            let mark = if sel { "› " } else { "  " };
            let body = execution_run_list_line(item, inner_w.saturating_sub(4));
            let style = if sel {
                Theme::tool_call()
            } else {
                Theme::text()
            };
            slice.push(Line::from(vec![
                Span::styled(mark, Theme::tool_done()),
                Span::styled(body, style),
            ]));
        }
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" runs (this thread) ", Theme::tool_done()))
        .border_style(Theme::dim());
    (Paragraph::new(Text::from(slice)).block(block), max_scroll)
}

pub fn executions_ensure_list_shows_selection(app: &mut App, list_inner_height: usize) {
    let v = list_inner_height.max(1);
    let Some(sel) = app.executions_selected_idx else {
        return;
    };
    if sel < app.executions_list_scroll_top {
        app.executions_list_scroll_top = sel;
    }
    if sel >= app.executions_list_scroll_top.saturating_add(v) {
        app.executions_list_scroll_top = sel.saturating_sub(v.saturating_sub(1));
    }
}

pub fn executions_code_paragraph(
    detail: &execution_browser::ExecutionRunDetail,
    area: Rect,
    scroll_top: usize,
) -> (Paragraph<'static>, usize) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let lines = execution_code_lines(&detail.source, inner_w.max(8));
    let visible = area.height.saturating_sub(2) as usize;
    let total = lines.len();
    let max_scroll = total.saturating_sub(visible.max(1));
    let st = scroll_top.min(max_scroll);
    let slice: Vec<Line<'static>> = lines.into_iter().skip(st).take(visible.max(1)).collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" source ", Theme::dim()))
        .border_style(Theme::dim());
    (Paragraph::new(Text::from(slice)).block(block), max_scroll)
}

pub fn executions_output_paragraph(
    journal: &RunJournal,
    area: Rect,
    scroll_top: usize,
) -> (Paragraph<'static>, usize) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let lines = execution_output_lines(journal, inner_w.max(8));
    let visible = area.height.saturating_sub(2) as usize;
    let total = lines.len();
    let max_scroll = total.saturating_sub(visible.max(1));
    let st = scroll_top.min(max_scroll);
    let slice: Vec<Line<'static>> = lines.into_iter().skip(st).take(visible.max(1)).collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" output ", Theme::dim()))
        .border_style(Theme::dim());
    (Paragraph::new(Text::from(slice)).block(block), max_scroll)
}
