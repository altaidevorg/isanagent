use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::cell_render::flatten_cells_to_lines;
use crate::channels::terminal_ui::{Cell, Theme, TranscriptSelection};

/// Returns (paragraph, max_scroll, visible_start_index).
pub fn transcript_paragraph(
    cells: &[Cell],
    transcript_area: Rect,
    scroll_from_bottom: u16,
    selection: Option<&TranscriptSelection>,
) -> (Paragraph<'static>, u16, usize) {
    let inner_w = transcript_area.width.saturating_sub(2) as usize;
    let lines = flatten_cells_to_lines(cells, inner_w.max(8));
    let visible = transcript_area.height.saturating_sub(2) as usize;
    let total = lines.len();
    let max_scroll = total.saturating_sub(visible).min(u16::MAX as usize) as u16;
    let start = total
        .saturating_sub(visible)
        .saturating_sub(scroll_from_bottom as usize);
    let take = visible.max(1);
    let slice: Vec<Line<'static>> = lines.into_iter().skip(start).take(take).collect();

    let slice = if let Some(sel) = selection {
        let (sl, sc, el, ec) = sel.normalized();
        apply_selection_to_lines(slice, start, sl, sc, el, ec)
    } else {
        slice
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" transcript ", Theme::dim()))
        .border_style(Theme::dim());
    (
        Paragraph::new(Text::from(slice)).block(block),
        max_scroll,
        start,
    )
}

/// Extract plain text from the selected range of flattened transcript lines.
pub fn extract_selection_text(
    cells: &[Cell],
    inner_width: usize,
    sel: &TranscriptSelection,
) -> String {
    let lines = flatten_cells_to_lines(cells, inner_width.max(8));
    let (start_line, start_col, end_line, end_col) = sel.normalized();

    let mut result = String::new();
    for (idx, line) in lines.iter().enumerate() {
        if idx < start_line || idx > end_line {
            continue;
        }
        let plain: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        let line_dw = UnicodeWidthStr::width(plain.as_str());
        let col_start = if idx == start_line { start_col } else { 0 };
        let col_end = if idx == end_line { end_col } else { line_dw };

        let mut col = 0usize;
        let mut extracted = String::new();
        for ch in plain.chars() {
            let ch_w = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
            if col >= col_start && col < col_end {
                extracted.push(ch);
            }
            col += ch_w;
            if col > col_end {
                break;
            }
        }

        if idx > start_line {
            result.push('\n');
        }
        result.push_str(&extracted);
    }
    result
}

// ── selection highlighting helpers ────────────────────────────────────────────

fn apply_selection_to_lines(
    lines: Vec<Line<'static>>,
    start_idx: usize,
    sel_start_line: usize,
    sel_start_col: usize,
    sel_end_line: usize,
    sel_end_col: usize,
) -> Vec<Line<'static>> {
    lines
        .into_iter()
        .enumerate()
        .map(|(i, line)| {
            let abs_idx = start_idx + i;
            if abs_idx < sel_start_line || abs_idx > sel_end_line {
                return line;
            }
            let line_width: usize = line
                .spans
                .iter()
                .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
                .sum();
            let col_start = if abs_idx == sel_start_line {
                sel_start_col
            } else {
                0
            };
            let col_end = if abs_idx == sel_end_line {
                sel_end_col
            } else {
                line_width
            };
            if col_start >= col_end {
                return line;
            }
            highlight_line_range(line, col_start, col_end)
        })
        .collect()
}

fn highlight_line_range(line: Line<'static>, start_col: usize, end_col: usize) -> Line<'static> {
    let highlight = Theme::selection();
    let mut new_spans: Vec<Span<'static>> = Vec::new();
    let mut col: usize = 0;

    for span in line.spans {
        let span_text: &str = span.content.as_ref();
        let span_dw = UnicodeWidthStr::width(span_text);
        let span_start = col;
        let span_end = col + span_dw;

        if span_end <= start_col || span_start >= end_col {
            // Entirely outside selection.
            new_spans.push(span);
        } else {
            // Intersects selection — split into before / selected / after.
            let mut before = String::new();
            let mut selected = String::new();
            let mut after = String::new();
            let mut char_col = span_start;

            for ch in span_text.chars() {
                let ch_w = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
                if char_col + ch_w <= start_col {
                    before.push(ch);
                } else if char_col >= end_col {
                    after.push(ch);
                } else {
                    selected.push(ch);
                }
                char_col += ch_w;
            }

            if !before.is_empty() {
                new_spans.push(Span::styled(before, span.style));
            }
            if !selected.is_empty() {
                new_spans.push(Span::styled(selected, span.style.patch(highlight)));
            }
            if !after.is_empty() {
                new_spans.push(Span::styled(after, span.style));
            }
        }

        col = span_end;
    }

    Line::from(new_spans)
}
