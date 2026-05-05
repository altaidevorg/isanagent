use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use unicode_width::UnicodeWidthChar;

use super::cell_render::flatten_cells_to_lines;
use crate::channels::terminal_ui::app::TranscriptSelection;
use crate::channels::terminal_ui::Cell;
use crate::channels::terminal_ui::Theme;

/// Build the transcript widget, optionally highlighting a mouse selection.
///
/// Returns `(paragraph, max_scroll, visible_start_index)`.
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
    let mut slice: Vec<Line<'static>> = lines.into_iter().skip(start).take(take).collect();

    // Apply selection highlighting if active.
    if let Some(sel) = selection {
        if !sel.is_empty() {
            apply_selection_to_lines(&mut slice, start, sel);
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" transcript ", Theme::dim()))
        .border_style(Theme::dim());
    (Paragraph::new(Text::from(slice)).block(block), max_scroll, start)
}

/// Extract plain text from the selected region for clipboard copy.
pub fn extract_selection_text(
    cells: &[Cell],
    inner_width: usize,
    sel: &TranscriptSelection,
) -> String {
    let lines = flatten_cells_to_lines(cells, inner_width.max(8));
    let (sl, sc, el, ec) = sel.normalized();
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        if i < sl || i > el {
            continue;
        }
        let plain: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        let start_col = if i == sl { sc } else { 0 };
        let end_col = if i == el { ec } else { usize::MAX };
        // Map display columns to byte offsets.
        let mut col = 0usize;
        let mut byte_start = plain.len();
        let mut byte_end = plain.len();
        for (bi, ch) in plain.char_indices() {
            if col >= start_col && byte_start == plain.len() {
                byte_start = bi;
            }
            col += ch.width().unwrap_or(0);
            if col >= end_col && byte_end == plain.len() {
                byte_end = bi + ch.len_utf8();
                break;
            }
        }
        if i > sl {
            out.push('\n');
        }
        out.push_str(&plain[byte_start..byte_end]);
    }
    out
}

// ── Selection highlighting ──────────────────────────────────────────

fn apply_selection_to_lines(
    visible_lines: &mut [Line<'static>],
    visible_start: usize,
    sel: &TranscriptSelection,
) {
    let (sl, sc, el, ec) = sel.normalized();
    for (idx, line) in visible_lines.iter_mut().enumerate() {
        let abs = visible_start + idx;
        if abs < sl || abs > el {
            continue;
        }
        let line_sc = if abs == sl { sc } else { 0 };
        let line_ec = if abs == el { ec } else { usize::MAX };
        if line_sc == line_ec {
            continue;
        }
        *line = highlight_line_range(line, line_sc, line_ec);
    }
}

/// Clone `line` with `Theme::selection()` patched onto the display-column range `[sc..ec)`.
fn highlight_line_range(line: &Line<'static>, sc: usize, ec: usize) -> Line<'static> {
    let highlight = Theme::selection();
    let mut out_spans: Vec<Span<'static>> = Vec::new();
    let mut col: usize = 0;
    for span in &line.spans {
        let span_start = col;
        let span_w: usize = span
            .content
            .chars()
            .map(|c| c.width().unwrap_or(0))
            .sum();
        let span_end = span_start + span_w;
        col = span_end;

        if span_end <= sc || span_start >= ec {
            // Entirely outside selection.
            out_spans.push(span.clone());
            continue;
        }
        if span_start >= sc && span_end <= ec {
            // Entirely inside selection.
            out_spans.push(Span::styled(
                span.content.clone(),
                span.style.patch(highlight),
            ));
            continue;
        }
        // Partial overlap — split at column boundaries.
        let mut byte_off = 0usize;
        let mut dcol = span_start;
        let text = span.content.as_ref();
        let mut seg_start = 0usize;
        let mut in_sel = dcol >= sc && dcol < ec;

        for ch in text.chars() {
            let cw = ch.width().unwrap_or(0);
            let next_dcol = dcol + cw;
            // Is this char inside [sc..ec)?
            let char_inside = dcol >= sc && dcol < ec;
            if char_inside != in_sel {
                // Flush segment.
                let seg = &text[seg_start..byte_off];
                if !seg.is_empty() {
                    let st = if in_sel {
                        span.style.patch(highlight)
                    } else {
                        span.style
                    };
                    out_spans.push(Span::styled(seg.to_owned(), st));
                }
                seg_start = byte_off;
                in_sel = char_inside;
            }
            byte_off += ch.len_utf8();
            dcol = next_dcol;
        }
        // Flush remaining.
        let seg = &text[seg_start..];
        if !seg.is_empty() {
            let st = if in_sel {
                span.style.patch(highlight)
            } else {
                span.style
            };
            out_spans.push(Span::styled(seg.to_owned(), st));
        }
    }
    Line::from(out_spans)
}
