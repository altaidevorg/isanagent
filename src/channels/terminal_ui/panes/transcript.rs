use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use super::cell_render::flatten_cells_to_lines;
use crate::channels::terminal_ui::Cell;
use crate::channels::terminal_ui::Theme;

pub fn transcript_paragraph(
    cells: &[Cell],
    transcript_area: Rect,
    scroll_from_bottom: u16,
) -> (Paragraph<'static>, u16) {
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
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" transcript ", Theme::dim()))
        .border_style(Theme::dim());
    (Paragraph::new(Text::from(slice)).block(block), max_scroll)
}
