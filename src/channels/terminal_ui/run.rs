//! Fullscreen Ratatui loop (alternate screen): transcript cells, status strip, composed input.
//! Inspired by Xerxes-style terminal agents — scrollable rail, role-colored labels, calm chrome.

use std::io::{self, stdout};
use std::path::PathBuf;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::layout::{Constraint, Layout, Margin, Rect};
use ratatui::prelude::*;
use ratatui::style::Modifier;
use ratatui::widgets::{Block, Borders, Paragraph};
use tokio::sync::mpsc::Sender;

use crate::bus::{BusMessage, InboundMessage, OutboundMessage};
use crate::channels::terminal_ui::attachments::parse_terminal_attachments;
use crate::channels::terminal_ui::markdown;
use crate::channels::terminal_ui::protocol::ISANAGENT_AGENT_THOUGHT;
use crate::channels::terminal_ui::{App, Cell, Theme, ToolNoticePhase};
use crate::clarification::METADATA_CLARIFICATION;

const ISANAGENT_TOOL_NOTIFY: &str = "isanagent_tool_notify";
const ISANAGENT_TOOL_PHASE: &str = "isanagent_tool_phase";

/// Coalesce consecutive model-thought lines into one cell (streaming-style UX).
fn append_cell_merging_thought(cells: &mut Vec<Cell>, cell: Cell) {
    match (&cell, cells.last_mut()) {
        (Cell::Thinking { text: new_t }, Some(Cell::Thinking { text: acc })) => {
            if !acc.is_empty() {
                acc.push('\n');
            }
            acc.push_str(new_t);
        }
        _ => cells.push(cell),
    }
}

fn outbound_to_cell(msg: &OutboundMessage) -> Cell {
    let thought = msg
        .metadata
        .get(ISANAGENT_AGENT_THOUGHT)
        .and_then(|v| v.as_bool())
        == Some(true);
    let tool_notify = msg
        .metadata
        .get(ISANAGENT_TOOL_NOTIFY)
        .and_then(|v| v.as_bool())
        == Some(true);
    let clarification = msg
        .metadata
        .get(METADATA_CLARIFICATION)
        .and_then(|v| v.as_bool())
        == Some(true);
    let phase = msg
        .metadata
        .get(ISANAGENT_TOOL_PHASE)
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if thought {
        Cell::Thinking {
            text: msg.content.clone(),
        }
    } else if tool_notify {
        let ph = match phase {
            "call" => ToolNoticePhase::Call,
            "result" => ToolNoticePhase::Result,
            _ => ToolNoticePhase::Other,
        };
        Cell::ToolNotice {
            phase: ph,
            content: msg.content.clone(),
        }
    } else if clarification {
        Cell::Clarification {
            text: msg.content.clone(),
        }
    } else {
        Cell::Assistant {
            markdown: msg.content.clone(),
        }
    }
}

/// Greedy wrap by display width (`unicode_width`); preserves explicit newlines.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
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

fn cell_block_lines(cell: &Cell, inner_width: usize) -> Vec<Line<'static>> {
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
        Cell::ToolNotice { phase, content } => {
            let label_style = match phase {
                ToolNoticePhase::Call => Theme::tool_call().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Result => Theme::tool_done().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Other => Theme::tool_call().add_modifier(Modifier::BOLD),
            };
            let label = match phase {
                ToolNoticePhase::Call => "tool",
                ToolNoticePhase::Result => "done",
                ToolNoticePhase::Other => "tool",
            };
            let mut v = vec![Line::from(vec![
                Span::styled(" ⚡ ", label_style),
                Span::styled(label, label_style),
            ])];
            for ln in wrap_text(content, w.saturating_sub(2)) {
                v.push(Line::from(Span::styled(ln, Theme::text())));
            }
            v.push(Line::from(""));
            v
        }
        Cell::Clarification { text } => {
            let mut v = vec![Line::from(vec![
                Span::styled(" ? ", Theme::clarification()),
                Span::styled("question", Theme::clarification()),
            ])];
            for ln in wrap_text(text, w.saturating_sub(2)) {
                v.push(Line::from(Span::styled(ln, Theme::clarification())));
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
    }
}

fn flatten_cells_to_lines(cells: &[Cell], inner_width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for cell in cells {
        lines.extend(cell_block_lines(cell, inner_width));
    }
    lines
}

fn layout_chunks(area: Rect) -> [Rect; 4] {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(0)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(4),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .split(area);
    [chunks[0], chunks[1], chunks[2], chunks[3]]
}

fn transcript_paragraph(
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

/// Run until user quits. Restores terminal on exit.
pub(crate) fn run_ratatui_main(
    bus_tx: Sender<BusMessage>,
    outbound_rx: std::sync::mpsc::Receiver<OutboundMessage>,
    shutdown_tx: tokio::sync::mpsc::UnboundedSender<()>,
    sandbox_dir: PathBuf,
    mut chat_id: String,
    channel_name: String,
    session_banner: String,
) -> io::Result<()> {
    let mut stdout = stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen, crossterm::cursor::Hide)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    app.cells.push(Cell::System {
        message: session_banner,
    });

    let tick = Duration::from_millis(80);
    let max_scroll_holder = std::cell::Cell::new(0u16);

    loop {
        while let Ok(msg) = outbound_rx.try_recv() {
            let cell = outbound_to_cell(&msg);
            append_cell_merging_thought(&mut app.cells, cell);
            if app.following_tail() {
                app.scroll_offset = 0;
            }
        }

        if app.should_quit {
            break;
        }

        terminal.draw(|f| {
            let area = f.area();
            let ch = layout_chunks(area);

            let title = Paragraph::new(Line::from(vec![
                Span::styled(" isanagent ", Theme::input_prompt()),
                Span::styled(
                    "  alt-screen · /exit · /new · ↑↓ history · PgUp/PgDn scroll",
                    Theme::dim(),
                ),
            ]));
            f.render_widget(title, ch[0]);

            let (transcript_widget, max_s) =
                transcript_paragraph(&app.cells, ch[1], app.scroll_offset);
            max_scroll_holder.set(max_s);
            f.render_widget(transcript_widget, ch[1]);

            let hint = "Enter send · Ctrl+W word · Ctrl+U clear line · Ctrl+C exit";
            let status = format!(
                "{}  ·  session {}…  ·  {} cells",
                hint,
                &chat_id[..8.min(chat_id.len())],
                app.cells.len()
            );
            let status_w = Paragraph::new(Span::styled(status, Theme::status_bar()));
            f.render_widget(status_w, ch[2]);

            let input_block = Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(" compose ", Theme::dim()))
                .border_style(Theme::dim());
            let input_para = Paragraph::new(Line::from(vec![
                Span::styled("> ", Theme::input_prompt()),
                Span::styled(app.input.as_str(), Theme::text()),
            ]))
            .block(input_block);
            f.render_widget(input_para, ch[3]);

            let inner_area = ch[3].inner(Margin::new(1, 1));
            let prefix_w = crate::channels::terminal_ui::display_width("> ")
                + crate::channels::terminal_ui::display_width(
                    &app.input[..app.cursor.min(app.input.len())],
                );
            let cx = inner_area
                .x
                .saturating_add((prefix_w.min(inner_area.width.saturating_sub(1) as usize)) as u16);
            let cy = inner_area.y;
            f.set_cursor_position((cx, cy));
        })?;

        app.max_scroll = max_scroll_holder.get();
        if app.scroll_offset > app.max_scroll {
            app.scroll_offset = app.max_scroll;
        }

        if !event::poll(tick)? {
            continue;
        }

        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.request_quit();
                    let _ = shutdown_tx.send(());
                }
                KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.delete_word();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.clear_line();
                }
                KeyCode::Enter => {
                    let raw = app.take_input();
                    let text = raw.trim();
                    if text.is_empty() {
                        continue;
                    }
                    if text.starts_with('/') {
                        if text.eq_ignore_ascii_case("/exit") || text.eq_ignore_ascii_case("/quit")
                        {
                            app.request_quit();
                            let _ = shutdown_tx.send(());
                            continue;
                        }
                        if text.eq_ignore_ascii_case("/new") {
                            chat_id = uuid::Uuid::new_v4().to_string();
                            app.cells.push(Cell::System {
                                message: format!("New session: {}", chat_id),
                            });
                            continue;
                        }
                        app.cells.push(Cell::System {
                            message: "Unknown command. Try /exit, /new.".into(),
                        });
                        continue;
                    }
                    if text.eq_ignore_ascii_case("exit") || text.eq_ignore_ascii_case("quit") {
                        app.request_quit();
                        let _ = shutdown_tx.send(());
                        continue;
                    }

                    app.cells.push(Cell::User { text: raw.clone() });
                    let (clean_text, attachments) = parse_terminal_attachments(text, &sandbox_dir);
                    let msg = InboundMessage {
                        channel: channel_name.clone(),
                        sender_id: "local_user".to_string(),
                        chat_id: chat_id.clone(),
                        thread_id: None,
                        content: clean_text,
                        attachments,
                        metadata: Default::default(),
                    };
                    if bus_tx.blocking_send(BusMessage::Inbound(msg)).is_err() {
                        app.cells.push(Cell::System {
                            message: "Bus closed; exiting.".into(),
                        });
                        app.request_quit();
                    }
                    app.scroll_to_bottom();
                }
                KeyCode::Char(c) => app.insert_char(c),
                KeyCode::Backspace => app.backspace(),
                KeyCode::Delete => app.delete_forward(),
                KeyCode::Left => app.move_left(),
                KeyCode::Right => app.move_right(),
                KeyCode::Home => app.home(),
                KeyCode::End => app.end(),
                KeyCode::Up => app.history_up(),
                KeyCode::Down => app.history_down(),
                KeyCode::PageUp => app.scroll_up(8),
                KeyCode::PageDown => app.scroll_down(8),
                _ => {}
            },
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        crossterm::cursor::Show
    )?;
    Ok(())
}
