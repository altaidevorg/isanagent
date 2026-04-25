//! Fullscreen Ratatui loop (alternate screen): transcript cells, status strip, composed input.
//! Inspired by Xerxes-style terminal agents — scrollable rail, role-colored labels, calm chrome.

use std::io::{self, stdout};
use std::path::PathBuf;
use std::time::Duration;

use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
        MouseEventKind,
    },
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
use crate::channels::terminal_ui::protocol::{
    ISANAGENT_AGENT_THOUGHT, ISANAGENT_EXECUTION_JOB, ISANAGENT_EXECUTION_JOB_STARTED,
    ISANAGENT_EXECUTION_STREAM, ISANAGENT_TERMINAL_ERROR, METADATA_EXECUTION_DESCRIPTION,
    METADATA_EXECUTION_JOB_ID, METADATA_EXECUTION_JOB_STATUS, METADATA_EXECUTION_JOB_TOOL_NAME,
    METADATA_EXECUTION_RUN_ID, METADATA_EXECUTION_SESSION_ID, METADATA_TOOL_CALL_PREVIEW,
    METADATA_TOOL_NAME, METADATA_TOOL_RESULT_PREVIEW,
};
use crate::channels::terminal_ui::{
    init_from_env, uses_ansi_color, App, Cell, JobStripStatus, TerminalUiFocus, Theme, ToastKind,
    ToolNoticePhase, ToolRailEntry,
};
use crate::clarification::{METADATA_CLARIFICATION, METADATA_CLARIFICATION_CHOICES};

/// Second component of `execution_stream_label`: prefer model-provided description, else short id.
pub(crate) fn execution_strip_subtitle(description: Option<&str>, id: &str) -> String {
    const MAX_CHARS: usize = 96;
    if let Some(d) = description.map(str::trim).filter(|s| !s.is_empty()) {
        let n = d.chars().count();
        if n <= MAX_CHARS {
            return d.to_string();
        }
        return format!(
            "{}…",
            d.chars()
                .take(MAX_CHARS.saturating_sub(1))
                .collect::<String>()
        );
    }
    let short: String = id.chars().filter(|c| *c != '-').take(8).collect();
    if short.is_empty() {
        "…".to_string()
    } else {
        format!("…{short}")
    }
}

const ISANAGENT_TOOL_NOTIFY: &str = "isanagent_tool_notify";
const ISANAGENT_TOOL_PHASE: &str = "isanagent_tool_phase";
/// Lines to scroll per mouse wheel notch over the transcript.
const MOUSE_SCROLL_LINES: u16 = 3;
const TOAST_COPY_OK_SECS: u64 = 3;
const TOAST_COPY_ERR_SECS: u64 = 5;

const TERMINAL_HELP: &str = r#"Commands (leading slash):
  /exit, /quit   Quit and restore the terminal
  /new           Start a new session (new chat id)
  /copy          Copy the last assistant reply to the clipboard
  /install-python Install uv (best effort) in the background; UI stays responsive
  /cancel, /stop Stop the in-flight reply for this chat (drops queued prompts)
  /background, /bg Promote the in-flight execution_run / colab_mcp_tool_call to a background job
  /tools         Open the tool activity pane (same as Tab)
  /help, /?      Show this help

Keys:
  Enter             Send the compose line
  Tab / Ctrl+T      Switch focus: transcript <-> tool activity
  Esc               From tool activity: return to transcript (no-op on transcript)
  PgUp / PgDn       Scroll the focused pane (transcript or tool activity)
  Mouse wheel       Scroll when the pointer is over the focused pane (horizontal wheel scrolls too)
  Ctrl+Shift+Y      Copy last assistant reply
  Ctrl+W / Ctrl+U   Delete word / clear line
  Ctrl+C            Exit (same idea as /exit)

Environment:
  NO_COLOR          If set to a non-empty value, ANSI foreground colors in the TUI are disabled.
"#;

/// Coalesce consecutive model-thought lines into one cell (streaming-style UX).
fn tool_notice_display_content(msg: &OutboundMessage, phase_str: &str) -> String {
    let tn = msg
        .metadata
        .get(METADATA_TOOL_NAME)
        .and_then(|v| v.as_str());
    match (tn, phase_str) {
        (Some(name), "call") => msg
            .metadata
            .get(METADATA_TOOL_CALL_PREVIEW)
            .and_then(|v| v.as_str())
            .map(|pv| {
                if pv.is_empty() {
                    name.to_string()
                } else {
                    format!("{name} {pv}")
                }
            })
            .unwrap_or_else(|| msg.content.clone()),
        (Some(name), "result" | "fail") => msg
            .metadata
            .get(METADATA_TOOL_RESULT_PREVIEW)
            .and_then(|v| v.as_str())
            .map(|pv| format!("{name} → {pv}"))
            .unwrap_or_else(|| msg.content.clone()),
        _ => msg.content.clone(),
    }
}

fn apply_terminal_tool_aux(app: &mut App, msg: &OutboundMessage) {
    let phase_str = msg
        .metadata
        .get(ISANAGENT_TOOL_PHASE)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let ph = match phase_str {
        "call" => ToolNoticePhase::Call,
        "result" => ToolNoticePhase::Result,
        "fail" => ToolNoticePhase::Failed,
        _ => ToolNoticePhase::Other,
    };
    let display = tool_notice_display_content(msg, phase_str);
    let tool_name = msg
        .metadata
        .get(METADATA_TOOL_NAME)
        .and_then(|v| v.as_str())
        .unwrap_or("tool")
        .to_string();
    app.push_tool_rail(ToolRailEntry {
        tool_name,
        phase: ph,
        summary: display.clone(),
    });
    match phase_str {
        "call" => {
            app.active_tool_line = Some(display);
        }
        "result" | "fail" => {
            app.active_tool_line = None;
        }
        _ => {}
    }
}

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
    let terminal_error = msg
        .metadata
        .get(ISANAGENT_TERMINAL_ERROR)
        .and_then(|v| v.as_bool())
        == Some(true);
    if terminal_error {
        return Cell::Error {
            message: msg.content.clone(),
        };
    }

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
            "fail" => ToolNoticePhase::Failed,
            _ => ToolNoticePhase::Other,
        };
        let content = tool_notice_display_content(msg, phase);
        Cell::ToolNotice { phase: ph, content }
    } else if clarification {
        let choices = msg
            .metadata
            .get(METADATA_CLARIFICATION_CHOICES)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Cell::Clarification {
            text: msg.content.clone(),
            choices,
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
                ToolNoticePhase::Failed => Theme::error().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Other => Theme::tool_call().add_modifier(Modifier::BOLD),
            };
            let label = match phase {
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

fn flatten_cells_to_lines(cells: &[Cell], inner_width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for cell in cells {
        lines.extend(cell_block_lines(cell, inner_width));
    }
    lines
}

fn layout_chunks(area: Rect, exec_panel_h: u16, active_tool_h: u16) -> [Rect; 6] {
    let exec_constraint = if exec_panel_h > 0 {
        Constraint::Length(exec_panel_h)
    } else {
        Constraint::Length(0)
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(0)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(4),
            exec_constraint,
            Constraint::Length(1),
            Constraint::Length(active_tool_h),
            Constraint::Length(3),
        ])
        .split(area);
    [
        chunks[0], chunks[1], chunks[2], chunks[3], chunks[4], chunks[5],
    ]
}

fn chunks_line_width(spans: &[Span<'static>]) -> usize {
    spans
        .iter()
        .map(|s| super::display_width(s.content.as_ref()))
        .sum()
}

/// Truncate `s` to at most `max` display columns; appends `…` when shortened (`…` uses one column).
fn truncate_chars_display(s: &str, max: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    if max == 0 {
        return String::new();
    }
    if super::display_width(s) <= max {
        return s.to_string();
    }
    if max == 1 {
        return "…".to_string();
    }
    let budget = max.saturating_sub(1);
    let mut out = String::new();
    let mut col = 0usize;
    for ch in s.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        if col + cw > budget {
            break;
        }
        out.push(ch);
        col += cw;
    }
    if out.is_empty() {
        "…".to_string()
    } else {
        out.push('…');
        out
    }
}

/// Build one line per job in the multi-job strip, plus an optional Jupyter-stream tail
/// (the latest line of `execution_stream_recent`) so single-stream UX is preserved when
/// there are no Colab jobs racing.
fn jobs_strip_lines(app: &App, max_width: usize, include_stream_tail: bool) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    if include_stream_tail {
        if let Some(last) = app
            .execution_stream_recent
            .lines()
            .filter(|s| !s.trim().is_empty())
            .last()
        {
            let label = app
                .execution_stream_label
                .as_ref()
                .map(|(_, sub)| sub.as_str())
                .unwrap_or("jupyter");
            out.push(job_strip_line(
                "·",
                label,
                Some("(stream)"),
                last,
                None,
                max_width,
                Theme::tool_call(),
            ));
        }
    }
    for entry in app.jobs_strip.iter() {
        let style = match entry.status {
            JobStripStatus::Running => Theme::tool_call(),
            JobStripStatus::Completed => Theme::dim(),
            JobStripStatus::Failed | JobStripStatus::Timeout => Theme::input_prompt(),
            JobStripStatus::Cancelled => Theme::dim(),
        };
        let age = entry.started_at.elapsed();
        out.push(job_strip_line(
            entry.status.icon(),
            &entry.tool_name,
            entry.description.as_deref(),
            entry.last_line.as_str(),
            Some(format_age(age)),
            max_width,
            style,
        ));
    }
    out
}

fn job_strip_line(
    icon: &str,
    tool_name: &str,
    description: Option<&str>,
    last_line: &str,
    age: Option<String>,
    max_width: usize,
    head_style: ratatui::style::Style,
) -> Line<'static> {
    let dim = Theme::dim();
    let mut head = format!("{icon} {tool_name}");
    if let Some(d) = description.map(str::trim).filter(|s| !s.is_empty()) {
        head.push_str(": ");
        head.push_str(d);
    }
    let mut groups: Vec<Vec<Span<'static>>> = Vec::new();
    groups.push(vec![Span::styled(head, head_style)]);
    if let Some(a) = age {
        groups.push(vec![Span::styled(format!("  ·  {a}"), dim)]);
    }
    let trimmed = last_line.trim_end();
    if !trimmed.is_empty() {
        groups.push(vec![Span::styled(format!("  ·  {trimmed}"), dim)]);
    }
    line_from_chunk_groups(groups, max_width)
}

fn format_age(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Merge span groups left-to-right until `max_width` would be exceeded; drops trailing groups.
/// If the first group is wider than `max_width`, truncates its text (first group must be one span).
fn line_from_chunk_groups(groups: Vec<Vec<Span<'static>>>, max_width: usize) -> Line<'static> {
    if max_width < 1 {
        return Line::from(Span::raw(""));
    }
    let Some(first) = groups.first() else {
        return Line::from(Span::raw(""));
    };
    let w0 = chunks_line_width(first);
    if w0 > max_width {
        if first.len() == 1 {
            let st = first[0].style;
            let t = first[0].content.to_string();
            let cut = truncate_chars_display(&t, max_width);
            return Line::from(Span::styled(cut, st));
        }
        let mut flat: Vec<Span<'static>> = Vec::new();
        let mut used = 0usize;
        for sp in first {
            let cw = super::display_width(sp.content.as_ref());
            if used + cw <= max_width {
                flat.push(sp.clone());
                used += cw;
            } else {
                break;
            }
        }
        if flat.is_empty() {
            return Line::from(Span::styled("…", Theme::dim()));
        }
        return Line::from(flat);
    }

    let mut flat: Vec<Span<'static>> = Vec::new();
    let mut used = 0usize;
    for g in groups {
        let gw = chunks_line_width(&g);
        if used + gw <= max_width {
            flat.extend(g);
            used += gw;
        } else {
            break;
        }
    }

    if flat.is_empty() {
        Line::from(Span::styled("…", Theme::dim()))
    } else {
        Line::from(flat)
    }
}

fn build_title_line(max_width: usize) -> Line<'static> {
    let dim = Theme::dim();
    let groups = vec![
        vec![Span::styled(" isanagent ", Theme::input_prompt())],
        vec![Span::styled(
            format!(" {} ", env!("CARGO_PKG_VERSION")),
            dim,
        )],
        vec![Span::styled(
            "· /exit · /new · /copy · /cancel · /background · /tools · /help · Tab · Esc · ↑↓ · wheel · PgUp/PgDn",
            dim,
        )],
    ];
    line_from_chunk_groups(groups, max_width)
}

fn build_status_line(
    max_width: usize,
    status_model: &str,
    thinking: bool,
    chat_id: &str,
    cell_count: usize,
    toast: Option<(&str, ToastKind)>,
) -> Line<'static> {
    let dim = Theme::dim();
    let (activity_label, activity_style) = if thinking {
        ("thinking", Theme::tool_call())
    } else {
        ("idle", Theme::dim())
    };
    let sid = &chat_id[..8.min(chat_id.len())];
    let mut first_row = vec![Span::styled(status_model.to_string(), Theme::text())];
    if !uses_ansi_color() {
        first_row.push(Span::styled(" [plain]", Theme::dim()));
    }
    let mut groups: Vec<Vec<Span<'static>>> = vec![
        first_row,
        vec![
            Span::styled(" · ", dim),
            Span::styled(activity_label, activity_style),
        ],
        vec![
            Span::styled(" · ", dim),
            Span::styled(format!("session {sid}…"), dim),
        ],
        vec![
            Span::styled(" · ", dim),
            Span::styled(format!("{cell_count} cells"), dim),
        ],
        vec![
            Span::styled(" · ", dim),
            Span::styled(
                "Enter send · ^Shift+Y copy last · wheel · ^W word · ^U clear · ^C exit",
                Theme::status_bar(),
            ),
        ],
    ];
    if let Some((msg, kind)) = toast {
        let style = match kind {
            ToastKind::Ok => Theme::tool_done(),
            ToastKind::Err => Theme::error(),
        };
        let t = truncate_chars_display(msg, max_width.clamp(12, 120));
        groups.insert(0, vec![Span::styled(t, style), Span::styled(" · ", dim)]);
    }
    line_from_chunk_groups(groups, max_width)
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

fn tool_history_paragraph(
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
                ToolNoticePhase::Call => Theme::tool_call().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Result => Theme::tool_done().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Failed => Theme::error().add_modifier(Modifier::BOLD),
                ToolNoticePhase::Other => Theme::tool_call().add_modifier(Modifier::BOLD),
            };
            let label = match e.phase {
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

fn outbound_clears_thinking(msg: &OutboundMessage) -> bool {
    let is_thought = msg
        .metadata
        .get(ISANAGENT_AGENT_THOUGHT)
        .and_then(|v| v.as_bool())
        == Some(true);
    let is_tool = msg
        .metadata
        .get(ISANAGENT_TOOL_NOTIFY)
        .and_then(|v| v.as_bool())
        == Some(true);
    let is_exec = msg
        .metadata
        .get(ISANAGENT_EXECUTION_STREAM)
        .and_then(|v| v.as_bool())
        == Some(true);
    let is_exec_job = msg
        .metadata
        .get(ISANAGENT_EXECUTION_JOB)
        .and_then(|v| v.as_bool())
        == Some(true);
    let is_err = msg
        .metadata
        .get(ISANAGENT_TERMINAL_ERROR)
        .and_then(|v| v.as_bool())
        == Some(true);
    let is_clar = msg
        .metadata
        .get(METADATA_CLARIFICATION)
        .and_then(|v| v.as_bool())
        == Some(true);
    is_err || is_clar || (!is_thought && !is_tool && !is_exec && !is_exec_job)
}

fn handle_execution_job_started_notice(app: &mut App, msg: &OutboundMessage) {
    let jid = msg
        .metadata
        .get(METADATA_EXECUTION_JOB_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if jid.is_empty() {
        return;
    }
    let sid = msg
        .metadata
        .get(METADATA_EXECUTION_SESSION_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let tool = msg
        .metadata
        .get(METADATA_EXECUTION_JOB_TOOL_NAME)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let desc = msg
        .metadata
        .get(METADATA_EXECUTION_DESCRIPTION)
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty());
    app.job_strip_started(jid, sid, tool, desc);
}

fn handle_execution_job_finished_notice(app: &mut App, msg: &OutboundMessage) {
    let jid = msg
        .metadata
        .get(METADATA_EXECUTION_JOB_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if jid.is_empty() {
        return;
    }
    let status = msg
        .metadata
        .get(METADATA_EXECUTION_JOB_STATUS)
        .and_then(|v| v.as_str())
        .unwrap_or("completed");
    let summary = msg.content.trim();
    app.job_strip_finished(jid, status, summary);
}

fn append_execution_job_panel(app: &mut App, msg: &OutboundMessage) {
    let sid = msg
        .metadata
        .get(METADATA_EXECUTION_SESSION_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let jid = msg
        .metadata
        .get(METADATA_EXECUTION_JOB_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let desc = msg
        .metadata
        .get(METADATA_EXECUTION_DESCRIPTION)
        .and_then(|v| v.as_str());
    let label = (sid, execution_strip_subtitle(desc, &jid));
    if app.execution_stream_label != Some(label.clone()) {
        app.execution_stream_recent.clear();
        app.execution_stream_label = Some(label);
    }
    app.execution_stream_recent.push_str(msg.content.trim_end());
    app.execution_stream_recent.push('\n');
    const MAX: usize = 24_000;
    if app.execution_stream_recent.len() > MAX {
        let drop = app.execution_stream_recent.len() - MAX;
        let mut cut = drop;
        while cut < app.execution_stream_recent.len()
            && !app.execution_stream_recent.is_char_boundary(cut)
        {
            cut += 1;
        }
        app.execution_stream_recent.drain(..cut);
    }
}

fn append_execution_stream_panel(app: &mut App, msg: &OutboundMessage) {
    let sid = msg
        .metadata
        .get(METADATA_EXECUTION_SESSION_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let rid = msg
        .metadata
        .get(METADATA_EXECUTION_RUN_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let desc = msg
        .metadata
        .get(METADATA_EXECUTION_DESCRIPTION)
        .and_then(|v| v.as_str());
    let label = (sid, execution_strip_subtitle(desc, &rid));
    if app.execution_stream_label != Some(label.clone()) {
        app.execution_stream_recent.clear();
        app.execution_stream_label = Some(label);
    }
    app.execution_stream_recent.push_str(msg.content.trim_end());
    app.execution_stream_recent.push('\n');
    const MAX: usize = 24_000;
    if app.execution_stream_recent.len() > MAX {
        let drop = app.execution_stream_recent.len() - MAX;
        let mut cut = drop;
        while cut < app.execution_stream_recent.len()
            && !app.execution_stream_recent.is_char_boundary(cut)
        {
            cut += 1;
        }
        app.execution_stream_recent.drain(..cut);
    }
}

fn rect_contains(r: Rect, col: u16, row: u16) -> bool {
    let x1 = r.x.saturating_add(r.width);
    let y1 = r.y.saturating_add(r.height);
    col >= r.x && col < x1 && row >= r.y && row < y1
}

fn last_assistant_markdown(cells: &[Cell]) -> Option<&str> {
    cells.iter().rev().find_map(|c| {
        if let Cell::Assistant { markdown } = c {
            Some(markdown.as_str())
        } else {
            None
        }
    })
}

fn copy_last_assistant_to_clipboard(cells: &[Cell]) -> Result<usize, String> {
    let text = last_assistant_markdown(cells)
        .ok_or_else(|| "No assistant reply in this transcript yet.".to_string())?;
    let mut clip = arboard::Clipboard::new().map_err(|e| format!("clipboard init: {e}"))?;
    clip.set_text(text)
        .map_err(|e| format!("clipboard set: {e}"))?;
    Ok(text.len())
}

/// Arguments for [`run_ratatui_main`].
pub(crate) struct RatatuiMainConfig {
    pub bus_tx: Sender<BusMessage>,
    pub outbound_rx: std::sync::mpsc::Receiver<OutboundMessage>,
    pub shutdown_tx: tokio::sync::mpsc::UnboundedSender<()>,
    pub sandbox_dir: PathBuf,
    pub chat_id: String,
    pub channel_name: String,
    pub session_banner: String,
    pub status_model: String,
}

/// Run until user quits. Restores terminal on exit.
pub(crate) fn run_ratatui_main(config: RatatuiMainConfig) -> io::Result<()> {
    let RatatuiMainConfig {
        bus_tx,
        outbound_rx,
        shutdown_tx,
        sandbox_dir,
        mut chat_id,
        channel_name,
        session_banner,
        status_model,
    } = config;

    init_from_env();

    let mut stdout = stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen, crossterm::cursor::Hide)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    if let Err(e) = execute!(terminal.backend_mut(), EnableMouseCapture) {
        log::warn!("Terminal UI: mouse capture unavailable ({e}); wheel scroll disabled.");
    }

    let mut app = App::new();
    app.cells.push(Cell::System {
        message: session_banner,
    });

    // Result of a background `install_uv_best_effort` started from `/install-python`.
    let mut uv_install_rx: Option<std::sync::mpsc::Receiver<Result<String, String>>> = None;

    let tick = Duration::from_millis(80);
    let max_transcript_scroll_holder = std::cell::Cell::new(0u16);
    let max_tool_history_scroll_holder = std::cell::Cell::new(0u16);

    loop {
        app.clear_expired_toast();

        if let Some(ref rx) = uv_install_rx {
            match rx.try_recv() {
                Ok(Ok(msg)) => {
                    uv_install_rx = None;
                    app.cells.push(Cell::System { message: msg });
                }
                Ok(Err(err)) => {
                    uv_install_rx = None;
                    app.cells.push(Cell::Error { message: err });
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    uv_install_rx = None;
                    app.cells.push(Cell::System {
                        message: "uv install finished (no result message).".into(),
                    });
                }
            }
        }

        while let Ok(msg) = outbound_rx.try_recv() {
            if msg
                .metadata
                .get(ISANAGENT_EXECUTION_JOB_STARTED)
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                handle_execution_job_started_notice(&mut app, &msg);
                continue;
            }
            if msg
                .metadata
                .get(ISANAGENT_EXECUTION_STREAM)
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                append_execution_stream_panel(&mut app, &msg);
                if let Some(jid) = msg
                    .metadata
                    .get(METADATA_EXECUTION_JOB_ID)
                    .and_then(|v| v.as_str())
                {
                    if let Some(line) = msg.content.lines().last() {
                        app.job_strip_set_last_line(jid, line);
                    }
                }
                continue;
            }
            if msg
                .metadata
                .get(ISANAGENT_EXECUTION_JOB)
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                handle_execution_job_finished_notice(&mut app, &msg);
                append_execution_job_panel(&mut app, &msg);
                continue;
            }
            if outbound_clears_thinking(&msg) {
                app.thinking = false;
                app.active_tool_line = None;
            }
            let is_tool_notify = msg
                .metadata
                .get(ISANAGENT_TOOL_NOTIFY)
                .and_then(|v| v.as_bool())
                == Some(true);
            if is_tool_notify {
                apply_terminal_tool_aux(&mut app, &msg);
                if app.ui_focus == TerminalUiFocus::ToolHistory && app.tool_history_following_tail()
                {
                    app.tool_history_scroll = 0;
                }
            }
            let cell = outbound_to_cell(&msg);
            append_cell_merging_thought(&mut app.cells, cell);
            if app.following_tail() {
                app.scroll_offset = 0;
            }
        }

        if app.should_quit {
            break;
        }

        // Drop terminal strip rows that are older than the linger window.
        app.evict_expired_jobs(Duration::from_secs(10));

        terminal.draw(|f| {
            let area = f.area();
            let stream_active = !app.execution_stream_recent.is_empty();
            let strip_active = !app.jobs_strip.is_empty();
            let exec_h = if !stream_active && !strip_active {
                0u16
            } else {
                let base = (area.height.saturating_mul(18) / 100).clamp(6, 18);
                if strip_active && !stream_active {
                    base.min(10)
                } else {
                    base
                }
            };
            let active_strip_h: u16 = 1;
            let ch = layout_chunks(area, exec_h, active_strip_h);

            let title_w = ch[0].width as usize;
            let title = Paragraph::new(build_title_line(title_w.max(1)));
            f.render_widget(title, ch[0]);

            match app.ui_focus {
                TerminalUiFocus::Transcript => {
                    let (w, max_s) = transcript_paragraph(&app.cells, ch[1], app.scroll_offset);
                    max_transcript_scroll_holder.set(max_s);
                    f.render_widget(w, ch[1]);
                    app.last_transcript_rect = Some(ch[1]);
                    app.last_tool_history_rect = None;
                }
                TerminalUiFocus::ToolHistory => {
                    let (w, max_s) =
                        tool_history_paragraph(&app.tool_rail, ch[1], app.tool_history_scroll);
                    max_tool_history_scroll_holder.set(max_s);
                    f.render_widget(w, ch[1]);
                    app.last_tool_history_rect = Some(ch[1]);
                    app.last_transcript_rect = None;
                }
            }

            if exec_h > 0 {
                let exec_block = Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(" execution ", Theme::tool_call()))
                    .border_style(Theme::dim());
                if strip_active {
                    let inner = exec_block.inner(ch[2]);
                    f.render_widget(exec_block, ch[2]);
                    let lines = jobs_strip_lines(&app, inner.width as usize, stream_active);
                    let para = Paragraph::new(Text::from(lines));
                    f.render_widget(para, inner);
                } else {
                    let exec_para = Paragraph::new(Text::raw(app.execution_stream_recent.as_str()))
                        .block(exec_block);
                    f.render_widget(exec_para, ch[2]);
                }
            }

            let status_w_px = ch[3].width as usize;
            let status_line = build_status_line(
                status_w_px.max(1),
                status_model.as_str(),
                app.thinking,
                &chat_id,
                app.cells.len(),
                app.active_toast(),
            );
            let status_w = Paragraph::new(status_line);
            f.render_widget(status_w, ch[3]);

            let active_w = ch[4].width as usize;
            let idle = "Idle (no running tool)";
            let active_text = app.active_tool_line.as_deref().unwrap_or(idle);
            let t = truncate_chars_display(active_text, active_w.max(8).saturating_sub(6));
            let active_row = Line::from(vec![
                Span::styled(" tool ", Theme::tool_call()),
                Span::styled(t, Theme::dim()),
            ]);
            f.render_widget(Paragraph::new(active_row), ch[4]);

            let input_block = Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(" compose ", Theme::dim()))
                .border_style(Theme::dim());
            let input_para = Paragraph::new(Line::from(vec![
                Span::styled("> ", Theme::input_prompt()),
                Span::styled(app.input.as_str(), Theme::text()),
            ]))
            .block(input_block);
            f.render_widget(input_para, ch[5]);

            let inner_area = ch[5].inner(Margin::new(1, 1));
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

        app.max_scroll = max_transcript_scroll_holder.get();
        app.tool_history_max_scroll = max_tool_history_scroll_holder.get();
        if app.scroll_offset > app.max_scroll {
            app.scroll_offset = app.max_scroll;
        }
        if app.tool_history_scroll > app.tool_history_max_scroll {
            app.tool_history_scroll = app.tool_history_max_scroll;
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
                            app.thinking = false;
                            app.cells.push(Cell::System {
                                message: format!("New session: {}", chat_id),
                            });
                            continue;
                        }
                        if text.eq_ignore_ascii_case("/copy") {
                            match copy_last_assistant_to_clipboard(&app.cells) {
                                Ok(n) => {
                                    app.set_toast(
                                        ToastKind::Ok,
                                        format!("Copied last reply ({n} chars)"),
                                        Duration::from_secs(TOAST_COPY_OK_SECS),
                                    );
                                }
                                Err(e) => {
                                    app.set_toast(
                                        ToastKind::Err,
                                        format!("Copy failed: {e}"),
                                        Duration::from_secs(TOAST_COPY_ERR_SECS),
                                    );
                                }
                            }
                            continue;
                        }
                        if text.eq_ignore_ascii_case("/install-python") {
                            if uv_install_rx.is_some() {
                                app.set_toast(
                                    ToastKind::Err,
                                    "uv install already running".into(),
                                    Duration::from_secs(4),
                                );
                                continue;
                            }
                            let (tx, rx) = std::sync::mpsc::channel();
                            uv_install_rx = Some(rx);
                            std::thread::spawn(move || {
                                let out = crate::execution::install_uv_best_effort();
                                let _ = tx.send(out);
                            });
                            app.set_toast(
                                ToastKind::Ok,
                                "Installing uv in the background…".into(),
                                Duration::from_secs(5),
                            );
                            continue;
                        }
                        if text.eq_ignore_ascii_case("/help") || text.eq_ignore_ascii_case("/?") {
                            app.cells.push(Cell::System {
                                message: TERMINAL_HELP.trim().to_string(),
                            });
                            continue;
                        }
                        if text.eq_ignore_ascii_case("/tools") {
                            app.focus_tool_history();
                            continue;
                        }
                        if text.eq_ignore_ascii_case("/cancel")
                            || text.eq_ignore_ascii_case("/stop")
                        {
                            app.thinking = false;
                            if bus_tx
                                .blocking_send(BusMessage::Cancel(chat_id.clone()))
                                .is_err()
                            {
                                app.cells.push(Cell::System {
                                    message: "Bus closed; exiting.".into(),
                                });
                                app.request_quit();
                            } else {
                                app.cells.push(Cell::System {
                                    message: "Cancel sent for this chat (queued prompts cleared)."
                                        .into(),
                                });
                            }
                            continue;
                        }
                        if text.eq_ignore_ascii_case("/background")
                            || text.eq_ignore_ascii_case("/bg")
                        {
                            if bus_tx
                                .blocking_send(BusMessage::PromoteSyncToBackground(
                                    chat_id.clone(),
                                ))
                                .is_err()
                            {
                                app.cells.push(Cell::System {
                                    message: "Bus closed; exiting.".into(),
                                });
                                app.request_quit();
                            } else {
                                app.cells.push(Cell::System {
                                    message: "Promote-to-background requested. If a sync execution_run / colab_mcp_tool_call is in flight it will return a job_id you can poll with execution_job_status.".into(),
                                });
                            }
                            continue;
                        }
                        app.cells.push(Cell::System {
                            message:
                                "Unknown command. Try /help, /exit, /new, /copy, /install-python, /cancel, /background, /tools."
                                    .into(),
                        });
                        continue;
                    }
                    if text.eq_ignore_ascii_case("exit") || text.eq_ignore_ascii_case("quit") {
                        app.request_quit();
                        let _ = shutdown_tx.send(());
                        continue;
                    }

                    app.cells.push(Cell::User { text: raw.clone() });
                    app.thinking = true;
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
                        app.thinking = false;
                        app.cells.push(Cell::System {
                            message: "Bus closed; exiting.".into(),
                        });
                        app.request_quit();
                    }
                    app.scroll_to_bottom();
                }
                KeyCode::Backspace => app.backspace(),
                KeyCode::Delete => app.delete_forward(),
                KeyCode::Left => app.move_left(),
                KeyCode::Right => app.move_right(),
                KeyCode::Home => app.home(),
                KeyCode::End => app.end(),
                KeyCode::Up => app.history_up(),
                KeyCode::Down => app.history_down(),
                KeyCode::Esc => {
                    if app.ui_focus == TerminalUiFocus::ToolHistory {
                        app.focus_transcript();
                    }
                }
                KeyCode::Tab => {
                    app.toggle_ui_focus();
                    if app.following_tail() {
                        app.scroll_offset = 0;
                    }
                    if app.tool_history_following_tail() {
                        app.tool_history_scroll = 0;
                    }
                }
                KeyCode::Char('t') | KeyCode::Char('T')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    app.toggle_ui_focus();
                    if app.following_tail() {
                        app.scroll_offset = 0;
                    }
                    if app.tool_history_following_tail() {
                        app.tool_history_scroll = 0;
                    }
                }
                KeyCode::PageUp => {
                    if app.ui_focus == TerminalUiFocus::ToolHistory {
                        app.tool_history_scroll_up(8);
                    } else {
                        app.scroll_up(8);
                    }
                }
                KeyCode::PageDown => {
                    if app.ui_focus == TerminalUiFocus::ToolHistory {
                        app.tool_history_scroll_down(8);
                    } else {
                        app.scroll_down(8);
                    }
                }
                KeyCode::Char(c)
                    if matches!(c, 'y' | 'Y')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.modifiers.contains(KeyModifiers::SHIFT) =>
                {
                    match copy_last_assistant_to_clipboard(&app.cells) {
                        Ok(n) => {
                            app.set_toast(
                                ToastKind::Ok,
                                format!("Copied last reply ({n} chars)"),
                                Duration::from_secs(TOAST_COPY_OK_SECS),
                            );
                            if app.following_tail() {
                                app.scroll_offset = 0;
                            }
                        }
                        Err(e) => {
                            app.set_toast(
                                ToastKind::Err,
                                format!("Copy failed: {e}"),
                                Duration::from_secs(TOAST_COPY_ERR_SECS),
                            );
                        }
                    }
                }
                KeyCode::Char(c) => app.insert_char(c),
                _ => {}
            },
            Event::Mouse(me) => {
                let over_transcript = app
                    .last_transcript_rect
                    .map(|r| rect_contains(r, me.column, me.row))
                    .unwrap_or(false);
                let over_tool_history = app
                    .last_tool_history_rect
                    .map(|r| rect_contains(r, me.column, me.row))
                    .unwrap_or(false);
                if over_transcript {
                    match me.kind {
                        MouseEventKind::ScrollUp => app.scroll_up(MOUSE_SCROLL_LINES),
                        MouseEventKind::ScrollDown => app.scroll_down(MOUSE_SCROLL_LINES),
                        // Trackpads: horizontal wheel maps to vertical transcript scroll.
                        MouseEventKind::ScrollLeft => app.scroll_up(MOUSE_SCROLL_LINES),
                        MouseEventKind::ScrollRight => app.scroll_down(MOUSE_SCROLL_LINES),
                        _ => {}
                    }
                } else if over_tool_history {
                    match me.kind {
                        MouseEventKind::ScrollUp => app.tool_history_scroll_up(MOUSE_SCROLL_LINES),
                        MouseEventKind::ScrollDown => {
                            app.tool_history_scroll_down(MOUSE_SCROLL_LINES)
                        }
                        MouseEventKind::ScrollLeft => {
                            app.tool_history_scroll_up(MOUSE_SCROLL_LINES)
                        }
                        MouseEventKind::ScrollRight => {
                            app.tool_history_scroll_down(MOUSE_SCROLL_LINES)
                        }
                        _ => {}
                    }
                }
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    disable_raw_mode()?;
    let _ = execute!(terminal.backend_mut(), DisableMouseCapture);
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        crossterm::cursor::Show
    )?;
    Ok(())
}

#[cfg(test)]
mod width_fit_tests {
    use super::{build_status_line, build_title_line, truncate_chars_display};
    use crate::channels::terminal_ui::display_width;
    use ratatui::text::Line;

    fn flat(line: &Line) -> String {
        line.spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<Vec<_>>()
            .join("")
    }

    #[test]
    fn truncate_display_never_exceeds_budget() {
        let t = truncate_chars_display("hello world", 5);
        assert!(
            display_width(&t) <= 5,
            "got {t:?} width {}",
            display_width(&t)
        );
        assert!(t.contains('…'));
    }

    #[test]
    fn status_drops_low_priority_when_narrow() {
        let line = build_status_line(26, "gemini-2.5-flash", false, "uuid-here-ok", 3, None);
        let t = flat(&line);
        assert!(t.contains("gemini"));
        assert!(t.contains("idle"));
        assert!(
            !t.contains("Enter send"),
            "hints should drop first when tight: {t}"
        );
    }

    #[test]
    fn title_drops_version_and_hints_when_very_narrow() {
        let line = build_title_line(14);
        let t = flat(&line);
        assert!(t.contains("isanagent"), "{t}");
        assert!(
            !t.contains("PgUp"),
            "keyboard hint lives in last chunk: {t}"
        );
    }
}

#[cfg(test)]
mod execution_strip_tests {
    use super::execution_strip_subtitle;

    #[test]
    fn prefers_description_over_id() {
        assert_eq!(
            execution_strip_subtitle(Some("  MCQ generation  "), "abc-def-0123"),
            "MCQ generation"
        );
    }

    #[test]
    fn truncates_long_description() {
        let d: String = (0..120).map(|_| 'x').collect();
        let out = execution_strip_subtitle(Some(&d), "id");
        assert!(out.ends_with('…'));
        assert!(out.chars().count() <= 96);
    }

    #[test]
    fn fallback_short_id() {
        let out = execution_strip_subtitle(None, "683c0fdc-a2bd");
        assert!(out.starts_with('…'));
        assert!(out.contains("683c0fdc"));
    }
}
