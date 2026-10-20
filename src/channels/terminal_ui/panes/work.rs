//! The work drawer: jobs, notifications, sub-agents and workspace files in one list.

use std::time::Duration;

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::channels::terminal_ui::text_format::truncate_chars_display;
use crate::channels::terminal_ui::{AgentTaskStatus, App, JobStripStatus, Theme};

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

fn age_since_ms(ms: i64) -> String {
    let at = std::time::UNIX_EPOCH + Duration::from_millis(ms.max(0) as u64);
    format_age(
        std::time::SystemTime::now()
            .duration_since(at)
            .unwrap_or_default(),
    )
}

fn section_header(label: &str, count: usize, width: usize) -> Line<'static> {
    let text = if count == 0 {
        format!("{label} · none")
    } else {
        format!("{label} · {count}")
    };
    Line::from(Span::styled(
        truncate_chars_display(&text, width),
        Theme::tool_call().add_modifier(Modifier::BOLD),
    ))
}

/// Drawer rows, each exactly one terminal line (truncated to `width`), and the line index of
/// the selected item. Selectable rows follow [`App::work_items`] order.
pub fn work_drawer_lines(app: &App, width: usize) -> (Vec<Line<'static>>, Option<usize>) {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut selected_line = None;
    let mut row = 0usize;
    let spinner = app.get_spinner_frame().to_string();
    let label_w = width.saturating_sub(2);
    let mut push_item = |out: &mut Vec<Line<'static>>, label: String, style: Style| {
        let label = truncate_chars_display(&label, label_w);
        if app.work_selected_idx == Some(row) {
            selected_line = Some(out.len());
            out.push(Line::from(Span::styled(
                format!("› {label}"),
                style.add_modifier(Modifier::REVERSED),
            )));
        } else {
            out.push(Line::from(Span::styled(format!("  {label}"), style)));
        }
        row += 1;
    };

    if !app.notifications.is_empty() {
        out.push(section_header(
            "notifications",
            app.notifications.len(),
            width,
        ));
        for n in &app.notifications {
            let style = if n.kind == "clarification" {
                Theme::tool_pending()
            } else {
                Theme::dim()
            };
            let label = format!(
                "[{}] {} ({})",
                n.kind,
                n.title,
                age_since_ms(n.created_at_ms)
            );
            push_item(&mut out, label, style);
        }
    }

    let live: Vec<_> = app.work_live_jobs().collect();
    out.push(section_header(
        "jobs",
        live.len() + app.background_jobs.len(),
        width,
    ));
    for e in live {
        let (style, icon) = match e.status {
            JobStripStatus::Running => (Theme::tool_call(), spinner.as_str()),
            JobStripStatus::Waiting => (Theme::tool_pending(), "⏳"),
            JobStripStatus::Completed => (Theme::dim(), "✓"),
            JobStripStatus::Failed | JobStripStatus::Timeout => (Theme::error(), "✗"),
            JobStripStatus::Cancelled => (Theme::dim(), "⨯"),
        };
        let mut label = format!("{icon} {}", e.tool_name);
        if let Some(d) = e.description.as_deref().filter(|d| !d.trim().is_empty()) {
            label.push_str(": ");
            label.push_str(d.trim());
        }
        label.push_str(&format!(" · {}", format_age(e.started_at.elapsed())));
        if !e.last_line.is_empty() {
            label.push_str(" · ");
            label.push_str(&e.last_line);
        }
        push_item(&mut out, label, style);
    }
    for j in &app.background_jobs {
        let (style, icon) = match j.state.as_str() {
            "running" => (Theme::tool_call(), spinner.as_str()),
            "waiting" => (Theme::tool_pending(), "⏲"),
            "completed" => (Theme::dim(), "✓"),
            "failed" => (Theme::error(), "✗"),
            _ => (Theme::dim(), "·"),
        };
        let label = format!(
            "{icon} {} [{}] ({})",
            j.kind,
            j.state,
            age_since_ms(j.created_at_ms)
        );
        push_item(&mut out, label, style);
    }

    out.push(section_header("sub-agents", app.agent_tasks.len(), width));
    for t in &app.agent_tasks {
        let (style, icon) = match t.status {
            AgentTaskStatus::Running => (Theme::tool_call(), spinner.as_str()),
            AgentTaskStatus::Completed => (Theme::dim(), "✓"),
            AgentTaskStatus::Failed => (Theme::error(), "✗"),
            AgentTaskStatus::Cancelled => (Theme::dim(), "⨯"),
        };
        let name = match (&t.agent_name, &t.display_name) {
            (Some(a), Some(d)) => format!("{a}: {d}"),
            (Some(a), None) => a.clone(),
            (None, Some(d)) => d.clone(),
            (None, None) => format!("task-{}", t.task_id.chars().take(8).collect::<String>()),
        };
        let mut label = format!("{icon} {name} · {}", format_age(t.started_at.elapsed()));
        if !t.last_line.is_empty() {
            label.push_str(" · ");
            label.push_str(&t.last_line);
        }
        push_item(&mut out, label, style);
    }

    out.push(section_header("files", app.workspace_files_total, width));
    for f in &app.workspace_files {
        push_item(&mut out, f.clone(), Theme::text());
    }
    let hidden = app
        .workspace_files_total
        .saturating_sub(app.workspace_files.len());
    if hidden > 0 {
        out.push(Line::from(Span::styled(
            truncate_chars_display(&format!("  … {hidden} older"), width),
            Theme::dim(),
        )));
    }
    (out, selected_line)
}

/// Bordered work drawer for `area`. `scroll_top` is the previous first visible line; the
/// returned value is the scroll actually used (clamped, and moved to keep the selection
/// visible).
pub fn work_drawer_paragraph(
    app: &App,
    area: Rect,
    scroll_top: usize,
) -> (Paragraph<'static>, usize) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let visible = area.height.saturating_sub(2) as usize;
    let (lines, selected) = work_drawer_lines(app, inner_w);
    let max_scroll = lines.len().saturating_sub(visible.max(1));
    let mut top = scroll_top.min(max_scroll);
    if let Some(sel) = selected {
        if sel < top {
            top = sel;
        } else if visible > 0 && sel >= top + visible {
            top = sel + 1 - visible;
        }
    }
    let title = format!(" work · {} ", app.work_count());
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(title, Theme::tool_call()))
        .title_bottom(Line::from(Span::styled(
            " ↑↓ Enter · Esc chat ",
            Theme::dim(),
        )))
        .border_style(Theme::dim());
    let body: Vec<Line<'static>> = lines.into_iter().skip(top).take(visible.max(1)).collect();
    (Paragraph::new(Text::from(body)).block(block), top)
}
