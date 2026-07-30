//! Fullscreen Ratatui loop (alternate screen): transcript cells, status strip, composed input.
//! Inspired by Xerxes-style terminal agents — scrollable rail, role-colored labels, calm chrome.

use std::io::{self, stdout};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::layout::{Constraint, Direction, Layout, Margin, Rect};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use tokio::sync::mpsc::Sender;

use crate::bus::{BusMessage, InboundMessage, OutboundMessage};
use crate::channels::terminal_ui::attachments::parse_terminal_attachments;
use crate::channels::terminal_ui::history_cells;
use crate::channels::terminal_ui::panes::{
    conversations_ensure_list_shows_selection, conversations_list_paragraph,
    executions_code_paragraph, executions_ensure_list_shows_selection, executions_list_paragraph,
    executions_output_paragraph, extract_selection_text, tool_history_paragraph,
    transcript_paragraph,
};
use crate::channels::terminal_ui::protocol::{
    ISANAGENT_AGENT_THOUGHT, ISANAGENT_BACKGROUND_JOB_FINISHED, ISANAGENT_BACKGROUND_JOB_STARTED,
    ISANAGENT_EXECUTION_JOB, ISANAGENT_EXECUTION_JOB_STARTED, ISANAGENT_EXECUTION_STREAM,
    ISANAGENT_LLM_RETRY_AVAILABLE, ISANAGENT_SUBAGENT_TASK_FINISHED,
    ISANAGENT_SUBAGENT_TASK_STARTED, ISANAGENT_TERMINAL_ERROR, ISANAGENT_TOOL_PROGRESS,
    METADATA_BACKGROUND_JOB_DESCRIPTION, METADATA_BACKGROUND_JOB_STATUS,
    METADATA_BACKGROUND_JOB_TOOL_NAME, METADATA_EXECUTION_DESCRIPTION, METADATA_EXECUTION_JOB_ID,
    METADATA_EXECUTION_JOB_STATUS, METADATA_EXECUTION_JOB_TOOL_NAME, METADATA_EXECUTION_RUN_ID,
    METADATA_EXECUTION_SESSION_ID, METADATA_SUBAGENT_AGENT_NAME, METADATA_SUBAGENT_CHILD_CHAT_ID,
    METADATA_SUBAGENT_DISPLAY_NAME, METADATA_SUBAGENT_STATUS, METADATA_SUBAGENT_TASK_ID,
    METADATA_TOOL_CALL_ID, METADATA_TOOL_CALL_PREVIEW, METADATA_TOOL_NAME,
    METADATA_TOOL_RESULT_CHAR_COUNT, METADATA_TOOL_RESULT_PREVIEW,
};
use crate::channels::terminal_ui::text_format::truncate_chars_display;
use crate::channels::terminal_ui::{
    execution_browser, init, uses_ansi_color, AgentTaskStatus, App, Cell, JobStripStatus,
    ModelSelector, TerminalUiFocus, Theme, ToastKind, ToolNoticePhase, ToolRailEntry,
    TranscriptSelection,
};
use crate::clarification::{METADATA_CLARIFICATION, METADATA_CLARIFICATION_CHOICES};
use crate::memory::{chat_id_from_root_thread_id, MemoryMessage, SharedReply};
use crate::NodeHandle;

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
/// If set to a falsy value (0/false/no/off), disable application mouse capture at start.
/// Default ON; use **Ctrl+Shift+M** to toggle in-session.
const ISANAGENT_TUI_MOUSE: &str = "ISANAGENT_TUI_MOUSE";
/// Lines to scroll per mouse wheel notch over the transcript.
const MOUSE_SCROLL_LINES: u16 = 3;
const TOAST_COPY_OK_SECS: u64 = 3;
const TOAST_COPY_ERR_SECS: u64 = 5;
const TOAST_MOUSE_TOGGLE_OK_SECS: u64 = 4;
const TOAST_MOUSE_TOGGLE_ERR_SECS: u64 = 5;
/// Relative path inside workspace_dir where we persist the last chosen model config key.
const LAST_MODEL_FILE: &str = ".system_generated/last_model";

/// Best-effort write of the chosen config key so the next startup remembers it.
fn persist_last_model(workspace_dir: &Path, config_key: &str) {
    let path = workspace_dir.join(LAST_MODEL_FILE);
    let _ = std::fs::write(&path, config_key);
}

/// Shared helper: resolve a provider config by key, send SwitchModel, and update UI.
/// Used by both the interactive model selector (Enter key) and the `/model <name>` command.
fn try_switch_model(
    app: &mut App,
    bus_tx: &Sender<BusMessage>,
    providers: &std::collections::HashMap<String, crate::config::ProviderConfig>,
    workspace_dir: &Path,
    config_key: &str,
) {
    if let Some(cfg) = providers.get(config_key) {
        let resolved_url = cfg.resolved_base_url().unwrap_or_default();
        match cfg.resolve_api_key() {
            Ok(api_key) => {
                let msg = BusMessage::SwitchModel {
                    provider_name: cfg.provider_name.clone(),
                    model_name: cfg.model_name.clone(),
                    base_url: resolved_url,
                    api_key,
                };
                if bus_tx.blocking_send(msg).is_err() {
                    app.cells.push(Cell::System {
                        message: "Bus closed; exiting.".into(),
                    });
                    app.request_quit();
                } else {
                    app.status_model = cfg.model_name.clone();
                    persist_last_model(workspace_dir, config_key);
                    app.set_toast(
                        ToastKind::Ok,
                        format!("Model: {}", app.status_model),
                        Duration::from_secs(3),
                    );
                }
            }
            Err(e) => {
                let err_msg = format!(
                    "No API key for '{}': {}\n\
                     Set the env var or add api_key = \"...\" under [providers.{}] in config.toml.",
                    config_key, e, config_key
                );
                app.cells.push(Cell::Error { message: err_msg });
                app.set_toast(
                    ToastKind::Err,
                    format!("No API key for {}", config_key),
                    Duration::from_secs(5),
                );
            }
        }
    }
}

const TERMINAL_HELP: &str = r#"Commands (leading slash):
  /exit, /quit   Quit and restore the terminal
  /new           Start a new thread (new chat id)
  /copy          Copy the last assistant reply to the clipboard
  /install-python Install uv (best effort) in the background; UI stays responsive
  /cancel, /stop Stop the in-flight reply for this chat (drops queued prompts)
  /background, /bg Promote the in-flight execution_run to a background job
  /retry         Re-submit the last user message after an LLM-failed banner
  /tools         Open the tool activity pane
  /exec          Open the executions browser (workspace execution_runs.jsonl)
  /agents        Open the sub-agent task pane (running / finished named agents, plan steps)
  /chats         Open past sessions (saved terminal threads from workspace memory)
  /help, /?      Show this help
  /skills add <url_or_shorthand> [skill_name] Add remote skills

Keys:
  Enter             Send the compose line; in past-sessions pane: load selected and continue
  Tab / Ctrl+T      Next pane: transcript → past sessions → executions → tool activity → sub-agents
  Shift+Tab         Previous pane (reverse of Tab)
  Esc               From any pane: return to transcript
  PgUp / PgDn       Scroll the focused pane; on executions: output pane (Ctrl+Pg*: code pane)
  F5                Refresh list (executions or past-sessions pane)
  Mouse drag        Select text in transcript; auto-copies to clipboard on release
  Ctrl+Shift+M      Toggle application mouse: on = wheel scrolls panes; off = native selection/copy
  Ctrl+Shift+Y      Copy last assistant reply
  Ctrl+W / Ctrl+U   Delete word / clear line
  Ctrl+G            Open background jobs & notifications pane
  Ctrl+C            Cancel in-flight work (same as /cancel; does not exit)
  Ctrl+D            Exit if compose line is empty (like /exit); else delete forward (readline-style)

Environment:
  NO_COLOR            If set to a non-empty value, ANSI foreground colors in the TUI are disabled.
  ISANAGENT_TUI_MOUSE Mouse wheel scroll is ON by default. Set to 0/false/no/off to disable.
"#;

fn env_truthy(var: &str) -> bool {
    std::env::var_os(var).is_some_and(|v| {
        matches!(
            v.to_string_lossy().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn env_falsy(var: &str) -> bool {
    std::env::var_os(var).is_some_and(|v| {
        matches!(
            v.to_string_lossy().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

/// Slash commands with descriptions for the autocomplete popup.
const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/model", "Switch LLM model"),
    ("/new", "Start a new thread"),
    ("/exit", "Quit the terminal"),
    ("/copy", "Copy last reply to clipboard"),
    ("/retry", "Re-submit after LLM failure"),
    ("/cancel", "Cancel in-flight work"),
    ("/background", "Promote to background job"),
    ("/tools", "Open tool activity pane"),
    ("/exec", "Open executions browser"),
    ("/agents", "Open sub-agent tasks"),
    ("/chats", "Open past sessions"),
    ("/help", "Show full help"),
    ("/skills", "Add remote skills"),
    ("/install-python", "Install uv runtime"),
];

/// Same as `/cancel`: cancel in-flight work; quit only if the bus is gone.
fn try_cancel_inflight(app: &mut App, bus_tx: &Sender<BusMessage>, chat_id: &str) {
    app.thinking = false;
    if bus_tx
        .blocking_send(BusMessage::Cancel(chat_id.to_string()))
        .is_err()
    {
        app.cells.push(Cell::System {
            message: "Bus closed; exiting.".into(),
        });
        app.request_quit();
    } else {
        app.cells.push(Cell::System {
            message: "Cancel sent for this thread (queued prompts cleared).".into(),
        });
    }
}

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
        (Some(name), "result" | "fail") => {
            let core = msg
                .metadata
                .get(METADATA_TOOL_RESULT_PREVIEW)
                .and_then(|v| v.as_str())
                .map(|pv| format!("{name} → {pv}"))
                .unwrap_or_else(|| msg.content.clone());
            let char_count = msg
                .metadata
                .get(METADATA_TOOL_RESULT_CHAR_COUNT)
                .and_then(|v| v.as_u64());
            if let Some(n) = char_count {
                format!("{core} ({n} chars)")
            } else {
                core
            }
        }
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
        let tool_call_id = msg
            .metadata
            .get(METADATA_TOOL_CALL_ID)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let ph = match phase {
            // When we have a stable id we can collapse the two-cell display into one mutating
            // cell; render the in-flight call as Pending (yellow/dim) and let
            // `upsert_tool_notice` flip it to Result/Failed when the matching result arrives.
            "call" if tool_call_id.is_some() => ToolNoticePhase::Pending,
            "call" => ToolNoticePhase::Call,
            "result" => ToolNoticePhase::Result,
            "fail" => ToolNoticePhase::Failed,
            _ => ToolNoticePhase::Other,
        };
        let content = tool_notice_display_content(msg, phase);
        Cell::ToolNotice {
            phase: ph,
            content,
            tool_call_id,
        }
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

fn layout_chunks(area: Rect, exec_panel_h: u16, active_tool_h: u16, input_h: u16) -> [Rect; 6] {
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
            Constraint::Length(input_h),
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

/// Build lines for the agent-tasks pane (sub-agent lifecycle).
fn background_pane_paragraph(app: &App) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();

    // 1. Notifications (High Priority)
    if !app.notifications.is_empty() {
        out.push(Line::from(Span::styled(
            "--- NOTIFICATIONS ---",
            Theme::tool_call(),
        )));
        for entry in &app.notifications {
            let style = if entry.kind == "clarification" {
                Theme::tool_pending()
            } else {
                Theme::dim()
            };
            let created_at = std::time::UNIX_EPOCH
                + std::time::Duration::from_millis(entry.created_at_ms as u64);
            let age = std::time::SystemTime::now()
                .duration_since(created_at)
                .unwrap_or_default();
            let label = format!("[{}] {} ({})", entry.kind, entry.title, format_age(age));
            out.push(Line::from(Span::styled(label, style)));
        }
        out.push(Line::from(""));
    }

    // 2. Background Jobs
    if !app.background_jobs.is_empty() {
        out.push(Line::from(Span::styled(
            "--- BACKGROUND JOBS ---",
            Theme::tool_call(),
        )));
        let spinner_str = app.get_spinner_frame().to_string();
        for entry in &app.background_jobs {
            let (style, icon) = match entry.state.as_str() {
                "running" => (Theme::tool_call(), spinner_str.as_str()),
                "waiting" => (Theme::tool_pending(), "⏲"),
                "completed" => (Theme::dim(), "✓"),
                "failed" => (Theme::error(), "✗"),
                _ => (Theme::dim(), "·"),
            };
            let created_at = std::time::UNIX_EPOCH
                + std::time::Duration::from_millis(entry.created_at_ms as u64);
            let age = std::time::SystemTime::now()
                .duration_since(created_at)
                .unwrap_or_default();
            let label = format!(
                "{} {} [{}] ({})",
                icon,
                entry.kind,
                entry.state,
                format_age(age)
            );
            let line = Line::from(Span::styled(label, style));
            out.push(line);
        }
        out.push(Line::from(""));
    }

    // 3. Sub-Agent Tasks (Legacy AgentTasks)
    if !app.agent_tasks.is_empty() {
        out.push(Line::from(Span::styled(
            "--- SUB-AGENTS ---",
            Theme::tool_call(),
        )));
        let spinner_str = app.get_spinner_frame().to_string();
        for entry in app.agent_tasks.iter() {
            let (style, icon) = match entry.status {
                AgentTaskStatus::Running => (Theme::tool_call(), spinner_str.as_str()),
                AgentTaskStatus::Completed => (Theme::dim(), "✓"),
                AgentTaskStatus::Failed => (Theme::error(), "✗"),
                AgentTaskStatus::Cancelled => (Theme::dim(), "⨯"),
            };
            let label = match (&entry.agent_name, &entry.display_name) {
                (Some(a), Some(d)) => format!("{a}: {d}"),
                (Some(a), None) => a.clone(),
                (None, Some(d)) => d.clone(),
                (None, None) => {
                    let short = &entry.task_id[..8.min(entry.task_id.len())];
                    format!("task-{short}")
                }
            };
            let age = entry.started_at.elapsed();
            let mut line_text = format!("{icon} {label}");
            if !entry.last_line.is_empty() {
                line_text.push_str("  ·  ");
                line_text.push_str(&entry.last_line);
            }
            line_text.push_str("  ·  ");
            line_text.push_str(&format_age(age));
            out.push(Line::from(Span::styled(line_text, style)));
        }
    }

    if out.is_empty() {
        return vec![Line::from(Span::styled(
            "No background jobs or sub-agent tasks found.",
            Theme::dim(),
        ))];
    }
    out
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
            .rfind(|s| !s.trim().is_empty())
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
    let spinner_str = app.get_spinner_frame().to_string();
    for entry in app.jobs_strip.iter() {
        let (style, icon) = match entry.status {
            JobStripStatus::Running => (Theme::tool_call(), spinner_str.as_str()),
            JobStripStatus::Waiting => (Theme::input_prompt(), "⏳"),
            JobStripStatus::Completed => (Theme::dim(), "✓"),
            JobStripStatus::Failed | JobStripStatus::Timeout => (Theme::input_prompt(), "✗"),
            JobStripStatus::Cancelled => (Theme::dim(), "⨯"),
        };
        let age = entry.started_at.elapsed();
        out.push(job_strip_line(
            icon,
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
    let groups = vec![vec![Span::styled(
        " ALTAI isanagent ",
        Theme::input_prompt(),
    )]];
    line_from_chunk_groups(groups, max_width)
}

fn build_status_line(
    max_width: usize,
    thinking: bool,
    cell_count: usize,
    toast: Option<(&str, ToastKind)>,
    app: &App,
) -> Line<'static> {
    let dim = Theme::dim();
    let activity_label = if thinking {
        format!("🦾 {} thinking", app.get_spinner_frame())
    } else {
        "🦾 idle".to_string()
    };
    let activity_style = if thinking {
        Theme::tool_call()
    } else {
        Theme::dim()
    };
    let mut first_row = vec![Span::styled(activity_label, activity_style)];
    if !uses_ansi_color() {
        first_row.push(Span::styled(" [plain]", Theme::dim()));
    }
    let mut groups: Vec<Vec<Span<'static>>> = vec![
        first_row,
        vec![
            Span::styled(" · ", dim),
            Span::styled(format!("[ 📋 {} Todos ]", app.todos_count), dim),
        ],
        vec![
            Span::styled(" · ", dim),
            Span::styled(format!("[ 🕒 {} Crons ]", app.crons_count), dim),
        ],
        vec![
            Span::styled(" · ", dim),
            Span::styled(format!("[ 🛠 {} Jobs ]", app.jobs_strip.len()), dim),
        ],
        vec![
            Span::styled(" · ", dim),
            Span::styled(
                format!(
                    "[ 🤖 {} Agents ]",
                    app.agent_tasks.iter().filter(|e| !e.status.is_terminal()).count()
                ),
                dim,
            ),
        ],
        vec![
            Span::styled(" · ", dim),
            Span::styled(format!("{cell_count} cells"), dim),
        ],
        vec![
            Span::styled(" · ", dim),
            Span::styled(
                "Enter send · ^G background · ^Shift+Y copy last · ^Shift+M wheel · ^W word · ^U clear · ^C cancel · ^D exit",
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
    app.job_strip_started(jid, sid, tool, desc, &msg.chat_id);
}

fn handle_background_job_started_notice(app: &mut App, msg: &OutboundMessage) {
    let jid = msg
        .metadata
        .get(crate::bus::METADATA_BACKGROUND_JOB_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if jid.is_empty() {
        return;
    }
    let tool = msg
        .metadata
        .get(METADATA_BACKGROUND_JOB_TOOL_NAME)
        .and_then(|v| v.as_str())
        .unwrap_or("job");
    let desc = msg
        .metadata
        .get(METADATA_BACKGROUND_JOB_DESCRIPTION)
        .and_then(|v| v.as_str());
    app.job_strip_started(jid, "background", tool, desc, &msg.chat_id);
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
    app.job_strip_finished(jid, status, summary, &msg.chat_id);
}

fn handle_background_job_finished_notice(app: &mut App, msg: &OutboundMessage) {
    let jid = msg
        .metadata
        .get(crate::bus::METADATA_BACKGROUND_JOB_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if jid.is_empty() {
        return;
    }
    let status = msg
        .metadata
        .get(METADATA_BACKGROUND_JOB_STATUS)
        .and_then(|v| v.as_str())
        .unwrap_or("completed");
    let summary = msg.content.trim();
    app.job_strip_finished(jid, status, summary, &msg.chat_id);
}

fn handle_subagent_task_started_notice(app: &mut App, msg: &OutboundMessage) {
    let tid = msg
        .metadata
        .get(METADATA_SUBAGENT_TASK_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if tid.is_empty() {
        return;
    }
    let cid = msg
        .metadata
        .get(METADATA_SUBAGENT_CHILD_CHAT_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let agent = msg
        .metadata
        .get(METADATA_SUBAGENT_AGENT_NAME)
        .and_then(|v| v.as_str());
    let display = msg
        .metadata
        .get(METADATA_SUBAGENT_DISPLAY_NAME)
        .and_then(|v| v.as_str());
    app.agent_task_started(tid, cid, agent, display);
}

fn handle_subagent_task_finished_notice(app: &mut App, msg: &OutboundMessage) {
    let tid = msg
        .metadata
        .get(METADATA_SUBAGENT_TASK_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if tid.is_empty() {
        return;
    }
    let status = msg
        .metadata
        .get(METADATA_SUBAGENT_STATUS)
        .and_then(|v| v.as_str())
        .unwrap_or("completed");
    let summary = msg.content.trim();
    app.agent_task_finished(tid, status, summary);
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

/// Character-level wrapping for the compose input box.
///
/// Splits each logical line (delimited by `'\n'` in `input`) into visual rows
/// that are at most `width` display-columns wide.  The first logical line gets
/// the `"> "` prompt; continuations get `"  "`.  Each row is a complete
/// [`Line`] so the caller can render the result **without** `Wrap`.
///
/// This keeps cursor-position arithmetic (`total_cols / width`) perfectly
/// aligned with what is actually drawn on screen.
fn char_wrap_compose(
    input: &str,
    width: usize,
    prompt_style: Style,
    text_style: Style,
) -> Vec<Line<'static>> {
    use unicode_width::UnicodeWidthChar;

    let w = width.max(1);
    let mut out: Vec<Line<'static>> = Vec::new();

    for (li, logical) in input.split('\n').enumerate() {
        let prefix: &str = if li == 0 { "> " } else { "  " };
        let prefix_w: usize = 2; // both "> " and "  " are 2 display-columns
        let text_budget = w.saturating_sub(prefix_w);

        // First visual row: prefix + as much text as fits.
        let mut chars = logical.chars().peekable();
        let mut first_chunk = String::new();
        let mut col: usize = 0;
        while let Some(&ch) = chars.peek() {
            let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
            if col + cw > text_budget {
                break;
            }
            first_chunk.push(ch);
            col += cw;
            chars.next();
        }
        out.push(Line::from(vec![
            Span::styled(prefix.to_string(), prompt_style),
            Span::styled(first_chunk, text_style),
        ]));

        // Subsequent visual rows: up to `w` columns of text each (no prefix).
        while chars.peek().is_some() {
            let mut chunk = String::new();
            let mut col: usize = 0;
            while let Some(&ch) = chars.peek() {
                let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
                if col + cw > w {
                    break;
                }
                chunk.push(ch);
                col += cw;
                chars.next();
            }
            if chunk.is_empty() {
                // Single character wider than the whole line — force it through.
                if let Some(ch) = chars.next() {
                    chunk.push(ch);
                }
            }
            out.push(Line::from(Span::styled(chunk, text_style)));
        }
    }

    out
}

fn rect_contains(r: Rect, col: u16, row: u16) -> bool {
    let x1 = r.x.saturating_add(r.width);
    let y1 = r.y.saturating_add(r.height);
    col >= r.x && col < x1 && row >= r.y && row < y1
}

/// Map mouse screen coordinates to (flattened_line_index, display_col) in the transcript,
/// clamping to the inner content area (excluding borders).
fn mouse_to_transcript_coords(
    mouse_col: u16,
    mouse_row: u16,
    rect: Rect,
    visible_start: usize,
) -> (usize, usize) {
    let content_y = rect.y.saturating_add(1);
    let content_x = rect.x.saturating_add(1);
    let max_row = rect.y.saturating_add(rect.height).saturating_sub(2);
    let max_col = rect.x.saturating_add(rect.width).saturating_sub(2);

    let row = mouse_row.clamp(content_y, content_y.max(max_row));
    let col = mouse_col.clamp(content_x, content_x.max(max_col));

    let content_row = (row - content_y) as usize;
    let content_col = (col - content_x) as usize;
    (visible_start + content_row, content_col)
}

fn clamp_executions_selection(app: &mut App) {
    if app.executions_runs.is_empty() {
        app.executions_selected_idx = None;
        return;
    }
    let n = app.executions_runs.len();
    match app.executions_selected_idx {
        None => app.executions_selected_idx = Some(0),
        Some(i) if i >= n => app.executions_selected_idx = Some(n - 1),
        _ => {}
    }
}

fn load_selected_execution_detail(workspace_dir: &Path, app: &mut App) {
    app.executions_detail = None;
    app.executions_detail_error = None;
    let Some(idx) = app.executions_selected_idx else {
        return;
    };
    let Some(item) = app.executions_runs.get(idx) else {
        return;
    };
    match execution_browser::load_run_detail(workspace_dir, item) {
        Ok(d) => {
            app.executions_code_scroll_top = 0;
            app.executions_output_scroll_top = 0;
            app.executions_detail = Some(d);
        }
        Err(e) => app.executions_detail_error = Some(e),
    }
}

fn rescan_executions_manifest(workspace_dir: &Path, chat_id: &str, app: &mut App) {
    match execution_browser::load_runs_for_chat(workspace_dir, chat_id) {
        Ok(runs) => {
            app.executions_runs_error = None;
            app.executions_runs = runs;
            clamp_executions_selection(app);
            load_selected_execution_detail(workspace_dir, app);
        }
        Err(e) => app.executions_runs_error = Some(e),
    }
}

struct FocusCycleContext<'a> {
    workspace_dir: &'a Path,
    chat_id: &'a str,
    rt: &'a tokio::runtime::Runtime,
    memory_node: &'a NodeHandle<MemoryMessage>,
    last_exec_poll: &'a mut Instant,
    last_conversations_poll: &'a mut Instant,
}

impl<'a> FocusCycleContext<'a> {
    fn new(
        workspace_dir: &'a Path,
        chat_id: &'a str,
        rt: &'a tokio::runtime::Runtime,
        memory_node: &'a NodeHandle<MemoryMessage>,
        last_exec_poll: &'a mut Instant,
        last_conversations_poll: &'a mut Instant,
    ) -> Self {
        Self {
            workspace_dir,
            chat_id,
            rt,
            memory_node,
            last_exec_poll,
            last_conversations_poll,
        }
    }
}

fn apply_ui_focus_cycle(app: &mut App, forward: bool, ctx: &mut FocusCycleContext<'_>) {
    if forward {
        app.toggle_ui_focus();
    } else {
        app.toggle_ui_focus_back();
    }
    if app.ui_focus == TerminalUiFocus::Executions {
        rescan_executions_manifest(ctx.workspace_dir, ctx.chat_id, app);
        *ctx.last_exec_poll = Instant::now();
    }
    if app.ui_focus == TerminalUiFocus::Conversations {
        refresh_conversations_list(ctx.rt, ctx.memory_node, app);
        *ctx.last_conversations_poll = Instant::now();
    }
    if app.following_tail() {
        app.scroll_offset = 0;
    }
    if app.tool_history_following_tail() {
        app.tool_history_scroll = 0;
    }
}

fn last_copyable_text(cells: &[Cell], streaming_assistant: &str) -> Option<String> {
    // 1. If there's an active streaming assistant buffer, copy that.
    if !streaming_assistant.is_empty() {
        return Some(streaming_assistant.to_string());
    }
    // 2. Search cells in reverse for the last copyable cell.
    cells.iter().rev().find_map(|c| match c {
        Cell::Assistant { markdown } => Some(markdown.clone()),
        Cell::Thinking { text } => Some(text.clone()),
        Cell::Error { message } => Some(message.clone()),
        Cell::System { message } => Some(message.clone()),
        _ => None,
    })
}

fn copy_last_assistant_to_clipboard(
    cells: &[Cell],
    streaming_assistant: &str,
) -> Result<usize, String> {
    let text = last_copyable_text(cells, streaming_assistant)
        .ok_or_else(|| "No assistant reply in this transcript yet.".to_string())?;
    let mut clip = arboard::Clipboard::new().map_err(|e| format!("clipboard init: {e}"))?;
    clip.set_text(&text)
        .map_err(|e| format!("clipboard set: {e}"))?;
    Ok(text.len())
}

fn sync_terminal_session_chat(bus_tx: &Sender<BusMessage>, chat_id: &str) {
    let _ = bus_tx.blocking_send(BusMessage::SetTerminalSessionChat {
        chat_id: chat_id.to_string(),
    });
}

fn load_thread_transcript_cells(
    rt: &tokio::runtime::Runtime,
    memory_node: &NodeHandle<MemoryMessage>,
    thread_id: &str,
) -> Result<Vec<Cell>, String> {
    use tokio::sync::oneshot;
    let messages: Result<Vec<crate::utils::ChatMessage>, String> = rt.block_on(async {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::GetContext {
            thread_id: thread_id.to_string(),
            reply: SharedReply::new(tx),
        };
        memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;
        rx.await
            .map_err(|_| "memory actor channel closed".to_string())?
    });
    let messages = messages?;
    Ok(history_cells::chat_messages_to_terminal_cells(&messages))
}

fn refresh_conversations_list(
    rt: &tokio::runtime::Runtime,
    memory_node: &NodeHandle<MemoryMessage>,
    app: &mut App,
) {
    use tokio::sync::oneshot;
    let res: Result<Vec<crate::memory::RootThreadListItem>, String> = rt.block_on(async {
        let (tx, rx) = oneshot::channel();
        let msg = MemoryMessage::ListRootThreadsForChannelWithPreviews {
            channel: "terminal".to_string(),
            limit: 100,
            reply: SharedReply::new(tx),
        };
        memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;
        rx.await
            .map_err(|_| "memory actor channel closed".to_string())?
    });
    match res {
        Ok(rows) => {
            app.conversations_error = None;
            app.conversations_items = rows;
            if !app.conversations_items.is_empty() {
                if app.conversations_selected_idx.is_none() {
                    app.conversations_selected_idx = Some(0);
                } else {
                    let max = app.conversations_items.len().saturating_sub(1);
                    if let Some(i) = app.conversations_selected_idx.as_mut() {
                        *i = (*i).min(max);
                    }
                }
            } else {
                app.conversations_selected_idx = None;
            }
        }
        Err(e) => {
            app.conversations_error = Some(e);
            app.conversations_items.clear();
        }
    }
}

/// Arguments for [`run_ratatui_main`].
pub(crate) struct RatatuiMainConfig {
    pub bus_tx: Sender<BusMessage>,
    pub outbound_rx: std::sync::mpsc::Receiver<OutboundMessage>,
    pub shutdown_tx: tokio::sync::mpsc::UnboundedSender<()>,
    pub workspace_dir: PathBuf,
    pub sandbox_dir: PathBuf,
    pub chat_id: String,
    pub channel_name: String,
    pub opening_banner: String,
    pub status_model: String,
    /// Workspace memory (same as agent) for past-session list and transcript load.
    pub memory_node: NodeHandle<MemoryMessage>,
    /// Named alternative providers for `/model` switching.
    pub providers: std::collections::HashMap<String, crate::config::ProviderConfig>,
    /// Whether the host permits ANSI foreground colors for this session.
    pub color_enabled: bool,
    /// Whether `chat_id` names a persisted chat that should be loaded.
    pub resume_session: bool,
    /// File references composed into the first user message.
    pub initial_files: Vec<PathBuf>,
}

/// Run until user quits. Restores terminal on exit.
pub(crate) fn run_ratatui_main(config: RatatuiMainConfig) -> io::Result<()> {
    let RatatuiMainConfig {
        bus_tx,
        outbound_rx,
        shutdown_tx,
        workspace_dir,
        sandbox_dir,
        mut chat_id,
        channel_name,
        opening_banner,
        status_model,
        memory_node,
        providers,
        color_enabled,
        resume_session,
        initial_files,
    } = config;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(io::Error::other)?;

    init(color_enabled);

    let mut stdout = stdout();
    enable_raw_mode()?;
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableBracketedPaste,
        crossterm::cursor::Hide
    )?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut application_mouse_enabled = !env_falsy(ISANAGENT_TUI_MOUSE);
    if application_mouse_enabled {
        if let Err(e) = execute!(terminal.backend_mut(), EnableMouseCapture) {
            log::warn!(
                "Terminal UI: mouse capture unavailable ({e}); start with native selection."
            );
            application_mouse_enabled = false;
        }
    }

    let mut app = App::new();
    app.status_model = status_model;
    app.cells.push(Cell::System {
        message: opening_banner,
    });
    if !initial_files.is_empty() {
        let refs = initial_files
            .iter()
            .map(|path| format!("@{}", path.display()))
            .collect::<Vec<_>>()
            .join(" ");
        app.input = format!("{refs} ");
        app.cursor = app.input.len();
        app.cells.push(Cell::System {
            message: "Attached file references were added to the composer for the next message."
                .into(),
        });
    }
    if resume_session {
        match load_thread_transcript_cells(&rt, &memory_node, &chat_id) {
            Ok(mut cells) => {
                let sid = &chat_id[..8.min(chat_id.len())];
                cells.insert(
                    0,
                    Cell::System {
                        message: format!("Resumed session {sid}… — loaded from workspace memory."),
                    },
                );
                app.cells = cells;
            }
            Err(error) => app.cells.push(Cell::System {
                message: format!("Could not load session history: {error}"),
            }),
        }
    }

    sync_terminal_session_chat(&bus_tx, &chat_id);
    refresh_conversations_list(&rt, &memory_node, &mut app);

    // Result of a background `install_uv_best_effort` started from `/install-python`.
    let mut uv_install_rx: Option<std::sync::mpsc::Receiver<Result<String, String>>> = None;

    let tick = Duration::from_millis(80);
    let max_transcript_scroll_holder = std::cell::Cell::new(0u16);
    let max_tool_history_scroll_holder = std::cell::Cell::new(0u16);
    let max_exec_list_scroll_holder = std::cell::Cell::new(0usize);
    let max_exec_code_scroll_holder = std::cell::Cell::new(0usize);
    let max_exec_out_scroll_holder = std::cell::Cell::new(0usize);
    let max_conversations_list_scroll_holder = std::cell::Cell::new(0usize);
    let mut last_exec_poll = Instant::now() - Duration::from_secs(60);
    let mut last_conversations_poll = Instant::now() - Duration::from_secs(60);
    let mut last_todos_poll = Instant::now() - Duration::from_secs(60);

    let start_time = Instant::now();
    let (todos_tx, todos_rx) = std::sync::mpsc::channel();
    let (crons_tx, crons_rx) = std::sync::mpsc::channel::<usize>();
    let (jobs_tx, jobs_rx) = std::sync::mpsc::channel::<Vec<crate::memory::BackgroundJobRecord>>();
    let (notifications_tx, notifications_rx) =
        std::sync::mpsc::channel::<Vec<crate::memory::NotificationRecord>>();

    // Spawn a single long-lived background thread for periodic DB polling (todos + crons + jobs + notifications).
    // Receives tick signals via a channel; avoids spawning a new OS thread every poll interval.
    let (poll_trigger_tx, poll_trigger_rx) = std::sync::mpsc::channel::<String>();
    {
        let rt_handle = rt.handle().clone();
        let memory_node = memory_node.clone();
        let todos_tx = todos_tx.clone();
        let crons_tx = crons_tx.clone();
        let jobs_tx = jobs_tx.clone();
        let notifications_tx = notifications_tx.clone();
        let spawn_result = std::thread::Builder::new()
            .name("ui-db-poller".into())
            .spawn(move || {
                while let Ok(cid) = poll_trigger_rx.recv() {
                    rt_handle.block_on(async {
                        // 1. Todos
                        let (otx, orx) = tokio::sync::oneshot::channel();
                        let _ = memory_node
                            .send_packet(crate::memory::MemoryMessage::LoadHarnessTodos {
                                chat_id: cid.clone(),
                                reply: crate::memory::SharedReply::new(otx),
                            })
                            .await;

                        if let Ok(Ok(Some(todos))) = orx.await {
                            let active = todos
                                .into_iter()
                                .filter(|t| t.status != "completed")
                                .count();
                            let _ = todos_tx.send(active);
                        } else {
                            let _ = todos_tx.send(0);
                        }

                        // 2. Crons
                        let (ctx, crx) = tokio::sync::oneshot::channel();
                        let _ = memory_node
                            .send_packet(crate::memory::MemoryMessage::GetActiveCronsCount {
                                reply: crate::memory::SharedReply::new(ctx),
                            })
                            .await;

                        if let Ok(Ok(count)) = crx.await {
                            let _ = crons_tx.send(count);
                        } else {
                            let _ = crons_tx.send(0);
                        }

                        // 3. Background Jobs
                        let (jtx, jrx) = tokio::sync::oneshot::channel();
                        let _ = memory_node
                            .send_packet(crate::memory::MemoryMessage::ListBackgroundJobs {
                                chat_id: None,
                                // The terminal's inbox is intentionally
                                // workspace-global: jobs may originate in a
                                // different channel and must stay visible.
                                channel: None,
                                limit: 50,
                                reply: crate::memory::SharedReply::new(jtx),
                            })
                            .await;
                        if let Ok(Ok(jobs)) = jrx.await {
                            let _ = jobs_tx.send(jobs);
                        }

                        // 4. Notifications
                        let (ntx, nrx) = tokio::sync::oneshot::channel();
                        let _ = memory_node
                            .send_packet(crate::memory::MemoryMessage::ListNotifications {
                                chat_id: None,
                                channel: None,
                                limit: 50,
                                unseen_only: false,
                                reply: crate::memory::SharedReply::new(ntx),
                            })
                            .await;
                        if let Ok(Ok(notifs)) = nrx.await {
                            let _ = notifications_tx.send(notifs);
                        }
                    });
                }
            });
        if let Err(e) = spawn_result {
            log::error!("Failed to spawn ui-db-poller thread: {e}");
        }
    }

    loop {
        app.spinner_tick = (start_time.elapsed().as_millis() / 80) as usize;

        while let Ok(active_count) = todos_rx.try_recv() {
            app.todos_count = active_count;
        }
        while let Ok(active_count) = crons_rx.try_recv() {
            app.crons_count = active_count;
        }
        while let Ok(jobs) = jobs_rx.try_recv() {
            app.background_jobs = jobs;
        }
        while let Ok(notifs) = notifications_rx.try_recv() {
            app.notifications = notifs;
        }

        if last_todos_poll.elapsed() >= Duration::from_secs(2) {
            last_todos_poll = Instant::now();
            let _ = poll_trigger_tx.send(chat_id.clone());
        }

        app.clear_expired_toast();

        if app.ui_focus == TerminalUiFocus::Executions
            && last_exec_poll.elapsed() >= Duration::from_secs(2)
        {
            last_exec_poll = Instant::now();
            rescan_executions_manifest(&workspace_dir, &chat_id, &mut app);
        }
        if app.ui_focus == TerminalUiFocus::Conversations
            && last_conversations_poll.elapsed() >= Duration::from_secs(2)
        {
            last_conversations_poll = Instant::now();
            refresh_conversations_list(&rt, &memory_node, &mut app);
        }
        // Background jobs and notifications are now polled by the background thread.

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
                .get(ISANAGENT_SUBAGENT_TASK_STARTED)
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                handle_subagent_task_started_notice(&mut app, &msg);
                continue;
            }
            if msg
                .metadata
                .get(ISANAGENT_SUBAGENT_TASK_FINISHED)
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                handle_subagent_task_finished_notice(&mut app, &msg);
                continue;
            }
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
            if msg
                .metadata
                .get(ISANAGENT_BACKGROUND_JOB_STARTED)
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                handle_background_job_started_notice(&mut app, &msg);
                continue;
            }
            if msg
                .metadata
                .get(ISANAGENT_BACKGROUND_JOB_FINISHED)
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                handle_background_job_finished_notice(&mut app, &msg);
                continue;
            }
            if let Some(jid) = msg
                .metadata
                .get(crate::bus::METADATA_BACKGROUND_JOB_ID)
                .and_then(|v| v.as_str())
            {
                // General background job output: update strip, don't add to transcript.
                if let Some(line) = msg.content.lines().rfind(|l| !l.trim().is_empty()) {
                    app.job_strip_set_last_line(jid, line);
                }

                let is_notify = msg
                    .metadata
                    .get("isanagent_notification")
                    .and_then(|v| v.as_bool())
                    == Some(true);
                if !is_notify {
                    continue;
                }
            }
            if msg
                .metadata
                .get(ISANAGENT_TOOL_PROGRESS)
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                app.active_tool_line = Some(msg.content.clone());
                continue;
            }
            if outbound_clears_thinking(&msg) {
                app.thinking = false;
                app.active_tool_line = None;
            }
            if msg
                .metadata
                .get(ISANAGENT_LLM_RETRY_AVAILABLE)
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                app.llm_retry_available = true;
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
            match cell {
                Cell::ToolNotice {
                    phase,
                    content,
                    tool_call_id,
                } => {
                    app.upsert_tool_notice(tool_call_id, phase, content);
                }
                other => {
                    append_cell_merging_thought(&mut app.cells, other);
                }
            }
            if app.following_tail() {
                app.scroll_offset = 0;
            }
        }

        if app.should_quit {
            break;
        }

        // Drop terminal strip rows that are older than the linger window.
        app.evict_expired_jobs(Duration::from_secs(10));
        app.evict_expired_agent_tasks(Duration::from_secs(30));

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
            // Calculate visual line count accounting for wrapping at inner width.
            let compose_inner_w = area.width.saturating_sub(2).max(1) as usize; // borders
            let mut visual_lines: u16 = 0;
            for line in app.input.split('\n') {
                let prefix_len: usize = 2; // "> " or "  "
                let cols = prefix_len + super::display_width(line);
                if cols == 0 {
                    visual_lines += 1;
                } else {
                    visual_lines += ((cols as u16).div_ceil(compose_inner_w as u16)).max(1);
                }
            }
            if visual_lines == 0 {
                visual_lines = 1;
            }
            let input_h = (visual_lines + 2).clamp(3, 12);
            let ch = layout_chunks(area, exec_h, active_strip_h, input_h);

            let title_w = ch[0].width as usize;
            let title = Paragraph::new(build_title_line(title_w.max(1)));
            f.render_widget(title, ch[0]);

            match app.ui_focus {
                TerminalUiFocus::Transcript => {
                    let (w, max_s, vis_start) = transcript_paragraph(
                        &app.cells,
                        ch[1],
                        app.scroll_offset,
                        app.transcript_selection.as_ref(),
                    );
                    max_transcript_scroll_holder.set(max_s);
                    app.last_transcript_visible_start = vis_start;
                    f.render_widget(w, ch[1]);
                    app.last_transcript_rect = Some(ch[1]);
                    app.last_tool_history_rect = None;
                    app.last_executions_list_rect = None;
                    app.last_executions_code_rect = None;
                    app.last_executions_output_rect = None;
                    app.last_conversations_list_rect = None;
                    app.last_agent_tasks_rect = None;
                }
                TerminalUiFocus::Conversations => {
                    let (w, max_s) = conversations_list_paragraph(&app, ch[1]);
                    max_conversations_list_scroll_holder.set(max_s);
                    f.render_widget(w, ch[1]);
                    app.last_conversations_list_rect = Some(ch[1]);
                    app.last_transcript_rect = None;
                    app.last_tool_history_rect = None;
                    app.last_executions_list_rect = None;
                    app.last_executions_code_rect = None;
                    app.last_executions_output_rect = None;
                    app.last_agent_tasks_rect = None;
                }
                TerminalUiFocus::Executions => {
                    let list_w = (ch[1].width / 3).clamp(26, 46);
                    let hareas = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Length(list_w), Constraint::Min(8)])
                        .split(ch[1]);
                    let list_r = hareas[0];
                    let detail_area = hareas[1];
                    let vareas = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
                        .split(detail_area);
                    let code_r = vareas[0];
                    let out_r = vareas[1];

                    let (pl, max_l) = executions_list_paragraph(&app, list_r);
                    max_exec_list_scroll_holder.set(max_l);
                    f.render_widget(pl, list_r);

                    match &app.executions_detail {
                        Some(d) => {
                            let (pc, max_c) = executions_code_paragraph(
                                d,
                                code_r,
                                app.executions_code_scroll_top,
                            );
                            max_exec_code_scroll_holder.set(max_c);
                            f.render_widget(pc, code_r);
                            let (po, max_o) = executions_output_paragraph(
                                &d.journal,
                                out_r,
                                app.executions_output_scroll_top,
                            );
                            max_exec_out_scroll_holder.set(max_o);
                            f.render_widget(po, out_r);
                            app.last_executions_code_rect = Some(code_r);
                            app.last_executions_output_rect = Some(out_r);
                        }
                        None => {
                            max_exec_code_scroll_holder.set(0);
                            max_exec_out_scroll_holder.set(0);
                            app.last_executions_code_rect = None;
                            app.last_executions_output_rect = None;
                            let msg = app
                                .executions_detail_error
                                .as_deref()
                                .unwrap_or("Pick a run from the list (↑↓).");
                            let empty = Paragraph::new(Line::from(Span::styled(msg, Theme::dim())))
                                .block(
                                    Block::default()
                                        .borders(Borders::ALL)
                                        .title(Span::styled(" detail ", Theme::dim()))
                                        .border_style(Theme::dim()),
                                );
                            f.render_widget(empty, detail_area);
                        }
                    }
                    app.last_transcript_rect = None;
                    app.last_tool_history_rect = None;
                    app.last_executions_list_rect = Some(list_r);
                    app.last_conversations_list_rect = None;
                    app.last_agent_tasks_rect = None;
                }
                TerminalUiFocus::ToolHistory => {
                    let (w, max_s) =
                        tool_history_paragraph(&app.tool_rail, ch[1], app.tool_history_scroll);
                    max_tool_history_scroll_holder.set(max_s);
                    f.render_widget(w, ch[1]);
                    app.last_tool_history_rect = Some(ch[1]);
                    app.last_transcript_rect = None;
                    app.last_executions_list_rect = None;
                    app.last_executions_code_rect = None;
                    app.last_executions_output_rect = None;
                    app.last_conversations_list_rect = None;
                    app.last_agent_tasks_rect = None;
                }
                TerminalUiFocus::AgentTasks | TerminalUiFocus::BackgroundJobs => {
                    let list = background_pane_paragraph(&app);
                    let title = if app.ui_focus == TerminalUiFocus::BackgroundJobs {
                        " background "
                    } else {
                        " sub-agents "
                    };
                    let w = Paragraph::new(Text::from(list))
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .title(Span::styled(title, Theme::tool_call()))
                                .border_style(Theme::dim()),
                        )
                        .scroll((app.agent_tasks_scroll_top as u16, 0));
                    f.render_widget(w, ch[1]);
                    app.last_agent_tasks_rect = Some(ch[1]);
                    app.last_transcript_rect = None;
                    app.last_tool_history_rect = None;
                    app.last_executions_list_rect = None;
                    app.last_executions_code_rect = None;
                    app.last_executions_output_rect = None;
                    app.last_conversations_list_rect = None;
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
                app.thinking,
                app.cells.len(),
                app.active_toast(),
                &app,
            );
            let status_w = Paragraph::new(status_line);
            f.render_widget(status_w, ch[3]);

            let active_w = ch[4].width as usize;
            let idle = "Idle (no running tool)";
            let active_text = app.active_tool_line.as_deref().unwrap_or(idle);
            let icon = if app.active_tool_line.is_some() {
                app.get_spinner_frame().to_string()
            } else {
                "·".to_string()
            };
            let t = truncate_chars_display(active_text, active_w.max(8).saturating_sub(6));
            let active_row = Line::from(vec![
                Span::styled(format!(" {} ", icon), Theme::tool_call()),
                Span::styled(t, Theme::dim()),
            ]);
            f.render_widget(Paragraph::new(active_row), ch[4]);

            let input_block = Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(" compose ", Theme::dim()))
                .border_style(Theme::dim());

            let inner_area = ch[5].inner(Margin::new(1, 1));
            let inner_w = inner_area.width.max(1); // guard against zero-width after margin

            // Build visual lines with character-level wrapping so that the cursor
            // calculation (`total_cols / inner_w`) matches the rendered layout
            // exactly.  Previously the code relied on ratatui's Wrap which uses
            // word-level breaking, causing a mismatch.
            let text_lines = char_wrap_compose(
                &app.input,
                inner_w as usize,
                Theme::input_prompt(),
                Theme::text(),
            );

            // Calculate visual line of cursor for wrapping.
            // Uses display_width (Unicode column width) instead of char count so that
            // CJK characters and emoji that occupy two cells are measured correctly.
            let text_before_cursor = &app.input[..app.cursor];
            let mut cursor_visual_line: u16 = 0;
            let mut cursor_col_visual: u16 = 0;
            let lines_before_cursor: Vec<&str> = text_before_cursor.split('\n').collect();
            let total_lines = lines_before_cursor.len();
            for (i, line) in lines_before_cursor.into_iter().enumerate() {
                let prefix_len: u16 = 2; // "> " on first line, "  " on continuations
                let col_width = super::display_width(line) as u16;
                let total_cols = prefix_len + col_width;

                if i == total_lines - 1 {
                    cursor_visual_line += total_cols / inner_w;
                    cursor_col_visual = total_cols % inner_w;
                } else {
                    let visual_lines = if total_cols == 0 {
                        1
                    } else {
                        total_cols.div_ceil(inner_w)
                    };
                    cursor_visual_line += visual_lines;
                }
            }

            let mut input_v_scroll = 0;
            let available_h = input_h.saturating_sub(2);
            if cursor_visual_line >= available_h {
                input_v_scroll = (cursor_visual_line + 1).saturating_sub(available_h);
            }

            let input_para = Paragraph::new(Text::from(text_lines))
                .block(input_block)
                .scroll((input_v_scroll, 0));
            f.render_widget(input_para, ch[5]);

            let cx = inner_area.x.saturating_add(cursor_col_visual);
            let cy = inner_area
                .y
                .saturating_add(cursor_visual_line.saturating_sub(input_v_scroll));
            let cx = cx.clamp(
                inner_area.x,
                inner_area
                    .x
                    .saturating_add(inner_area.width.saturating_sub(1)),
            );
            let cy = cy.clamp(
                inner_area.y,
                inner_area
                    .y
                    .saturating_add(inner_area.height.saturating_sub(1)),
            );
            // Model selector popup overlay.
            if let Some(selector) = &app.model_selector {
                let popup_h = (selector.items.len() as u16 + 4).min(area.height.saturating_sub(4));
                let popup_w = 50u16.min(area.width.saturating_sub(4));
                let popup_x = area.x + (area.width.saturating_sub(popup_w)) / 2;
                let popup_y = area.y + (area.height.saturating_sub(popup_h)) / 2;
                let popup_area = Rect::new(popup_x, popup_y, popup_w, popup_h);

                f.render_widget(Clear, popup_area);

                let inner_block = Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(
                        " Select Model (↑↓ Enter Esc) ",
                        Theme::tool_call(),
                    ))
                    .border_style(Style::default().fg(Color::Cyan));
                let inner = inner_block.inner(popup_area);
                f.render_widget(inner_block, popup_area);

                let visible_h = inner.height as usize;
                let scroll_offset = if selector.selected >= visible_h {
                    selector.selected - visible_h + 1
                } else {
                    0
                };

                let mut lines: Vec<Line> = Vec::new();
                for (i, entry) in selector
                    .items
                    .iter()
                    .enumerate()
                    .skip(scroll_offset)
                    .take(visible_h)
                {
                    let marker = if i == selector.selected { "▶ " } else { "  " };
                    let style = if i == selector.selected {
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::White)
                    };
                    let line_text = format!("{}{}", marker, entry.label);
                    lines.push(Line::from(Span::styled(line_text, style)));
                }
                let list_para = Paragraph::new(Text::from(lines));
                f.render_widget(list_para, inner);
            }

            // Slash command hint popup (shown when input starts with "/" and no model selector).
            if app.model_selector.is_none()
                && app.input.starts_with('/')
                && !app.input.contains(' ')
            {
                let prefix = app.input.to_ascii_lowercase();
                let matching: Vec<(&str, &str)> = SLASH_COMMANDS
                    .iter()
                    .filter(|(cmd, _)| cmd.starts_with(&prefix))
                    .copied()
                    .collect();
                if !matching.is_empty() {
                    let popup_h = (matching.len() as u16 + 2).min(16);
                    let popup_w = 40u16.min(area.width.saturating_sub(4));
                    // Position above the compose box
                    let popup_y = ch[5].y.saturating_sub(popup_h);
                    let popup_x = ch[5].x + 1;
                    let popup_area = Rect::new(popup_x, popup_y, popup_w, popup_h);

                    f.render_widget(Clear, popup_area);
                    let hint_block = Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::DarkGray));
                    let hint_inner = hint_block.inner(popup_area);
                    f.render_widget(hint_block, popup_area);

                    let mut lines: Vec<Line> = Vec::new();
                    for (cmd, desc) in matching.iter().take(hint_inner.height as usize) {
                        lines.push(Line::from(vec![
                            Span::styled(format!("{:<16}", cmd), Style::default().fg(Color::Cyan)),
                            Span::styled(*desc, Style::default().fg(Color::Gray)),
                        ]));
                    }
                    f.render_widget(Paragraph::new(Text::from(lines)), hint_inner);
                }
            }

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
        if app.ui_focus == TerminalUiFocus::Executions {
            app.executions_list_scroll_top = app
                .executions_list_scroll_top
                .min(max_exec_list_scroll_holder.get());
            app.executions_code_scroll_top = app
                .executions_code_scroll_top
                .min(max_exec_code_scroll_holder.get());
            app.executions_output_scroll_top = app
                .executions_output_scroll_top
                .min(max_exec_out_scroll_holder.get());
        }
        if app.ui_focus == TerminalUiFocus::Conversations {
            app.conversations_list_scroll_top = app
                .conversations_list_scroll_top
                .min(max_conversations_list_scroll_holder.get());
        }

        if !event::poll(tick)? {
            continue;
        }

        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                // Model selector popup intercepts all keys when active.
                if app.model_selector.is_some() {
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            if let Some(sel) = &mut app.model_selector {
                                sel.move_up();
                            }
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            if let Some(sel) = &mut app.model_selector {
                                sel.move_down();
                            }
                        }
                        KeyCode::Enter => {
                            let selection = app
                                .model_selector
                                .as_ref()
                                .and_then(|s| s.selected_entry().map(|e| e.config_key.clone()));
                            app.model_selector = None;
                            if let Some(config_key) = selection {
                                if let Some(cfg) = providers.get(&config_key) {
                                    let resolved_url = cfg.resolved_base_url().unwrap_or_default();
                                    match cfg.resolve_api_key() {
                                        Ok(api_key) => {
                                            let msg = BusMessage::SwitchModel {
                                                provider_name: cfg.provider_name.clone(),
                                                model_name: cfg.model_name.clone(),
                                                base_url: resolved_url,
                                                api_key,
                                            };
                                            if bus_tx.blocking_send(msg).is_err() {
                                                app.cells.push(Cell::System {
                                                    message: "Bus closed; exiting.".into(),
                                                });
                                                app.request_quit();
                                            } else {
                                                app.status_model = cfg.model_name.clone();
                                                persist_last_model(&workspace_dir, &config_key);
                                                app.set_toast(
                                                    ToastKind::Ok,
                                                    format!("Model: {}", app.status_model),
                                                    Duration::from_secs(3),
                                                );
                                            }
                                        }
                                        Err(e) => {
                                            let err_msg = format!(
                                                "No API key for '{}': {}\n\
                                                Set the env var or add api_key = \"...\" under [providers.{}] in config.toml.",
                                                config_key, e, config_key
                                            );
                                            app.cells.push(Cell::Error { message: err_msg });
                                            app.set_toast(
                                                ToastKind::Err,
                                                format!("No API key for {}", config_key),
                                                Duration::from_secs(5),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        KeyCode::Esc | KeyCode::Char('q') => {
                            app.model_selector = None;
                        }
                        _ => {}
                    }
                    continue;
                }
                match key.code {
                    KeyCode::Char('c')
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && !key.modifiers.contains(KeyModifiers::SHIFT) =>
                    {
                        try_cancel_inflight(&mut app, &bus_tx, &chat_id);
                    }
                    KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if app.input.is_empty() {
                            app.request_quit();
                            let _ = shutdown_tx.send(());
                        } else {
                            app.delete_forward();
                        }
                    }
                    KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        app.delete_word();
                    }
                    KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        app.clear_line();
                    }
                    KeyCode::Left if key.modifiers.contains(KeyModifiers::ALT) => {
                        app.move_left_word()
                    }
                    KeyCode::Right if key.modifiers.contains(KeyModifiers::ALT) => {
                        app.move_right_word()
                    }
                    KeyCode::Char('b') | KeyCode::Char('B')
                        if key.modifiers.contains(KeyModifiers::ALT) =>
                    {
                        app.move_left_word()
                    }
                    KeyCode::Char('f') | KeyCode::Char('F')
                        if key.modifiers.contains(KeyModifiers::ALT) =>
                    {
                        app.move_right_word()
                    }
                    KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                        app.insert_char('\n');
                    }
                    KeyCode::Enter => {
                        // Windows Terminal / Conhost simulated paste detection.
                        // If another event is queued immediately within 2ms, it is mathematically impossible
                        // to be a human typing. It must be a simulated paste chunk, so we insert a newline
                        // instead of submitting the prompt.
                        if let Ok(true) = event::poll(Duration::from_millis(2)) {
                            app.insert_char('\n');
                            continue;
                        }

                        if app.ui_focus == TerminalUiFocus::Conversations
                            || app.ui_focus == TerminalUiFocus::BackgroundJobs
                        {
                            let selected_id = if app.ui_focus == TerminalUiFocus::Conversations {
                                if let Some(idx) = app.conversations_selected_idx {
                                    if idx < app.conversations_items.len() {
                                        Some((
                                            app.conversations_items[idx].thread_id.clone(),
                                            false,
                                        ))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                // BackgroundJobs jump-to-thread
                                if let Some(idx) = app.background_jobs_selected_idx {
                                    app.background_panel_items()
                                        .get(idx)
                                        .map(|item| (item.chat_id().to_string(), true))
                                } else {
                                    None
                                }
                            };

                            if let Some((target_id, is_chat_id)) = selected_id {
                                let (thread_id, new_cid) = if is_chat_id {
                                    let tid = crate::bus::clarification_session_key(
                                        channel_name.as_str(),
                                        &target_id,
                                        None,
                                    );
                                    (tid, target_id)
                                } else {
                                    let cid = match chat_id_from_root_thread_id(
                                        channel_name.as_str(),
                                        &target_id,
                                    ) {
                                        Some(c) => c,
                                        None => {
                                            app.set_toast(
                                                ToastKind::Err,
                                                "Invalid session row.".into(),
                                                Duration::from_secs(4),
                                            );
                                            continue;
                                        }
                                    };
                                    (target_id, cid)
                                };

                                try_cancel_inflight(&mut app, &bus_tx, &chat_id);
                                match load_thread_transcript_cells(&rt, &memory_node, &thread_id) {
                                    Ok(mut cells) => {
                                        chat_id = new_cid;
                                        sync_terminal_session_chat(&bus_tx, &chat_id);
                                        let sid = &chat_id[..8.min(chat_id.len())];
                                        cells.insert(
                                            0,
                                            Cell::System {
                                                message: format!(
                                                    "Resumed session {sid}… — loaded from workspace memory."
                                                ),
                                            },
                                        );
                                        app.cells = cells;
                                        app.thinking = false;
                                        app.llm_retry_available = false;
                                        app.last_inbound_text = None;
                                        app.tool_rail.clear();
                                        app.scroll_offset = 0;
                                        rescan_executions_manifest(
                                            &workspace_dir,
                                            &chat_id,
                                            &mut app,
                                        );
                                        last_exec_poll = Instant::now();
                                        app.focus_transcript();
                                        app.set_toast(
                                            ToastKind::Ok,
                                            format!("Continuing session {sid}…"),
                                            Duration::from_secs(2),
                                        );
                                    }
                                    Err(e) => {
                                        app.set_toast(
                                            ToastKind::Err,
                                            format!("Could not load history: {e}"),
                                            Duration::from_secs(5),
                                        );
                                    }
                                }
                            }
                            continue;
                        }
                        let raw = app.take_input();
                        let text = raw.trim();
                        if text.is_empty() {
                            continue;
                        }
                        if text.starts_with('/') {
                            if text.eq_ignore_ascii_case("/exit")
                                || text.eq_ignore_ascii_case("/quit")
                            {
                                app.request_quit();
                                let _ = shutdown_tx.send(());
                                continue;
                            }
                            if text.eq_ignore_ascii_case("/new") {
                                chat_id = uuid::Uuid::new_v4().to_string();
                                sync_terminal_session_chat(&bus_tx, &chat_id);
                                app.thinking = false;
                                app.cells.push(Cell::System {
                                    message: format!("New thread: {}", chat_id),
                                });
                                rescan_executions_manifest(&workspace_dir, &chat_id, &mut app);
                                last_exec_poll = Instant::now();
                                continue;
                            }
                            if text.eq_ignore_ascii_case("/copy") {
                                match copy_last_assistant_to_clipboard(
                                    &app.cells,
                                    &app.streaming_assistant,
                                ) {
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
                            if text.eq_ignore_ascii_case("/help") || text.eq_ignore_ascii_case("/?")
                            {
                                app.cells.push(Cell::System {
                                    message: TERMINAL_HELP.trim().to_string(),
                                });
                                continue;
                            }
                            if text.eq_ignore_ascii_case("/tools") {
                                app.focus_tool_history();
                                continue;
                            }
                            if text.eq_ignore_ascii_case("/exec") {
                                app.focus_executions();
                                rescan_executions_manifest(&workspace_dir, &chat_id, &mut app);
                                last_exec_poll = Instant::now();
                                continue;
                            }
                            if text.eq_ignore_ascii_case("/agents") {
                                app.focus_agent_tasks();
                                continue;
                            }
                            if text.eq_ignore_ascii_case("/background")
                                || text.eq_ignore_ascii_case("/bg")
                            {
                                app.focus_background_jobs();
                                let _ = poll_trigger_tx.send(chat_id.clone());

                                continue;
                            }
                            if text.eq_ignore_ascii_case("/chats") {
                                app.focus_conversations();
                                refresh_conversations_list(&rt, &memory_node, &mut app);
                                last_conversations_poll = Instant::now();
                                continue;
                            }
                            if text.eq_ignore_ascii_case("/cancel")
                                || text.eq_ignore_ascii_case("/stop")
                            {
                                try_cancel_inflight(&mut app, &bus_tx, &chat_id);
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
                                    message: "Promote-to-background requested. If a sync execution_run is in flight it will return a job_id you can poll with execution_job_status.".into(),
                                });
                                }
                                continue;
                            }
                            if text.eq_ignore_ascii_case("/retry") {
                                if !app.llm_retry_available {
                                    app.cells.push(Cell::System {
                                    message: "Nothing to retry. /retry is only available right after an LLM-failed banner.".into(),
                                });
                                    continue;
                                }
                                let Some(prev) = app.last_inbound_text.clone() else {
                                    app.llm_retry_available = false;
                                    app.cells.push(Cell::System {
                                        message: "No previous user message to re-submit.".into(),
                                    });
                                    continue;
                                };
                                app.llm_retry_available = false;
                                app.cells.push(Cell::User { text: prev.clone() });
                                app.thinking = true;
                                let (clean_text, attachments) =
                                    parse_terminal_attachments(&prev, &sandbox_dir);
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
                                continue;
                            }
                            // PR-5.1.1: `/context` — report the current chat's
                            // context size. Read-only memory query; runs even while
                            // a reasoning turn is active (the read goes through the
                            // SqliteMemoryActor which serializes against writes).
                            if text.eq_ignore_ascii_case("/context") {
                                let session_key = crate::bus::clarification_session_key(
                                    &channel_name,
                                    &chat_id,
                                    None,
                                );
                                let messages = rt.block_on(async {
                                    let (tx, rx) = tokio::sync::oneshot::channel();
                                    let _ = memory_node
                                        .send_packet(crate::memory::MemoryMessage::GetContext {
                                            thread_id: session_key.clone(),
                                            reply: crate::memory::SharedReply::new(tx),
                                        })
                                        .await;
                                    rx.await.ok().and_then(|r| r.ok()).unwrap_or_default()
                                });
                                let user_turns =
                                    messages.iter().filter(|m| m.role == "user").count();
                                let approx_tokens: usize = messages
                                    .iter()
                                    .map(|m| {
                                        m.content.as_ref().map_or(0, |c| c.text_content().len()) / 4
                                    })
                                    .sum();
                                app.cells.push(Cell::System {
                                    message: format!(
                                        "Context: {} message(s) · {} user turn(s) · ~{} tokens (rough estimate, 4 chars per token). Use /compact to force a compaction.",
                                        messages.len(),
                                        user_turns,
                                        approx_tokens
                                    ),
                                });
                                continue;
                            }
                            // PR-5.1: `/compact [focus instructions]` — fires a
                            // manual compaction for the current terminal chat by
                            // posting `BusMessage::TriggerCompaction`. The agent's
                            // per-chat FIFO guard runs the compaction between turns
                            // (drops it with a warning if a reasoning turn is
                            // currently active).
                            if text.eq_ignore_ascii_case("/compact")
                                || text.to_ascii_lowercase().starts_with("/compact ")
                            {
                                let focus = text
                                    .strip_prefix("/compact")
                                    .or_else(|| text.strip_prefix("/COMPACT"))
                                    .unwrap_or("")
                                    .trim()
                                    .to_string();
                                let session_key = crate::bus::clarification_session_key(
                                    &channel_name,
                                    &chat_id,
                                    None,
                                );
                                let msg = BusMessage::TriggerCompaction {
                                    session_key,
                                    focus_instructions: if focus.is_empty() {
                                        None
                                    } else {
                                        Some(focus.clone())
                                    },
                                    trigger: Some(crate::bus::CompactionTrigger::Manual),
                                };
                                if bus_tx.blocking_send(msg).is_err() {
                                    app.cells.push(Cell::System {
                                        message: "Bus closed; cannot trigger compaction.".into(),
                                    });
                                } else {
                                    let note = if focus.is_empty() {
                                        "Compaction requested. It will run between turns; next inbound will see a smaller context.".to_string()
                                    } else {
                                        format!(
                                            "Compaction requested with focus: \"{}\". Next inbound will see a smaller context.",
                                            focus
                                        )
                                    };
                                    app.cells.push(Cell::System { message: note });
                                }
                                continue;
                            }
                            if text.eq_ignore_ascii_case("/model")
                                || text.to_ascii_lowercase().starts_with("/model ")
                            {
                                let arg = text.strip_prefix("/model").unwrap_or("").trim();
                                if arg.is_empty() {
                                    // Open interactive model selector popup
                                    if providers.is_empty() {
                                        app.cells.push(Cell::System {
                                        message: "No models configured. Add [providers.*] sections to config.toml.".into(),
                                    });
                                    } else {
                                        app.model_selector =
                                            Some(ModelSelector::from_providers(&providers));
                                    }
                                } else if providers.contains_key(arg) {
                                    try_switch_model(
                                        &mut app,
                                        &bus_tx,
                                        &providers,
                                        &workspace_dir,
                                        arg,
                                    );
                                } else {
                                    let available: Vec<&str> =
                                        providers.keys().map(|s| s.as_str()).collect();
                                    app.cells.push(Cell::System {
                                        message: format!(
                                            "Unknown model '{}'. Available: {}",
                                            arg,
                                            available.join(", ")
                                        ),
                                    });
                                }
                                continue;
                            }
                            if text.to_ascii_lowercase().starts_with("/skills ") {
                                let arg = text.strip_prefix("/skills ").unwrap_or("").trim();
                                if arg.starts_with("add ") {
                                    let repo_part = arg.strip_prefix("add ").unwrap_or("").trim();
                                    if repo_part.is_empty() {
                                        app.cells.push(Cell::System {
                                            message: "Usage: /skills add <repo_url_or_shorthand> [skill_name]".into(),
                                        });
                                    } else {
                                        let mut parts = repo_part.split_whitespace();
                                        let repo_url = parts.next().unwrap_or("").to_string();
                                        let skill_name = parts.next().map(|s| s.to_string());

                                        let msg = BusMessage::InstallSkill {
                                            repo_url: repo_url.clone(),
                                            skill_name,
                                        };
                                        if bus_tx.blocking_send(msg).is_err() {
                                            app.cells.push(Cell::System {
                                                message: "Bus closed; cannot install skills."
                                                    .into(),
                                            });
                                        } else {
                                            app.cells.push(Cell::System {
                                                message: format!("Skill installation requested for repository: {}. Check logs for progress.", repo_url),
                                            });
                                        }
                                    }
                                } else {
                                    app.cells.push(Cell::System {
                                        message: "Usage: /skills add <repo_url_or_shorthand> [skill_name]".into(),
                                    });
                                }
                                continue;
                            }
                            app.cells.push(Cell::System {
                            message:
                                "Unknown command. Try /help, /exit, /new, /chats, /copy, /install-python, /cancel, /background, /retry, /tools, /exec, /agents, /model, /compact, /context, /skills."
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
                        app.last_inbound_text = Some(text.to_string());
                        app.llm_retry_available = false;
                        let (clean_text, attachments) =
                            parse_terminal_attachments(text, &sandbox_dir);
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
                    KeyCode::Up => {
                        if app.ui_focus == TerminalUiFocus::Conversations
                            && !app.conversations_items.is_empty()
                        {
                            if let Some(i) = app.conversations_selected_idx {
                                if i > 0 {
                                    app.conversations_selected_idx = Some(i - 1);
                                    let list_h = app
                                        .last_conversations_list_rect
                                        .map(|r| r.height.saturating_sub(2) as usize)
                                        .unwrap_or(8);
                                    conversations_ensure_list_shows_selection(&mut app, list_h);
                                }
                            }
                        } else if app.ui_focus == TerminalUiFocus::Executions
                            && !app.executions_runs.is_empty()
                        {
                            if let Some(i) = app.executions_selected_idx {
                                if i > 0 {
                                    app.executions_selected_idx = Some(i - 1);
                                    load_selected_execution_detail(&workspace_dir, &mut app);
                                    let list_h = app
                                        .last_executions_list_rect
                                        .map(|r| r.height.saturating_sub(2) as usize)
                                        .unwrap_or(8);
                                    executions_ensure_list_shows_selection(&mut app, list_h);
                                }
                            }
                        } else if app.ui_focus == TerminalUiFocus::BackgroundJobs {
                            if let Some(i) = app.background_jobs_selected_idx {
                                if i > 0 {
                                    app.background_jobs_selected_idx = Some(i - 1);
                                }
                            }
                        } else if !matches!(
                            app.ui_focus,
                            TerminalUiFocus::Executions
                                | TerminalUiFocus::Conversations
                                | TerminalUiFocus::BackgroundJobs
                        ) {
                            app.history_up();
                        }
                    }
                    KeyCode::Down => {
                        if app.ui_focus == TerminalUiFocus::Conversations
                            && !app.conversations_items.is_empty()
                        {
                            if let Some(i) = app.conversations_selected_idx {
                                if i + 1 < app.conversations_items.len() {
                                    app.conversations_selected_idx = Some(i + 1);
                                    let list_h = app
                                        .last_conversations_list_rect
                                        .map(|r| r.height.saturating_sub(2) as usize)
                                        .unwrap_or(8);
                                    conversations_ensure_list_shows_selection(&mut app, list_h);
                                }
                            }
                        } else if app.ui_focus == TerminalUiFocus::Executions
                            && !app.executions_runs.is_empty()
                        {
                            if let Some(i) = app.executions_selected_idx {
                                if i + 1 < app.executions_runs.len() {
                                    app.executions_selected_idx = Some(i + 1);
                                    load_selected_execution_detail(&workspace_dir, &mut app);
                                    let list_h = app
                                        .last_executions_list_rect
                                        .map(|r| r.height.saturating_sub(2) as usize)
                                        .unwrap_or(8);
                                    executions_ensure_list_shows_selection(&mut app, list_h);
                                }
                            }
                        } else if app.ui_focus == TerminalUiFocus::BackgroundJobs
                            && (!app.background_jobs.is_empty()
                                || !app.notifications.is_empty()
                                || !app.agent_tasks.is_empty())
                        {
                            if let Some(i) = app.background_jobs_selected_idx {
                                let total = app.notifications.len()
                                    + app.background_jobs.len()
                                    + app.agent_tasks.len();
                                if i + 1 < total {
                                    app.background_jobs_selected_idx = Some(i + 1);
                                }
                            }
                        } else if !matches!(
                            app.ui_focus,
                            TerminalUiFocus::Executions
                                | TerminalUiFocus::Conversations
                                | TerminalUiFocus::BackgroundJobs
                        ) {
                            app.history_down();
                        }
                    }
                    KeyCode::Esc => {
                        if matches!(
                            app.ui_focus,
                            TerminalUiFocus::ToolHistory
                                | TerminalUiFocus::Executions
                                | TerminalUiFocus::Conversations
                                | TerminalUiFocus::AgentTasks
                        ) {
                            app.focus_transcript();
                        }
                    }
                    KeyCode::BackTab => {
                        let mut focus_ctx = FocusCycleContext::new(
                            &workspace_dir,
                            &chat_id,
                            &rt,
                            &memory_node,
                            &mut last_exec_poll,
                            &mut last_conversations_poll,
                        );
                        apply_ui_focus_cycle(&mut app, false, &mut focus_ctx);
                    }
                    KeyCode::Tab => {
                        let forward = !key.modifiers.contains(KeyModifiers::SHIFT);
                        let mut focus_ctx = FocusCycleContext::new(
                            &workspace_dir,
                            &chat_id,
                            &rt,
                            &memory_node,
                            &mut last_exec_poll,
                            &mut last_conversations_poll,
                        );
                        apply_ui_focus_cycle(&mut app, forward, &mut focus_ctx);
                    }
                    KeyCode::Char('t') | KeyCode::Char('T')
                        if key.modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        let mut focus_ctx = FocusCycleContext::new(
                            &workspace_dir,
                            &chat_id,
                            &rt,
                            &memory_node,
                            &mut last_exec_poll,
                            &mut last_conversations_poll,
                        );
                        apply_ui_focus_cycle(&mut app, true, &mut focus_ctx);
                    }
                    KeyCode::Char('g') | KeyCode::Char('G')
                        if key.modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        app.focus_background_jobs();
                        let _ = poll_trigger_tx.send(chat_id.clone());
                    }
                    KeyCode::PageUp => match app.ui_focus {
                        TerminalUiFocus::ToolHistory => app.tool_history_scroll_up(8),
                        TerminalUiFocus::AgentTasks | TerminalUiFocus::BackgroundJobs => {
                            app.agent_tasks_scroll_top =
                                app.agent_tasks_scroll_top.saturating_sub(1);
                        }
                        TerminalUiFocus::Executions => {
                            if key.modifiers.contains(KeyModifiers::CONTROL) {
                                app.executions_code_scroll_top =
                                    app.executions_code_scroll_top.saturating_sub(3);
                            } else if key.modifiers.contains(KeyModifiers::SHIFT) {
                                app.executions_list_scroll_top =
                                    app.executions_list_scroll_top.saturating_sub(1);
                            } else {
                                app.executions_output_scroll_top =
                                    app.executions_output_scroll_top.saturating_sub(3);
                            }
                        }
                        TerminalUiFocus::Conversations => {
                            app.conversations_list_scroll_top =
                                app.conversations_list_scroll_top.saturating_sub(1);
                        }
                        TerminalUiFocus::Transcript => app.scroll_up(8),
                    },
                    KeyCode::PageDown => match app.ui_focus {
                        TerminalUiFocus::ToolHistory => app.tool_history_scroll_down(8),
                        TerminalUiFocus::AgentTasks | TerminalUiFocus::BackgroundJobs => {
                            app.agent_tasks_scroll_top =
                                app.agent_tasks_scroll_top.saturating_add(1);
                        }
                        TerminalUiFocus::Executions => {
                            if key.modifiers.contains(KeyModifiers::CONTROL) {
                                app.executions_code_scroll_top =
                                    app.executions_code_scroll_top.saturating_add(3);
                            } else if key.modifiers.contains(KeyModifiers::SHIFT) {
                                app.executions_list_scroll_top =
                                    app.executions_list_scroll_top.saturating_add(1);
                            } else {
                                app.executions_output_scroll_top =
                                    app.executions_output_scroll_top.saturating_add(3);
                            }
                        }
                        TerminalUiFocus::Conversations => {
                            app.conversations_list_scroll_top =
                                app.conversations_list_scroll_top.saturating_add(1);
                        }
                        TerminalUiFocus::Transcript => app.scroll_down(8),
                    },
                    KeyCode::F(5) => {
                        if app.ui_focus == TerminalUiFocus::Executions {
                            rescan_executions_manifest(&workspace_dir, &chat_id, &mut app);
                            last_exec_poll = Instant::now();
                        }
                        if app.ui_focus == TerminalUiFocus::Conversations {
                            refresh_conversations_list(&rt, &memory_node, &mut app);
                            last_conversations_poll = Instant::now();
                        }
                    }
                    KeyCode::Char(c)
                        if matches!(c, 'y' | 'Y')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.modifiers.contains(KeyModifiers::SHIFT) =>
                    {
                        match copy_last_assistant_to_clipboard(&app.cells, &app.streaming_assistant)
                        {
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
                    KeyCode::Char(mch)
                        if matches!(mch, 'm' | 'M')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.modifiers.contains(KeyModifiers::SHIFT) =>
                    {
                        if application_mouse_enabled {
                            match execute!(terminal.backend_mut(), DisableMouseCapture) {
                                Ok(()) => {
                                    application_mouse_enabled = false;
                                    app.set_toast(
                                        ToastKind::Ok,
                                        "Mouse wheel off — native selection works.".into(),
                                        Duration::from_secs(TOAST_MOUSE_TOGGLE_OK_SECS),
                                    );
                                }
                                Err(e) => {
                                    app.set_toast(
                                        ToastKind::Err,
                                        format!("Could not release mouse: {e}"),
                                        Duration::from_secs(TOAST_MOUSE_TOGGLE_ERR_SECS),
                                    );
                                }
                            }
                        } else {
                            match execute!(terminal.backend_mut(), EnableMouseCapture) {
                                Ok(()) => {
                                    application_mouse_enabled = true;
                                    app.set_toast(
                                        ToastKind::Ok,
                                        "Mouse wheel on — Ctrl+Shift+M again for native selection."
                                            .into(),
                                        Duration::from_secs(TOAST_MOUSE_TOGGLE_OK_SECS),
                                    );
                                }
                                Err(e) => {
                                    app.set_toast(
                                        ToastKind::Err,
                                        format!("Mouse capture failed: {e}"),
                                        Duration::from_secs(TOAST_MOUSE_TOGGLE_ERR_SECS),
                                    );
                                }
                            }
                        }
                    }
                    KeyCode::Char(c) => app.insert_char(c),
                    _ => {}
                }
            } // end Event::Key block
            Event::Mouse(me) => {
                // ── Text selection: drag / release (fires regardless of position) ──
                if app.selecting {
                    match me.kind {
                        MouseEventKind::Drag(MouseButton::Left) => {
                            if let Some(rect) = app.last_transcript_rect {
                                let (line_idx, col) = mouse_to_transcript_coords(
                                    me.column,
                                    me.row,
                                    rect,
                                    app.last_transcript_visible_start,
                                );
                                if let Some(sel) = &mut app.transcript_selection {
                                    sel.end_line = line_idx;
                                    sel.end_col = col;
                                }
                            }
                        }
                        MouseEventKind::Up(MouseButton::Left) => {
                            // Update end position from release coordinates
                            // (handles terminals that don't send Drag events).
                            if let Some(rect) = app.last_transcript_rect {
                                let (line_idx, col) = mouse_to_transcript_coords(
                                    me.column,
                                    me.row,
                                    rect,
                                    app.last_transcript_visible_start,
                                );
                                if let Some(sel) = &mut app.transcript_selection {
                                    sel.end_line = line_idx;
                                    sel.end_col = col;
                                }
                            }
                            app.selecting = false;
                            if let Some(sel) = &app.transcript_selection {
                                if !sel.is_empty() {
                                    if let Some(rect) = app.last_transcript_rect {
                                        let inner_w = rect.width.saturating_sub(2) as usize;
                                        let text = extract_selection_text(&app.cells, inner_w, sel);
                                        let n = text.len();
                                        match arboard::Clipboard::new()
                                            .and_then(|mut c| c.set_text(&text))
                                        {
                                            Ok(()) => {
                                                app.set_toast(
                                                    ToastKind::Ok,
                                                    format!("Copied selection ({n} chars)"),
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
                                    }
                                } else {
                                    app.transcript_selection = None;
                                }
                            }
                        }
                        _ => {}
                    }
                }

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
                        MouseEventKind::Down(MouseButton::Left) => {
                            if let Some(rect) = app.last_transcript_rect {
                                let (line_idx, col) = mouse_to_transcript_coords(
                                    me.column,
                                    me.row,
                                    rect,
                                    app.last_transcript_visible_start,
                                );
                                app.transcript_selection = Some(TranscriptSelection {
                                    anchor_line: line_idx,
                                    anchor_col: col,
                                    end_line: line_idx,
                                    end_col: col,
                                });
                                app.selecting = true;
                            }
                        }
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
                } else if app
                    .last_conversations_list_rect
                    .map(|r| rect_contains(r, me.column, me.row))
                    .unwrap_or(false)
                {
                    match me.kind {
                        MouseEventKind::ScrollUp => {
                            app.conversations_list_scroll_top =
                                app.conversations_list_scroll_top.saturating_sub(1);
                        }
                        MouseEventKind::ScrollDown => {
                            app.conversations_list_scroll_top =
                                app.conversations_list_scroll_top.saturating_add(1);
                        }
                        MouseEventKind::ScrollLeft => {
                            app.conversations_list_scroll_top =
                                app.conversations_list_scroll_top.saturating_sub(1);
                        }
                        MouseEventKind::ScrollRight => {
                            app.conversations_list_scroll_top =
                                app.conversations_list_scroll_top.saturating_add(1);
                        }
                        _ => {}
                    }
                } else if app
                    .last_agent_tasks_rect
                    .map(|r| rect_contains(r, me.column, me.row))
                    .unwrap_or(false)
                {
                    match me.kind {
                        MouseEventKind::ScrollUp => {
                            app.agent_tasks_scroll_top =
                                app.agent_tasks_scroll_top.saturating_sub(1);
                        }
                        MouseEventKind::ScrollDown => {
                            app.agent_tasks_scroll_top =
                                app.agent_tasks_scroll_top.saturating_add(1);
                        }
                        MouseEventKind::ScrollLeft => {
                            app.agent_tasks_scroll_top =
                                app.agent_tasks_scroll_top.saturating_sub(1);
                        }
                        MouseEventKind::ScrollRight => {
                            app.agent_tasks_scroll_top =
                                app.agent_tasks_scroll_top.saturating_add(1);
                        }
                        _ => {}
                    }
                } else if app.ui_focus == TerminalUiFocus::Executions {
                    let over_list = app
                        .last_executions_list_rect
                        .map(|r| rect_contains(r, me.column, me.row))
                        .unwrap_or(false);
                    let over_code = app
                        .last_executions_code_rect
                        .map(|r| rect_contains(r, me.column, me.row))
                        .unwrap_or(false);
                    let over_out = app
                        .last_executions_output_rect
                        .map(|r| rect_contains(r, me.column, me.row))
                        .unwrap_or(false);
                    let n = MOUSE_SCROLL_LINES as usize;
                    if over_list {
                        match me.kind {
                            MouseEventKind::ScrollUp => {
                                app.executions_list_scroll_top =
                                    app.executions_list_scroll_top.saturating_sub(1);
                            }
                            MouseEventKind::ScrollDown => {
                                app.executions_list_scroll_top =
                                    app.executions_list_scroll_top.saturating_add(1);
                            }
                            MouseEventKind::ScrollLeft => {
                                app.executions_list_scroll_top =
                                    app.executions_list_scroll_top.saturating_sub(1);
                            }
                            MouseEventKind::ScrollRight => {
                                app.executions_list_scroll_top =
                                    app.executions_list_scroll_top.saturating_add(1);
                            }
                            _ => {}
                        }
                    } else if over_code {
                        match me.kind {
                            MouseEventKind::ScrollUp => {
                                app.executions_code_scroll_top =
                                    app.executions_code_scroll_top.saturating_sub(n);
                            }
                            MouseEventKind::ScrollDown => {
                                app.executions_code_scroll_top =
                                    app.executions_code_scroll_top.saturating_add(n);
                            }
                            MouseEventKind::ScrollLeft => {
                                app.executions_code_scroll_top =
                                    app.executions_code_scroll_top.saturating_sub(n);
                            }
                            MouseEventKind::ScrollRight => {
                                app.executions_code_scroll_top =
                                    app.executions_code_scroll_top.saturating_add(n);
                            }
                            _ => {}
                        }
                    } else if over_out {
                        match me.kind {
                            MouseEventKind::ScrollUp => {
                                app.executions_output_scroll_top =
                                    app.executions_output_scroll_top.saturating_sub(n);
                            }
                            MouseEventKind::ScrollDown => {
                                app.executions_output_scroll_top =
                                    app.executions_output_scroll_top.saturating_add(n);
                            }
                            MouseEventKind::ScrollLeft => {
                                app.executions_output_scroll_top =
                                    app.executions_output_scroll_top.saturating_sub(n);
                            }
                            MouseEventKind::ScrollRight => {
                                app.executions_output_scroll_top =
                                    app.executions_output_scroll_top.saturating_add(n);
                            }
                            _ => {}
                        }
                    }
                } else {
                    // Fallback: scroll transcript for any unhandled area (e.g. compose box).
                    match me.kind {
                        MouseEventKind::ScrollUp => app.scroll_up(MOUSE_SCROLL_LINES),
                        MouseEventKind::ScrollDown => app.scroll_down(MOUSE_SCROLL_LINES),
                        MouseEventKind::ScrollLeft => app.scroll_up(MOUSE_SCROLL_LINES),
                        MouseEventKind::ScrollRight => app.scroll_down(MOUSE_SCROLL_LINES),
                        _ => {}
                    }
                }
            }
            Event::Paste(ref s) => {
                for c in s.chars() {
                    // Filter out carriage returns to avoid issues, just keep newlines
                    if c != '\r' {
                        app.insert_char(c);
                    }
                }
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    disable_raw_mode()?;
    if application_mouse_enabled {
        let _ = execute!(terminal.backend_mut(), DisableMouseCapture);
    }
    execute!(
        terminal.backend_mut(),
        DisableBracketedPaste,
        LeaveAlternateScreen,
        crossterm::cursor::Show
    )?;
    Ok(())
}

#[cfg(test)]
mod width_fit_tests {
    use super::{build_status_line, build_title_line};
    use crate::channels::terminal_ui::app::App;
    use crate::channels::terminal_ui::display_width;
    use crate::channels::terminal_ui::text_format::truncate_chars_display;
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
        let app = App::new();
        let line = build_status_line(26, false, 3, None, &app);
        let t = flat(&line);
        assert!(t.contains("🦾"), "{t}");
        assert!(t.contains("idle"));
        assert!(!t.contains("thread"), "{t}");
        assert!(
            !t.contains("Enter send"),
            "hints should drop first when tight: {t}"
        );
    }

    #[test]
    fn title_shows_brand_only() {
        let line = build_title_line(120);
        let t = flat(&line);
        assert!(t.contains("ALTAI isanagent"), "{t}");
        assert!(!t.contains("thread "), "{t}");
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
