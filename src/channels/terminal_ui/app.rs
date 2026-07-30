#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ratatui::layout::Rect;

use crate::config::ProviderConfig;
use crate::memory::RootThreadListItem;

use super::execution_browser::{ExecutionRunDetail, ExecutionRunListItem};

/// An entry in the model selector popup.
#[derive(Debug, Clone)]
pub struct ModelSelectorEntry {
    /// Display label (e.g. "claude-opus-4-7")
    pub label: String,
    /// Config key in `[providers.*]` map
    pub config_key: String,
    /// Provider name (e.g. "anthropic")
    pub provider_name: String,
    /// Model ID (e.g. "claude-opus-4-7")
    pub model_name: String,
}

/// Popup state for interactive model selection via `/model`.
#[derive(Debug, Clone)]
pub struct ModelSelector {
    pub items: Vec<ModelSelectorEntry>,
    pub selected: usize,
}

impl ModelSelector {
    pub fn from_providers(providers: &std::collections::HashMap<String, ProviderConfig>) -> Self {
        let mut items: Vec<ModelSelectorEntry> = providers
            .iter()
            .map(|(key, cfg)| ModelSelectorEntry {
                label: format!("{} ({})", cfg.model_name, cfg.provider_name),
                config_key: key.clone(),
                provider_name: cfg.provider_name.clone(),
                model_name: cfg.model_name.clone(),
            })
            .collect();
        items.sort_by(|a, b| a.config_key.cmp(&b.config_key));
        Self { items, selected: 0 }
    }

    pub fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    pub fn move_down(&mut self) {
        if self.selected + 1 < self.items.len() {
            self.selected += 1;
        }
    }

    pub fn selected_entry(&self) -> Option<&ModelSelectorEntry> {
        self.items.get(self.selected)
    }
}

/// High-level UI mode for the terminal front-end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalUiMode {
    #[default]
    Input,
    /// Agent is streaming tokens into `streaming_*` buffers.
    Streaming,
}

/// Telemetry-style tool line phase (mirrors `terminal` channel metadata).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolNoticePhase {
    /// Initial in-flight phase: a tool call has been emitted but no result has arrived yet.
    /// Rendered yellow/dim. When a matching result/fail arrives with the same `tool_call_id`,
    /// `upsert_tool_notice` mutates this cell in place rather than appending a second row.
    Pending,
    Call,
    Result,
    /// Tool returned an error string (e.g. `Error: …`).
    Failed,
    Other,
}

/// Which main pane has keyboard and scroll focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalUiFocus {
    #[default]
    Transcript,
    /// List root terminal sessions from workspace memory; Enter loads and continues.
    Conversations,
    /// Browse `execution_runs.jsonl` + per-run journals for this terminal thread (`chat_id`).
    Executions,
    ToolHistory,
    /// Sub-agent task lifecycle pane (running / finished named agents, plan steps).
    AgentTasks,
    /// Detailed list of background jobs and notifications.
    BackgroundJobs,
}

const TOOL_RAIL_CAP: usize = 150;

/// One line in the tool-activity ring buffer (call / result / fail).
#[derive(Debug, Clone)]
pub struct ToolRailEntry {
    pub tool_name: String,
    pub phase: ToolNoticePhase,
    pub summary: String,
}

/// Lifecycle of a job tracked by the multi-job execution strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStripStatus {
    Running,
    Waiting,
    Completed,
    Failed,
    Cancelled,
    Timeout,
}

impl JobStripStatus {
    pub fn from_str(s: &str) -> Self {
        match s {
            "completed" => JobStripStatus::Completed,
            "failed" => JobStripStatus::Failed,
            "cancelled" => JobStripStatus::Cancelled,
            "timeout" => JobStripStatus::Timeout,
            "waiting" => JobStripStatus::Waiting,
            _ => JobStripStatus::Running,
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            JobStripStatus::Running => "·",
            JobStripStatus::Waiting => "⏳",
            JobStripStatus::Completed => "✓",
            JobStripStatus::Failed => "✗",
            JobStripStatus::Cancelled => "⨯",
            JobStripStatus::Timeout => "⏱",
        }
    }

    pub fn is_terminal(&self) -> bool {
        !matches!(self, JobStripStatus::Running | JobStripStatus::Waiting)
    }
}

/// Lifecycle of a sub-agent task tracked by the agent-tasks pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentTaskStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl AgentTaskStatus {
    pub fn from_str(s: &str) -> Self {
        match s {
            "completed" => AgentTaskStatus::Completed,
            "failed" => AgentTaskStatus::Failed,
            "cancelled" => AgentTaskStatus::Cancelled,
            _ => AgentTaskStatus::Running,
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            AgentTaskStatus::Running => "·",
            AgentTaskStatus::Completed => "✓",
            AgentTaskStatus::Failed => "✗",
            AgentTaskStatus::Cancelled => "⨯",
        }
    }

    pub fn is_terminal(&self) -> bool {
        !matches!(self, AgentTaskStatus::Running)
    }
}

/// One row in the agent-tasks pane (sub-agent lifecycle).
#[derive(Debug, Clone)]
pub struct AgentTaskEntry {
    pub task_id: String,
    pub child_chat_id: String,
    pub agent_name: Option<String>,
    pub display_name: Option<String>,
    pub started_at: Instant,
    pub status: AgentTaskStatus,
    pub last_line: String,
    pub terminal_at: Option<Instant>,
}

/// One row in the multi-job execution strip below the transcript.
#[derive(Debug, Clone)]
pub struct JobStripEntry {
    pub job_id: String,
    pub session_id: String,
    pub chat_id: String,
    pub tool_name: String,
    pub description: Option<String>,
    pub started_at: Instant,
    pub status: JobStripStatus,
    pub last_line: String,
    /// Set when the job reached a terminal status; used to evict finished rows after a delay.
    pub terminal_at: Option<Instant>,
}

/// One renderable unit in the transcript (Xerxes-style cell model: labeled blocks, tool rail).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cell {
    User {
        text: String,
    },
    Assistant {
        /// Model text; may be plain text or markdown depending on provider output.
        markdown: String,
    },
    Thinking {
        text: String,
    },
    ToolNotice {
        phase: ToolNoticePhase,
        content: String,
        /// LLM-supplied stable id for the originating tool invocation.
        ///
        /// When present, lets the UI mutate a single cell from `Pending` → `Result`/`Failed`
        /// instead of pushing two cells per call. Optional for backwards compatibility with
        /// older bus messages and synthetic notices that have no upstream id.
        tool_call_id: Option<String>,
    },
    Clarification {
        text: String,
        /// From `ask_user` when provided; shown as a numbered list in the terminal.
        choices: Vec<String>,
        /// Optional unified-diff payload for file-edit approvals.
        edit_diff: Option<crate::channels::terminal_ui::EditDiffPayload>,
    },
    System {
        message: String,
    },
    /// Agent / provider failure surfaced in-session (not only logs).
    Error {
        message: String,
    },
}

/// Mouse text selection state within the transcript pane.
/// Coordinates are (line_index, display_col) in flattened transcript lines.
#[derive(Debug, Clone, Copy)]
pub struct TranscriptSelection {
    pub anchor_line: usize,
    pub anchor_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl TranscriptSelection {
    /// Returns (start_line, start_col, end_line, end_col) normalized so start <= end.
    pub fn normalized(&self) -> (usize, usize, usize, usize) {
        if self.anchor_line < self.end_line
            || (self.anchor_line == self.end_line && self.anchor_col <= self.end_col)
        {
            (
                self.anchor_line,
                self.anchor_col,
                self.end_line,
                self.end_col,
            )
        } else {
            (
                self.end_line,
                self.end_col,
                self.anchor_line,
                self.anchor_col,
            )
        }
    }

    pub fn is_empty(&self) -> bool {
        self.anchor_line == self.end_line && self.anchor_col == self.end_col
    }
}

/// Ephemeral status-strip message (e.g. copy confirmation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Ok,
    Err,
}

#[derive(Debug, Clone)]
pub struct Toast {
    pub kind: ToastKind,
    pub message: String,
    pub until: Instant,
}

/// Mutable TUI application state: transcript, single-line input, scroll, streaming buffers.
#[derive(Debug, Clone)]
pub struct App {
    pub mode: TerminalUiMode,
    pub cells: Vec<Cell>,
    pub input: String,
    pub cursor: usize,
    pub history: Vec<String>,
    pub history_idx: Option<usize>,
    pub streaming_thinking: String,
    pub streaming_assistant: String,
    pub active_tool_line: Option<String>,
    pub ui_focus: TerminalUiFocus,
    /// Recent tool telemetry (ring buffer, newest at end).
    pub tool_rail: Vec<ToolRailEntry>,
    /// Lines hidden below the tool-history viewport bottom (`0` = follow tail).
    pub tool_history_scroll: u16,
    /// Upper bound on `tool_history_scroll`, set from wrapped tool-rail line counts.
    pub tool_history_max_scroll: u16,
    pub last_tool_history_rect: Option<Rect>,
    /// Lines hidden below the viewport bottom. `0` = follow latest output.
    pub scroll_offset: u16,
    /// Upper bound on `scroll_offset`, set by the renderer from wrapped line counts.
    pub max_scroll: u16,
    pub should_quit: bool,
    /// LLM loop active for this chat thread (cleared on assistant reply, error, or clarification).
    pub thinking: bool,
    /// Last drawn transcript widget area (for mouse wheel hit-testing).
    pub last_transcript_rect: Option<Rect>,
    /// First visible line index in the transcript (set by renderer for mouse coordinate mapping).
    pub last_transcript_visible_start: usize,
    /// Active mouse text selection in the transcript pane.
    pub transcript_selection: Option<TranscriptSelection>,
    /// True while the left mouse button is held during a drag selection.
    pub selecting: bool,
    /// Short-lived message shown in the status strip (not stored in the transcript).
    pub toast: Option<Toast>,
    /// Latest `execution_run` stream (Jupyter); shown in a dedicated strip below the transcript.
    pub execution_stream_recent: String,
    pub execution_stream_label: Option<(String, String)>,
    /// Multi-job execution strip rows (Colab MCP background calls, auto-promoted runs, etc.).
    pub jobs_strip: VecDeque<JobStripEntry>,
    /// Sub-agent task rows (named agents, plan steps) — shown in the AgentTasks pane.
    pub agent_tasks: VecDeque<AgentTaskEntry>,
    /// Lines hidden below the agent-tasks viewport top (`0` = follow tail).
    pub agent_tasks_scroll_top: usize,
    pub last_agent_tasks_rect: Option<Rect>,
    pub background_jobs: Vec<crate::memory::BackgroundJobRecord>,
    pub notifications: Vec<crate::memory::NotificationRecord>,
    pub background_jobs_selected_idx: Option<usize>,
    pub last_background_jobs_rect: Option<Rect>,
    /// True when the most recent reasoning turn ended with an exhausted-retry LLM failure
    /// banner. Cleared once the next user message (or `/retry`) is sent.
    pub llm_retry_available: bool,
    /// Remembers the last user-sent inbound text so `/retry` can re-submit it after an
    /// exhausted-retry LLM failure.
    pub last_inbound_text: Option<String>,
    /// Runs from `execution_runs.jsonl` filtered by this terminal `chat_id` (newest first).
    pub executions_runs: Vec<ExecutionRunListItem>,
    pub executions_runs_error: Option<String>,
    pub executions_selected_idx: Option<usize>,
    /// Scroll within the run list (lines hidden above the viewport top).
    pub executions_list_scroll_top: usize,
    /// Lines skipped above the code pane viewport.
    pub executions_code_scroll_top: usize,
    /// Lines skipped above the stdout/stderr pane viewport.
    pub executions_output_scroll_top: usize,
    pub executions_detail: Option<ExecutionRunDetail>,
    pub executions_detail_error: Option<String>,
    /// Last frame's run-list block (wheel hit-test).
    pub last_executions_list_rect: Option<Rect>,
    pub last_executions_code_rect: Option<Rect>,
    pub last_executions_output_rect: Option<Rect>,
    /// Root terminal threads from `messages` (newest activity first).
    pub conversations_items: Vec<RootThreadListItem>,
    pub conversations_error: Option<String>,
    pub conversations_selected_idx: Option<usize>,
    /// Lines hidden above the past-sessions list viewport.
    pub conversations_list_scroll_top: usize,
    pub last_conversations_list_rect: Option<Rect>,
    pub draft_input: Option<String>,
    pub spinner_tick: usize,
    pub todos_count: usize,
    pub crons_count: usize,
    /// Active model selector popup (shown via `/model`).
    pub model_selector: Option<ModelSelector>,
    /// Active model name displayed in the status bar (updated on `/model` switch).
    pub status_model: String,
    /// Workspace label (usually sandbox basename) for the dense status header.
    pub status_workspace: String,
    /// Permission mode label (`ask`, `plan`, …).
    pub status_permission: String,
    /// Short session / chat id for the status header.
    pub status_session: String,
    /// Blocking approval overlay (shell / edit) awaiting a four-way reply.
    pub pending_approval: bool,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

pub enum BackgroundPanelItem<'a> {
    Notification(&'a crate::memory::NotificationRecord),
    Job(&'a crate::memory::BackgroundJobRecord),
    AgentTask(&'a crate::channels::terminal_ui::app::AgentTaskEntry),
}

impl<'a> BackgroundPanelItem<'a> {
    pub fn chat_id(&self) -> &'a str {
        match self {
            Self::Notification(n) => &n.chat_id,
            Self::Job(j) => &j.chat_id,
            Self::AgentTask(t) => &t.child_chat_id,
        }
    }
}

impl App {
    pub fn background_panel_items(&self) -> Vec<BackgroundPanelItem<'_>> {
        let mut items = Vec::new();
        for n in &self.notifications {
            items.push(BackgroundPanelItem::Notification(n));
        }
        for j in &self.background_jobs {
            items.push(BackgroundPanelItem::Job(j));
        }
        for a in &self.agent_tasks {
            items.push(BackgroundPanelItem::AgentTask(a));
        }
        items
    }

    pub fn new() -> Self {
        Self {
            mode: TerminalUiMode::default(),
            cells: Vec::new(),
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_idx: None,
            streaming_thinking: String::new(),
            streaming_assistant: String::new(),
            active_tool_line: None,
            ui_focus: TerminalUiFocus::default(),
            tool_rail: Vec::new(),
            tool_history_scroll: 0,
            tool_history_max_scroll: 0,
            last_tool_history_rect: None,
            scroll_offset: 0,
            max_scroll: 0,
            should_quit: false,
            thinking: false,
            last_transcript_rect: None,
            last_transcript_visible_start: 0,
            transcript_selection: None,
            selecting: false,
            toast: None,
            execution_stream_recent: String::new(),
            execution_stream_label: None,
            jobs_strip: VecDeque::new(),
            agent_tasks: VecDeque::new(),
            agent_tasks_scroll_top: 0,
            last_agent_tasks_rect: None,
            background_jobs: Vec::new(),
            notifications: Vec::new(),
            background_jobs_selected_idx: None,
            last_background_jobs_rect: None,
            llm_retry_available: false,
            last_inbound_text: None,
            executions_runs: Vec::new(),
            executions_runs_error: None,
            executions_selected_idx: None,
            executions_list_scroll_top: 0,
            executions_code_scroll_top: 0,
            executions_output_scroll_top: 0,
            executions_detail: None,
            executions_detail_error: None,
            last_executions_list_rect: None,
            last_executions_code_rect: None,
            last_executions_output_rect: None,
            conversations_items: Vec::new(),
            conversations_error: None,
            conversations_selected_idx: None,
            conversations_list_scroll_top: 0,
            last_conversations_list_rect: None,
            draft_input: None,
            spinner_tick: 0,
            todos_count: 0,
            crons_count: 0,
            model_selector: None,
            status_model: String::new(),
            status_workspace: String::new(),
            status_permission: String::new(),
            status_session: String::new(),
            pending_approval: false,
        }
    }

    pub fn set_toast(&mut self, kind: ToastKind, message: String, visible_for: Duration) {
        self.toast = Some(Toast {
            kind,
            message,
            until: Instant::now() + visible_for,
        });
    }

    pub fn clear_expired_toast(&mut self) {
        if let Some(t) = &self.toast {
            if Instant::now() >= t.until {
                self.toast = None;
            }
        }
    }

    /// Text and style for the status line when the toast is still active.
    pub fn active_toast(&self) -> Option<(&str, ToastKind)> {
        self.toast.as_ref().and_then(|t| {
            if Instant::now() < t.until {
                Some((t.message.as_str(), t.kind))
            } else {
                None
            }
        })
    }

    pub fn following_tail(&self) -> bool {
        self.scroll_offset == 0
    }

    fn reduce_motion_enabled() -> bool {
        matches!(
            std::env::var("ALTAI_TUI_REDUCE_MOTION").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("on")
        )
    }

    pub fn get_spinner_frame(&self) -> char {
        if Self::reduce_motion_enabled() {
            return '●';
        }
        const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        FRAMES[self.spinner_tick % FRAMES.len()]
    }

    pub fn toggle_ui_focus(&mut self) {
        self.ui_focus = match self.ui_focus {
            TerminalUiFocus::Transcript => TerminalUiFocus::Conversations,
            TerminalUiFocus::Conversations => TerminalUiFocus::Executions,
            TerminalUiFocus::Executions => TerminalUiFocus::ToolHistory,
            TerminalUiFocus::ToolHistory => TerminalUiFocus::AgentTasks,
            TerminalUiFocus::AgentTasks => TerminalUiFocus::BackgroundJobs,
            TerminalUiFocus::BackgroundJobs => TerminalUiFocus::Transcript,
        };
        self.tool_history_scroll = 0;
    }

    pub fn toggle_ui_focus_back(&mut self) {
        self.ui_focus = match self.ui_focus {
            TerminalUiFocus::Transcript => TerminalUiFocus::BackgroundJobs,
            TerminalUiFocus::BackgroundJobs => TerminalUiFocus::AgentTasks,
            TerminalUiFocus::AgentTasks => TerminalUiFocus::ToolHistory,
            TerminalUiFocus::ToolHistory => TerminalUiFocus::Executions,
            TerminalUiFocus::Executions => TerminalUiFocus::Conversations,
            TerminalUiFocus::Conversations => TerminalUiFocus::Transcript,
        };
        self.tool_history_scroll = 0;
    }

    pub fn focus_tool_history(&mut self) {
        self.ui_focus = TerminalUiFocus::ToolHistory;
        self.tool_history_scroll = 0;
    }

    pub fn focus_transcript(&mut self) {
        self.ui_focus = TerminalUiFocus::Transcript;
        self.tool_history_scroll = 0;
    }

    pub fn focus_agent_tasks(&mut self) {
        self.ui_focus = TerminalUiFocus::AgentTasks;
    }

    pub fn focus_background_jobs(&mut self) {
        self.ui_focus = TerminalUiFocus::BackgroundJobs;
        if self.background_jobs_selected_idx.is_none()
            && (!self.background_jobs.is_empty() || !self.notifications.is_empty())
        {
            self.background_jobs_selected_idx = Some(0);
        }
    }

    pub fn focus_executions(&mut self) {
        self.ui_focus = TerminalUiFocus::Executions;
        self.tool_history_scroll = 0;
    }

    pub fn focus_conversations(&mut self) {
        self.ui_focus = TerminalUiFocus::Conversations;
        self.tool_history_scroll = 0;
    }

    pub fn tool_history_following_tail(&self) -> bool {
        self.tool_history_scroll == 0
    }

    pub fn push_tool_rail(&mut self, entry: ToolRailEntry) {
        self.tool_rail.push(entry);
        while self.tool_rail.len() > TOOL_RAIL_CAP {
            self.tool_rail.remove(0);
        }
    }

    pub fn tool_history_scroll_up(&mut self, n: u16) {
        self.tool_history_scroll = self
            .tool_history_scroll
            .saturating_add(n)
            .min(self.tool_history_max_scroll);
    }

    pub fn tool_history_scroll_down(&mut self, n: u16) {
        self.tool_history_scroll = self.tool_history_scroll.saturating_sub(n);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }

    pub fn scroll_to_top(&mut self) {
        self.scroll_offset = self.max_scroll;
    }

    pub fn scroll_up(&mut self, n: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(n).min(self.max_scroll);
    }

    pub fn scroll_down(&mut self, n: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(n);
    }

    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.input[..self.cursor]
            .chars()
            .last()
            .map(|c| c.len_utf8())
            .unwrap_or(0);
        self.cursor -= prev;
        self.input.remove(self.cursor);
    }

    pub fn delete_forward(&mut self) {
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.input[..self.cursor]
            .chars()
            .last()
            .map(|c| c.len_utf8())
            .unwrap_or(0);
        self.cursor -= prev;
    }

    pub fn move_right(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let next = self.input[self.cursor..]
            .chars()
            .next()
            .map(|c| c.len_utf8())
            .unwrap_or(0);
        self.cursor += next;
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.input.len();
    }

    /// Byte offset of the previous word boundary before `self.cursor`.
    fn prev_word_boundary(&self) -> usize {
        if self.cursor == 0 {
            return 0;
        }
        let before = &self.input[..self.cursor];
        let trimmed = before.trim_end();
        trimmed
            .rfind(|c: char| c.is_whitespace() || c == '/')
            .map(|i| i + 1)
            .unwrap_or(0)
    }

    pub fn delete_word(&mut self) {
        let boundary = self.prev_word_boundary();
        if boundary < self.cursor {
            self.input.drain(boundary..self.cursor);
            self.cursor = boundary;
        }
    }

    pub fn move_left_word(&mut self) {
        self.cursor = self.prev_word_boundary();
    }

    pub fn move_right_word(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let after = &self.input[self.cursor..];
        let mut found_non_whitespace = false;
        let mut adv = 0;
        for c in after.chars() {
            let is_ws = c.is_whitespace() || c == '/';
            if found_non_whitespace && is_ws {
                break;
            }
            if !is_ws {
                found_non_whitespace = true;
            }
            adv += c.len_utf8();
        }
        self.cursor += adv;
    }

    /// Returns (line_index, display_col) for the current cursor position.
    pub fn cursor_line_col(&self) -> (usize, usize) {
        let before = &self.input[..self.cursor.min(self.input.len())];
        let lines: Vec<&str> = before.split('\n').collect();
        let line_idx = lines.len().saturating_sub(1);
        let col = super::display_width(lines.last().unwrap_or(&""));
        (line_idx, col)
    }

    pub fn clear_line(&mut self) {
        self.input.clear();
        self.cursor = 0;
    }

    /// Take the current input for submission; records history when non-empty.
    pub fn take_input(&mut self) -> String {
        let text = self.input.clone();
        if !text.trim().is_empty() {
            self.history.push(text.clone());
        }
        self.clear_line();
        self.history_idx = None;
        self.draft_input = None;
        text
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.history_idx {
            Some(0) => 0,
            Some(i) => i.saturating_sub(1),
            None => {
                self.draft_input = Some(self.input.clone());
                self.history.len() - 1
            }
        };
        self.history_idx = Some(idx);
        self.input = self.history[idx].clone();
        self.cursor = self.input.len();
    }

    pub fn history_down(&mut self) {
        match self.history_idx {
            Some(i) if i + 1 < self.history.len() => {
                let idx = i + 1;
                self.history_idx = Some(idx);
                self.input = self.history[idx].clone();
                self.cursor = self.input.len();
            }
            Some(_) => {
                self.history_idx = None;
                if let Some(draft) = self.draft_input.take() {
                    self.input = draft;
                } else {
                    self.clear_line();
                }
                self.cursor = self.input.len();
            }
            None => {}
        }
    }

    /// Move streaming buffers into `cells` as `Thinking` / `Assistant` rows.
    pub fn flush_streaming(&mut self) {
        if !self.streaming_thinking.is_empty() {
            self.cells.push(Cell::Thinking {
                text: std::mem::take(&mut self.streaming_thinking),
            });
        }
        if !self.streaming_assistant.is_empty() {
            self.cells.push(Cell::Assistant {
                markdown: std::mem::take(&mut self.streaming_assistant),
            });
        }
    }

    pub fn request_quit(&mut self) {
        self.should_quit = true;
    }

    /// Insert (or refresh) a Running row for `job_id` in the multi-job strip.
    pub fn job_strip_started(
        &mut self,
        job_id: &str,
        session_id: &str,
        tool_name: &str,
        description: Option<&str>,
        chat_id: &str,
    ) {
        if let Some(entry) = self.jobs_strip.iter_mut().find(|e| e.job_id == job_id) {
            entry.status = JobStripStatus::Running;
            entry.started_at = Instant::now();
            entry.terminal_at = None;
            entry.chat_id = chat_id.to_string();
            return;
        }
        self.jobs_strip.push_back(JobStripEntry {
            job_id: job_id.to_string(),
            session_id: session_id.to_string(),
            chat_id: chat_id.to_string(),
            tool_name: tool_name.to_string(),
            description: description.map(|s| s.to_string()),
            started_at: Instant::now(),
            status: JobStripStatus::Running,
            last_line: String::new(),
            terminal_at: None,
        });
        self.cap_jobs_strip();
    }

    pub fn job_strip_set_last_line(&mut self, job_id: &str, line: &str) {
        if let Some(existing) = self.jobs_strip.iter_mut().find(|e| e.job_id == job_id) {
            let trimmed = line.trim_end();
            if !trimmed.is_empty() {
                existing.last_line = trimmed.to_string();
            }
        }
    }

    pub fn job_strip_finished(&mut self, job_id: &str, status: &str, summary: &str, chat_id: &str) {
        let new_status = JobStripStatus::from_str(status);
        if let Some(existing) = self.jobs_strip.iter_mut().find(|e| e.job_id == job_id) {
            existing.status = new_status;
            existing.terminal_at = Some(Instant::now());
            let trimmed = summary.trim_end();
            if !trimmed.is_empty() {
                existing.last_line = trimmed.to_string();
            }
        } else {
            // Finished notice for an unknown job (rare): synthesize a row so the user sees the result.
            self.jobs_strip.push_back(JobStripEntry {
                job_id: job_id.to_string(),
                session_id: String::new(),
                chat_id: chat_id.to_string(),
                tool_name: String::new(),
                description: None,
                started_at: Instant::now(),
                status: new_status,
                last_line: summary.trim_end().to_string(),
                terminal_at: Some(Instant::now()),
            });
            self.cap_jobs_strip();
        }
    }

    /// Drop terminal rows older than `linger`, and any extras over `running_cap` running rows.
    pub fn evict_expired_jobs(&mut self, linger: Duration) {
        let now = Instant::now();
        self.jobs_strip.retain(|e| match e.terminal_at {
            Some(t) => now.saturating_duration_since(t) < linger,
            None => true,
        });
    }

    /// Cap the strip length so it cannot grow unbounded across long sessions.
    fn cap_jobs_strip(&mut self) {
        const MAX_ROWS: usize = 16;
        while self.jobs_strip.len() > MAX_ROWS {
            self.jobs_strip.pop_front();
        }
    }

    pub fn jobs_strip_has_running(&self) -> bool {
        self.jobs_strip
            .iter()
            .any(|e| e.status == JobStripStatus::Running)
    }

    /// Insert (or refresh) a Running row for `task_id` in the agent-tasks pane.
    pub fn agent_task_started(
        &mut self,
        task_id: &str,
        child_chat_id: &str,
        agent_name: Option<&str>,
        display_name: Option<&str>,
    ) {
        if let Some(existing) = self.agent_tasks.iter_mut().find(|e| e.task_id == task_id) {
            existing.status = AgentTaskStatus::Running;
            existing.terminal_at = None;
            existing.child_chat_id = child_chat_id.to_string();
            existing.agent_name = agent_name.map(|s| s.to_string());
            existing.display_name = display_name.map(|s| s.to_string());
            return;
        }
        self.agent_tasks.push_back(AgentTaskEntry {
            task_id: task_id.to_string(),
            child_chat_id: child_chat_id.to_string(),
            agent_name: agent_name.map(|s| s.to_string()),
            display_name: display_name.map(|s| s.to_string()),
            started_at: Instant::now(),
            status: AgentTaskStatus::Running,
            last_line: String::new(),
            terminal_at: None,
        });
        self.cap_agent_tasks();
    }

    /// Mark a task as finished with its terminal status and a short summary.
    pub fn agent_task_finished(&mut self, task_id: &str, status: &str, summary: &str) {
        let new_status = AgentTaskStatus::from_str(status);
        if let Some(existing) = self.agent_tasks.iter_mut().find(|e| e.task_id == task_id) {
            existing.status = new_status;
            existing.terminal_at = Some(Instant::now());
            let trimmed = summary.trim_end();
            if !trimmed.is_empty() {
                existing.last_line = trimmed.to_string();
            }
        } else {
            self.agent_tasks.push_back(AgentTaskEntry {
                task_id: task_id.to_string(),
                child_chat_id: String::new(),
                agent_name: None,
                display_name: None,
                started_at: Instant::now(),
                status: new_status,
                last_line: summary.trim_end().to_string(),
                terminal_at: Some(Instant::now()),
            });
            self.cap_agent_tasks();
        }
    }

    /// Drop terminal rows older than `linger`.
    pub fn evict_expired_agent_tasks(&mut self, linger: Duration) {
        let now = Instant::now();
        self.agent_tasks.retain(|e| match e.terminal_at {
            Some(t) => now.saturating_duration_since(t) < linger,
            None => true,
        });
    }

    fn cap_agent_tasks(&mut self) {
        const MAX_ROWS: usize = 32;
        while self.agent_tasks.len() > MAX_ROWS {
            self.agent_tasks.pop_front();
        }
    }

    pub fn agent_tasks_has_running(&self) -> bool {
        self.agent_tasks
            .iter()
            .any(|e| e.status == AgentTaskStatus::Running)
    }

    /// Insert or mutate a `Cell::ToolNotice` keyed by `tool_call_id`.
    ///
    /// - On `Pending` / `Call`: append a new cell (no in-place merge).
    /// - On `Result` / `Failed`: walk `cells` from the back and mutate the most recent
    ///   `ToolNotice` whose `tool_call_id` matches. If none is found (race / id missing),
    ///   append as a fresh cell — preserves visibility for orphan results.
    /// - When `tool_call_id` is `None`, always appends (legacy behavior).
    pub fn upsert_tool_notice(
        &mut self,
        tool_call_id: Option<String>,
        phase: ToolNoticePhase,
        content: String,
    ) {
        let id = match (tool_call_id.as_deref(), phase) {
            (Some(id), ToolNoticePhase::Result | ToolNoticePhase::Failed) if !id.is_empty() => id,
            _ => {
                self.cells.push(Cell::ToolNotice {
                    phase,
                    content,
                    tool_call_id,
                });
                return;
            }
        };

        for cell in self.cells.iter_mut().rev() {
            if let Cell::ToolNotice {
                phase: existing_phase,
                content: existing_content,
                tool_call_id: existing_id,
            } = cell
            {
                if existing_id.as_deref() == Some(id) {
                    let merged_content =
                        if matches!(phase, ToolNoticePhase::Result | ToolNoticePhase::Failed)
                            && matches!(
                                *existing_phase,
                                ToolNoticePhase::Pending | ToolNoticePhase::Call
                            )
                            && !existing_content.trim().is_empty()
                        {
                            let result_tail = content
                                .split_once("→")
                                .map(|(_, tail)| tail.trim())
                                .unwrap_or(content.trim());
                            if result_tail.is_empty() {
                                existing_content.clone()
                            } else {
                                format!("{} → {}", existing_content.trim(), result_tail)
                            }
                        } else {
                            content.clone()
                        };
                    *existing_phase = phase;
                    *existing_content = merged_content;
                    return;
                }
            }
        }

        self.cells.push(Cell::ToolNotice {
            phase,
            content,
            tool_call_id,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_clamps() {
        let mut app = App::new();
        app.max_scroll = 5;
        app.scroll_up(10);
        assert_eq!(app.scroll_offset, 5);
        app.scroll_down(3);
        assert_eq!(app.scroll_offset, 2);
        app.scroll_to_bottom();
        assert!(app.following_tail());
    }

    #[test]
    fn insert_and_backspace_utf8() {
        let mut app = App::new();
        app.insert_char('a');
        app.insert_char('β');
        assert_eq!(app.input, "aβ");
        app.end();
        app.backspace();
        assert_eq!(app.input, "a");
    }

    #[test]
    fn take_input_records_history() {
        let mut app = App::new();
        app.insert_char('x');
        assert_eq!(app.take_input(), "x");
        assert!(app.input.is_empty());
        app.history_up();
        assert_eq!(app.input, "x");
    }

    #[test]
    fn flush_streaming_creates_cells() {
        let mut app = App::new();
        app.streaming_thinking = "plan".into();
        app.streaming_assistant = "hello".into();
        app.flush_streaming();
        assert_eq!(
            app.cells,
            vec![
                Cell::Thinking {
                    text: "plan".into()
                },
                Cell::Assistant {
                    markdown: "hello".into()
                },
            ]
        );
    }

    #[test]
    fn display_width_counts_wide_chars() {
        assert!(super::super::display_width("你好") >= 4);
        assert_eq!(super::super::display_width("ab"), 2);
    }

    #[test]
    fn job_strip_started_inserts_running_row() {
        let mut app = App::new();
        app.job_strip_started(
            "job-1",
            "sess-1",
            "execution_run",
            Some("training"),
            "chat-1",
        );
        assert_eq!(app.jobs_strip.len(), 1);
        let row = app.jobs_strip.front().unwrap();
        assert_eq!(row.job_id, "job-1");
        assert_eq!(row.tool_name, "execution_run");
        assert_eq!(row.description.as_deref(), Some("training"));
        assert_eq!(row.status, JobStripStatus::Running);
        assert!(app.jobs_strip_has_running());
    }

    #[test]
    fn job_strip_finished_marks_terminal_status() {
        let mut app = App::new();
        app.job_strip_started("job-1", "sess-1", "execution_run", None, "chat-1");
        app.job_strip_set_last_line("job-1", "epoch 4 / 8");
        app.job_strip_finished("job-1", "completed", "exit 0 in 1234ms", "chat-1");
        let row = app.jobs_strip.front().unwrap();
        assert_eq!(row.status, JobStripStatus::Completed);
        assert!(row.terminal_at.is_some());
        assert_eq!(row.last_line, "exit 0 in 1234ms");
    }

    #[test]
    fn evict_expired_jobs_drops_only_old_terminal_rows() {
        let mut app = App::new();
        app.job_strip_started("running", "s1", "execution_run", None, "chat-1");
        app.job_strip_started("done", "s1", "execution_run", None, "chat-1");
        app.job_strip_finished("done", "completed", "ok", "chat-1");
        if let Some(row) = app.jobs_strip.iter_mut().find(|e| e.job_id == "done") {
            row.terminal_at = Some(Instant::now() - Duration::from_secs(60));
        }
        app.evict_expired_jobs(Duration::from_secs(10));
        assert_eq!(app.jobs_strip.len(), 1);
        assert_eq!(app.jobs_strip.front().unwrap().job_id, "running");
    }

    #[test]
    fn upsert_tool_notice_appends_pending_then_mutates_to_result() {
        let mut app = App::new();
        app.upsert_tool_notice(
            Some("call-1".into()),
            ToolNoticePhase::Pending,
            "execution_run epoch=1".into(),
        );
        assert_eq!(app.cells.len(), 1);
        match &app.cells[0] {
            Cell::ToolNotice {
                phase,
                tool_call_id,
                ..
            } => {
                assert_eq!(*phase, ToolNoticePhase::Pending);
                assert_eq!(tool_call_id.as_deref(), Some("call-1"));
            }
            _ => panic!("expected ToolNotice"),
        }

        app.upsert_tool_notice(
            Some("call-1".into()),
            ToolNoticePhase::Result,
            "execution_run → exit 0".into(),
        );
        assert_eq!(app.cells.len(), 1, "result should mutate in place");
        match &app.cells[0] {
            Cell::ToolNotice { phase, content, .. } => {
                assert_eq!(*phase, ToolNoticePhase::Result);
                assert!(content.contains("exit 0"));
            }
            _ => panic!("expected ToolNotice"),
        }
    }

    #[test]
    fn upsert_tool_notice_mutates_to_failed_for_matching_id() {
        let mut app = App::new();
        app.upsert_tool_notice(
            Some("call-2".into()),
            ToolNoticePhase::Pending,
            "tool x".into(),
        );
        app.upsert_tool_notice(
            Some("call-2".into()),
            ToolNoticePhase::Failed,
            "tool x → boom".into(),
        );
        assert_eq!(app.cells.len(), 1);
        match &app.cells[0] {
            Cell::ToolNotice { phase, .. } => assert_eq!(*phase, ToolNoticePhase::Failed),
            _ => panic!("expected ToolNotice"),
        }
    }

    #[test]
    fn upsert_tool_notice_appends_orphan_result_when_no_pending_match() {
        let mut app = App::new();
        // No prior Pending cell with this id — orphan result should still surface.
        app.upsert_tool_notice(
            Some("ghost".into()),
            ToolNoticePhase::Result,
            "ghost-tool → late ack".into(),
        );
        assert_eq!(app.cells.len(), 1);
    }

    #[test]
    fn upsert_tool_notice_uses_most_recent_match_when_id_repeats() {
        // Defensive: if the same id were ever recycled (e.g. retry path), the back-walk should
        // resolve to the most recent Pending cell, not the older completed one.
        let mut app = App::new();
        app.upsert_tool_notice(Some("dup".into()), ToolNoticePhase::Pending, "first".into());
        app.upsert_tool_notice(
            Some("dup".into()),
            ToolNoticePhase::Result,
            "first done".into(),
        );
        app.upsert_tool_notice(
            Some("dup".into()),
            ToolNoticePhase::Pending,
            "second".into(),
        );
        app.upsert_tool_notice(
            Some("dup".into()),
            ToolNoticePhase::Failed,
            "second failed".into(),
        );
        assert_eq!(app.cells.len(), 2);
        match &app.cells[1] {
            Cell::ToolNotice { phase, content, .. } => {
                assert_eq!(*phase, ToolNoticePhase::Failed);
                assert_eq!(content, "second → second failed");
            }
            _ => panic!("expected ToolNotice"),
        }
        match &app.cells[0] {
            Cell::ToolNotice { phase, content, .. } => {
                assert_eq!(*phase, ToolNoticePhase::Result);
                assert_eq!(content, "first → first done");
            }
            _ => panic!("expected ToolNotice"),
        }
    }

    #[test]
    fn upsert_tool_notice_without_id_always_appends() {
        let mut app = App::new();
        app.upsert_tool_notice(None, ToolNoticePhase::Call, "a".into());
        app.upsert_tool_notice(None, ToolNoticePhase::Result, "b".into());
        assert_eq!(app.cells.len(), 2);
    }

    #[test]
    fn job_strip_status_from_str_maps_known_states() {
        assert_eq!(
            JobStripStatus::from_str("completed"),
            JobStripStatus::Completed
        );
        assert_eq!(JobStripStatus::from_str("failed"), JobStripStatus::Failed);
        assert_eq!(
            JobStripStatus::from_str("cancelled"),
            JobStripStatus::Cancelled
        );
        assert_eq!(JobStripStatus::from_str("timeout"), JobStripStatus::Timeout);
        assert_eq!(
            JobStripStatus::from_str("anything-else"),
            JobStripStatus::Running
        );
    }
}
