#![allow(dead_code)]

use std::time::{Duration, Instant};

use ratatui::layout::Rect;

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
    ToolHistory,
}

const TOOL_RAIL_CAP: usize = 150;

/// One line in the tool-activity ring buffer (call / result / fail).
#[derive(Debug, Clone)]
pub struct ToolRailEntry {
    pub tool_name: String,
    pub phase: ToolNoticePhase,
    pub summary: String,
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
    },
    Clarification {
        text: String,
        /// From `ask_user` when provided; shown as a numbered list in the terminal.
        choices: Vec<String>,
    },
    System {
        message: String,
    },
    /// Agent / provider failure surfaced in-session (not only logs).
    Error {
        message: String,
    },
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
    /// LLM loop active for this session (cleared on assistant reply, error, or clarification).
    pub thinking: bool,
    /// Last drawn transcript widget area (for mouse wheel hit-testing).
    pub last_transcript_rect: Option<Rect>,
    /// Short-lived message shown in the status strip (not stored in the transcript).
    pub toast: Option<Toast>,
    /// Latest `execution_run` stream (Jupyter); shown in a dedicated strip below the transcript.
    pub execution_stream_recent: String,
    pub execution_stream_label: Option<(String, String)>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
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
            toast: None,
            execution_stream_recent: String::new(),
            execution_stream_label: None,
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

    pub fn toggle_ui_focus(&mut self) {
        self.ui_focus = match self.ui_focus {
            TerminalUiFocus::Transcript => TerminalUiFocus::ToolHistory,
            TerminalUiFocus::ToolHistory => TerminalUiFocus::Transcript,
        };
        self.tool_history_scroll = 0;
    }

    pub fn focus_tool_history(&mut self) {
        self.ui_focus = TerminalUiFocus::ToolHistory;
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

    pub fn delete_word(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let before = &self.input[..self.cursor];
        let trimmed = before.trim_end();
        let new_end = trimmed
            .rfind(|c: char| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        self.input.drain(new_end..self.cursor);
        self.cursor = new_end;
    }

    pub fn clear_line(&mut self) {
        self.input.clear();
        self.cursor = 0;
    }

    /// Take the current input for submission; records history when non-empty.
    pub fn take_input(&mut self) -> String {
        let text = self.input.clone();
        if !text.is_empty() {
            self.history.push(text.clone());
        }
        self.clear_line();
        self.history_idx = None;
        text
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.history_idx {
            Some(0) => 0,
            Some(i) => i.saturating_sub(1),
            None => self.history.len() - 1,
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
                self.clear_line();
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
}
