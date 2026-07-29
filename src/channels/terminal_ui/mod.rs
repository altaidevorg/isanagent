//! Ratatui-based terminal transcript (`run`) plus shared input helpers (`attachments`).
//! Cell model and theme follow a Xerxes-style layout (labels, rail, scroll).

#![allow(dead_code, unused_imports)]

mod app;
pub(crate) mod approval;
mod attachments;
mod execution_browser;
mod history_cells;
mod markdown;
pub mod panes;
pub(crate) mod protocol;
mod run;
mod syntect_highlight;
mod text_format;
mod theme;

pub use app::{
    AgentTaskEntry, AgentTaskStatus, App, Cell, JobStripEntry, JobStripStatus, ModelSelector,
    TerminalUiFocus, TerminalUiMode, ToastKind, ToolNoticePhase, ToolRailEntry,
    TranscriptSelection,
};
pub use approval::{approval_hotkey_reply, EditDiffPayload, APPROVAL_CHOICES};
pub use attachments::{
    load_host_file_attachments, load_sandbox_file_attachment, parse_terminal_attachments,
};
pub use theme::{
    current_appearance, init, init_appearance, init_from_env, init_from_host,
    resolve_host_appearance, uses_ansi_color, HostThemeMode, Theme, ThemeAppearance,
};

pub(crate) use run::{run_ratatui_main, RatatuiMainConfig};

use unicode_width::UnicodeWidthStr;

/// Display width of `s` in terminal columns (used by future wrapping / cursor alignment).
pub fn display_width(s: &str) -> usize {
    s.width()
}
