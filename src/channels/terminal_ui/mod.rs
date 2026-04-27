//! Ratatui-based terminal transcript (`run`) plus shared input helpers (`attachments`).
//! Cell model and theme follow a Xerxes-style layout (labels, rail, scroll).

#![allow(dead_code, unused_imports)]

mod app;
mod attachments;
mod execution_browser;
mod markdown;
pub(crate) mod protocol;
mod run;
mod syntect_highlight;
mod theme;

pub use app::{
    App, Cell, JobStripEntry, JobStripStatus, TerminalUiFocus, TerminalUiMode, ToastKind,
    ToolNoticePhase, ToolRailEntry,
};
pub use theme::{init_from_env, uses_ansi_color, Theme};

pub(crate) use attachments::parse_terminal_attachments;
pub(crate) use run::{run_ratatui_main, RatatuiMainConfig};

use unicode_width::UnicodeWidthStr;

/// Display width of `s` in terminal columns (used by future wrapping / cursor alignment).
pub fn display_width(s: &str) -> usize {
    s.width()
}
