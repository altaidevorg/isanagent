#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};

use ratatui::style::{Color, Modifier, Style};

static USE_ANSI_COLOR: AtomicBool = AtomicBool::new(true);

/// Read [`NO_COLOR`](https://no-color.org/) and store whether to emit ANSI colors. Call once from the Ratatui entry before the first draw.
pub fn init_from_env() {
    let allow = !matches!(
        std::env::var_os("NO_COLOR"),
        Some(s) if !s.is_empty()
    );
    init(allow);
}

/// Set the color capability chosen by an embedding host for this TUI session.
pub fn init(allow: bool) {
    USE_ANSI_COLOR.store(allow, Ordering::Relaxed);
}

#[inline]
fn ansi_color() -> bool {
    USE_ANSI_COLOR.load(Ordering::Relaxed)
}

/// Whether the TUI will emit ANSI foreground colors (false after [`init_from_env`] when `NO_COLOR` is set).
#[inline]
pub fn uses_ansi_color() -> bool {
    ansi_color()
}

#[inline]
fn fg(c: Color) -> Style {
    if ansi_color() {
        Style::default().fg(c)
    } else {
        Style::default()
    }
}

#[inline]
fn fg_mod(c: Color, m: Modifier) -> Style {
    if ansi_color() {
        Style::default().fg(c).add_modifier(m)
    } else {
        Style::default().add_modifier(m)
    }
}

/// Default palette for the TUI (tuned for dark terminals). Honors `NO_COLOR` (no foreground colors; modifiers kept for structure).
#[derive(Debug, Clone, Copy)]
pub struct Theme;

impl Theme {
    pub fn text() -> Style {
        fg(Color::White)
    }

    pub fn dim() -> Style {
        fg(Color::DarkGray)
    }

    pub fn user_prefix() -> Style {
        fg_mod(Color::Cyan, Modifier::BOLD)
    }

    pub fn assistant_bullet() -> Style {
        fg(Color::DarkGray)
    }

    pub fn thinking() -> Style {
        fg_mod(Color::DarkGray, Modifier::ITALIC)
    }

    pub fn tool_call() -> Style {
        fg(Color::Yellow)
    }

    /// Yellow + dim/italic, used for in-flight `Cell::ToolNotice` cells whose result
    /// has not yet arrived. Distinguishes "waiting" from a finished green/red result.
    pub fn tool_pending() -> Style {
        fg_mod(Color::Yellow, Modifier::DIM | Modifier::ITALIC)
    }

    pub fn tool_done() -> Style {
        fg(Color::Green)
    }

    pub fn clarification() -> Style {
        fg(Color::Magenta)
    }

    pub fn error() -> Style {
        fg(Color::Red)
    }

    pub fn status_bar() -> Style {
        fg(Color::Gray)
    }

    pub fn input_prompt() -> Style {
        fg_mod(Color::Green, Modifier::BOLD)
    }

    /// Highlight style for mouse-selected text in the transcript.
    pub fn selection() -> Style {
        Style::default().add_modifier(Modifier::REVERSED)
    }
}
