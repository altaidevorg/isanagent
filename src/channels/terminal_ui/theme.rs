#![allow(dead_code)]

use ratatui::style::{Color, Modifier, Style};

/// Default palette for the TUI (tuned for dark terminals; respects plain `Style` fallbacks).
#[derive(Debug, Clone, Copy)]
pub struct Theme;

impl Theme {
    pub fn text() -> Style {
        Style::default().fg(Color::White)
    }

    pub fn dim() -> Style {
        Style::default().fg(Color::DarkGray)
    }

    pub fn user_prefix() -> Style {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    }

    pub fn assistant_bullet() -> Style {
        Style::default().fg(Color::DarkGray)
    }

    pub fn thinking() -> Style {
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC)
    }

    pub fn tool_call() -> Style {
        Style::default().fg(Color::Yellow)
    }

    pub fn tool_done() -> Style {
        Style::default().fg(Color::Green)
    }

    pub fn clarification() -> Style {
        Style::default().fg(Color::Magenta)
    }

    pub fn error() -> Style {
        Style::default().fg(Color::Red)
    }

    pub fn status_bar() -> Style {
        Style::default().fg(Color::Gray)
    }

    pub fn input_prompt() -> Style {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    }
}
