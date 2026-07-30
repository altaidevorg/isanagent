#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use ratatui::style::{Color, Modifier, Style};

static USE_ANSI_COLOR: AtomicBool = AtomicBool::new(true);
/// 0 = dark, 1 = light, 2 = no-color (structure only).
static APPEARANCE: AtomicU8 = AtomicU8::new(0);

/// Host-selected theme mode before auto / NO_COLOR resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HostThemeMode {
    #[default]
    Auto,
    Dark,
    Light,
    NoColor,
}

/// Effective appearance applied to the TUI for this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeAppearance {
    Dark,
    Light,
    NoColor,
}

impl ThemeAppearance {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Light,
            2 => Self::NoColor,
            _ => Self::Dark,
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            Self::Dark => 0,
            Self::Light => 1,
            Self::NoColor => 2,
        }
    }
}

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
    if !allow {
        APPEARANCE.store(ThemeAppearance::NoColor.as_u8(), Ordering::Relaxed);
    }
}

/// Apply a resolved ALTAI appearance (dark / light / no-color).
pub fn init_appearance(appearance: ThemeAppearance) {
    APPEARANCE.store(appearance.as_u8(), Ordering::Relaxed);
    USE_ANSI_COLOR.store(appearance != ThemeAppearance::NoColor, Ordering::Relaxed);
}

/// Resolve host theme + NO_COLOR into an appearance and apply it.
pub fn init_from_host(theme: HostThemeMode, no_color: bool) {
    let appearance = resolve_host_appearance(theme, no_color);
    init_appearance(appearance);
}

/// Resolve without mutating global theme state (for tests and adapters).
pub fn resolve_host_appearance(theme: HostThemeMode, no_color: bool) -> ThemeAppearance {
    if no_color || theme == HostThemeMode::NoColor {
        return ThemeAppearance::NoColor;
    }
    match theme {
        HostThemeMode::Dark => ThemeAppearance::Dark,
        HostThemeMode::Light => ThemeAppearance::Light,
        HostThemeMode::NoColor => ThemeAppearance::NoColor,
        HostThemeMode::Auto => detect_auto_appearance(),
    }
}

fn detect_auto_appearance() -> ThemeAppearance {
    if let Ok(raw) = std::env::var("COLORFGBG") {
        if let Some(bg) = raw
            .split(';')
            .nth(1)
            .and_then(|part| part.trim().parse::<u8>().ok())
        {
            return if bg >= 8 {
                ThemeAppearance::Light
            } else {
                ThemeAppearance::Dark
            };
        }
    }
    ThemeAppearance::Dark
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
pub fn current_appearance() -> ThemeAppearance {
    ThemeAppearance::from_u8(APPEARANCE.load(Ordering::Relaxed))
}

/// Truecolor roles derived from ALTAI App `globals.css` OKLCH tokens.
#[derive(Debug, Clone, Copy)]
struct Palette {
    text: Color,
    muted: Color,
    active: Color,
    warning: Color,
    success: Color,
    info: Color,
    error: Color,
    focus: Color,
}

const DARK: Palette = Palette {
    text: Color::Rgb(240, 242, 244),
    muted: Color::Rgb(137, 140, 146),
    active: Color::Rgb(181, 234, 38),
    warning: Color::Rgb(245, 174, 57),
    success: Color::Rgb(81, 198, 114),
    info: Color::Rgb(75, 174, 237),
    error: Color::Rgb(248, 75, 75),
    focus: Color::Rgb(93, 114, 149),
};

const LIGHT: Palette = Palette {
    text: Color::Rgb(19, 22, 28),
    muted: Color::Rgb(81, 85, 92),
    active: Color::Rgb(154, 211, 53),
    warning: Color::Rgb(201, 105, 0),
    success: Color::Rgb(0, 127, 53),
    info: Color::Rgb(0, 106, 175),
    error: Color::Rgb(212, 9, 36),
    focus: Color::Rgb(79, 100, 134),
};

fn palette() -> Option<&'static Palette> {
    match current_appearance() {
        ThemeAppearance::Dark => Some(&DARK),
        ThemeAppearance::Light => Some(&LIGHT),
        ThemeAppearance::NoColor => None,
    }
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

/// Default palette for the TUI. Honors host theme + `NO_COLOR`.
#[derive(Debug, Clone, Copy)]
pub struct Theme;

impl Theme {
    pub fn text() -> Style {
        match palette() {
            Some(p) => fg(p.text),
            None => Style::default(),
        }
    }

    pub fn dim() -> Style {
        match palette() {
            Some(p) => fg(p.muted),
            None => Style::default().add_modifier(Modifier::DIM),
        }
    }

    pub fn user_prefix() -> Style {
        match palette() {
            Some(p) => fg_mod(p.info, Modifier::BOLD),
            None => Style::default().add_modifier(Modifier::BOLD),
        }
    }

    pub fn assistant_bullet() -> Style {
        Theme::dim()
    }

    pub fn thinking() -> Style {
        match palette() {
            Some(p) => fg_mod(p.muted, Modifier::ITALIC),
            None => Style::default().add_modifier(Modifier::ITALIC | Modifier::DIM),
        }
    }

    pub fn tool_call() -> Style {
        match palette() {
            Some(p) => fg(p.warning),
            None => Style::default().add_modifier(Modifier::BOLD),
        }
    }

    /// Yellow + dim/italic, used for in-flight `Cell::ToolNotice` cells whose result
    /// has not yet arrived. Distinguishes "waiting" from a finished green/red result.
    pub fn tool_pending() -> Style {
        match palette() {
            Some(p) => fg_mod(p.warning, Modifier::DIM | Modifier::ITALIC),
            None => Style::default().add_modifier(Modifier::DIM | Modifier::ITALIC),
        }
    }

    pub fn tool_done() -> Style {
        match palette() {
            Some(p) => fg(p.success),
            None => Style::default().add_modifier(Modifier::BOLD),
        }
    }

    pub fn clarification() -> Style {
        match palette() {
            Some(p) => fg(p.focus),
            None => Style::default().add_modifier(Modifier::BOLD),
        }
    }

    pub fn error() -> Style {
        match palette() {
            Some(p) => fg(p.error),
            None => Style::default().add_modifier(Modifier::BOLD),
        }
    }

    pub fn status_bar() -> Style {
        Theme::dim()
    }

    /// Lime/active accent — focused control, brand mark, progress.
    pub fn active() -> Style {
        match palette() {
            Some(p) => fg_mod(p.active, Modifier::BOLD),
            None => Style::default().add_modifier(Modifier::BOLD),
        }
    }

    pub fn input_prompt() -> Style {
        Theme::active()
    }

    /// Highlight style for mouse-selected text in the transcript.
    pub fn selection() -> Style {
        Style::default().add_modifier(Modifier::REVERSED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_color_disables_ansi() {
        assert_eq!(
            resolve_host_appearance(HostThemeMode::Dark, true),
            ThemeAppearance::NoColor
        );
        assert_eq!(
            resolve_host_appearance(HostThemeMode::NoColor, false),
            ThemeAppearance::NoColor
        );
    }

    #[test]
    fn light_theme_sets_appearance() {
        assert_eq!(
            resolve_host_appearance(HostThemeMode::Light, false),
            ThemeAppearance::Light
        );
        assert_eq!(
            resolve_host_appearance(HostThemeMode::Dark, false),
            ThemeAppearance::Dark
        );
    }

    #[test]
    fn resolve_respects_explicit_modes() {
        assert_eq!(
            resolve_host_appearance(HostThemeMode::Dark, false),
            ThemeAppearance::Dark
        );
        assert_eq!(
            resolve_host_appearance(HostThemeMode::Light, false),
            ThemeAppearance::Light
        );
        assert_eq!(
            resolve_host_appearance(HostThemeMode::Auto, true),
            ThemeAppearance::NoColor
        );
    }

    #[test]
    fn init_from_host_applies_no_color() {
        init_from_host(HostThemeMode::Dark, true);
        assert!(!uses_ansi_color());
        assert_eq!(current_appearance(), ThemeAppearance::NoColor);
        // Restore a colored dark default for other single-threaded callers.
        init_from_host(HostThemeMode::Dark, false);
        assert!(uses_ansi_color());
    }
}
