//! Token-highlight source text for the executions pane (`syntect` + Ratatui `Line`s).

use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

use super::uses_ansi_color;
use super::Theme;

const MAX_HIGHLIGHT_SOURCE_CHARS: usize = 256 * 1024;

static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
static THEME_SET: OnceLock<ThemeSet> = OnceLock::new();

fn syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn theme_set() -> &'static ThemeSet {
    THEME_SET.get_or_init(ThemeSet::load_defaults)
}

fn pick_theme() -> &'static syntect::highlighting::Theme {
    const NAMES: &[&str] = &[
        "Solarized (dark)",
        "base16-ocean.dark",
        "InspiredGitHub",
        "base16-eighties.dark",
    ];
    let ts = theme_set();
    for n in NAMES {
        if let Some(t) = ts.themes.get(*n) {
            return t;
        }
    }
    ts.themes.values().next().expect("syntect default themes")
}

fn syntax_for_hint<'a>(ps: &'a SyntaxSet, hint: &str) -> &'a syntect::parsing::SyntaxReference {
    let h = hint.trim().trim_start_matches('.');
    if !h.is_empty() {
        if let Some(s) = ps.find_syntax_by_extension(h) {
            return s;
        }
        if let Some(s) = ps.find_syntax_by_name(h) {
            return s;
        }
    }
    if let Some(s) = ps.find_syntax_by_extension("py") {
        return s;
    }
    ps.find_syntax_plain_text()
}

/// Infer a syntax-highlight grammar token from the first line (shebang) or fall back to `py`.
pub fn grammar_hint_from_source(source: &str) -> &'static str {
    let first = source.lines().next().unwrap_or("").trim();
    if let Some(rest) = first.strip_prefix("#!") {
        let rest = rest.trim();
        if rest.contains("python") {
            return "py";
        }
        if rest.contains("bash") || rest.contains("sh") {
            return "sh";
        }
        if rest.contains("node") {
            return "js";
        }
    }
    "py"
}

fn syntect_style_to_ratatui(st: syntect::highlighting::Style) -> Style {
    if !uses_ansi_color() {
        return Theme::dim().add_modifier(Modifier::ITALIC);
    }
    let mut out = Style::default().fg(Color::Rgb(
        st.foreground.r,
        st.foreground.g,
        st.foreground.b,
    ));
    let fs = st.font_style;
    if fs.contains(FontStyle::BOLD) {
        out = out.add_modifier(Modifier::BOLD);
    }
    if fs.contains(FontStyle::ITALIC) {
        out = out.add_modifier(Modifier::ITALIC);
    }
    if fs.contains(FontStyle::UNDERLINE) {
        out = out.add_modifier(Modifier::UNDERLINED);
    }
    out
}

/// Greedy wrap by display width for a sequence of styled pieces.
fn wrap_styled_line(pieces: Vec<(Style, String)>, width: usize) -> Vec<Line<'static>> {
    if width < 4 {
        return vec![Line::from(
            pieces
                .into_iter()
                .map(|(st, s)| Span::styled(s, st))
                .collect::<Vec<_>>(),
        )];
    }
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut current: Vec<Span<'static>> = Vec::new();
    let mut col = 0usize;

    fn flush(out: &mut Vec<Line<'static>>, current: &mut Vec<Span<'static>>) {
        if current.is_empty() {
            return;
        }
        out.push(Line::from(std::mem::take(current)));
    }

    for (style, text) in pieces {
        for ch in text.chars() {
            let w = unicode_width::UnicodeWidthChar::width(ch)
                .unwrap_or(0)
                .max(1);
            if col + w > width && !current.is_empty() {
                flush(&mut out, &mut current);
                col = 0;
            }
            if col == 0 && ch.is_whitespace() {
                continue;
            }
            if let Some(last) = current.last_mut() {
                if last.style == style {
                    last.content = format!("{}{}", last.content, ch).into();
                } else {
                    current.push(Span::styled(ch.to_string(), style));
                }
            } else {
                current.push(Span::styled(ch.to_string(), style));
            }
            col += w;
        }
    }
    flush(&mut out, &mut current);
    if out.is_empty() {
        out.push(Line::default());
    }
    out
}

/// Highlight `source` into wrapped Ratatui lines; on failure or oversized input, dim monospace lines.
pub fn highlight_source_wrapped(source: &str, width: usize) -> Vec<Line<'static>> {
    let w = width.max(8);
    let (source, truncated) = if source.chars().count() > MAX_HIGHLIGHT_SOURCE_CHARS {
        let end = source
            .char_indices()
            .nth(MAX_HIGHLIGHT_SOURCE_CHARS)
            .map(|(i, _)| i)
            .unwrap_or(source.len());
        (&source[..end], true)
    } else {
        (source, false)
    };

    let ps = syntax_set();
    let theme = pick_theme();
    let hint = grammar_hint_from_source(source);
    let syntax = syntax_for_hint(ps, hint);
    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut lines: Vec<Line<'static>> = Vec::new();

    for line in LinesWithEndings::from(source) {
        let regions = match highlighter.highlight_line(line, ps) {
            Ok(r) => r,
            Err(_) => {
                return fallback_dim_lines(source, w, truncated);
            }
        };
        let mut pieces: Vec<(Style, String)> = Vec::new();
        for (st, text) in regions {
            let rs = syntect_style_to_ratatui(st);
            if let Some((last_st, last_s)) = pieces.last_mut() {
                if *last_st == rs {
                    last_s.push_str(text);
                } else {
                    pieces.push((rs, text.to_string()));
                }
            } else {
                pieces.push((rs, text.to_string()));
            }
        }
        lines.extend(wrap_styled_line(pieces, w));
    }

    if truncated {
        lines.push(Line::from(Span::styled(
            "… (source truncated for highlighting)",
            Theme::tool_pending(),
        )));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(" ", Theme::dim())));
    }
    lines
}

fn fallback_dim_lines(source: &str, width: usize, truncated: bool) -> Vec<Line<'static>> {
    use crate::channels::terminal_ui::panes::wrap_text;
    let mut lines: Vec<Line<'static>> = wrap_text(source, width)
        .into_iter()
        .map(|s| Line::from(Span::styled(s, Theme::dim())))
        .collect();
    if truncated {
        lines.push(Line::from(Span::styled(
            "… (source truncated for highlighting)",
            Theme::tool_pending(),
        )));
    }
    lines
}
