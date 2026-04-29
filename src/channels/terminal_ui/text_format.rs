/// Truncate `s` to at most `max` display columns; appends `…` when shortened (`…` uses one column).
pub fn truncate_chars_display(s: &str, max: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    if max == 0 {
        return String::new();
    }
    if super::display_width(s) <= max {
        return s.to_string();
    }
    if max == 1 {
        return "…".to_string();
    }
    let budget = max.saturating_sub(1);
    let mut out = String::new();
    let mut col = 0usize;
    for ch in s.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        if col + cw > budget {
            break;
        }
        out.push(ch);
        col += cw;
    }
    if out.is_empty() {
        "…".to_string()
    } else {
        out.push('…');
        out
    }
}
