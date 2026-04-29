use chrono::{DateTime, Local};

/// Compact local time for past-session rows (`last_activity_ms` from SQLite `created_at`).
pub fn format_last_activity(last_activity_ms: i64) -> String {
    if last_activity_ms <= 0 {
        return "—".to_string();
    }
    let Some(utc) = DateTime::from_timestamp_millis(last_activity_ms) else {
        return "—".to_string();
    };
    let local = utc.with_timezone(&Local);
    local.format("%Y-%m-%d %H:%M").to_string()
}

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
