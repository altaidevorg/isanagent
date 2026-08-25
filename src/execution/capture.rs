//! Audit X6b: single canonical output-truncation primitive for all execution
//! providers. Replaces the divergent schemes (byte-cap halves in `local`,
//! capture-time halves plus a redundant second truncation in `ssh`, and a
//! byte-slicing appender in `jupyter`) with one type that enforces a combined
//! stdout+stderr byte budget on UTF-8 character boundaries.

use super::repl_framing::truncate_utf8_str_cap;

/// Bounded sink pair for one run's output sharing one byte budget.
///
/// Semantics (single source of truth):
/// - the **combined** captured output never exceeds `max_total_bytes`
///   (plus truncation markers),
/// - every append is cut on a UTF-8 character boundary (never panics),
/// - a cut stream ends once with `\n... (truncated)`.
pub struct OutputCapture {
    stdout: String,
    stderr: String,
    remaining: usize,
}

fn push_bounded(buf: &mut String, remaining: &mut usize, chunk: &str) {
    if *remaining == 0 || chunk.is_empty() {
        return;
    }
    let appended = truncate_utf8_str_cap(chunk, *remaining);
    *remaining = remaining.saturating_sub(appended.len());
    buf.push_str(&appended);
}

impl OutputCapture {
    /// Wrap already-captured strings, applying the shared-budget policy post hoc
    /// (used where a provider streams internally before it can bound).
    pub fn from_captured(stdout: String, stderr: String, max_total_bytes: usize) -> Self {
        let mut capture = Self::empty(max_total_bytes);
        capture.push_stdout(&stdout);
        capture.push_stderr(&stderr);
        capture
    }

    fn empty(max_total_bytes: usize) -> Self {
        Self {
            stdout: String::new(),
            stderr: String::new(),
            remaining: max_total_bytes,
        }
    }

    pub fn push_stdout(&mut self, chunk: &str) {
        push_bounded(&mut self.stdout, &mut self.remaining, chunk);
    }

    pub fn push_stderr(&mut self, chunk: &str) {
        push_bounded(&mut self.stderr, &mut self.remaining, chunk);
    }

    pub fn into_parts(self) -> (String, String) {
        (self.stdout, self.stderr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_budget_bounds_combined_output_on_char_boundaries() {
        // Regression for the audit finding: jupyter's old byte-slice appender
        // could panic cutting inside a multi-byte character.
        let mut capture = OutputCapture::from_captured(String::new(), String::new(), 16);
        capture.push_stdout("aaaaaaaaaabbb");
        capture.push_stderr("üüüüüü"); // 10 bytes of 2-byte chars
        let (stdout, stderr) = capture.into_parts();
        assert_eq!(stdout, "aaaaaaaaaabbb");
        assert!(stderr.starts_with("ü"));
        assert!(stderr.ends_with("\n... (truncated)"));
        assert!(stdout.len() + stderr.len() <= 16 + "\n... (truncated)".len());
    }

    #[test]
    fn exact_fit_never_marks_truncation() {
        let mut capture = OutputCapture::from_captured(String::new(), String::new(), 5);
        capture.push_stdout("abcde");
        assert_eq!(capture.into_parts().0, "abcde");

        let mut capture = OutputCapture::from_captured(String::new(), String::new(), 5);
        capture.push_stdout("abcde");
        capture.push_stderr("x");
        assert_eq!(capture.into_parts().1, "");
    }

    #[test]
    fn from_captured_applies_shared_budget_to_existing_strings() {
        let capture = OutputCapture::from_captured("long-stdout-value".into(), "err".into(), 12);
        let (stdout, stderr) = capture.into_parts();
        assert!(stdout.starts_with("long-stdout"));
        assert!(stdout.ends_with("\n... (truncated)"));
        // Shared budget: the first stream consumed the whole allowance.
        assert_eq!(stderr, "");
    }
}
