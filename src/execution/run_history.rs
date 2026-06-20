//! Durable per-run journal under workspace `.system_generated/execution_history/` (P1).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::ids::SessionId;
use super::run::RunResult;

const MAX_INLINE_TEXT: usize = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunJournal {
    pub schema_version: u32,
    pub provider_id: String,
    pub session_id: String,
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jupyter_kernel_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jupyter_notebook_path: Option<String>,
    pub started_rfc3339: String,
    pub finished_rfc3339: String,
    pub duration_ms: u64,
    pub exit_code: Option<i32>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub stdout: String,
    pub stderr: String,
    pub attachments: Vec<AttachmentJournal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentJournal {
    pub id: String,
    pub path: Option<String>,
    pub mime_hint: Option<String>,
}

fn truncate_field(s: String, max: usize) -> (String, bool) {
    if s.len() <= max {
        (s, false)
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        let mut t = s[..end].to_string();
        t.push_str("\n... (truncated in run.json; full output in tool result)");
        (t, true)
    }
}

/// `workspace_dir/.system_generated/execution_history/{provider}/{session_seg}/{run_id}/`
pub fn run_history_dir(
    workspace_dir: &Path,
    provider_id: &str,
    session_id: &SessionId,
    run_id: &str,
) -> PathBuf {
    let session_seg = super::artifacts::sanitize_session_segment(session_id);
    workspace_dir
        .join(".system_generated")
        .join("execution_history")
        .join(provider_id)
        .join(&session_seg)
        .join(run_id)
}

/// Inputs for [`write_run_journal`].
#[derive(Debug, Clone, Copy)]
pub struct RunJournalParams<'a> {
    pub workspace_dir: &'a Path,
    pub provider_id: &'a str,
    pub session_id: &'a SessionId,
    pub run_id: &'a str,
    pub code: &'a str,
    pub result: &'a RunResult,
    pub jupyter_kernel_id: Option<&'a str>,
    pub jupyter_notebook_path: Option<&'a str>,
    pub started_rfc3339: &'a str,
    pub finished_rfc3339: &'a str,
    pub duration_ms: u64,
}

/// Writes `run.json` and `source.txt`; returns journal directory.
pub async fn write_run_journal(p: RunJournalParams<'_>) -> Result<PathBuf, String> {
    let dir = run_history_dir(p.workspace_dir, p.provider_id, p.session_id, p.run_id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("run journal mkdir {}: {e}", dir.display()))?;

    // Redact secrets before they hit disk: executed code and captured stdout/stderr routinely carry
    // keys (an `env` dump, an echoed `$OPENAI_API_KEY`, a token printed by a script).
    let redactor = crate::redact::shared();
    let code = redactor.redact(p.code);
    tokio::fs::write(dir.join("source.txt"), code.as_bytes())
        .await
        .map_err(|e| format!("run journal source write: {e}"))?;

    let (stdout, stdout_truncated) = truncate_field(
        redactor.redact(&p.result.stdout).into_owned(),
        MAX_INLINE_TEXT,
    );
    let (stderr, stderr_truncated) = truncate_field(
        redactor.redact(&p.result.stderr).into_owned(),
        MAX_INLINE_TEXT,
    );

    let journal = RunJournal {
        schema_version: 1,
        provider_id: p.provider_id.to_string(),
        session_id: p.session_id.to_string(),
        run_id: p.run_id.to_string(),
        jupyter_kernel_id: p.jupyter_kernel_id.map(str::to_string),
        jupyter_notebook_path: p.jupyter_notebook_path.map(str::to_string),
        started_rfc3339: p.started_rfc3339.to_string(),
        finished_rfc3339: p.finished_rfc3339.to_string(),
        duration_ms: p.duration_ms,
        exit_code: p.result.exit_code,
        stdout_truncated,
        stderr_truncated,
        stdout,
        stderr,
        attachments: p
            .result
            .attachments
            .iter()
            .map(|a| AttachmentJournal {
                id: a.id.clone(),
                path: a.path.clone(),
                mime_hint: a.mime_hint.clone(),
            })
            .collect(),
    };
    let json = serde_json::to_string_pretty(&journal).map_err(|e| e.to_string())?;
    tokio::fs::write(dir.join("run.json"), json.as_bytes())
        .await
        .map_err(|e| format!("run journal run.json: {e}"))?;

    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn run_journal_redacts_secrets_in_code_stdout_stderr() {
        // Use format-identifiable secrets so the test does not depend on the process environment.
        let ws =
            std::env::temp_dir().join(format!("isanagent_runjournal_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&ws).unwrap();
        let sid = SessionId::new("sess1");
        let result = RunResult::new(
            "leaked AKIAIOSFODNN7EXAMPLE in stdout",
            "trace ghp_0123456789abcdefghijABCDEFGHIJ here",
            Some(0),
        );

        write_run_journal(RunJournalParams {
            workspace_dir: &ws,
            provider_id: "local",
            session_id: &sid,
            run_id: "run1",
            code: "print('sk_live_0123456789abcdefghij')",
            result: &result,
            jupyter_kernel_id: None,
            jupyter_notebook_path: None,
            started_rfc3339: "t0",
            finished_rfc3339: "t1",
            duration_ms: 1,
        })
        .await
        .unwrap();

        let dir = run_history_dir(&ws, "local", &sid, "run1");
        let source = std::fs::read_to_string(dir.join("source.txt")).unwrap();
        let run_json = std::fs::read_to_string(dir.join("run.json")).unwrap();

        // Executed code (source.txt)
        assert!(source.contains("[REDACTED_STRIPE_KEY]"), "source: {source}");
        assert!(!source.contains("sk_live_0123456789"), "source: {source}");
        // stdout/stderr (run.json)
        assert!(
            run_json.contains("[REDACTED_AWS_KEY]"),
            "run.json: {run_json}"
        );
        assert!(
            !run_json.contains("AKIAIOSFODNN7EXAMPLE"),
            "run.json: {run_json}"
        );
        assert!(
            run_json.contains("[REDACTED_GITHUB_TOKEN]"),
            "run.json: {run_json}"
        );
        assert!(!run_json.contains("ghp_0123456789"), "run.json: {run_json}");

        let _ = std::fs::remove_dir_all(&ws);
    }
}
