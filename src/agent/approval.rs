//! Audit X9: shell/code-exec approval policy, split out of the former
//! `agent/mod.rs` god-file.
//!
//! Contents: destructive-command matching, four-way reply classification,
//! per-session policy-mode resolution, code-exec tool gating, and the
//! process-scoped "always for this run" grant registry.

use std::collections::HashSet;

use serde_json::Value;

use crate::config::{ResolvedShellPolicy, ShellPolicyMode};

/// Lowercase and collapse every run of whitespace (spaces, tabs, newlines) to a single space so a
/// destructive command can't slip past a single-spaced approval pattern via `rm  -rf`, a tab, or a
/// mid-command line break.
fn normalize_command_for_matching(s: &str) -> String {
    let lowercase = s.to_ascii_lowercase();
    let mut result = String::with_capacity(lowercase.len());
    let mut words = lowercase.split_whitespace();
    if let Some(first) = words.next() {
        result.push_str(first);
        for word in words {
            result.push(' ');
            result.push_str(word);
        }
    }
    result
}

pub(crate) fn should_require_shell_approval(command: &str, patterns: &[String]) -> bool {
    // Pad with spaces so matching is on whole-word boundaries: a bare `.contains()` on the
    // normalized command would let a pattern like `rm` match any command containing `terminal`,
    // `platform`, `firmware`, `alarm`, `warm`, `harm`, etc. ("terminal".contains("rm") is true),
    // forcing spurious approval prompts. Padding both sides keeps the whitespace-robustness (a
    // run of spaces/tabs/newlines was already collapsed to one) while only matching real tokens.
    let normalized = format!(" {} ", normalize_command_for_matching(command));
    patterns.iter().any(|p| {
        let np = normalize_command_for_matching(p);
        // Ignore empty/whitespace-only patterns: `contains("")` is always true and would otherwise
        // force approval on *every* command — a silent config footgun.
        if np.is_empty() {
            return false;
        }
        normalized.contains(&format!(" {np} "))
    })
}

/// Process-scoped "always for this run" grants (`shell:…` / `edit:…`).
fn process_approval_grants() -> &'static tokio::sync::Mutex<HashSet<String>> {
    static GRANTS: std::sync::OnceLock<tokio::sync::Mutex<HashSet<String>>> =
        std::sync::OnceLock::new();
    GRANTS.get_or_init(|| tokio::sync::Mutex::new(HashSet::new()))
}

pub(crate) async fn approval_already_granted(key: &str) -> bool {
    process_approval_grants().lock().await.contains(key)
}

pub(crate) async fn remember_approval_grant(key: String) {
    process_approval_grants().lock().await.insert(key);
}

/// Parse a user's reply to a shell-approval prompt. **Deny by default**: the command runs only when
/// the reply is composed *entirely* of affirmative or neutral-filler words AND carries at least one
/// explicit affirmative. Any unrecognized token — a negation ("never", "nope", "can't"), a caveat,
/// or stray prose — forces a deny.
///
/// This allowlist posture (rather than a denylist) is deliberate: it fixes the original
/// `contains("approve") && !contains("deny")` parse that read "do not approve" as APPROVED, and it
/// also closes the broader class a denylist misses, where an affirmative word is buried in a
/// negative sentence ("never approve", "i can't approve", "approve? actually nope"). The prompt
/// constrains the choices to approve/deny with `allow_empty = false`, so the strictness is
/// UX-compatible; an unrecognized reply simply skips execution and the user can re-confirm.
#[cfg(test)]
fn shell_approval_reply_is_grant(reply: &str) -> bool {
    matches!(
        classify_approval_reply(reply),
        ApprovalReply::Grant | ApprovalReply::AlwaysThisRun
    )
}

/// Four-way approval reply classification for ALTAI CLI / TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApprovalReply {
    Grant,
    AlwaysThisRun,
    Deny,
    Abort,
}

pub(crate) fn classify_approval_reply(reply: &str) -> ApprovalReply {
    const AFFIRM: &[&str] = &[
        "approve",
        "approved",
        "approves",
        "yes",
        "yep",
        "yeah",
        "y",
        "ok",
        "okay",
        "k",
        "allow",
        "allowed",
        "confirm",
        "confirmed",
        "accept",
        "accepted",
        "proceed",
        "go",
        "sure",
    ];
    const FILLER: &[&str] = &[
        "please", "it", "this", "that", "ahead", "now", "run", "do", "for",
    ];
    const ALWAYS: &[&str] = &["always"];
    const ABORT: &[&str] = &["abort", "cancel", "quit", "stop"];

    let r = reply.trim().to_ascii_lowercase();
    if r.is_empty() {
        return ApprovalReply::Deny;
    }

    let tokens: Vec<&str> = r
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();

    if tokens.iter().any(|t| ABORT.contains(t))
        && !tokens
            .iter()
            .any(|t| AFFIRM.contains(t) || ALWAYS.contains(t))
    {
        return ApprovalReply::Abort;
    }

    // "always" / "always for this run" — allow only known filler around always.
    if tokens.iter().any(|t| ALWAYS.contains(t)) {
        let ok = tokens
            .iter()
            .all(|t| ALWAYS.contains(t) || FILLER.contains(t) || AFFIRM.contains(t));
        if ok {
            return ApprovalReply::AlwaysThisRun;
        }
        return ApprovalReply::Deny;
    }

    let mut saw_affirmative = false;
    for tok in &tokens {
        if AFFIRM.contains(tok) {
            saw_affirmative = true;
        } else if !FILLER.contains(tok) {
            return ApprovalReply::Deny;
        }
    }
    if saw_affirmative {
        ApprovalReply::Grant
    } else {
        ApprovalReply::Deny
    }
}

pub(crate) fn command_preview_with_flag(command: &str) -> (String, bool) {
    const MAX_PREVIEW: usize = 160;
    if command.len() <= MAX_PREVIEW {
        (command.to_string(), false)
    } else {
        (format!("{}…", &command[..MAX_PREVIEW]), true)
    }
}

pub(crate) fn command_preview(command: &str) -> String {
    command_preview_with_flag(command).0
}

pub(crate) fn shell_policy_mode_for_session(
    policy: &ResolvedShellPolicy,
    unattended_session: bool,
) -> ShellPolicyMode {
    if unattended_session {
        policy.unattended_mode
    } else {
        policy.interactive_mode
    }
}

pub(crate) fn edit_policy_mode_for_session(
    policy: &ResolvedShellPolicy,
    unattended_session: bool,
) -> ShellPolicyMode {
    if unattended_session {
        policy.unattended_edit_mode
    } else {
        policy.interactive_edit_mode
    }
}

/// Reason shown when the edit policy blocks a mutation. Unattended sessions
/// default to Deny independently of plan mode, so the message distinguishes the
/// two cases rather than hardcoding "plan mode active" everywhere (PR #62 review #2).
pub(crate) fn edit_policy_block_reason(unattended_session: bool) -> &'static str {
    if unattended_session {
        "File edit blocked by policy: unattended edit mode is active."
    } else {
        "File edit blocked by policy: plan mode active — finalize or apply the plan first."
    }
}

/// Tools that execute model-authored code/commands on the host or a session. All of these run
/// arbitrary code, so they share the shell-policy approval gate — not just `exec`. Keying the
/// gate on this category (rather than the literal name `"exec"`) is what stops `execution_run`
/// / `execution_run_background` from bypassing approval entirely.
pub(crate) fn is_code_exec_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "exec" | "exec_send" | "execution_run" | "execution_run_background"
    )
}

/// Tools that mutate a workspace file and therefore need the edit policy gate.
pub(crate) fn is_file_mutate_tool(tool_name: &str) -> bool {
    matches!(tool_name, "write_file" | "edit_file")
}

/// Code-exec tools that run *arbitrary* code (Python source / session cells) where the
/// destructive-shell-pattern heuristic does not meaningfully apply, so any such call is
/// treated as approval-worthy in ask/deny mode.
fn is_arbitrary_code_tool(tool_name: &str) -> bool {
    matches!(tool_name, "execution_run" | "execution_run_background")
}

/// Extract the command/code a code-exec tool will run. `exec` carries it in `command`; `exec_send`
/// carries it in `input`; execution tools carry it in `code`.
pub(crate) fn extract_code_exec_command(tool_name: &str, args: &Value) -> Option<String> {
    let key = match tool_name {
        "exec" => "command",
        "exec_send" => "input",
        _ => "code",
    };
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Whether a code-exec call needs approval in ask/deny mode. Arbitrary-code tools always do;
/// shell `exec` only when the command matches a destructive pattern (preserves existing UX).
pub(crate) fn code_exec_requires_approval(
    tool_name: &str,
    command: &str,
    patterns: &[String],
) -> bool {
    is_arbitrary_code_tool(tool_name) || should_require_shell_approval(command, patterns)
}

#[cfg(test)]
mod code_exec_gate_tests {
    use super::*;
    use crate::agent::tool_dispatch::append_post_tool_output;
    use crate::traits::{ToolErrorCode, ToolResult};
    use serde_json::json;

    #[test]
    fn category_covers_all_code_exec_tools() {
        assert!(is_code_exec_tool("exec"));
        assert!(is_code_exec_tool("execution_run"));
        assert!(is_code_exec_tool("execution_run_background"));
        assert!(!is_code_exec_tool("read_file"));
        assert!(!is_code_exec_tool("web_search"));
    }

    #[test]
    fn file_mutate_category_covers_write_and_edit_only() {
        assert!(is_file_mutate_tool("write_file"));
        assert!(is_file_mutate_tool("edit_file"));
        assert!(!is_file_mutate_tool("read_file"));
        assert!(!is_file_mutate_tool("exec"));
        assert!(!is_file_mutate_tool("list_dir"));
        assert!(!is_file_mutate_tool("search_text"));
    }

    #[test]
    fn edit_block_reason_distinguishes_unattended_and_plan_mode() {
        // PR #62 review #2: the Deny message must match why the edit was blocked.
        let unattended = edit_policy_block_reason(true);
        assert!(unattended.contains("unattended"), "{unattended}");
        assert!(!unattended.contains("plan mode"), "{unattended}");
        let plan = edit_policy_block_reason(false);
        assert!(plan.contains("plan mode"), "{plan}");
    }

    #[test]
    fn extracts_command_for_exec_and_code_for_execution_tools() {
        assert_eq!(
            extract_code_exec_command("exec", &json!({"command": " ls -la "})).as_deref(),
            Some("ls -la")
        );
        assert_eq!(
            extract_code_exec_command("execution_run", &json!({"code": "print(1)"})).as_deref(),
            Some("print(1)")
        );
        // wrong key / empty -> None
        assert!(extract_code_exec_command("execution_run", &json!({"command": "x"})).is_none());
        assert!(extract_code_exec_command("exec", &json!({"command": "  "})).is_none());
    }

    #[test]
    fn arbitrary_code_always_requires_approval_benign_exec_does_not() {
        let patterns = vec!["rm -rf".to_string()];
        // Arbitrary-code tools: even benign code requires approval (closes the bypass).
        assert!(code_exec_requires_approval(
            "execution_run",
            "print('hi')",
            &patterns
        ));
        // Shell `exec`: benign command does NOT require approval (preserves existing UX)...
        assert!(!code_exec_requires_approval("exec", "ls -la", &patterns));
        // ...but a destructive one does.
        assert!(code_exec_requires_approval(
            "exec",
            "rm -rf /tmp/x",
            &patterns
        ));
    }

    #[test]
    fn approval_match_is_whitespace_insensitive() {
        let patterns = vec!["rm -rf".to_string()];
        // Extra spaces, tabs, and a mid-command newline must not bypass the gate.
        assert!(should_require_shell_approval("rm  -rf /tmp/x", &patterns));
        assert!(should_require_shell_approval("rm\t-rf /tmp/x", &patterns));
        assert!(should_require_shell_approval("rm\n-rf /tmp/x", &patterns));
        assert!(should_require_shell_approval("RM -RF /tmp/x", &patterns));
        // A benign command still does not match.
        assert!(!should_require_shell_approval("ls -la", &patterns));
    }

    #[test]
    fn approval_match_is_word_boundary_not_substring() {
        // A bare `rm` pattern must match the real command but NOT words that merely contain "rm"
        // (the substring false positive: "terminal".contains("rm") is true).
        let patterns = vec!["rm".to_string()];
        assert!(should_require_shell_approval("rm -rf /tmp/x", &patterns));
        assert!(should_require_shell_approval(
            "echo hi && rm file",
            &patterns
        ));
        assert!(!should_require_shell_approval("terminal --help", &patterns));
        assert!(!should_require_shell_approval(
            "warm up the cache",
            &patterns
        ));
        assert!(!should_require_shell_approval(
            "npm run platform",
            &patterns
        ));
        assert!(!should_require_shell_approval("check firmware", &patterns));
    }

    #[test]
    fn empty_pattern_does_not_force_approval_on_everything() {
        // `contains("")` is always true; an empty/whitespace pattern must be ignored.
        let patterns = vec!["".to_string(), "   ".to_string()];
        assert!(!should_require_shell_approval("ls -la", &patterns));
        // An empty pattern alongside a real one must not suppress the real match.
        let mixed = vec!["".to_string(), "rm -rf".to_string()];
        assert!(should_require_shell_approval("rm -rf /tmp/x", &mixed));
        assert!(!should_require_shell_approval("ls", &mixed));
    }

    #[test]
    fn approval_reply_is_deny_default() {
        // The regression: "do not approve" must NOT grant.
        assert!(!shell_approval_reply_is_grant("do not approve"));
        assert!(!shell_approval_reply_is_grant("don't approve"));
        assert!(!shell_approval_reply_is_grant("deny"));
        assert!(!shell_approval_reply_is_grant("no"));
        assert!(!shell_approval_reply_is_grant("reject this"));
        assert!(!shell_approval_reply_is_grant(""));
        assert!(!shell_approval_reply_is_grant("   "));
        // Ambiguous / unrelated text is denied (safe default).
        assert!(!shell_approval_reply_is_grant("hmm let me think"));
        // Affirmative word buried in a negative reply must NOT grant (the denylist gap).
        assert!(!shell_approval_reply_is_grant("never approve"));
        assert!(!shell_approval_reply_is_grant("i can't approve"));
        assert!(!shell_approval_reply_is_grant("approve? actually nope"));
        assert!(!shell_approval_reply_is_grant("disapprove"));
        assert!(!shell_approval_reply_is_grant("i approve... no wait, deny"));
        // Explicit grants (incl. affirmative + neutral filler).
        assert!(shell_approval_reply_is_grant("approve"));
        assert!(shell_approval_reply_is_grant("Approved"));
        assert!(shell_approval_reply_is_grant("yes"));
        assert!(shell_approval_reply_is_grant("ok"));
        assert!(shell_approval_reply_is_grant("allow"));
        assert!(shell_approval_reply_is_grant("approve please"));
        assert!(shell_approval_reply_is_grant("approve this"));
        assert!(shell_approval_reply_is_grant("go ahead"));
        // "do" is neutral filler: it rescues genuine affirmatives ("yes do it") without weakening
        // deny-default — a negation always carries another non-filler token (see "do not approve"
        // above), and "do" on its own is not affirmative.
        assert!(shell_approval_reply_is_grant("yes do it"));
        assert!(shell_approval_reply_is_grant("yes, please do"));
        assert!(!shell_approval_reply_is_grant("do it"));
    }

    #[test]
    fn approval_reply_classifies_always_and_abort() {
        assert_eq!(
            classify_approval_reply("always"),
            ApprovalReply::AlwaysThisRun
        );
        assert_eq!(
            classify_approval_reply("always for this run"),
            ApprovalReply::AlwaysThisRun
        );
        assert_eq!(classify_approval_reply("abort"), ApprovalReply::Abort);
        assert_eq!(classify_approval_reply("cancel"), ApprovalReply::Abort);
        assert_eq!(classify_approval_reply("deny"), ApprovalReply::Deny);
        assert!(shell_approval_reply_is_grant("always"));
        assert!(!shell_approval_reply_is_grant("abort"));
    }

    #[test]
    fn append_post_tool_output_preserves_polarity() {
        // Appends to a success, preserving its typed status.
        let ok = append_post_tool_output(ToolResult::success("applied"), "tests passed");
        assert_eq!(ok.content, "applied\n\n[post-tool hook]\ntests passed");
        assert!(!ok.is_error());
        // Appends to an error without replacing its typed root cause.
        let err = append_post_tool_output(
            ToolResult::error(ToolErrorCode::ExecutionFailed, "boom"),
            "lint output",
        );
        assert_eq!(err.content, "Error: boom\n\n[post-tool hook]\nlint output");
        assert_eq!(err.error_code(), Some(ToolErrorCode::ExecutionFailed));
    }
}
