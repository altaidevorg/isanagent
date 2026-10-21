//! Audit X9: shell/code-exec approval policy, split out of the former
//! `agent/mod.rs` god-file.
//!
//! Contents: destructive-command matching, four-way reply classification,
//! per-session policy-mode resolution, code-exec tool gating, and the
//! process-scoped "always for this run" grant registry.

use std::collections::HashSet;

use serde_json::Value;

use crate::clarification::ASK_USER_REPLY_PREFIX;
use crate::config::{ResolvedToolPermissions, ShellPolicyMode};
use crate::traits::{PathAccess, PathAccessKind};
use crate::utils::path_starts_with;
use std::path::{Path, PathBuf};

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

    // Approval prompts go through `ask_user`, whose result wraps the answer for the model.
    let reply = reply.strip_prefix(ASK_USER_REPLY_PREFIX).unwrap_or(reply);
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

/// Reason shown when the legacy edit ceiling (`[harness.shell_policy].edit_mode` /
/// `edit_unattended_default`) blocks a mutation. Plan mode sets `edit_mode = "deny"`; the
/// unattended ceiling is independent of it, so the message distinguishes the two cases rather
/// than hardcoding "plan mode active" everywhere (PR #62 review #2).
pub(crate) fn edit_policy_block_reason(unattended_session: bool) -> &'static str {
    if unattended_session {
        "File edit blocked by policy: unattended edit mode is active."
    } else {
        "File edit blocked by policy: plan mode active — finalize or apply the plan first."
    }
}

/// What `[harness.permissions]` decides for one call, before any prompt is shown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PermissionGate {
    /// A `[harness.permissions.tools]` entry; it replaces the location and default rules.
    /// Never produced for code-exec tools (their entry sets the shell-policy mode instead).
    /// `ceiling_denied` is true when only the legacy edit ceiling turned the entry into a deny.
    ToolOverride {
        mode: ShellPolicyMode,
        ceiling_denied: bool,
    },
    /// A file tool: the strictest location rule over the paths the call touches. `access` is
    /// the path that decided it; `ceiling_denied` is true when only the legacy edit ceiling
    /// (`edit_mode` / plan mode, `edit_unattended_default`) turned it into a deny.
    Location {
        mode: ShellPolicyMode,
        access: PathAccess,
        ceiling_denied: bool,
    },
    /// A file tool (one that reports its paths) reported none for this call. It is not let
    /// through under `default`: the stricter of the inside and outside rules applies.
    UnreportedPaths {
        mode: ShellPolicyMode,
        ceiling_denied: bool,
    },
    /// Any other tool: `[harness.permissions].default`.
    Default(ShellPolicyMode),
    /// Not gated here: `ask_user` (the approval channel itself), code-exec tools without a
    /// working directory (the shell policy gates them), and calls whose paths could not be
    /// resolved (the tool reports that error without touching anything).
    Exempt,
}

/// Classify one call against `[harness.permissions]`. `accesses` is `None` when the tool could
/// not resolve its arguments; `reports_paths` is the tool's [`crate::traits::Tool::reports_path_accesses`].
pub(crate) fn permission_gate(
    perms: &ResolvedToolPermissions,
    tool_name: &str,
    accesses: Option<&[PathAccess]>,
    reports_paths: bool,
    unattended: bool,
) -> PermissionGate {
    if tool_name == "ask_user" {
        return PermissionGate::Exempt;
    }
    let is_edit = is_file_mutate_tool(tool_name);
    if let Some(mode) = perms.rule_override(tool_name, is_edit, unattended) {
        return PermissionGate::ToolOverride {
            mode,
            ceiling_denied: mode == ShellPolicyMode::Deny
                && perms.rule_override(tool_name, false, unattended) != Some(ShellPolicyMode::Deny),
        };
    }
    let Some(accesses) = accesses else {
        return PermissionGate::Exempt;
    };
    if accesses.is_empty() && reports_paths {
        let strictest = |edit: bool| {
            perms
                .location_mode(true, edit, unattended)
                .stricter(perms.location_mode(false, edit, unattended))
        };
        let mode = strictest(is_edit);
        return PermissionGate::UnreportedPaths {
            mode,
            ceiling_denied: mode == ShellPolicyMode::Deny
                && strictest(false) != ShellPolicyMode::Deny,
        };
    }
    let mut decided: Option<(ShellPolicyMode, &PathAccess)> = None;
    for access in accesses {
        let edit = is_edit && access.kind == PathAccessKind::Write;
        let mode = perms.location_mode(access.inside_sandbox, edit, unattended);
        let replace = match decided {
            None => true,
            Some((current, current_access)) => {
                let stricter = mode.stricter(current);
                (stricter == mode && stricter != current)
                    || (mode == current && current_access.inside_sandbox && !access.inside_sandbox)
            }
        };
        if replace {
            decided = Some((mode, access));
        }
    }
    match decided {
        Some((mode, access)) => PermissionGate::Location {
            mode,
            ceiling_denied: mode == ShellPolicyMode::Deny
                && perms.location_mode(access.inside_sandbox, false, unattended)
                    != ShellPolicyMode::Deny,
            access: access.clone(),
        },
        None if is_code_exec_tool(tool_name) => PermissionGate::Exempt,
        None => PermissionGate::Default(perms.default_mode_for(tool_name, unattended)),
    }
}

/// Path matching for grants follows the platform's default filesystem case rules.
const GRANT_CASE_INSENSITIVE: bool = cfg!(any(windows, target_os = "macos"));

/// An "always" answer to a location prompt (or an outside-edit prompt).
///
/// Outside the sandbox the grant covers the same operation kind (read / write+edit / working
/// directory, kept separate) for the target's directory and everything below it: the parent
/// directory for a file, the directory itself for a directory target. Inside the sandbox it covers
/// only the exact path. A directory grant never widens onto a protected path (the outer
/// workspace's `config.toml` / `.system_generated/`) unless the granted directory itself lies
/// inside that protected path; the exact target is always covered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PathGrant {
    kind: PathAccessKind,
    exact: PathBuf,
    exact_is_dir: bool,
    scope: Option<PathBuf>,
    /// The scope contains a protected path (shown as an exception in the prompt).
    shielded: bool,
}

impl PathGrant {
    /// The grant an "always" answer to `access` stores. A folder that would be a filesystem
    /// root, the user's home directory, or an ancestor of `workspace_dir` is too broad: the grant
    /// then falls back to the exact path.
    pub(crate) fn for_access(perms: &ResolvedToolPermissions, access: &PathAccess) -> Self {
        let exact_is_dir = access.kind == PathAccessKind::WorkingDir || access.path.is_dir();
        let same =
            |a: &Path, b: &Path| path_starts_with(a, b, true) && path_starts_with(b, a, true);
        let scope = (!access.inside_sandbox)
            .then(|| {
                if exact_is_dir {
                    access.path.clone()
                } else {
                    access
                        .path
                        .parent()
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|| access.path.clone())
                }
            })
            .filter(|scope| {
                let is_root = scope.parent().is_none();
                let is_home = perms
                    .grant_home_dir
                    .as_deref()
                    .is_some_and(|home| same(scope, home));
                let above_workspace = perms
                    .grant_workspace_dir
                    .as_deref()
                    .is_some_and(|ws| path_starts_with(ws, scope, true) && !same(ws, scope));
                !(is_root || is_home || above_workspace)
            });
        let shielded = scope.as_deref().is_some_and(|scope| {
            perms
                .protected_paths
                .iter()
                .any(|guarded| path_starts_with(guarded, scope, true))
        });
        Self {
            kind: access.kind,
            exact: access.path.clone(),
            exact_is_dir,
            scope,
            shielded,
        }
    }

    pub(crate) fn covers(&self, access: &PathAccess, protected: &[PathBuf]) -> bool {
        self.covers_with(access, protected, GRANT_CASE_INSENSITIVE)
    }

    fn covers_with(
        &self,
        access: &PathAccess,
        protected: &[PathBuf],
        case_insensitive: bool,
    ) -> bool {
        if access.kind != self.kind {
            return false;
        }
        let same = |a: &Path, b: &Path| {
            path_starts_with(a, b, case_insensitive) && path_starts_with(b, a, case_insensitive)
        };
        if same(&access.path, &self.exact) {
            return true;
        }
        let Some(scope) = self.scope.as_deref() else {
            return false;
        };
        if !path_starts_with(&access.path, scope, case_insensitive) {
            return false;
        }
        // Shield checks always ignore case: erring toward a fresh prompt is the safe side.
        !protected.iter().any(|guarded| {
            path_starts_with(&access.path, guarded, true) && !path_starts_with(scope, guarded, true)
        })
    }

    /// What answering "always" will allow, for the prompt text.
    pub(crate) fn describe(&self) -> String {
        let what = match self.kind {
            PathAccessKind::Read => "reads",
            PathAccessKind::Write => "writes and edits",
            PathAccessKind::WorkingDir => "running commands",
        };
        match &self.scope {
            Some(scope) => format!(
                "always = allow {what} under `{}`{} until isanagent restarts",
                scope.display(),
                if self.shielded {
                    " (except isanagent config and data)"
                } else {
                    ""
                }
            ),
            None => format!(
                "always = allow {what} of this {} only (`{}`) until isanagent restarts",
                if self.exact_is_dir { "folder" } else { "file" },
                self.exact.display()
            ),
        }
    }
}

fn process_path_grants() -> &'static tokio::sync::Mutex<Vec<PathGrant>> {
    static GRANTS: std::sync::OnceLock<tokio::sync::Mutex<Vec<PathGrant>>> =
        std::sync::OnceLock::new();
    GRANTS.get_or_init(|| tokio::sync::Mutex::new(Vec::new()))
}

pub(crate) async fn path_already_granted(access: &PathAccess, protected: &[PathBuf]) -> bool {
    process_path_grants()
        .lock()
        .await
        .iter()
        .any(|grant| grant.covers(access, protected))
}

pub(crate) async fn remember_path_grant(grant: PathGrant) {
    let mut grants = process_path_grants().lock().await;
    if !grants.contains(&grant) {
        grants.push(grant);
    }
}

pub(crate) use crate::config::is_code_exec_tool;

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

    fn no_limits() -> ResolvedToolPermissions {
        ResolvedToolPermissions::default()
    }

    /// Home `/home/u`, outer workspace `/home/u/.isanagent`.
    fn limits() -> ResolvedToolPermissions {
        ResolvedToolPermissions {
            protected_paths: vec![
                PathBuf::from("/home/u/.isanagent/config.toml"),
                PathBuf::from("/home/u/.isanagent/.system_generated"),
            ],
            grant_home_dir: Some(PathBuf::from("/home/u")),
            grant_workspace_dir: Some(PathBuf::from("/home/u/.isanagent")),
            ..ResolvedToolPermissions::default()
        }
    }

    #[test]
    fn too_broad_folders_fall_back_to_the_exact_path() {
        let p = limits();
        let exact_only = |path: &str, kind| {
            let grant = PathGrant::for_access(&p, &access(path, kind));
            assert!(grant.covers_with(&access(path, kind), &p.protected_paths, false));
            grant
        };
        // A file directly under `/`, a `/` working directory, a file in the home directory.
        for (path, kind) in [
            ("/x.txt", PathAccessKind::Read),
            ("/", PathAccessKind::WorkingDir),
            ("/home/u/notes.md", PathAccessKind::Read),
            ("/home/u", PathAccessKind::WorkingDir),
        ] {
            let grant = exact_only(path, kind);
            assert!(!grant.covers_with(
                &access("/home/u/other.md", kind),
                &p.protected_paths,
                false
            ));
            assert!(grant.describe().contains("only"), "{}", grant.describe());
        }
        assert!(exact_only("/home/u/notes.md", PathAccessKind::Read)
            .describe()
            .contains("always = allow reads of this file only (`/home/u/notes.md`)"));
        assert!(exact_only("/", PathAccessKind::WorkingDir)
            .describe()
            .contains("of this folder only"));
        // An ancestor of workspace_dir that is not home (workspace elsewhere).
        let mut elsewhere = limits();
        elsewhere.grant_workspace_dir = Some(PathBuf::from("/srv/agents/ws"));
        let grant = PathGrant::for_access(
            &elsewhere,
            &access("/srv/agents/readme", PathAccessKind::Read),
        );
        assert!(
            grant.describe().contains("of this file only"),
            "{}",
            grant.describe()
        );
        // Case variants of home are still home.
        let grant = PathGrant::for_access(&p, &access("/HOME/U/a.txt", PathAccessKind::Read));
        assert!(
            grant.describe().contains("of this file only"),
            "{}",
            grant.describe()
        );
        // A normal folder below home still widens.
        let grant = PathGrant::for_access(&p, &access("/home/u/data/a.txt", PathAccessKind::Read));
        assert!(
            grant.describe().contains("under `/home/u/data`"),
            "{}",
            grant.describe()
        );
        assert!(!grant.describe().contains("except"), "{}", grant.describe());
    }

    #[test]
    fn folder_containing_isanagent_data_says_so() {
        let p = limits();
        // Approving a file directly in workspace_dir grants workspace_dir minus config and data.
        let grant = PathGrant::for_access(
            &p,
            &access("/home/u/.isanagent/notes.md", PathAccessKind::Read),
        );
        let text = grant.describe();
        assert!(
            text.contains("under `/home/u/.isanagent` (except isanagent config and data)"),
            "{text}"
        );
        assert!(grant.covers_with(
            &access("/home/u/.isanagent/other.md", PathAccessKind::Read),
            &p.protected_paths,
            false
        ));
        assert!(!grant.covers_with(
            &access("/home/u/.isanagent/config.toml", PathAccessKind::Read),
            &p.protected_paths,
            false
        ));
        assert!(!grant.covers_with(
            &access(
                "/home/u/.isanagent/.system_generated/agent_memory.db",
                PathAccessKind::Read
            ),
            &p.protected_paths,
            false
        ));
    }

    fn access(path: &str, kind: PathAccessKind) -> PathAccess {
        PathAccess {
            path: PathBuf::from(path),
            kind,
            inside_sandbox: false,
        }
    }

    #[test]
    fn outside_file_grant_covers_its_directory_tree_for_the_same_kind() {
        let grant =
            PathGrant::for_access(&no_limits(), &access("/srv/x/a.txt", PathAccessKind::Read));
        let covers = |p: &str, kind| grant.covers_with(&access(p, kind), &[], false);
        assert!(covers("/srv/x/a.txt", PathAccessKind::Read));
        assert!(covers("/srv/x/b.txt", PathAccessKind::Read));
        assert!(covers("/srv/x/deep/c.txt", PathAccessKind::Read));
        // Sibling with the same string prefix is not inside the grant.
        assert!(!covers("/srv/xy/c.txt", PathAccessKind::Read));
        assert!(!covers("/srv/x2", PathAccessKind::Read));
        assert!(!covers("/srv/a.txt", PathAccessKind::Read));
        // Kinds stay separate.
        assert!(!covers("/srv/x/b.txt", PathAccessKind::Write));
        assert!(!covers("/srv/x/b.txt", PathAccessKind::WorkingDir));
        assert!(
            grant.describe().contains("allow reads under `/srv/x`"),
            "{}",
            grant.describe()
        );
    }

    #[test]
    fn working_dir_grant_covers_that_directory_and_below() {
        let grant =
            PathGrant::for_access(&no_limits(), &access("/data", PathAccessKind::WorkingDir));
        assert!(grant.covers_with(&access("/data/sub", PathAccessKind::WorkingDir), &[], false));
        assert!(!grant.covers_with(&access("/data2", PathAccessKind::WorkingDir), &[], false));
    }

    #[test]
    fn inside_grants_stay_exact() {
        let mut inside = access("/ws/workspace/a.txt", PathAccessKind::Read);
        inside.inside_sandbox = true;
        let grant = PathGrant::for_access(&no_limits(), &inside);
        assert!(grant.covers_with(&inside, &[], false));
        let mut sibling = access("/ws/workspace/b.txt", PathAccessKind::Read);
        sibling.inside_sandbox = true;
        assert!(!grant.covers_with(&sibling, &[], false));
        assert!(grant
            .describe()
            .contains("of this file only (`/ws/workspace/a.txt`)"));
    }

    #[test]
    fn grant_never_widens_onto_the_outer_workspace_secrets() {
        let protected = vec![
            PathBuf::from("/home/u/.isanagent/config.toml"),
            PathBuf::from("/home/u/.isanagent/.system_generated"),
        ];
        // "always" on ~/x widens to ~, but not onto config.toml / .system_generated.
        let grant = PathGrant::for_access(&no_limits(), &access("/home/u/x", PathAccessKind::Read));
        let covers =
            |p: &str| grant.covers_with(&access(p, PathAccessKind::Read), &protected, false);
        assert!(covers("/home/u/notes.md"));
        assert!(!covers("/home/u/.isanagent/config.toml"));
        assert!(!covers(
            "/home/u/.isanagent/.system_generated/agent_memory.db"
        ));
        // Case games do not get past the shield.
        assert!(!covers("/home/u/.isanagent/CONFIG.toml"));
        // A grant whose directory lies inside a protected path covers it.
        let inner = PathGrant::for_access(
            &no_limits(),
            &access(
                "/home/u/.isanagent/.system_generated/logs/runtime.log",
                PathAccessKind::Read,
            ),
        );
        assert!(inner.covers_with(
            &access(
                "/home/u/.isanagent/.system_generated/logs/old.log",
                PathAccessKind::Read
            ),
            &protected,
            false
        ));
        // Approving config.toml itself covers that exact file only.
        let cfg = PathGrant::for_access(
            &no_limits(),
            &access("/home/u/.isanagent/config.toml", PathAccessKind::Read),
        );
        assert!(cfg.covers_with(
            &access("/home/u/.isanagent/config.toml", PathAccessKind::Read),
            &protected,
            false
        ));
        assert!(!cfg.covers_with(
            &access(
                "/home/u/.isanagent/.system_generated/x",
                PathAccessKind::Read
            ),
            &protected,
            false
        ));
    }

    #[test]
    fn grant_case_handling_follows_the_platform_flag() {
        let grant = PathGrant::for_access(
            &no_limits(),
            &access("/Users/a/Data/a.txt", PathAccessKind::Read),
        );
        let other_case = access("/users/a/data/b.txt", PathAccessKind::Read);
        assert!(!grant.covers_with(&other_case, &[], false));
        assert!(grant.covers_with(&other_case, &[], true));
        assert!(!grant.covers_with(
            &access("/users/a/data2/b.txt", PathAccessKind::Read),
            &[],
            true
        ));
        assert_eq!(
            GRANT_CASE_INSENSITIVE,
            cfg!(any(windows, target_os = "macos"))
        );
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
    fn approval_reply_reads_through_the_ask_user_result_wrapper() {
        let wrapped = |answer: &str| format!("{ASK_USER_REPLY_PREFIX}{answer}");
        assert_eq!(
            classify_approval_reply(&wrapped("approve")),
            ApprovalReply::Grant
        );
        assert_eq!(
            classify_approval_reply(&wrapped("always")),
            ApprovalReply::AlwaysThisRun
        );
        assert_eq!(
            classify_approval_reply(&wrapped("abort")),
            ApprovalReply::Abort
        );
        assert_eq!(
            classify_approval_reply(&wrapped("deny")),
            ApprovalReply::Deny
        );
        // An off-list answer stays a deny.
        assert_eq!(
            classify_approval_reply(
                "User reply (not among listed choices): maybe\n\nListed options were: []"
            ),
            ApprovalReply::Deny
        );
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
