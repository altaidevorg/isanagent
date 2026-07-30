use async_trait::async_trait;
use chrono::Utc;
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::time::{timeout, Duration};
use walkdir::WalkDir;

use crate::config::JinaWebBackend;
use crate::tool_runtime::{current_tool_exec_ctx, ToolExecCtx};
use crate::traits::{MutationPreview, Tool, ToolErrorCode, ToolResult};
use crate::utils::{join_lexically_under_root, normalize_sandbox_relative_input};
use crate::NodeHandle;

/// Maximum paths returned by `glob_files`.
const MAX_GLOB_RESULTS: usize = 500;
/// Maximum characters returned by `search_text`.
const MAX_SEARCH_TEXT_CHARS: usize = 20_000;
/// Maximum unified-diff lines included in `edit_file` output.
const MAX_DIFF_OUTPUT_LINES: usize = 80;
/// Bound approval metadata so a large write cannot flood the UI event channel.
const MAX_EDIT_APPROVAL_DIFF_CHARS: usize = 16_000;

/// Resolves a path against the workspace and enforces boundary restrictions.
pub fn resolve_path(path: &str, workspace_dir: &Path, restrict: bool) -> Result<PathBuf, String> {
    // 1. Expand naive relativity to the workspace dir. When restricted, resolve `.` / `..`
    // lexically under the workspace root so `..` at the root stays inside (does not canonicalize
    // to the parent directory and fail the boundary check). Strip a leading `workspace/` segment
    // when it duplicates the sandbox directory name (models often pass `workspace/foo` while the
    // tool root is already `.../workspace`).
    let trimmed = path.trim();
    let base_path = Path::new(trimmed);
    let resolved = if base_path.is_absolute() {
        base_path.to_path_buf()
    } else {
        let rel = normalize_sandbox_relative_input(workspace_dir, trimmed);
        if restrict {
            join_lexically_under_root(workspace_dir, &rel)?
        } else {
            workspace_dir.join(rel)
        }
    };

    // 2. Canonicalize if it exists to cleanly remove `..` and `.`.
    // If it doesn't exist yet (e.g., writing a new file), we canonicalize the nearest existing parent
    // and append the remainder.
    let canonical = if resolved.exists() {
        std::fs::canonicalize(&resolved).map_err(|e| format!("Path normalization error: {}", e))?
    } else {
        // Find nearest existing parent
        let mut parent = resolved.parent();
        let mut missing_components = Vec::new();
        while let Some(p) = parent {
            if p.exists() {
                break;
            }
            missing_components.push(p.file_name().unwrap_or_default());
            parent = p.parent();
        }

        let mut safe_base = std::fs::canonicalize(parent.unwrap_or_else(|| Path::new(".")))
            .map_err(|e| format!("Base path normalization error: {}", e))?;

        for comp in missing_components.into_iter().rev() {
            safe_base.push(comp);
        }
        if let Some(name) = resolved.file_name() {
            safe_base.push(name);
        }
        safe_base
    };

    // 3. Enforce sandbox boundary if restricted.
    if restrict {
        let canonical_workspace = std::fs::canonicalize(workspace_dir)
            .map_err(|e| format!("Workspace normalization error: {}", e))?;

        if !canonical.starts_with(&canonical_workspace) {
            return Err(format!(
                "PermissionError: Path {} is outside allowed workspace directory {}",
                resolved.display(),
                workspace_dir.display()
            ));
        }
    }

    Ok(canonical)
}

/// Relative path from `base` to `path` using `/` separators, for glob matching.
///
/// `WalkDir` paths may not prefix-strip cleanly against a canonical `base` on Windows
/// (e.g. `\\?\`-prefixed vs non-verbatim paths). Normalize with [`fs::canonicalize`] when needed.
fn path_for_glob_match(base: &Path, path: &Path) -> Option<String> {
    fn rel_after_strip(base: &Path, path: &Path) -> Option<String> {
        let rel = path.strip_prefix(base).ok()?;
        let s = rel.to_string_lossy().replace('\\', "/");
        Some(if s.is_empty() { ".".to_string() } else { s })
    }

    if let Some(s) = rel_after_strip(base, path) {
        return Some(s);
    }
    let base_can = fs::canonicalize(base).ok()?;
    let path_can = fs::canonicalize(path).ok()?;
    rel_after_strip(&base_can, &path_can)
}

fn truncate_diff_output(diff: String) -> String {
    let lines: Vec<&str> = diff.lines().collect();
    if lines.len() <= MAX_DIFF_OUTPUT_LINES {
        return diff;
    }
    let head: Vec<_> = lines.iter().take(MAX_DIFF_OUTPUT_LINES).copied().collect();
    format!(
        "{}\n... ({} more diff lines omitted)",
        head.join("\n"),
        lines.len() - MAX_DIFF_OUTPUT_LINES
    )
}

fn unified_diff_snippet(old: &str, new: &str) -> String {
    let patch = diffy::create_patch(old, new);
    truncate_diff_output(patch.to_string())
}

fn truncate_approval_diff(mut diff: String) -> (String, bool) {
    if diff.len() <= MAX_EDIT_APPROVAL_DIFF_CHARS {
        return (diff, false);
    }
    let mut end = MAX_EDIT_APPROVAL_DIFF_CHARS;
    while !diff.is_char_boundary(end) {
        end -= 1;
    }
    diff.truncate(end);
    diff.push_str("\n... (approval diff truncated)");
    (diff, true)
}

fn content_fingerprint(content: Option<&str>) -> String {
    match content {
        Some(content) => format!("sha256:{:x}", Sha256::digest(content.as_bytes())),
        None => "absent".to_string(),
    }
}

fn display_mutation_path(actual_path: &Path, workspace_dir: &Path) -> String {
    // `resolve_path` returns a canonical target. Canonicalize the workspace too
    // so `/var` versus `/private/var` aliases on macOS still produce a relative
    // user-facing path and a stable approval identity.
    let canonical_workspace =
        fs::canonicalize(workspace_dir).unwrap_or_else(|_| workspace_dir.to_path_buf());
    actual_path
        .strip_prefix(&canonical_workspace)
        .unwrap_or(actual_path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn mutation_preview(
    workspace_dir: &Path,
    restrict_to_workspace: bool,
    path_str: &str,
    before: Option<&str>,
    after: &str,
) -> Result<MutationPreview, String> {
    let actual_path = resolve_path(path_str, workspace_dir, restrict_to_workspace)?;
    let display_path = display_mutation_path(&actual_path, workspace_dir);
    let (diff, diff_truncated) =
        truncate_approval_diff(unified_diff_snippet(before.unwrap_or(""), after));
    Ok(MutationPreview {
        path: display_path,
        diff,
        diff_truncated,
        base_fingerprint: content_fingerprint(before),
    })
}

fn validate_approved_preview(
    workspace_dir: &Path,
    restrict_to_workspace: bool,
    path_str: &str,
    approved_preview: &MutationPreview,
) -> Result<(), String> {
    let actual_path = resolve_path(path_str, workspace_dir, restrict_to_workspace)?;
    let display_path = display_mutation_path(&actual_path, workspace_dir);
    if display_path != approved_preview.path {
        return Err(
            "Edit approval no longer matches the requested path; request a new approval.".into(),
        );
    }
    let current = match fs::read_to_string(&actual_path) {
        Ok(content) => Some(content),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(format!("Could not re-read approved edit target: {error}")),
    };
    if content_fingerprint(current.as_deref()) != approved_preview.base_fingerprint {
        return Err(
            "Edit target changed after approval; request a new approval with an updated diff."
                .into(),
        );
    }
    Ok(())
}

fn ripgrep_available() -> bool {
    which::which("rg").is_ok()
}

pub struct ReadFileTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a slice of a local file. REQUIRED on every call: path, start_line, and end_line \
(1-indexed, inclusive). Each call returns at most 100 lines — never omit the line range, \
and never rely on a default. For longer files, issue multiple read_file calls with adjacent \
ranges (e.g. 1–100, then 101–200). Prefer absolute or workspace-relative paths."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or workspace-relative path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Required. First line to include (1-indexed, inclusive). Always pass explicitly — there is no default."
                },
                "end_line": {
                    "type": "integer",
                    "description": "Required. Last line to include (1-indexed, inclusive). Must be >= start_line. The tool caps each call at 100 lines even if the range is wider."
                }
            },
            "required": ["path", "start_line", "end_line"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid 'path' argument")?;

        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;

        if super::isanagent_ignore::is_ignored(&actual_path, false) {
            return Err(format!(
                "blocked by .isanagentignore: {}",
                actual_path.display()
            ));
        }

        let start_line = args.get("start_line").and_then(|v| v.as_u64()).ok_or(
            "Missing required 'start_line'. Every read_file call must pass start_line and \
end_line (1-indexed, inclusive); max 100 lines per call — e.g. start_line=1, end_line=100.",
        )?;
        let end_line = args.get("end_line").and_then(|v| v.as_u64()).ok_or(
            "Missing required 'end_line'. Every read_file call must pass start_line and \
end_line (1-indexed, inclusive); max 100 lines per call — e.g. start_line=1, end_line=100.",
        )?;

        let content = fs::read_to_string(&actual_path).map_err(|e| e.to_string())?;

        let start = start_line.max(1) as usize;
        let end = end_line as usize;

        if end < start {
            return Err("end_line must be greater than or equal to start_line".to_string());
        }

        let lines_to_read = (end - start + 1).min(100);

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let actual_start = start.min(total_lines.max(1));
        let actual_end = (actual_start + lines_to_read - 1).min(total_lines);

        let snippet: Vec<String> = lines[actual_start - 1..actual_end]
            .iter()
            .map(|l| l.to_string())
            .collect();

        Ok(snippet.join("\n"))
    }
}

pub struct WriteFileTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a local file. Be careful, this will overwrite the file if it exists."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write into the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'content' argument")?;

        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;

        if let Some(parent) = actual_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent directories: {}", e))?;
        }

        crate::checkpoint::snapshot_before(&actual_path, "write_file");
        fs::write(&actual_path, content)
            .map(|_| format!("Successfully wrote to {}", actual_path.display()))
            .map_err(|e| e.to_string())
    }

    async fn preview_mutation(&self, args: &Value) -> Result<Option<MutationPreview>, String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'content' argument")?;
        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;
        let before = match fs::read_to_string(&actual_path) {
            Ok(current_content) => {
                // No-op write: the file already holds the exact content. Skip the
                // approval prompt — there is no mutation to review.
                if current_content == content {
                    return Ok(None);
                }
                Some(current_content)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(format!("Could not preview write target: {error}")),
        };
        mutation_preview(
            &self.workspace_dir,
            self.restrict_to_workspace,
            path_str,
            before.as_deref(),
            content,
        )
        .map(Some)
    }

    async fn execute_with_approved_mutation(
        &self,
        args: Value,
        approved_preview: Option<&MutationPreview>,
    ) -> Result<String, String> {
        if let Some(preview) = approved_preview {
            let path_str = args
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'path' argument")?;
            validate_approved_preview(
                &self.workspace_dir,
                self.restrict_to_workspace,
                path_str,
                preview,
            )?;
        }
        self.execute(args).await
    }
}

pub struct EditFileTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing old_text with new_text. The old_text must appear exactly once unless replace_all is true. Returns a truncated unified diff of the change."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace it with"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace every occurrence of old_text. If false (default), old_text must be unique in the file."
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;

        let old_text = args
            .get("old_text")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'old_text' argument")?;

        let new_text = args
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'new_text' argument")?;

        let replace_all = args
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if old_text == new_text {
            return Ok("Error: old_text and new_text are identical.".to_string());
        }

        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;

        let content =
            fs::read_to_string(&actual_path).map_err(|e| format!("Error reading file: {}", e))?;

        if !content.contains(old_text) {
            return Ok("Error: old_text not found in file.".to_string());
        }

        let count = content.matches(old_text).count();
        if count > 1 && !replace_all {
            return Ok(format!(
                "Error: old_text appears {} times. Provide more surrounding context to make it unique, or set replace_all to true.",
                count
            ));
        }

        let old_content = content.clone();
        let new_content = if replace_all {
            content.replace(old_text, new_text)
        } else {
            content.replacen(old_text, new_text, 1)
        };

        crate::checkpoint::snapshot_before(&actual_path, "edit_file");
        fs::write(&actual_path, &new_content).map_err(|e| format!("Error saving edits: {}", e))?;

        let diff = unified_diff_snippet(&old_content, &new_content);
        let replacements = if replace_all { count } else { 1 };
        Ok(format!(
            "Applied {} replacement(s) to {}:\n\n{}",
            replacements,
            actual_path.display(),
            diff
        ))
    }

    async fn preview_mutation(&self, args: &Value) -> Result<Option<MutationPreview>, String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
        let old_text = args
            .get("old_text")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'old_text' argument")?;
        let new_text = args
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'new_text' argument")?;
        let replace_all = args
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // No-op edit: identical old/new text produces an empty diff. Skip the
        // approval prompt — there is no mutation to review. (execute() separately
        // returns an "identical" error so the model learns the edit was a no-op.)
        if old_text == new_text {
            return Ok(None);
        }
        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;
        let before = fs::read_to_string(&actual_path)
            .map_err(|error| format!("Could not preview edit target: {error}"))?;
        if !before.contains(old_text) {
            return Ok(None);
        }
        let count = before.matches(old_text).count();
        if count > 1 && !replace_all {
            return Ok(None);
        }
        let after = if replace_all {
            before.replace(old_text, new_text)
        } else {
            before.replacen(old_text, new_text, 1)
        };
        mutation_preview(
            &self.workspace_dir,
            self.restrict_to_workspace,
            path_str,
            Some(&before),
            &after,
        )
        .map(Some)
    }

    async fn execute_with_approved_mutation(
        &self,
        args: Value,
        approved_preview: Option<&MutationPreview>,
    ) -> Result<String, String> {
        if let Some(preview) = approved_preview {
            let path_str = args
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'path' argument")?;
            validate_approved_preview(
                &self.workspace_dir,
                self.restrict_to_workspace,
                path_str,
                preview,
            )?;
        }
        self.execute(args).await
    }
}

pub struct ListDirTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List the contents of a directory."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;

        let actual_path = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;

        if !actual_path.is_dir() {
            return Ok(format!("Error: Not a directory: {}", actual_path.display()));
        }

        let mut entries = match fs::read_dir(&actual_path) {
            Ok(iter) => iter,
            Err(e) => return Ok(format!("Error reading dir: {}", e)),
        };

        let mut items = Vec::new();
        while let Some(Ok(entry)) = entries.next() {
            let metadata = entry.metadata().map_err(|e| e.to_string())?;
            if super::isanagent_ignore::is_ignored(&entry.path(), metadata.is_dir()) {
                continue;
            }
            let prefix = if metadata.is_dir() { "📁" } else { "📄" };
            items.push(format!(
                "{} {}",
                prefix,
                entry.file_name().to_string_lossy()
            ));
        }

        items.sort();

        if items.is_empty() {
            return Ok(format!("Directory {} is empty", actual_path.display()));
        }

        Ok(items.join("\n"))
    }
}

/// Discover files under a directory using a glob pattern (e.g. `**/*.rs`, `src/**/*.toml`).
pub struct GlobFilesTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

fn compile_glob_single(pattern: &str) -> Result<GlobSet, String> {
    let glob = GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .map_err(|e| format!("Invalid glob pattern: {}", e))?;
    let mut builder = GlobSetBuilder::new();
    builder.add(glob);
    builder
        .build()
        .map_err(|e| format!("Invalid glob pattern: {}", e))
}

#[async_trait]
impl Tool for GlobFilesTool {
    fn name(&self) -> &str {
        "glob_files"
    }

    fn description(&self) -> &str {
        "Find files under a base directory matching a glob pattern. Returns sorted paths (capped). Use ** for recursive patterns."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern. Use ** for recursion (e.g. **/*.md matches markdown anywhere under the base). A bare *.md only matches files directly in the base directory, not in subfolders."
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search from (relative to workspace). Defaults to '.'"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'pattern' argument")?;

        let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let base = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;
        if !base.exists() {
            return Ok(format!("Error: path not found: {}", base.display()));
        }
        if !base.is_dir() {
            return Ok(format!(
                "Error: base path is not a directory: {}",
                base.display()
            ));
        }

        // Align with WalkDir output so `strip_prefix` works on all platforms (notably Windows).
        let walk_root = fs::canonicalize(&base).map_err(|e| {
            format!(
                "Could not canonicalize search base {}: {}",
                base.display(),
                e
            )
        })?;

        let matcher = compile_glob_single(pattern)?;
        let mut matches: Vec<PathBuf> = Vec::new();
        let mut truncated = false;

        for entry in WalkDir::new(&walk_root).follow_links(false) {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if super::isanagent_ignore::is_ignored(path, entry.file_type().is_dir()) {
                continue;
            }
            if !path.is_file() {
                continue;
            }
            let Some(rel) = path_for_glob_match(&walk_root, path) else {
                continue;
            };
            if !matcher.is_match(&rel) {
                continue;
            }
            matches.push(path.to_path_buf());
            if matches.len() >= MAX_GLOB_RESULTS {
                truncated = true;
                break;
            }
        }

        matches.sort();
        matches.dedup();

        if matches.is_empty() {
            return Ok("No files matched.".to_string());
        }

        let mut out = matches
            .into_iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join("\n");

        if truncated {
            out.push_str(&format!(
                "\n... (glob results capped at {} paths; refine the pattern or base path)",
                MAX_GLOB_RESULTS
            ));
        }

        Ok(out)
    }
}

/// Regex search across files under the workspace (ripgrep when available, otherwise a built-in walker).
pub struct SearchTextTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
    /// Default ripgrep subprocess timeout (seconds); per-call `timeout_secs` in tool args overrides when set.
    pub ripgrep_timeout_secs: u64,
}

async fn search_text_ripgrep(
    pattern: &str,
    search_path: &Path,
    file_glob: Option<&str>,
    output_mode: &str,
    case_insensitive: bool,
    context_lines: u32,
    timeout_secs: u64,
) -> Result<String, String> {
    let mut cmd = tokio::process::Command::new("rg");
    cmd.arg("--no-heading");
    if case_insensitive {
        cmd.arg("-i");
    }
    match output_mode {
        "files_with_matches" => {
            cmd.arg("-l");
        }
        "count" => {
            cmd.arg("-c");
        }
        _ => {
            cmd.arg("-n");
            if context_lines > 0 {
                cmd.arg("-C");
                cmd.arg(context_lines.to_string());
            }
        }
    }
    if let Some(g) = file_glob {
        cmd.arg("--glob");
        cmd.arg(g);
    }
    if let Some(ignore_file) = super::isanagent_ignore::find_ignore_file(search_path) {
        cmd.arg("--ignore-file");
        cmd.arg(ignore_file);
    }
    cmd.arg("--");
    cmd.arg(pattern);
    cmd.arg(search_path);

    let fut = cmd.output();
    let output = tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), fut)
        .await
        .map_err(|_| format!("Search timed out after {} seconds.", timeout_secs))?
        .map_err(|e| format!("Failed to run ripgrep: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let code = output.status.code();

    if code == Some(1) && stdout.is_empty() {
        return Ok("No matches found.".to_string());
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            return Ok("No matches found.".to_string());
        }
        return Err(format!("ripgrep error: {}", stderr));
    }

    if stdout.is_empty() {
        return Ok("No matches found.".to_string());
    }

    Ok(truncate_search_output(stdout))
}

fn truncate_search_output(mut s: String) -> String {
    if s.len() <= MAX_SEARCH_TEXT_CHARS {
        return s;
    }
    let mut end = MAX_SEARCH_TEXT_CHARS;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s.push_str(&format!(
        "\n... (truncated, output exceeded {} characters)",
        MAX_SEARCH_TEXT_CHARS
    ));
    s
}

fn search_text_native(
    regex: &regex::Regex,
    search_root: &Path,
    file_glob: Option<&GlobSet>,
    output_mode: &str,
) -> Result<String, String> {
    let mut lines_out: Vec<String> = Vec::new();
    let mut count_rows: Vec<String> = Vec::new();

    let mut visit_file = |abs: &Path, rel_key: &str| -> Result<(), String> {
        if let Some(gs) = file_glob {
            if !gs.is_match(rel_key) {
                return Ok(());
            }
        }
        let meta = match fs::metadata(abs) {
            Ok(m) => m,
            Err(_) => return Ok(()),
        };
        if !meta.is_file() || meta.len() > 2 * 1024 * 1024 {
            return Ok(());
        }
        let text = match fs::read_to_string(abs) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        match output_mode {
            "files_with_matches" => {
                if regex.is_match(&text) {
                    lines_out.push(abs.display().to_string());
                }
            }
            "count" => {
                let mut n: usize = 0;
                for line in text.lines() {
                    n += regex.find_iter(line).count();
                }
                if n > 0 {
                    count_rows.push(format!("{}:{}", abs.display(), n));
                }
            }
            _ => {
                for (i, line) in text.lines().enumerate() {
                    let line_no = i + 1;
                    if regex.is_match(line) {
                        lines_out.push(format!("{}:{}:{}", abs.display(), line_no, line));
                    }
                }
            }
        }
        Ok(())
    };

    if search_root.is_file() {
        let rel_key = search_root
            .file_name()
            .map(|s| s.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        visit_file(search_root, &rel_key)?;
    } else {
        for entry in WalkDir::new(search_root).follow_links(false) {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if super::isanagent_ignore::is_ignored(path, entry.file_type().is_dir()) {
                continue;
            }
            if !path.is_file() {
                continue;
            }
            let Some(rel) = path_for_glob_match(search_root, path) else {
                continue;
            };
            visit_file(path, &rel)?;
        }
    }

    let mut result = match output_mode {
        "count" => {
            if count_rows.is_empty() {
                return Ok("No matches found.".to_string());
            }
            count_rows.sort();
            count_rows.join("\n")
        }
        _ => {
            if lines_out.is_empty() {
                return Ok("No matches found.".to_string());
            }
            lines_out.sort();
            lines_out.join("\n")
        }
    };

    result = truncate_search_output(result);
    Ok(result)
}

#[async_trait]
impl Tool for SearchTextTool {
    fn name(&self) -> &str {
        "search_text"
    }

    fn description(&self) -> &str {
        "Search file contents with a Rust regex. Prefer this over shell grep/findstr pipelines when locating code or logs. Uses ripgrep when installed for speed and context-line support; otherwise scans files under the search path (skipping very large files in fallback mode)."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Rust regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (relative to workspace). Defaults to '.'"
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob filter for paths (e.g. *.rs, *.{ts,tsx})"
                },
                "output_mode": {
                    "type": "string",
                    "description": "One of: files_with_matches (default), content, count",
                    "enum": ["files_with_matches", "content", "count"]
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive matching (default false)"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around matches when output_mode is content (requires ripgrep when > 0)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Ripgrep subprocess timeout in seconds (1–3600; overrides workspace default when set)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'pattern' argument")?;

        let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let file_glob = args.get("glob").and_then(|v| v.as_str());
        let output_mode = args
            .get("output_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("files_with_matches");
        let case_insensitive = args
            .get("case_insensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let context_lines = args
            .get("context_lines")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        const RG_TIMEOUT_MAX: u64 = 3600;
        let ripgrep_timeout_secs = match args.get("timeout_secs").and_then(|v| v.as_u64()) {
            Some(t) => t.clamp(1, RG_TIMEOUT_MAX),
            None => self.ripgrep_timeout_secs.clamp(1, RG_TIMEOUT_MAX),
        };

        let resolved = resolve_path(path_str, &self.workspace_dir, self.restrict_to_workspace)?;

        if !resolved.exists() {
            return Ok(format!("Error: path not found: {}", resolved.display()));
        }

        let search_target = fs::canonicalize(&resolved).unwrap_or_else(|_| resolved.clone());

        if ripgrep_available() {
            let glob_arg = file_glob;
            return search_text_ripgrep(
                pattern,
                &search_target,
                glob_arg,
                output_mode,
                case_insensitive,
                context_lines,
                ripgrep_timeout_secs,
            )
            .await;
        }

        if context_lines > 0 && output_mode == "content" {
            return Err(
                "context_lines requires ripgrep (rg) on PATH for this host; install ripgrep or use context_lines 0."
                    .to_string(),
            );
        }

        let file_glob_set = if let Some(g) = file_glob {
            Some(compile_glob_single(g)?)
        } else {
            None
        };

        let regex = regex::RegexBuilder::new(pattern)
            .case_insensitive(case_insensitive)
            .build()
            .map_err(|e| format!("Invalid regex: {}", e))?;

        let search_root = search_target;
        let mode = output_mode.to_string();
        let glob_set = file_glob_set;
        let regex_owned = regex;
        tokio::task::spawn_blocking(move || {
            search_text_native(&regex_owned, &search_root, glob_set.as_ref(), &mode)
        })
        .await
        .map_err(|e| format!("search task failed: {}", e))?
    }
}

pub struct ShellExecTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
}

struct ShellExecOutcome {
    content: String,
    failure_exit_code: Option<i32>,
}

impl ShellExecTool {
    fn check_safety_guards(command: &str) -> Result<(), String> {
        let lower_cmd = command.to_lowercase();
        // Hard safety rails for catastrophic host operations. Workflow-level policy (ask/deny/allow)
        // handles common destructive workspace commands such as rm/del/git reset.
        let blocked_patterns = [
            "format ",
            "mkfs",
            "diskpart",
            "dd if=",
            "> /dev/sd",
            "shutdown",
            "reboot",
            "poweroff",
            ":(){ :|:& };:",
        ];

        for pattern in blocked_patterns.iter() {
            if lower_cmd.contains(pattern) {
                return Err(format!(
                    "Command blocked by safety guard (detected dangerous pattern: {})",
                    pattern
                ));
            }
        }
        Ok(())
    }

    async fn execute_command(&self, args: Value) -> Result<ShellExecOutcome, String> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'command' argument")?;
        let lower_cmd = command.to_ascii_lowercase();
        let grep_like = lower_cmd.contains("grep ")
            || lower_cmd.contains("| grep")
            || lower_cmd.contains("cat ")
            || lower_cmd.contains("wc ");

        Self::check_safety_guards(command)?;

        let cwd_str = args
            .get("working_dir")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(60)
            .clamp(1, 3600);

        let actual_dir = resolve_path(cwd_str, &self.workspace_dir, self.restrict_to_workspace)?;

        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = tokio::process::Command::new("cmd");
            c.arg("/C").arg(command);
            c
        } else {
            let mut c = tokio::process::Command::new("sh");
            c.arg("-c").arg(command);
            c
        };

        cmd.current_dir(actual_dir);
        cmd.envs(std::env::vars());

        let output =
            match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), cmd.output())
                .await
            {
                Ok(Ok(output)) => output,
                Ok(Err(error)) => return Err(format!("Failed to execute command: {error}")),
                Err(_) => return Err(format!("Command timed out after {timeout_secs} seconds")),
            };

        let mut result = String::new();
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            result.push_str(&stdout);
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            if !result.is_empty() {
                result.push_str("\nSTDERR:\n");
            }
            result.push_str(&stderr);
        }

        let failure_exit_code =
            (!output.status.success()).then(|| output.status.code().unwrap_or(-1));

        if result.is_empty() && failure_exit_code.is_none() {
            result = "(no output)".to_string();
        } else {
            if grep_like {
                result.push_str("\n\n[advisory] Prefer `search_text` for code/log discovery and `read_file` for file reads; shell grep/cat pipelines are less portable across hosts.");
            }
            if result.len() > 10000 {
                let mut cut = 10000;
                while cut > 0 && !result.is_char_boundary(cut) {
                    cut -= 1;
                }
                result = format!(
                    "{}\n... (truncated, {} more chars)",
                    &result[..cut],
                    result.len() - cut
                );
            }
            if let Some(code) = failure_exit_code {
                result.push_str(&format!("\nExit code: {code}"));
            }
        }

        Ok(ShellExecOutcome {
            content: result,
            failure_exit_code,
        })
    }
}

#[async_trait]
impl Tool for ShellExecTool {
    fn name(&self) -> &str {
        "exec"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output (60s timeout). Host details (OS/shell/path style) are provided in RUNTIME CONTEXT each turn; write commands for that host. Prefer first-class tools (`search_text`, `read_file`, `glob_files`, `web_fetch`) before shell one-liners, especially for grep/cat/wc style tasks. \
         On **Windows** this runs under **cmd /C** one string: nested double-quotes often break remote **ssh** compound commands (e.g. `ssh user@host \"mkdir -p /tmp/x && cmd\"`). Prefer a **single** remote argument without inner double-quotes, use **execution_* / SSH harness** for remote work, or run **two** short exec calls instead of one over-quoted line."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional relative working directory for the command"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (defaults to 60, max 3600)"
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what this command is trying to achieve (used for UI and audits)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        self.execute_command(args)
            .await
            .map(|outcome| outcome.content)
    }

    async fn execute_with_approved_mutation_typed(
        &self,
        args: Value,
        _approved_preview: Option<&MutationPreview>,
    ) -> ToolResult {
        match self.execute_command(args).await {
            Ok(outcome) => match outcome.failure_exit_code {
                Some(code) => ToolResult::error_with_content(
                    ToolErrorCode::NonZeroExit,
                    format!("exec exited with status {code}"),
                    outcome.content,
                ),
                None => ToolResult::success(outcome.content),
            },
            Err(error) => ToolResult::error(ToolErrorCode::ExecutionFailed, error),
        }
    }
}

fn web_http_client(timeout_secs: u64) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0")
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))
}

fn parse_scraper_selector(sel: &str) -> Result<scraper::Selector, String> {
    scraper::Selector::parse(sel).map_err(|e| format!("Invalid CSS selector {:?}: {}", sel, e))
}

fn truncate_web_output(text: String, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text;
    }
    let mut end = max_chars;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n... (truncated, {} more chars)",
        &text[..end],
        text.len() - end
    )
}

fn apply_jina_bearer(
    req: reqwest::RequestBuilder,
    jina: Option<&JinaWebBackend>,
) -> reqwest::RequestBuilder {
    if let Some(j) = jina {
        if let Some(key) = j.api_key.as_deref() {
            return req.header("Authorization", format!("Bearer {}", key));
        }
    }
    req
}

/// DuckDuckGo `/html/` often blocks scrapers; `/lite/` via POST is more reliable.
async fn web_search_duckduckgo(query: &str, max_output_chars: usize) -> Result<String, String> {
    let url = "https://lite.duckduckgo.com/lite/";
    let client = web_http_client(45)?;

    let res = client
        .post(url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!("q={}", urlencoding::encode(query)))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let body = res.text().await.map_err(|e| e.to_string())?;

    let document = scraper::Html::parse_document(&body);
    let title_selector = parse_scraper_selector(".result-link")?;
    let snippet_selector = parse_scraper_selector(".result-snippet")?;

    let mut results = String::new();

    let titles: Vec<_> = document.select(&title_selector).take(5).collect();
    let snippets: Vec<_> = document.select(&snippet_selector).take(5).collect();

    for (i, (title_elem, snippet_elem)) in titles.into_iter().zip(snippets).enumerate() {
        let title = title_elem.text().collect::<Vec<_>>().join(" ");
        let link = title_elem.value().attr("href").unwrap_or("");
        let snippet = snippet_elem
            .text()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();

        results.push_str(&format!(
            "{}. [{}]({})\n   {}\n\n",
            i + 1,
            title,
            link,
            snippet
        ));
    }

    if results.is_empty() {
        return Ok("No results found.".to_string());
    }

    Ok(truncate_web_output(results, max_output_chars))
}

/// [Jina Search](https://s.jina.ai/) — useful when DuckDuckGo is unreachable from the host.
async fn web_search_jina(
    query: &str,
    jina: &JinaWebBackend,
    max_output_chars: usize,
) -> Result<String, String> {
    let url = format!("https://s.jina.ai/{}", urlencoding::encode(query));
    let client = web_http_client(45)?;
    let req = apply_jina_bearer(client.get(&url), Some(jina));
    let res = req.send().await.map_err(|e| e.to_string())?;
    if !res.status().is_success() {
        return Err(format!("Jina search HTTP error: {}", res.status()));
    }
    let body = res.text().await.map_err(|e| e.to_string())?;
    if body.trim().is_empty() {
        return Ok("No results found.".to_string());
    }
    Ok(truncate_web_output(body, max_output_chars))
}

async fn web_fetch_direct(url: &str, max_output_chars: usize) -> Result<String, String> {
    let client = web_http_client(30)?;

    let res = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch URL: {}", e))?;

    if !res.status().is_success() {
        return Err(format!("HTTP Error: {}", res.status()));
    }

    let content_type = res
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if content_type.contains("application/json") {
        let json_body: Value = res
            .json()
            .await
            .map_err(|e| format!("Invalid JSON: {}", e))?;
        let s = serde_json::to_string_pretty(&json_body).unwrap_or_default();
        return Ok(truncate_web_output(s, max_output_chars));
    }

    let body = res
        .text()
        .await
        .map_err(|e| format!("Failed to decode text: {}", e))?;

    let document = scraper::Html::parse_document(&body);
    let mut text_output = String::new();

    // Heuristic HTML→text for direct fetches: skip non-content tags (scripts, chrome, SVG) and
    // treat block-level tags as line breaks / light markdown markers. Not configurable; Jina path
    // avoids this entirely.
    let elements_to_ignore = [
        "script", "style", "noscript", "svg", "nav", "footer", "header",
    ];
    let block_elements = [
        "p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br",
    ];

    let body_selector = parse_scraper_selector("body")?;
    if let Some(body_node) = document.select(&body_selector).next() {
        for node in body_node.descendants() {
            if let scraper::Node::Element(elem) = node.value() {
                let tag = elem.name();

                if elements_to_ignore.contains(&tag) {
                    continue;
                }
                if block_elements.contains(&tag) {
                    text_output.push('\n');
                    if tag.starts_with('h') {
                        text_output.push_str("### ");
                    }
                    if tag == "li" {
                        text_output.push_str("- ");
                    }
                }
            } else if let scraper::Node::Text(text_node) = node.value() {
                let text = text_node.trim();
                if !text.is_empty() {
                    let mut ignore = false;
                    let mut parent = node.parent();
                    while let Some(p) = parent {
                        if let scraper::Node::Element(e) = p.value() {
                            if elements_to_ignore.contains(&e.name()) {
                                ignore = true;
                                break;
                            }
                        }
                        parent = p.parent();
                    }

                    if !ignore {
                        text_output.push_str(text);
                        text_output.push(' ');
                    }
                }
            }
        }
    }

    let cleaned = text_output
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    Ok(truncate_web_output(cleaned, max_output_chars))
}

/// [Jina Reader](https://r.jina.ai/) returns LLM-friendly markdown for a target URL.
async fn web_fetch_jina(
    url: &str,
    jina: &JinaWebBackend,
    max_output_chars: usize,
) -> Result<String, String> {
    // Jina Reader expects the target URL as a path suffix with `:` and `/` intact — do not
    // percent-encode the whole URL or the service cannot resolve the target.
    let reader_url = format!("https://r.jina.ai/{}", url.trim());
    let client = web_http_client(60)?;
    let req = apply_jina_bearer(client.get(&reader_url), Some(jina));
    let res = req
        .send()
        .await
        .map_err(|e| format!("Failed to fetch URL via Jina: {}", e))?;
    if !res.status().is_success() {
        return Err(format!("HTTP Error (Jina reader): {}", res.status()));
    }

    let content_type = res
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if content_type.contains("application/json") {
        let json_body: Value = res
            .json()
            .await
            .map_err(|e| format!("Invalid JSON: {}", e))?;
        let s = serde_json::to_string_pretty(&json_body).unwrap_or_default();
        return Ok(truncate_web_output(s, max_output_chars));
    }

    let body = res
        .text()
        .await
        .map_err(|e| format!("Failed to decode text: {}", e))?;
    Ok(truncate_web_output(body, max_output_chars))
}

pub struct WebSearchTool {
    /// When `Some`, use [Jina Search](https://s.jina.ai/) (`[jina].enabled` in config).
    pub jina: Option<JinaWebBackend>,
    /// From `max_web_tool_output_chars` in config (see `AppConfig::effective_max_web_tool_output_chars`).
    pub max_output_chars: usize,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for **current** facts, docs, and release notes. Discovery tool only: use this to find candidate sources, then follow with `web_fetch` on authoritative URLs before concluding. Uses Jina (s.jina.ai) when [jina].enabled is true in config; otherwise DuckDuckGo Lite."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query' argument")?;

        if let Some(ref jina) = self.jina {
            web_search_jina(query, jina, self.max_output_chars).await
        } else {
            web_search_duckduckgo(query, self.max_output_chars).await
        }
    }
}

pub struct WebFetchTool {
    /// When `Some`, use [Jina Reader](https://r.jina.ai/) (`[jina].enabled` in config).
    pub jina: Option<JinaWebBackend>,
    /// From `max_web_tool_output_chars` in config (see `AppConfig::effective_max_web_tool_output_chars`).
    pub max_output_chars: usize,
    pub workspace_dir: std::path::PathBuf,
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL in detail (docs, raw GitHub, paper pages). Use after `web_search` to read primary sources and extract evidence. Uses Jina Reader (r.jina.ai) when [jina].enabled is true; otherwise direct GET with HTML text extraction or JSON pretty-print. Prefer official docs and pinned `raw.githubusercontent.com` sources when validating ML APIs."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'url' argument")?;

        let full_content = if let Some(ref jina) = self.jina {
            web_fetch_jina(url, jina, usize::MAX).await?
        } else {
            web_fetch_direct(url, usize::MAX).await?
        };

        let uuid = uuid::Uuid::new_v4().to_string();
        let downloads_dir = self
            .workspace_dir
            .join("workspace")
            .join("downloads")
            .join("web");
        let _ = tokio::fs::create_dir_all(&downloads_dir).await;
        let file_path = downloads_dir.join(format!("{uuid}.txt"));
        tokio::fs::write(&file_path, &full_content)
            .await
            .map_err(|e| e.to_string())?;

        let safe_limit = self.max_output_chars.saturating_sub(1000).max(1000);
        let preview = crate::execution::truncate_utf8_str_cap(&full_content, safe_limit);
        let total_lines = full_content.lines().count();

        Ok(format!(
            "{preview}\n\n---\nNote: The full response ({} lines, {} bytes) was saved to `{}`. \
            If this preview is truncated, use the `read_file` tool with `start_line` and `end_line` arguments \
            on that path to incrementally read the rest of the content and/or use the `search_text` tool to find specific information.",
            total_lines,
            full_content.len(),
            file_path.display()
        ))
    }
}

pub struct CronTool {
    pub cron_node: NodeHandle<String>,
    pub multi_tenant_edge_cron_enabled: bool,
    pub mte_cron_scheduler: Option<std::sync::Arc<crate::scheduler::MultiTenantEdgeCronScheduler>>,
    pub db_path: String,
}

/// Resolve the cron destination from the trusted per-invocation runtime
/// context when one exists. The `chat_id` / `channel` arguments predate
/// `ToolExecCtx` and remain supported only for callers that do not install a
/// context (for example, external integrations that invoke a tool directly).
///
/// A model must never be able to redirect a scheduled action to another
/// conversation by fabricating a destination in tool arguments.
fn cron_target_from_args(
    args: &Value,
    exec_ctx: Option<&ToolExecCtx>,
) -> Result<(String, String), String> {
    let supplied_chat_id = args.get("chat_id").and_then(|v| v.as_str());
    let supplied_channel = args.get("channel").and_then(|v| v.as_str());

    if let Some(ctx) = exec_ctx {
        if let Some(chat_id) = supplied_chat_id {
            if chat_id != ctx.chat_id {
                return Err("cron chat_id does not match the current tool session".to_string());
            }
        }
        if let Some(channel) = supplied_channel {
            if channel != ctx.channel {
                return Err("cron channel does not match the current tool session".to_string());
            }
        }
        return Ok((ctx.chat_id.clone(), ctx.channel.clone()));
    }

    let chat_id = supplied_chat_id.ok_or("Missing 'chat_id' for add action")?;
    let channel = supplied_channel.ok_or("Missing 'channel' for add action")?;
    Ok((chat_id.to_string(), channel.to_string()))
}

fn cron_job_is_in_scope(job: &crate::scheduler::ActiveJob, exec_ctx: &ToolExecCtx) -> bool {
    job.chat_id == exec_ctx.chat_id && job.channel == exec_ctx.channel
}

#[async_trait]
impl Tool for CronTool {
    fn name(&self) -> &str {
        "cron"
    }

    fn description(&self) -> &str {
        "Manage scheduled tasks. Supports repeating intervals, exact times, or standard cron expressions. When called by an agent, schedules are bound to the current conversation and cannot target another chat."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform: 'add', 'remove', or 'list'."
                },
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job. Required only for 'remove' action."
                },
                "message": {
                    "type": "string",
                    "description": "The message to send back to you when triggered. Required for 'add' action."
                },
                "chat_id": {
                    "type": "string",
                    "description": "Legacy direct-call destination. Agent calls are bound to their current session; if supplied, this must match that session."
                },
                "channel": {
                    "type": "string",
                    "description": "Legacy direct-call destination. Agent calls are bound to their current session; if supplied, this must match that session."
                },
                "every_seconds": {
                    "type": "integer",
                    "description": if self.multi_tenant_edge_cron_enabled {
                        "Execute repeatedly every N seconds. Mutually exclusive with 'at' and 'cron_expr'. Not supported when multi-tenant-edge cron scheduling is enabled."
                    } else {
                        "Execute repeatedly every N seconds. Mutually exclusive with 'at' and 'cron_expr'."
                    }
                },
                "at": {
                    "type": "string",
                    "description": "Execute once at a specific ISO datetime. You MUST include the exact correct timezone offset from your RUNTIME CONTEXT (e.g. 2026-03-04T13:45:53+03:00, NOT ending in Z unless you are in UTC). Mutually exclusive with 'every_seconds' and 'cron_expr'."
                },
                "cron_expr": {
                    "type": "string",
                    "description": if self.multi_tenant_edge_cron_enabled {
                        "Execute using a 6-part UTC cron string (`second minute hour day month day-of-week`). Mutually exclusive with 'every_seconds' and 'at'."
                    } else {
                        "Execute using a 7-part cron string. Mutually exclusive with 'every_seconds' and 'at'."
                    }
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("add");
        let exec_ctx = current_tool_exec_ctx();

        if action == "list" {
            let store = crate::scheduler::CronStore::new(&self.db_path)?;
            let mut jobs = store.load_jobs()?;
            if let Some(ctx) = exec_ctx.as_ref() {
                jobs.retain(|job| cron_job_is_in_scope(job, ctx));
            }

            if jobs.is_empty() {
                return Ok("No active cron jobs found.".to_string());
            }

            let mut out = String::new();
            out.push_str(&format!("Found {} active cron job(s):\n", jobs.len()));
            for job in jobs {
                let sched_str = match &job.schedule {
                    crate::scheduler::ScheduleKind::At { at_ms } => {
                        let dt =
                            chrono::DateTime::from_timestamp_millis(*at_ms).unwrap_or_default();
                        format!("At: {}", dt.to_rfc3339())
                    }
                    crate::scheduler::ScheduleKind::Every { every_ms } => {
                        format!("Every: {}s", every_ms / 1000)
                    }
                    crate::scheduler::ScheduleKind::Cron { cron_expr } => {
                        format!("Cron: {}", cron_expr)
                    }
                };
                out.push_str(&format!(
                    "- Job ID: {} | Schedule: {} | Target: {}:{} | Message: {}\n",
                    job.id, sched_str, job.channel, job.chat_id, job.message
                ));
            }
            return Ok(out);
        }

        if action == "remove" {
            let job_id = args
                .get("job_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'job_id' for remove action")?;
            if let Some(ctx) = exec_ctx.as_ref() {
                let job = if let Some(scheduler) = self.mte_cron_scheduler.as_ref() {
                    scheduler.find_job(job_id)?
                } else {
                    crate::scheduler::CronStore::new(&self.db_path)?.find_job(job_id)?
                };
                let job = job.ok_or("Job was not found in the current conversation")?;
                if !cron_job_is_in_scope(&job, ctx) {
                    return Err("Job does not belong to the current conversation".to_string());
                }
            }
            if let Some(scheduler) = self.mte_cron_scheduler.as_ref() {
                let removed = scheduler.remove_job(job_id, Utc::now()).await?;
                return if removed {
                    Ok(format!("Removed job {}", job_id))
                } else {
                    Ok(format!("Job {} was not found", job_id))
                };
            }
            let cmd = crate::scheduler::CronCommand::Remove {
                id: job_id.to_string(),
            };
            let json_str = serde_json::to_string(&cmd).map_err(|e| e.to_string())?;
            self.cron_node
                .send_packet(json_str)
                .await
                .map_err(|e| e.to_string())?;
            return Ok(format!("Requested removal of job {}", job_id));
        }

        if action == "add" {
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'message' for add action")?;
            let (chat_id, channel) = cron_target_from_args(&args, exec_ctx.as_ref())?;
            let id = uuid::Uuid::new_v4().to_string()[..8].to_string();
            let specified_schedule_count = [
                args.get("every_seconds").is_some(),
                args.get("at").is_some(),
                args.get("cron_expr").is_some(),
            ]
            .into_iter()
            .filter(|present| *present)
            .count();
            if specified_schedule_count != 1 {
                return Err("Must provide exactly one of 'every_seconds', 'at', or 'cron_expr' for add action".to_string());
            }

            let schedule = if let Some(secs) = args.get("every_seconds").and_then(|v| v.as_i64()) {
                if self.multi_tenant_edge_cron_enabled {
                    return Err("every_seconds is not supported when [multi_tenant_edge].cron_scheduling_enabled = true".to_string());
                }
                crate::scheduler::ScheduleKind::Every {
                    every_ms: secs * 1000,
                }
            } else if let Some(at) = args.get("at").and_then(|v| v.as_str()) {
                let dt = chrono::DateTime::parse_from_rfc3339(at).map_err(|_| "Invalid ISO format for 'at'. Make sure you include the proper UTC offset as provided in context.")?;
                let schedule = crate::scheduler::ScheduleKind::At {
                    at_ms: dt.timestamp_millis(),
                };
                if self.multi_tenant_edge_cron_enabled {
                    crate::scheduler::validate_multi_tenant_edge_schedule(&schedule, Utc::now())?;
                }
                schedule
            } else if let Some(expr) = args.get("cron_expr").and_then(|v| v.as_str()) {
                crate::scheduler::validate_cron_expression(expr)?;
                if self.multi_tenant_edge_cron_enabled
                    && !crate::scheduler::is_six_field_cron_expr(expr)
                {
                    return Err("cron_expr must be a 6-field UTC cron expression when [multi_tenant_edge].cron_scheduling_enabled = true".to_string());
                }
                crate::scheduler::ScheduleKind::Cron {
                    cron_expr: expr.to_string(),
                }
            } else {
                unreachable!("Exactly one schedule type is guaranteed by the check above.");
            };

            if let Some(scheduler) = self.mte_cron_scheduler.as_ref() {
                scheduler
                    .add_job(
                        crate::scheduler::ActiveJob {
                            id: id.clone(),
                            schedule,
                            message: message.to_string(),
                            last_run_at_ms: None,
                            chat_id: chat_id.clone(),
                            channel: channel.clone(),
                            webhook_token: crate::scheduler::generate_webhook_token(),
                        },
                        Utc::now(),
                    )
                    .await?;
                return Ok(format!(
                    "Successfully scheduled job {} with action '{}'",
                    id, message
                ));
            }

            let cmd = crate::scheduler::CronCommand::Add {
                id: id.clone(),
                schedule,
                message: message.to_string(),
                chat_id,
                channel,
            };

            let json_str = serde_json::to_string(&cmd).map_err(|e| e.to_string())?;
            self.cron_node
                .send_packet(json_str)
                .await
                .map_err(|e| e.to_string())?;
            return Ok(format!(
                "Successfully scheduled job {} with action '{}'",
                id, message
            ));
        }

        Err(format!("Unknown action '{}'", action))
    }
}

#[cfg(test)]
mod cron_tool_tests {
    use super::CronTool;
    use crate::logging::create_logger_channel;
    use crate::scheduler::{ActiveJob, CronActor, CronSchedulingMode, CronStore, ScheduleKind};
    use crate::tool_runtime::{with_tool_exec_scope, ToolExecCtx};
    use crate::traits::Tool;
    use crate::NodeHandle;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn cron_tool_scopes_list_and_rejects_cross_chat_destinations() {
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().join("agent.db");
        let db_path_str = db_path.to_string_lossy().to_string();
        let store = CronStore::new(&db_path_str).expect("cron store");
        for (id, chat_id) in [("own-job", "chat-1"), ("other-job", "chat-2")] {
            store
                .insert_job(&ActiveJob {
                    id: id.to_string(),
                    schedule: ScheduleKind::Every { every_ms: 60_000 },
                    message: "wake up".to_string(),
                    last_run_at_ms: None,
                    chat_id: chat_id.to_string(),
                    channel: "tauri".to_string(),
                    webhook_token: "test-token".to_string(),
                })
                .expect("insert cron job");
        }

        let (logger, _logger_rx) = create_logger_channel(8);
        let (bus_tx, _bus_rx) = mpsc::channel(1);
        let actor = CronActor::new(
            "test-cron",
            &db_path_str,
            logger,
            CronSchedulingMode::Local,
            bus_tx,
        )
        .expect("cron actor");
        let tool = CronTool {
            cron_node: NodeHandle::new(actor, 8, 1, std::time::Duration::from_millis(1)),
            multi_tenant_edge_cron_enabled: false,
            mte_cron_scheduler: None,
            db_path: db_path_str,
        };

        with_tool_exec_scope(ToolExecCtx::new("tauri", "chat-1", None), async {
            let list = tool
                .execute(serde_json::json!({ "action": "list" }))
                .await
                .expect("scoped list");
            assert!(list.contains("own-job"));
            assert!(!list.contains("other-job"));

            let err = tool
                .execute(serde_json::json!({
                    "action": "add",
                    "message": "wake up",
                    "chat_id": "chat-2",
                    "channel": "tauri",
                    "every_seconds": 60
                }))
                .await
                .expect_err("cross-chat add must fail");
            assert!(err.contains("does not match"));

            let err = tool
                .execute(serde_json::json!({ "action": "remove", "job_id": "other-job" }))
                .await
                .expect_err("cross-chat remove must fail");
            assert!(err.contains("does not belong"));
        })
        .await;
    }
}

/// Message Tool: allows the agent to asynchronously emit proactive status messages
/// directly to the user/channel before the primary generation loop completes.
pub struct MessageTool {
    pub outbound_tx: tokio::sync::mpsc::Sender<crate::bus::BusMessage>,
}

#[async_trait]
impl Tool for MessageTool {
    fn name(&self) -> &str {
        "message"
    }

    fn description(&self) -> &str {
        "Send a message to the user asynchronously. Use this to provide proactive updates or intermediate results while working on long multi-step tasks."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Target channel (e.g., terminal, slack, email)."
                },
                "chat_id": {
                    "type": "string",
                    "description": "Target chat/user ID."
                },
                "thread_id": {
                    "type": "string",
                    "description": "Target thread ID if applicable."
                }
            },
            "required": ["content", "channel", "chat_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'content'")?;
        let channel = args
            .get("channel")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'channel'")?;
        let chat_id = args
            .get("chat_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'chat_id'")?;
        let thread_id = args
            .get("thread_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let msg = crate::bus::BusMessage::Outbound(crate::bus::OutboundMessage {
            channel: channel.to_string(),
            chat_id: chat_id.to_string(),
            thread_id,
            content: content.to_string(),
            metadata: std::collections::HashMap::new(),
        });

        match self.outbound_tx.send(msg).await {
            Ok(_) => Ok(format!("Message sent to {}:{}", channel, chat_id)),
            Err(e) => Err(format!("Failed to send message: {}", e)),
        }
    }
}

/// Wall-clock limit for each `git` subprocess invoked by [`GitWorktreeTool`].
const GIT_WORKTREE_CMD_TIMEOUT_SECS: u64 = 60;
const GIT_WORKTREE_OUTPUT_MAX_CHARS: usize = 10_000;

fn resolve_git_worktree_agent_path(
    path_str: &str,
    workspace_dir: &Path,
    restrict_to_workspace: bool,
    allow_path_outside_sandbox: bool,
) -> Result<PathBuf, String> {
    let enforce_sandbox = restrict_to_workspace && !allow_path_outside_sandbox;
    resolve_path(path_str, workspace_dir, enforce_sandbox)
}

fn validate_optional_branch_name(branch: &str) -> Result<(), String> {
    if branch.is_empty() {
        return Ok(());
    }
    if branch.len() > 244 {
        return Err("branch name is too long (max 244 characters)".to_string());
    }
    if !branch
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
    {
        return Err(
            "branch name may only contain ASCII letters, digits, '-', '_', '.', or '/'".to_string(),
        );
    }
    Ok(())
}

fn truncate_git_worktree_output(mut s: String) -> String {
    if s.len() <= GIT_WORKTREE_OUTPUT_MAX_CHARS {
        return s;
    }
    let mut end = GIT_WORKTREE_OUTPUT_MAX_CHARS;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    let rest = s.len() - end;
    s.truncate(end);
    s.push_str(&format!("\n... (truncated, {} more chars)", rest));
    s
}

async fn run_git_output(
    cwd: &Path,
    args: &[String],
    timeout_secs: u64,
) -> Result<std::process::Output, String> {
    let mut cmd = tokio::process::Command::new("git");
    cmd.current_dir(cwd);
    for a in args {
        cmd.arg(a);
    }
    let fut = cmd.output();
    match timeout(Duration::from_secs(timeout_secs), fut).await {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(e)) => Err(format!("failed to spawn git: {}", e)),
        Err(_) => Err(format!("git command timed out after {}s", timeout_secs)),
    }
}

async fn run_git_checked(cwd: &Path, args: &[String], timeout_secs: u64) -> Result<String, String> {
    let output = run_git_output(cwd, args, timeout_secs).await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = if !stderr.trim().is_empty() {
            stderr.to_string()
        } else {
            stdout.to_string()
        };
        return Err(format!(
            "git {} failed (exit {}): {}",
            args.join(" "),
            output.status.code().unwrap_or(-1),
            detail.trim()
        ));
    }
    let mut s = String::from_utf8_lossy(&output.stdout).to_string();
    let e = String::from_utf8_lossy(&output.stderr);
    if !e.trim().is_empty() {
        if !s.is_empty() {
            s.push('\n');
        }
        s.push_str(&e);
    }
    Ok(s)
}

async fn git_rev_parse_show_toplevel(cwd: &Path) -> Result<PathBuf, String> {
    let out = run_git_checked(
        cwd,
        &["rev-parse".into(), "--show-toplevel".into()],
        GIT_WORKTREE_CMD_TIMEOUT_SECS,
    )
    .await?;
    let line = out.lines().next().unwrap_or("").trim();
    if line.is_empty() {
        return Err("git rev-parse --show-toplevel returned empty output".to_string());
    }
    let p = PathBuf::from(line);
    fs::canonicalize(&p).map_err(|e| format!("could not canonicalize git root: {}", e))
}

async fn git_common_dir_abs(wt_path: &Path) -> Result<PathBuf, String> {
    let out_abs = run_git_output(
        wt_path,
        &[
            "rev-parse".into(),
            "--path-format=absolute".into(),
            "--git-common-dir".into(),
        ],
        GIT_WORKTREE_CMD_TIMEOUT_SECS,
    )
    .await;

    if let Ok(output) = out_abs {
        if output.status.success() {
            let line = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !line.is_empty() {
                let p = PathBuf::from(line);
                if let Ok(c) = fs::canonicalize(&p) {
                    return Ok(c);
                }
            }
        }
    }

    let out = run_git_checked(
        wt_path,
        &["rev-parse".into(), "--git-common-dir".into()],
        GIT_WORKTREE_CMD_TIMEOUT_SECS,
    )
    .await?;
    let line = out.trim();
    if line.is_empty() {
        return Err("git rev-parse --git-common-dir returned empty output".to_string());
    }
    let p = if Path::new(line).is_absolute() {
        PathBuf::from(line)
    } else {
        wt_path.join(line)
    };
    fs::canonicalize(p).map_err(|e| format!("could not canonicalize git common dir: {}", e))
}

fn main_repo_dir_from_common_git_dir(common_dir: &Path) -> PathBuf {
    if common_dir.file_name().and_then(|n| n.to_str()) == Some(".git") && common_dir.is_dir() {
        common_dir
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| common_dir.to_path_buf())
    } else {
        common_dir.to_path_buf()
    }
}

/// Path string passed to `git worktree`. Uses a path relative to `git_root` when possible so
/// Git for Windows is not given `\\?\`-prefixed absolutes (those often fail with "Invalid argument").
fn git_worktree_path_argument(git_root: &Path, wt: &Path) -> String {
    let forward = |p: &Path| p.to_string_lossy().replace('\\', "/");
    if let Ok(r) = wt.strip_prefix(git_root) {
        return forward(r);
    }
    if let Some(parent) = git_root.parent() {
        if let Ok(tail) = wt.strip_prefix(parent) {
            return format!("../{}", forward(tail));
        }
    }
    forward(&strip_windows_extended_path_prefix(wt))
}

fn strip_windows_extended_path_prefix(path: &Path) -> PathBuf {
    #[cfg(not(windows))]
    {
        path.to_path_buf()
    }
    #[cfg(windows)]
    {
        let s = path.to_string_lossy();
        if let Some(rest) = s.strip_prefix("\\\\?\\") {
            if let Some(unc) = rest.strip_prefix("UNC\\") {
                return PathBuf::from(format!("\\\\{}", unc.replace('/', "\\")));
            }
            return PathBuf::from(rest.to_string());
        }
        path.to_path_buf()
    }
}

/// Config-gated `git worktree` helpers (`add`, `remove`, `list`). Paths respect `resolve_path` and
/// optional sandbox relaxation via config (`allow_path_outside_sandbox`).
pub struct GitWorktreeTool {
    pub workspace_dir: PathBuf,
    pub restrict_to_workspace: bool,
    pub allow_path_outside_sandbox: bool,
}

impl GitWorktreeTool {
    async fn action_list(&self, base_path: &str) -> Result<String, String> {
        let base = resolve_git_worktree_agent_path(
            base_path,
            &self.workspace_dir,
            self.restrict_to_workspace,
            self.allow_path_outside_sandbox,
        )?;
        if !base.is_dir() {
            return Err(format!(
                "base_path is not a directory: {:?}",
                base.display()
            ));
        }
        let out = run_git_checked(
            &base,
            &["worktree".into(), "list".into()],
            GIT_WORKTREE_CMD_TIMEOUT_SECS,
        )
        .await?;
        Ok(truncate_git_worktree_output(out))
    }

    async fn action_add(
        &self,
        base_path: &str,
        worktree_path: &str,
        branch: Option<&str>,
    ) -> Result<String, String> {
        let base = resolve_git_worktree_agent_path(
            base_path,
            &self.workspace_dir,
            self.restrict_to_workspace,
            self.allow_path_outside_sandbox,
        )?;
        if !base.is_dir() {
            return Err(format!(
                "base_path is not a directory: {:?}",
                base.display()
            ));
        }
        let git_root = git_rev_parse_show_toplevel(&base).await?;
        let wt = resolve_git_worktree_agent_path(
            worktree_path,
            &self.workspace_dir,
            self.restrict_to_workspace,
            self.allow_path_outside_sandbox,
        )?;
        if wt == git_root {
            return Err("worktree path must not be the same as the repository root".to_string());
        }
        if let Some(parent) = wt.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create parent directories: {}", e))?;
        }
        let branch_name = if let Some(b) = branch.filter(|s| !s.is_empty()) {
            validate_optional_branch_name(b)?;
            b.to_string()
        } else {
            format!("isanagent-wt-{}", uuid::Uuid::new_v4().simple())
        };
        let wt_arg = git_worktree_path_argument(&git_root, &wt);
        let args = vec![
            "worktree".into(),
            "add".into(),
            "-b".into(),
            branch_name.clone(),
            wt_arg,
        ];
        run_git_checked(&git_root, &args, GIT_WORKTREE_CMD_TIMEOUT_SECS).await?;
        let wt_canon = fs::canonicalize(&wt).unwrap_or(wt);
        Ok(format!(
            "Created git worktree.\n  Path: {}\n  Branch: {}\n  Git root: {}",
            wt_canon.display(),
            branch_name,
            git_root.display()
        ))
    }

    async fn action_remove(&self, worktree_path: &str, force: bool) -> Result<String, String> {
        let wt = resolve_git_worktree_agent_path(
            worktree_path,
            &self.workspace_dir,
            self.restrict_to_workspace,
            self.allow_path_outside_sandbox,
        )?;
        if !wt.exists() {
            return Err(format!("worktree path does not exist: {}", wt.display()));
        }
        let wt_canon = fs::canonicalize(&wt).map_err(|e| e.to_string())?;
        let common = git_common_dir_abs(&wt_canon).await?;
        let main_repo = main_repo_dir_from_common_git_dir(&common);
        let mut args = vec!["worktree".into(), "remove".into()];
        if force {
            args.push("--force".into());
        }
        args.push(git_worktree_path_argument(&main_repo, &wt_canon));
        run_git_checked(&main_repo, &args, GIT_WORKTREE_CMD_TIMEOUT_SECS).await?;
        Ok(format!("Removed git worktree at {}", wt_canon.display()))
    }
}

#[async_trait]
impl Tool for GitWorktreeTool {
    fn name(&self) -> &str {
        "git_worktree"
    }

    fn description(&self) -> &str {
        "Manage git worktrees: list linked worktrees, add a new worktree on a fresh branch, or remove one. Requires git on PATH. Only available when enabled in config ([harness.git_worktree]). Worktree paths follow the same sandbox rules as other filesystem tools unless allow_path_outside_sandbox is set there."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "One of: list, add, remove",
                    "enum": ["list", "add", "remove"]
                },
                "base_path": {
                    "type": "string",
                    "description": "Directory inside the repo for git commands (list, add). Defaults to \".\"."
                },
                "path": {
                    "type": "string",
                    "description": "For add: filesystem path for the new worktree. For remove: path of the worktree to remove."
                },
                "branch": {
                    "type": "string",
                    "description": "For add only: new branch name. If omitted, a unique name is generated (isanagent-wt-<hex>)."
                },
                "force": {
                    "type": "boolean",
                    "description": "For remove only: pass --force to git worktree remove."
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid 'action'")?;
        let base_path = args
            .get("base_path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        match action {
            "list" => self.action_list(base_path).await,
            "add" => {
                let path = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or("add requires 'path'")?;
                let branch = args.get("branch").and_then(|v| v.as_str());
                self.action_add(base_path, path, branch).await
            }
            "remove" => {
                let path = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or("remove requires 'path'")?;
                let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
                self.action_remove(path, force).await
            }
            _ => Err(format!(
                "Unknown action {:?}; expected list, add, or remove",
                action
            )),
        }
    }
}

pub struct SearchMemoryTool {
    pub memory_node: NodeHandle<crate::memory::MemoryMessage>,
}

#[async_trait]
impl Tool for SearchMemoryTool {
    fn name(&self) -> &str {
        "search_memory"
    }

    fn description(&self) -> &str {
        "Search your long-term and short-term memory (session summaries) for past context, facts, or keywords."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query'")?;

        // Use oneshot channel to await the reply from the MemoryActor
        let (tx, rx) = tokio::sync::oneshot::channel();
        let msg = crate::memory::MemoryMessage::SearchSummaries {
            query: query.to_string(),
            reply: crate::memory::SharedReply::new(tx),
        };

        self.memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;

        let results = rx
            .await
            .map_err(|_| "Memory Actor Channel Closed".to_string())??;

        if results.is_empty() {
            Ok(format!("No memory results found for '{}'.", query))
        } else {
            Ok(format!(
                "Memory Search Results:\n\n{}",
                results.join("\n\n---\n\n")
            ))
        }
    }
}

pub struct FetchMemoryByDateTool {
    pub memory_node: NodeHandle<crate::memory::MemoryMessage>,
}

#[async_trait]
impl Tool for FetchMemoryByDateTool {
    fn name(&self) -> &str {
        "fetch_memory_by_date"
    }

    fn description(&self) -> &str {
        "Fetch long-term and short-term memory (session summaries) from a specific relative time range, like the last 7 days."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "days_ago": {
                    "type": "integer",
                    "description": "Number of days in the past to search from. For example, 7 means 'within the last 7 days'."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of summaries to return."
                }
            },
            "required": ["days_ago", "limit"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let days_ago = args
            .get("days_ago")
            .and_then(|v| v.as_u64())
            .ok_or("Missing or invalid 'days_ago'")?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let msg = crate::memory::MemoryMessage::FetchSummariesByTimeRange {
            days_ago,
            limit,
            reply: crate::memory::SharedReply::new(tx),
        };

        self.memory_node
            .send_packet(msg)
            .await
            .map_err(|e| e.to_string())?;

        let results = rx
            .await
            .map_err(|_| "Memory Actor Channel Closed".to_string())??;

        if results.is_empty() {
            Ok(format!(
                "No memory results found in the last {} days.",
                days_ago
            ))
        } else {
            Ok(format!(
                "Memory Results (Last {} days):\n\n{}",
                days_ago,
                results.join("\n\n---\n\n")
            ))
        }
    }
}

pub struct GetEnvTool;

#[async_trait]
impl Tool for GetEnvTool {
    fn name(&self) -> &str {
        "get_env"
    }

    fn description(&self) -> &str {
        "Get all environment variables currently exposed to the agent. Sensitive values (tokens, secrets) are masked automatically."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _args: Value) -> Result<String, String> {
        let mut env_vars: Vec<(String, String)> = std::env::vars().collect();
        env_vars.sort_by(|a, b| a.0.cmp(&b.0));

        let mut result = String::from("Environment Variables:\n\n");
        for (k, v) in env_vars {
            let k_lower = k.to_lowercase();
            let masked = if k_lower.contains("token")
                || k_lower.contains("secret")
                || k_lower.contains("key")
                || k_lower.contains("password")
                || k_lower.contains("auth")
            {
                "********".to_string()
            } else {
                v
            };
            result.push_str(&format!("{}={}\n", k, masked));
        }
        Ok(result)
    }
}

pub struct PythonRunTool {
    pub workspace_dir: PathBuf,
}

#[async_trait]
impl Tool for PythonRunTool {
    fn name(&self) -> &str {
        "python_run"
    }

    fn description(&self) -> &str {
        "Run raw python code. The code will be piped to `uv run python -` via stdin, bypassing shell quoting issues while running inside the uv managed environment. Use this for quick calculations, and prefer writing Python scripts to execute with uv for more complex tasks. Outputs stdout/stderr."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let code = args
            .get("code")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'code' argument")?;

        let mut cmd = tokio::process::Command::new("uv");
        cmd.arg("run");
        cmd.arg("python");
        cmd.arg("-");
        cmd.current_dir(&self.workspace_dir);
        // Explicitly forward host environment so secrets/API keys are visible to the child.
        cmd.envs(std::env::vars());
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn python: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin
                .write_all(code.as_bytes())
                .await
                .map_err(|e| format!("Failed to write to python stdin: {}", e))?;
        }

        let output =
            tokio::time::timeout(std::time::Duration::from_secs(60), child.wait_with_output())
                .await
                .map_err(|_| "Python execution timed out after 60 seconds")?
                .map_err(|e| format!("Failed to wait for python: {}", e))?;

        let mut result = String::new();
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            result.push_str(&stdout);
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            if !result.is_empty() {
                result.push_str("\nSTDERR:\n");
            }
            result.push_str(&stderr);
        }

        // Keep this the LAST append to `result`: the agent tail-anchors on the final `Exit code:`
        // line to derive is_error (utils::tool_output_signals_failure). python_run currently has no
        // grep advisory and no size truncation, so the marker is already the final line; if either
        // is ever added here, append it BEFORE this marker (see the exec path in ShellExecTool,
        // which appends the marker last for exactly this invariant).
        if !output.status.success() {
            result.push_str(&format!(
                "\nExit code: {}",
                output.status.code().unwrap_or(-1)
            ));
        }

        if result.is_empty() {
            Ok("(no output)".to_string())
        } else {
            Ok(result)
        }
    }
}

#[cfg(test)]
mod resolve_path_tests {
    use super::resolve_path;
    use std::fs;

    #[test]
    fn parent_dir_at_workspace_root_stays_inside() {
        let tmp = std::env::temp_dir().join(format!("isanagent_rp_{}", uuid::Uuid::new_v4()));
        let ws = tmp.join("workspace");
        fs::create_dir_all(&ws).unwrap();
        let canon = ws.canonicalize().unwrap();
        let got = resolve_path("..", &canon, true).unwrap();
        assert_eq!(got, canon, "expected .. to clamp to workspace root");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn multiple_parents_clamp_to_workspace_root() {
        let tmp = std::env::temp_dir().join(format!("isanagent_rp2_{}", uuid::Uuid::new_v4()));
        let ws = tmp.join("workspace");
        fs::create_dir_all(ws.join("nested")).unwrap();
        let canon = ws.canonicalize().unwrap();
        let got = resolve_path("nested/../../..", &canon, true).unwrap();
        assert_eq!(got, canon);
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Models often pass `workspace/foo` while the tool root is already `.../workspace`.
    #[test]
    fn strips_redundant_workspace_prefix_when_sandbox_leaf_matches() {
        let tmp = std::env::temp_dir().join(format!("isanagent_rp3_{}", uuid::Uuid::new_v4()));
        let ws = tmp.join("workspace");
        fs::create_dir_all(ws.join("pkg")).unwrap();
        let f = ws.join("pkg").join("t.txt");
        fs::write(&f, "ok").unwrap();
        let canon = ws.canonicalize().unwrap();
        let got = resolve_path("workspace/pkg/t.txt", &canon, true).unwrap();
        assert_eq!(got, f.canonicalize().unwrap());
        let _ = fs::remove_dir_all(&tmp);
    }
}

#[cfg(test)]
mod glob_files_tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn glob_files_finds_nested_markdown() {
        let root = std::env::temp_dir().join(format!("isanagent_glob_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("skills").join("cron")).unwrap();
        fs::write(root.join("skills").join("cron").join("SKILL.md"), "# skill").unwrap();

        let tool = GlobFilesTool {
            workspace_dir: root.clone(),
            restrict_to_workspace: false,
        };

        let out = tool
            .execute(json!({ "pattern": "**/*.md", "path": "." }))
            .await
            .unwrap();
        assert!(
            out.contains("SKILL.md"),
            "expected SKILL.md in output, got:\n{}",
            out
        );

        let flat = tool
            .execute(json!({ "pattern": "*.md", "path": "." }))
            .await
            .unwrap();
        assert_eq!(
            flat, "No files matched.",
            "*.md should not match nested files"
        );

        let _ = fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
mod mutation_preview_tests {
    use super::*;
    use serde_json::json;

    fn workspace() -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("isanagent_mutation_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[tokio::test]
    async fn write_preview_records_absent_file_and_approved_write_succeeds() {
        let root = workspace();
        let tool = WriteFileTool {
            workspace_dir: root.clone(),
            restrict_to_workspace: true,
        };
        let args = json!({ "path": "new.txt", "content": "new content\n" });
        let preview = tool.preview_mutation(&args).await.unwrap().unwrap();
        assert_eq!(preview.path, "new.txt");
        assert_eq!(preview.base_fingerprint, "absent");
        assert!(preview.diff.contains("+new content"));

        tool.execute_with_approved_mutation(args, Some(&preview))
            .await
            .unwrap();
        assert_eq!(
            fs::read_to_string(root.join("new.txt")).unwrap(),
            "new content\n"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn approved_write_rejects_intervening_change() {
        let root = workspace();
        let target = root.join("notes.txt");
        fs::write(&target, "before\n").unwrap();
        let tool = WriteFileTool {
            workspace_dir: root.clone(),
            restrict_to_workspace: true,
        };
        let args = json!({ "path": "notes.txt", "content": "approved\n" });
        let preview = tool.preview_mutation(&args).await.unwrap().unwrap();
        fs::write(&target, "changed elsewhere\n").unwrap();

        let error = tool
            .execute_with_approved_mutation(args, Some(&preview))
            .await
            .expect_err("intervening change must invalidate approval");
        assert!(error.contains("changed after approval"), "{error}");
        assert_eq!(fs::read_to_string(&target).unwrap(), "changed elsewhere\n");
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn edit_preview_describes_exact_replacement() {
        let root = workspace();
        fs::write(root.join("notes.txt"), "alpha\nbeta\n").unwrap();
        let tool = EditFileTool {
            workspace_dir: root.clone(),
            restrict_to_workspace: true,
        };
        let args = json!({
            "path": "notes.txt",
            "old_text": "beta",
            "new_text": "gamma"
        });
        let preview = tool.preview_mutation(&args).await.unwrap().unwrap();
        assert!(preview.diff.contains("-beta"), "{}", preview.diff);
        assert!(preview.diff.contains("+gamma"), "{}", preview.diff);
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn write_preview_skips_noop_when_content_unchanged() {
        // PR #62 review #4: writing identical content is a no-op; skip the prompt.
        let root = workspace();
        fs::write(root.join("same.txt"), "identical\n").unwrap();
        let tool = WriteFileTool {
            workspace_dir: root.clone(),
            restrict_to_workspace: true,
        };
        let args = json!({ "path": "same.txt", "content": "identical\n" });
        assert!(
            tool.preview_mutation(&args).await.unwrap().is_none(),
            "identical-content write must be a no-op preview"
        );
        // A genuinely different write still produces a preview.
        let changed = json!({ "path": "same.txt", "content": "new\n" });
        assert!(tool.preview_mutation(&changed).await.unwrap().is_some());
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn edit_preview_skips_noop_when_old_equals_new() {
        // PR #62 review #3: identical old/new text is a no-op; skip the prompt.
        let root = workspace();
        fs::write(root.join("notes.txt"), "alpha\nbeta\n").unwrap();
        let tool = EditFileTool {
            workspace_dir: root.clone(),
            restrict_to_workspace: true,
        };
        let args = json!({
            "path": "notes.txt",
            "old_text": "beta",
            "new_text": "beta"
        });
        assert!(
            tool.preview_mutation(&args).await.unwrap().is_none(),
            "identical old/new text must be a no-op preview"
        );
        let _ = fs::remove_dir_all(root);
    }
}

#[cfg(test)]
mod git_worktree_path_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn outside_absolute_rejected_when_restrict_without_allow() {
        let sandbox =
            std::env::temp_dir().join(format!("isanagent_gwt_s_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&sandbox).unwrap();
        let outside =
            std::env::temp_dir().join(format!("isanagent_gwt_o_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&outside).unwrap();
        let abs = outside.join("wt").to_string_lossy().to_string();
        let res = resolve_git_worktree_agent_path(&abs, &sandbox, true, false);
        assert!(res.is_err(), "expected err, got {:?}", res);
        let _ = fs::remove_dir_all(&sandbox);
        let _ = fs::remove_dir_all(&outside);
    }

    #[test]
    fn outside_absolute_ok_when_allow_outside() {
        let sandbox =
            std::env::temp_dir().join(format!("isanagent_gwt_s2_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&sandbox).unwrap();
        let outside =
            std::env::temp_dir().join(format!("isanagent_gwt_o2_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&outside).unwrap();
        let abs = outside.join("wt").to_string_lossy().to_string();
        let res = resolve_git_worktree_agent_path(&abs, &sandbox, true, true);
        assert!(res.is_ok(), "{:?}", res);
        let _ = fs::remove_dir_all(&sandbox);
        let _ = fs::remove_dir_all(&outside);
    }

    #[tokio::test]
    async fn git_worktree_roundtrip_under_sandbox() {
        if which::which("git").is_err() {
            return;
        }
        let sandbox =
            std::env::temp_dir().join(format!("isanagent_gwt_git_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(sandbox.join("repo")).unwrap();
        let repo = sandbox.join("repo");
        assert!(std::process::Command::new("git")
            .args(["init", "-b", "main"])
            .current_dir(&repo)
            .status()
            .unwrap()
            .success());
        assert!(
            std::process::Command::new("git")
                .args(["commit", "--allow-empty", "-m", "init"])
                .current_dir(&repo)
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .status()
                .unwrap()
                .success(),
            "git commit failed"
        );

        let tool = GitWorktreeTool {
            workspace_dir: sandbox.clone(),
            restrict_to_workspace: true,
            allow_path_outside_sandbox: false,
        };
        let list1 = tool
            .execute(json!({ "action": "list", "base_path": "repo" }))
            .await
            .expect("list");
        assert!(!list1.trim().is_empty(), "list: {}", list1);

        tool.execute(json!({
            "action": "add",
            "base_path": "repo",
            "path": "wt-side",
            "branch": "wt-branch-test"
        }))
        .await
        .expect("add");

        let wt_path = sandbox.join("wt-side");
        assert!(wt_path.is_dir(), "worktree dir missing");

        let list2 = tool
            .execute(json!({ "action": "list", "base_path": "repo" }))
            .await
            .expect("list2");
        assert!(
            list2.contains("wt-side") || list2.contains("wt-branch"),
            "list2: {}",
            list2
        );

        tool.execute(json!({ "action": "remove", "path": "wt-side" }))
            .await
            .expect("remove");
        let _ = fs::remove_dir_all(&sandbox);
    }
}

#[cfg(test)]
mod get_env_tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_get_env_masks_secrets() {
        std::env::set_var("TEST_SAFE_VAR", "hello");
        std::env::set_var("TEST_SECRET_TOKEN", "super_secret");

        let tool = GetEnvTool;
        let out = tool.execute(json!({})).await.unwrap();

        assert!(out.contains("TEST_SAFE_VAR=hello"));
        assert!(out.contains("TEST_SECRET_TOKEN=********"));
        assert!(!out.contains("super_secret"));
    }
}

#[cfg(test)]
mod python_run_tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    #[tokio::test]
    async fn test_python_run_basic() {
        if which::which("python").is_err() && which::which("python3").is_err() {
            return;
        }
        let root = std::env::temp_dir().join(format!("isanagent_py_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();

        let tool = PythonRunTool {
            workspace_dir: root.clone(),
        };
        let out = tool
            .execute(json!({ "code": "print('hello from python')" }))
            .await
            .unwrap();

        println!("PYTHON RUN OUTPUT: {}", out);
        if out.contains("Microsoft Store") && out.contains("9009") {
            println!("Skipping test due to Windows python app execution alias.");
            return;
        }
        assert!(out.contains("hello from python"));
        let _ = fs::remove_dir_all(&root);
    }
}

#[cfg(all(test, unix))]
mod exec_failure_tests {
    use super::*;
    use serde_json::json;

    fn exec_tool() -> ShellExecTool {
        let root = std::env::temp_dir().join(format!("isanagent_exec_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        ShellExecTool {
            workspace_dir: root,
            restrict_to_workspace: false,
        }
    }

    fn last_nonempty_line(s: &str) -> &str {
        s.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("")
    }

    /// Large (>10 KB) failing output: the internal truncation must NOT cut the `Exit code:` marker,
    /// and the agent's `is_error` heuristic must flag it. Regression guard for the truncation gap.
    #[tokio::test]
    async fn large_failing_output_keeps_exit_marker_last() {
        let out = exec_tool()
            .execute(json!({
                "command": "i=0; while [ $i -lt 1500 ]; do echo xxxxxxxxxx; i=$((i+1)); done; exit 7"
            }))
            .await
            .expect("exec returns Ok with the captured output");
        assert!(
            out.len() > 10_000,
            "expected truncation to engage: {}",
            out.len()
        );
        assert!(out.contains("(truncated,"), "expected truncation notice");
        assert_eq!(last_nonempty_line(&out), "Exit code: 7");
        assert!(crate::utils::tool_output_signals_failure("exec", &out));
    }

    /// grep-like failing command: the advisory must not trail (and thus hide) the exit marker.
    #[tokio::test]
    async fn grep_like_failure_keeps_exit_marker_last() {
        let out = exec_tool()
            .execute(json!({
                "command": "grep zzz /isanagent/definitely/missing/path/xyz; exit 2"
            }))
            .await
            .expect("exec returns Ok");
        assert!(
            out.contains("[advisory]"),
            "grep-like advisory expected: {out}"
        );
        assert_eq!(last_nonempty_line(&out), "Exit code: 2");
        assert!(crate::utils::tool_output_signals_failure("exec", &out));
    }

    /// A successful command is never flagged and carries no exit marker.
    #[tokio::test]
    async fn successful_command_has_no_exit_marker() {
        let out = exec_tool()
            .execute(json!({ "command": "echo ok" }))
            .await
            .expect("exec returns Ok");
        assert!(!out.contains("Exit code:"), "no marker on success: {out}");
        assert!(!crate::utils::tool_output_signals_failure("exec", &out));
    }

    #[tokio::test]
    async fn native_result_uses_process_status_not_spoofable_output() {
        let result = exec_tool()
            .execute_with_approved_mutation_typed(
                json!({ "command": "printf 'Exit code: 7\\n'" }),
                None,
            )
            .await;

        assert!(!result.is_error());
        assert_eq!(result.content.trim(), "Exit code: 7");
        assert_eq!(result.error_code(), None);
    }

    #[tokio::test]
    async fn native_result_preserves_real_nonzero_exit() {
        let result = exec_tool()
            .execute_with_approved_mutation_typed(
                json!({ "command": "printf 'failed\\n'; exit 7" }),
                None,
            )
            .await;

        assert!(result.is_error());
        assert_eq!(result.error_code(), Some(ToolErrorCode::NonZeroExit));
        assert_eq!(last_nonempty_line(&result.content), "Exit code: 7");
    }

    /// Output whose 10 KB truncation point lands inside a multi-byte UTF-8 sequence must not panic.
    /// `yes ₺ | head -n 5000 | tr -d '\n'` emits 5000 × '₺' (3 bytes each) = 15000 bytes, so byte
    /// 10000 falls mid-character. Before the char-boundary step-back this panicked on `&result[..10000]`.
    #[tokio::test]
    async fn large_multibyte_output_truncates_on_a_char_boundary() {
        let out = exec_tool()
            .execute(json!({ "command": "yes ₺ | head -n 5000 | tr -d '\\n'" }))
            .await
            .expect("exec returns Ok without panicking on a mid-char truncation");
        // Reaching here at all proves no char-boundary panic (the String is valid UTF-8 by type).
        assert!(out.contains("(truncated,"), "expected truncation notice");
    }
}
