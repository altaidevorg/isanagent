//! `.isanagentignore` enforcement for the agent's own file tools
//! (`read_file`, `list_dir`, `glob_files`, `search_text`).
//!
//! A workspace-root file with gitignore-style patterns. Walkers honor nested
//! copies; single-path tools check the nearest ancestor file. Mirrors the
//! spec implemented on the host side (altai-app's `fs::isanagentignore`) so a
//! project's `.isanagentignore` governs both the editor surface and the agent.
//!
//! Built on the `ignore` crate so negation (`!`), trailing-slash directory
//! rules, and `matched_path_or_any_parents` semantics line up with `.gitignore`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use ignore::Match;

/// Filename the agent's file tools treat as an ignore file (gitignore syntax).
pub const IGNORE_FILENAME: &str = ".isanagentignore";

struct CachedMatcher {
    matcher: Gitignore,
    mtime: Option<SystemTime>,
}

/// Process-global cache keyed by the directory that owns the `.isanagentignore`.
/// `Gitignore` is not `Clone`, so matching is performed under the lock.
fn cache() -> &'static Mutex<HashMap<PathBuf, CachedMatcher>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, CachedMatcher>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Walk up from `target`'s parent chain to the nearest directory containing a
/// `.isanagentignore`. Returns `(owning_dir, ignore_file_path)`, or `None` when
/// no ancestor carries the file. Mirrors gitignore ancestor resolution.
fn find_ignore_root(target: &Path) -> Option<(PathBuf, PathBuf)> {
    let start = if target.is_dir() {
        target.to_path_buf()
    } else {
        target.parent()?.to_path_buf()
    };
    let mut cursor: Option<&Path> = Some(&start);
    while let Some(dir) = cursor {
        let candidate = dir.join(IGNORE_FILENAME);
        if candidate.is_file() {
            return Some((dir.to_path_buf(), candidate));
        }
        cursor = dir.parent();
    }
    None
}

/// Path of the nearest ancestor `.isanagentignore`, for ripgrep's `--ignore-file`.
/// `None` when no such file applies to `start`.
pub fn find_ignore_file(start: &Path) -> Option<PathBuf> {
    find_ignore_root(start).map(|(_, file)| file)
}

/// Compile a `Gitignore` matcher from `<dir>/.isanagentignore`. Comments and
/// blank lines are skipped (they would otherwise be treated as literal
/// patterns). `None` when the file is missing, unreadable, or holds no pattern.
fn build_matcher(dir: &Path) -> Option<Gitignore> {
    let path = dir.join(IGNORE_FILENAME);
    let body = std::fs::read_to_string(&path).ok()?;
    let mut builder = GitignoreBuilder::new(dir);
    let mut added = 0usize;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Err(e) = builder.add_line(None, line) {
            log::debug!(".isanagentignore pattern rejected {line:?}: {e}");
            continue;
        }
        added += 1;
    }
    if added == 0 {
        return None;
    }
    builder.build().ok()
}

/// True if `target` is blocked by a `.isanagentignore` in any ancestor.
/// `is_dir` controls directory-pattern matching (trailing-slash rules).
///
/// Returns `false` when no `.isanagentignore` applies, the matcher accepts the
/// path, or `target` is the owning root itself.
pub fn is_ignored(target: &Path, is_dir: bool) -> bool {
    let Some((dir, file_path)) = find_ignore_root(target) else {
        return false;
    };
    let mtime = std::fs::metadata(&file_path)
        .and_then(|m| m.modified())
        .ok();

    let mut guard = cache().lock().expect("isanagentignore cache poisoned");
    let needs_rebuild = guard.get(&dir).is_none_or(|entry| entry.mtime != mtime);
    if needs_rebuild {
        match build_matcher(&dir) {
            Some(m) => {
                guard.insert(dir.clone(), CachedMatcher { matcher: m, mtime });
            }
            None => {
                guard.remove(&dir);
                return false;
            }
        }
    }
    let entry = guard.get(&dir).expect("just inserted or kept");
    let rel = match target.strip_prefix(&dir) {
        Ok(r) => r,
        Err(_) => return false,
    };
    if rel.as_os_str().is_empty() {
        return false;
    }
    // `matched_path_or_any_parents` walks ancestor components so a file inside
    // an ignored directory is caught — matches `.gitignore` semantics.
    matches!(
        entry.matcher.matched_path_or_any_parents(rel, is_dir),
        Match::Ignore(_)
    )
}

/// Drop every cached matcher so the next call re-reads from disk. Tests call
/// this between cases so edits land immediately regardless of mtime.
pub fn invalidate_all() {
    let mut guard = cache().lock().expect("isanagentignore cache poisoned");
    guard.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_ignore(dir: &Path, body: &str) {
        fs::write(dir.join(IGNORE_FILENAME), body).unwrap();
    }

    #[test]
    fn no_ignore_file_means_not_ignored() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("note.txt");
        fs::write(&file, b"x").unwrap();
        invalidate_all();
        assert!(!is_ignored(&file, false));
    }

    #[test]
    fn denies_matching_path() {
        let dir = tempdir().unwrap();
        write_ignore(dir.path(), "secrets/**\n");
        let secret = dir.path().join("secrets").join("api.key");
        fs::create_dir_all(secret.parent().unwrap()).unwrap();
        fs::write(&secret, b"x").unwrap();
        invalidate_all();
        assert!(is_ignored(&secret, false));
    }

    #[test]
    fn allows_unrelated_path() {
        let dir = tempdir().unwrap();
        write_ignore(dir.path(), "secrets/**\n");
        let readme = dir.path().join("README.md");
        fs::write(&readme, b"x").unwrap();
        invalidate_all();
        assert!(!is_ignored(&readme, false));
    }

    #[test]
    fn honors_negation() {
        let dir = tempdir().unwrap();
        write_ignore(dir.path(), "*.log\n!important.log\n");
        let ordinary = dir.path().join("debug.log");
        let important = dir.path().join("important.log");
        fs::write(&ordinary, b"x").unwrap();
        fs::write(&important, b"x").unwrap();
        invalidate_all();
        assert!(is_ignored(&ordinary, false));
        assert!(!is_ignored(&important, false));
    }

    #[test]
    fn honors_nested_dir_pattern() {
        let dir = tempdir().unwrap();
        write_ignore(dir.path(), "build/\n");
        let artifact = dir.path().join("a").join("b").join("build").join("out.txt");
        fs::create_dir_all(artifact.parent().unwrap()).unwrap();
        fs::write(&artifact, b"x").unwrap();
        invalidate_all();
        assert!(is_ignored(&artifact, false));
        assert!(is_ignored(artifact.parent().unwrap(), true));
    }

    #[test]
    fn find_ignore_file_returns_nearest_ancestor() {
        let root = tempdir().unwrap();
        write_ignore(root.path(), "*.env\n");
        let sub = root.path().join("a").join("b");
        fs::create_dir_all(&sub).unwrap();
        invalidate_all();
        let found = find_ignore_file(&sub).unwrap();
        assert_eq!(found, root.path().join(IGNORE_FILENAME));
    }

    #[test]
    fn find_ignore_file_none_when_absent() {
        let dir = tempdir().unwrap();
        invalidate_all();
        assert!(find_ignore_file(dir.path()).is_none());
    }
}
