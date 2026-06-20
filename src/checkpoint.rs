//! Pre-edit file backups for one-step undo.
//!
//! Before `edit_file` / `write_file` mutate a file, the prior content (or its absence, for a
//! newly-created file) is saved here, so an edit can be rolled back via the `checkpoint` tool.
//! **Only files the agent actually touches are backed up** — datasets/models in an ML workspace are
//! never snapshotted — so this is safe regardless of workspace size, unlike a whole-tree snapshot.
//!
//! Opt-in via `checkpoint_enabled = true`; entirely inert otherwise (the global store is unset, so
//! [`snapshot_before`] is a no-op and the `checkpoint` tool reports that it's disabled).

use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::traits::Tool;

static STORE: OnceLock<CheckpointStore> = OnceLock::new();

/// Cap on retained checkpoints; the oldest beyond this are pruned on each new snapshot so an
/// always-on agent doing thousands of edits can't grow `.system_generated/checkpoints` without bound.
const MAX_CHECKPOINTS: usize = 200;

/// Initialise the process-wide checkpoint store. `root` holds the backups; `base`, when `Some`,
/// confines restores to within that directory (the sandbox, when the file tools are workspace-
/// restricted). Call once at startup when checkpointing is enabled.
pub fn init(root: PathBuf, base: Option<PathBuf>) {
    let _ = STORE.set(CheckpointStore::new(root, base));
}

/// Back up `path` before it is mutated. No-op when checkpointing is disabled. Best-effort: a backup
/// failure is logged, never propagated, so it can't break the edit.
pub fn snapshot_before(path: &Path, label: &str) {
    if let Some(store) = STORE.get() {
        if let Err(e) = store.snapshot(path, label) {
            log::warn!("checkpoint snapshot failed for {}: {}", path.display(), e);
        }
    }
}

/// The store, for the `checkpoint` tool. `None` when disabled.
pub fn store() -> Option<&'static CheckpointStore> {
    STORE.get()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Meta {
    /// Absolute original path that was (or would be) mutated.
    path: String,
    /// The tool that triggered the snapshot (e.g. `edit_file`).
    label: String,
    /// `false` when the file did not exist pre-edit — restoring such an entry removes the file.
    existed: bool,
    created_ms: u128,
}

/// One backed-up pre-edit state.
pub struct CheckpointEntry {
    pub id: String,
    pub path: String,
    pub label: String,
    pub created_ms: u128,
    pub existed: bool,
}

pub struct CheckpointStore {
    root: PathBuf,
    /// When `Some`, restores are confined to within this directory (workspace-restricted mode).
    base: Option<PathBuf>,
}

impl CheckpointStore {
    pub fn new(root: PathBuf, base: Option<PathBuf>) -> Self {
        Self { root, base }
    }

    fn snapshot(&self, path: &Path, label: &str) -> std::io::Result<()> {
        let id = uuid::Uuid::new_v4().to_string();
        let dir = self.root.join(&id);
        std::fs::create_dir_all(&dir)?;
        let existed = path.exists();
        if existed {
            std::fs::copy(path, dir.join("content"))?;
        }
        let meta = Meta {
            path: path.to_string_lossy().into_owned(),
            label: label.to_string(),
            existed,
            created_ms: now_ms(),
        };
        let bytes = serde_json::to_vec(&meta)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(dir.join("meta.json"), bytes)?;
        self.prune();
        Ok(())
    }

    /// Bound disk growth: keep only the most recent [`MAX_CHECKPOINTS`], pruning the oldest.
    /// Best-effort — a prune failure never fails the snapshot.
    fn prune(&self) {
        let Ok(rd) = std::fs::read_dir(&self.root) else {
            return;
        };
        // Sort checkpoint dirs by mtime (≈ creation time — a checkpoint is written once and never
        // modified afterwards) rather than reading and JSON-parsing every meta.json. prune runs on
        // every snapshot, so this keeps it O(n) syscalls instead of O(n) file reads + parses.
        let mut dirs: Vec<(PathBuf, std::time::SystemTime)> = rd
            .flatten()
            .filter_map(|e| {
                // Use the dirent `file_type()` (populated from `d_type` without a stat syscall on
                // Linux/macOS) instead of `path().is_dir()`. It also does NOT follow symlinks, so a
                // stray symlink dropped into the store is skipped rather than traversed. filter_map
                // resolves `e.path()` once instead of in both a filter and a map.
                if !e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                    return None;
                }
                let modified = e
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                Some((e.path(), modified))
            })
            .collect();
        if dirs.len() <= MAX_CHECKPOINTS {
            return;
        }
        dirs.sort_by_key(|(_, modified)| *modified); // oldest first
        let to_remove = dirs.len() - MAX_CHECKPOINTS;
        for (path, _) in dirs.into_iter().take(to_remove) {
            let _ = std::fs::remove_dir_all(path);
        }
    }

    /// All checkpoints, newest first.
    pub fn list(&self) -> Vec<CheckpointEntry> {
        let mut entries = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.root) else {
            return entries;
        };
        for e in rd.flatten() {
            let Ok(bytes) = std::fs::read(e.path().join("meta.json")) else {
                continue;
            };
            let Ok(m) = serde_json::from_slice::<Meta>(&bytes) else {
                continue;
            };
            if let Some(id) = e.file_name().to_str().map(String::from) {
                entries.push(CheckpointEntry {
                    id,
                    path: m.path,
                    label: m.label,
                    created_ms: m.created_ms,
                    existed: m.existed,
                });
            }
        }
        entries.sort_by_key(|e| std::cmp::Reverse(e.created_ms));
        entries
    }

    /// Restore the file recorded by checkpoint `id` to its pre-edit state.
    pub fn restore(&self, id: &str) -> Result<String, String> {
        // Confine the lookup to a direct child of the store.
        if id.is_empty() || id.contains('/') || id.contains('\\') || id.contains("..") {
            return Err("invalid checkpoint id".to_string());
        }
        let dir = self.root.join(id);
        let meta_bytes = std::fs::read(dir.join("meta.json"))
            .map_err(|e| format!("checkpoint {id} not found: {e}"))?;
        let m: Meta = serde_json::from_slice(&meta_bytes)
            .map_err(|e| format!("checkpoint meta parse: {e}"))?;
        let target = PathBuf::from(&m.path);
        if m.existed {
            // SECURITY: re-validate the write target against the LIVE filesystem, not just the
            // recorded string. The agent could have swapped the path (or a parent) for a symlink
            // after the snapshot, and `fs::copy` follows symlinks — a lexical check on the unchanged
            // string would let that redirect the write outside the sandbox (TOCTOU). So in restricted
            // mode we refuse a symlinked final component and re-resolve through the same boundary the
            // file tools use (which canonicalizes, catching a symlinked parent that resolves out).
            let dest = self.safe_write_target(&target, &m.path)?;
            std::fs::copy(dir.join("content"), &dest).map_err(|e| format!("restore copy: {e}"))?;
            Ok(format!("Restored {} from checkpoint {}.", m.path, id))
        } else {
            // The snapshotted edit created the file; undo = remove it. `remove_file` unlinks a
            // symlink itself (not its target) at the FINAL component, but it still traverses
            // symlinks in PARENT directories — so a lexical `starts_with(base)` check is not enough:
            // an agent could swap a parent dir for a symlink after the snapshot and redirect the
            // unlink outside the sandbox (TOCTOU). Resolve the parent through the canonicalizing
            // boundary and rejoin the final component before deleting.
            let safe_target = self.safe_delete_target(&target)?;
            match std::fs::remove_file(&safe_target) {
                Ok(()) => Ok(format!(
                    "Removed {} (it was created after checkpoint {}).",
                    m.path, id
                )),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    Ok(format!("{} is already absent.", m.path))
                }
                Err(e) => Err(format!("restore remove: {e}")),
            }
        }
    }

    /// Resolve a safe destination for a restore *write*, closing the symlink/TOCTOU escape. In
    /// unrestricted mode (`base == None`) edits already go anywhere, so the recorded path is used
    /// as-is. In restricted mode the final component must not be a symlink, and the path is
    /// re-resolved through `resolve_path` (the same canonicalizing boundary edits use).
    fn safe_write_target(&self, target: &Path, raw: &str) -> Result<PathBuf, String> {
        let Some(base) = &self.base else {
            return Ok(target.to_path_buf());
        };
        if std::fs::symlink_metadata(target)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false)
        {
            return Err("checkpoint target is now a symlink; refusing to restore".to_string());
        }
        crate::tools::builtin::resolve_path(raw, base, true)
            .map_err(|e| format!("checkpoint target rejected: {e}"))
    }

    /// Resolve a safe target for a restore *delete* (undo of a created file), closing the
    /// parent-symlink/TOCTOU escape. In unrestricted mode (`base == None`) the recorded path is used
    /// as-is. In restricted mode the parent directory is re-resolved through `resolve_path` (which
    /// canonicalizes, rejecting a parent that escapes the boundary) and the original final component
    /// is rejoined — so a parent symlink can't redirect the unlink, while a symlink *at* the final
    /// component is still unlinked itself (not its target), which is the correct undo.
    fn safe_delete_target(&self, target: &Path) -> Result<PathBuf, String> {
        let Some(base) = &self.base else {
            return Ok(target.to_path_buf());
        };
        let parent = target
            .parent()
            .ok_or_else(|| "checkpoint target has no parent directory".to_string())?;
        let safe_parent =
            crate::tools::builtin::resolve_path(&parent.to_string_lossy(), base, true)
                .map_err(|e| format!("checkpoint target parent rejected: {e}"))?;
        // Defense in depth: `resolve_path` canonicalizes *existing* components but appends a
        // non-existent trailing component lexically, so it can return a path whose final parent
        // segment was not actually resolved through the symlink boundary. Re-check the resolved
        // parent here and refuse if it is itself a symlink — mirroring the final-component guard on
        // the write path (`safe_write_target`). A missing parent is fine: the created file is then
        // already gone, and the caller's NotFound arm reports it as "already absent". This narrows
        // the residual window to a sub-`remove_file` race; a fully race-free unlink would require
        // `openat2(RESOLVE_NO_SYMLINKS)` + `unlinkat` (tracked as a follow-up).
        match std::fs::symlink_metadata(&safe_parent) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(
                    "checkpoint target parent is now a symlink; refusing to delete".to_string(),
                )
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("checkpoint target parent stat: {e}")),
        }
        let file_name = target
            .file_name()
            .ok_or_else(|| "checkpoint target has no file name".to_string())?;
        Ok(safe_parent.join(file_name))
    }
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// `checkpoint` tool: list pre-edit backups and restore one (one-step undo for `edit_file` /
/// `write_file`). Registered only when checkpointing is enabled.
pub struct CheckpointTool;

#[async_trait]
impl Tool for CheckpointTool {
    fn name(&self) -> &str {
        "checkpoint"
    }

    fn description(&self) -> &str {
        "List or restore pre-edit file checkpoints — a one-step undo for edit_file/write_file. \
         action 'list' shows recent checkpoints (newest first); action 'restore' with an 'id' rolls \
         that file back to its state before the edit (removing it if the edit had created it)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["list", "restore"], "description": "list (default) or restore" },
                "id": { "type": "string", "description": "checkpoint id to restore (required when action=restore)" }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let Some(store) = store() else {
            return Ok("Checkpointing is disabled (set checkpoint_enabled = true).".to_string());
        };
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("list");
        match action {
            "list" => {
                let entries = store.list();
                if entries.is_empty() {
                    return Ok("No checkpoints.".to_string());
                }
                let mut out = String::from("Checkpoints (newest first):\n");
                for e in entries.iter().take(50) {
                    out.push_str(&format!(
                        "- {} [{}] {}{}\n",
                        e.id,
                        e.label,
                        e.path,
                        if e.existed { "" } else { " (created)" }
                    ));
                }
                Ok(out)
            }
            "restore" => {
                let id = args
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("restore requires 'id'")?;
                store.restore(id)
            }
            other => Err(format!("unknown action '{other}' (use list or restore)")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp() -> (PathBuf, CheckpointStore) {
        let base = std::env::temp_dir().join(format!("isan_ckpt_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        let store = CheckpointStore::new(base.join(".checkpoints"), Some(base.clone()));
        (base, store)
    }

    #[test]
    fn snapshot_then_restore_recovers_prior_content() {
        let (base, store) = temp();
        let file = base.join("a.txt");
        std::fs::write(&file, "v1").unwrap();

        store.snapshot(&file, "edit_file").unwrap();
        std::fs::write(&file, "v2-broken").unwrap(); // simulate a bad edit

        let entries = store.list();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].label, "edit_file");
        assert!(entries[0].existed);

        store.restore(&entries[0].id).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "v1");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn restoring_a_created_file_removes_it() {
        let (base, store) = temp();
        let file = base.join("new.txt");
        // File does not exist yet -> snapshot records a creation.
        store.snapshot(&file, "write_file").unwrap();
        std::fs::write(&file, "created").unwrap();

        let id = store.list()[0].id.clone();
        assert!(!store.list()[0].existed);
        store.restore(&id).unwrap();
        assert!(
            !file.exists(),
            "restoring a creation should remove the file"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn restore_rejects_bad_ids() {
        let (base, store) = temp();
        assert!(store.restore("../etc").is_err());
        assert!(store.restore("a/b").is_err());
        assert!(store.restore("missing-id").is_err());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn restore_refuses_meta_path_outside_base() {
        let (base, store) = temp();
        // Craft a checkpoint whose meta.path points OUTSIDE the base (a tampered/forged meta).
        let outside =
            std::env::temp_dir().join(format!("isan_outside_{}.txt", uuid::Uuid::new_v4()));
        let dir = store.root.join("crafted");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("content"), b"payload").unwrap();
        let meta = format!(
            r#"{{"path":{:?},"label":"edit_file","existed":true,"created_ms":1}}"#,
            outside.to_string_lossy()
        );
        std::fs::write(dir.join("meta.json"), meta).unwrap();

        assert!(
            store.restore("crafted").is_err(),
            "must refuse out-of-base meta"
        );
        assert!(!outside.exists(), "must not write the payload outside base");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn restore_refuses_symlinked_target() {
        let (base, store) = temp();
        let file = base.join("real.txt");
        std::fs::write(&file, "v1").unwrap();
        store.snapshot(&file, "edit_file").unwrap();
        std::fs::write(&file, "v2").unwrap();
        let id = store.list()[0].id.clone();

        // Agent swaps the target for a symlink pointing outside the base (TOCTOU).
        let outside = std::env::temp_dir().join(format!("isan_symout_{}", uuid::Uuid::new_v4()));
        std::fs::remove_file(&file).unwrap();
        std::os::unix::fs::symlink(&outside, &file).unwrap();

        let res = store.restore(&id);
        assert!(res.is_err(), "must refuse a symlinked target: {res:?}");
        assert!(!outside.exists(), "must not write through the symlink");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn restore_delete_refuses_parent_symlink_escape() {
        // Undo of a *created* file deletes it. A lexical containment check would miss a TOCTOU where
        // a PARENT directory is swapped for a symlink pointing outside the sandbox: `remove_file`
        // follows parent symlinks, so it would unlink a victim file outside the workspace.
        let (base, store) = temp();
        let sub = base.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let created = sub.join("created.txt");
        store.snapshot(&created, "write_file").unwrap(); // file doesn't exist yet -> records creation
        std::fs::write(&created, "created").unwrap();
        let id = store.list()[0].id.clone();
        assert!(!store.list()[0].existed);

        // A victim outside the sandbox, reachable only by redirecting the parent dir.
        let outside_dir =
            std::env::temp_dir().join(format!("isan_victim_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&outside_dir).unwrap();
        let victim = outside_dir.join("created.txt");
        std::fs::write(&victim, "do not delete me").unwrap();

        // Swap base/sub for a symlink -> outside_dir (TOCTOU on a parent component).
        std::fs::remove_dir_all(&sub).unwrap();
        std::os::unix::fs::symlink(&outside_dir, &sub).unwrap();

        let res = store.restore(&id);
        assert!(res.is_err(), "must refuse a parent-symlink escape: {res:?}");
        assert!(
            victim.exists(),
            "must not delete the file outside the sandbox"
        );
        let _ = std::fs::remove_dir_all(&outside_dir);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn list_is_newest_first() {
        let (base, store) = temp();
        let f = base.join("x");
        std::fs::write(&f, "1").unwrap();
        store.snapshot(&f, "edit_file").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        store.snapshot(&f, "write_file").unwrap();
        let entries = store.list();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].created_ms >= entries[1].created_ms);
        let _ = std::fs::remove_dir_all(&base);
    }
}
