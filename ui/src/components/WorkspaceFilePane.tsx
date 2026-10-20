import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { buildErrorMessage, type ApiErrorPayload } from "@/lib/api";
import { cn } from "@/lib/utils";

type WorkspaceListEntryDto = {
  name: string;
  kind: string;
  size?: number;
};

type WorkspaceListDto = {
  path: string;
  entries: WorkspaceListEntryDto[];
};

type WorkspaceFileDto = {
  path: string;
  content: string;
};

function workspaceParentPath(rel: string): string {
  const parts = rel.split("/").filter(Boolean);
  parts.pop();
  return parts.join("/");
}

function workspaceJoinPath(dir: string, name: string): string {
  return dir ? `${dir}/${name}` : name;
}


function PencilIcon() {
  return (
    <svg aria-hidden className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
      />
    </svg>
  );
}


export function WorkspaceFilePane() {
  const [currentPath, setCurrentPath] = useState("");
  const [entries, setEntries] = useState<WorkspaceListEntryDto[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<WorkspaceFileDto | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editDraft, setEditDraft] = useState("");
  const [savePending, setSavePending] = useState(false);
  const [renamingRel, setRenamingRel] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [renamePending, setRenamePending] = useState(false);

  const loadDirectory = useCallback(async (rel: string) => {
    setLoading(true);
    setError(null);
    try {
      const qs = rel ? `?path=${encodeURIComponent(rel)}` : "";
      const response = await fetch(`/v1/workspace/list${qs}`);
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as ApiErrorPayload;
        const apiMessage = payload?.error?.message?.trim();
        if (apiMessage) {
          throw new Error(apiMessage);
        }
        const hint =
          response.status === 404
            ? " (is the API running on :8080? With Vite dev, the /v1 proxy targets 127.0.0.1:8080. Rebuild isanagent if you use the embedded UI.)"
            : "";
        throw new Error(`Workspace list failed (${response.status})${hint}`);
      }
      const data = (await response.json()) as WorkspaceListDto;
      setCurrentPath(data.path);
      setEntries(data.entries);
    } catch (err) {
      setError(buildErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDirectory("");
  }, [loadDirectory]);

  const openFile = async (rel: string) => {
    setPreviewLoading(true);
    setError(null);
    try {
      const response = await fetch(`/v1/workspace/file?path=${encodeURIComponent(rel)}`);
      if (!response.ok) {
        throw new Error(`Open file failed (${response.status})`);
      }
      const data = (await response.json()) as WorkspaceFileDto;
      setPreview(data);
      setEditing(false);
      setEditDraft("");
    } catch (err) {
      setError(buildErrorMessage(err));
      setPreview(null);
      setEditing(false);
      setEditDraft("");
    } finally {
      setPreviewLoading(false);
    }
  };

  const saveFile = async () => {
    if (!preview) {
      return;
    }
    setSavePending(true);
    setError(null);
    try {
      const response = await fetch("/v1/workspace/file/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: preview.path, content: editDraft }),
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as
          | { error?: { message?: string } }
          | null;
        throw new Error(payload?.error?.message || `Save failed (${response.status})`);
      }
      const data = (await response.json()) as WorkspaceFileDto;
      setPreview(data);
      setEditing(false);
      setEditDraft("");
      void loadDirectory(currentPath);
    } catch (err) {
      setError(buildErrorMessage(err));
    } finally {
      setSavePending(false);
    }
  };

  const cancelRename = () => {
    setRenamingRel(null);
    setRenameDraft("");
  };

  const applyRename = async () => {
    if (!renamingRel) {
      return;
    }
    const newName = renameDraft.trim();
    if (!newName) {
      setError("Enter a name.");
      return;
    }
    if (newName.includes("/") || newName.includes("\\")) {
      setError("Use a single file or folder name (no path separators).");
      return;
    }
    const newRel = workspaceJoinPath(currentPath, newName);
    if (newRel === renamingRel) {
      cancelRename();
      return;
    }
    setRenamePending(true);
    setError(null);
    try {
      const response = await fetch("/v1/workspace/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from: renamingRel, to: newRel }),
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as ApiErrorPayload;
        throw new Error(payload?.error?.message || `Rename failed (${response.status})`);
      }
      const data = (await response.json()) as { path: string };
      const hadPreview = preview?.path === renamingRel;
      cancelRename();
      void loadDirectory(currentPath);
      if (hadPreview) {
        void openFile(data.path);
      }
    } catch (err) {
      setError(buildErrorMessage(err));
    } finally {
      setRenamePending(false);
    }
  };

  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 pb-3">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            Workspace Files
          </p>
          <p className="mt-1 font-mono text-xs text-foreground/80 break-all">
            {currentPath || "/"}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={loading || !currentPath}
            onClick={() => void loadDirectory(workspaceParentPath(currentPath))}
          >
            Up
          </Button>
          <Button type="button" variant="outline" size="sm" disabled={loading} onClick={() => void loadDirectory(currentPath)}>
            Refresh
          </Button>
        </div>
      </div>

      {error ? (
        <p className="mt-3 text-sm text-destructive">{error}</p>
      ) : null}

      {loading ? (
        <p className="mt-4 text-sm text-muted-foreground">Loading workspace…</p>
      ) : (
        <ul className="mt-3 max-h-80 space-y-1 overflow-y-auto rounded-lg border border-border/50 bg-background/60 p-2">
          {entries.length === 0 ? (
            <li className="px-2 py-1 text-sm text-muted-foreground">This folder is empty.</li>
          ) : (
            entries.map((e) => {
              const rel = workspaceJoinPath(currentPath, e.name);
              const isDir = e.kind === "dir";
              const isRenaming = renamingRel === rel;
              return (
                <li key={rel}>
                  {isRenaming ? (
                    <div className="flex flex-col gap-2 rounded-md border border-border/60 bg-background/80 p-2">
                      <label
                        className="text-[10px] font-medium text-muted-foreground"
                        htmlFor={`rename-${rel.replace(/\//g, "-")}`}
                      >
                        New name
                      </label>
                      <div className="flex flex-wrap items-center gap-2">
                        <input
                          id={`rename-${rel.replace(/\//g, "-")}`}
                          type="text"
                          value={renameDraft}
                          onChange={(ev) => setRenameDraft(ev.target.value)}
                          onKeyDown={(ev) => {
                            if (ev.key === "Enter") {
                              ev.preventDefault();
                              void applyRename();
                            }
                            if (ev.key === "Escape") {
                              ev.preventDefault();
                              cancelRename();
                            }
                          }}
                          disabled={renamePending}
                          className={cn(
                            "min-w-0 flex-1 rounded-[6px] border border-border bg-card px-2 py-1.5 font-mono text-xs text-foreground shadow-none",
                            "focus-visible:border-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                            "disabled:cursor-not-allowed disabled:opacity-50",
                          )}
                          autoFocus
                        />
                        <Button
                          type="button"
                          size="sm"
                          className="h-7 text-xs"
                          disabled={renamePending}
                          onClick={() => void applyRename()}
                        >
                          {renamePending ? "…" : "Rename"}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="h-7 text-xs"
                          disabled={renamePending}
                          onClick={cancelRename}
                        >
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-stretch gap-1">
                      <button
                        type="button"
                        className={cn(
                          "flex min-w-0 flex-1 items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted",
                          preview?.path === rel && !isDir ? "bg-muted" : "",
                        )}
                        onClick={() => {
                          if (isDir) {
                            setPreview(null);
                            setEditing(false);
                            setEditDraft("");
                            void loadDirectory(rel);
                          } else {
                            void openFile(rel);
                          }
                        }}
                      >
                        <span className="min-w-0 truncate">
                          {isDir ? "📁 " : "📄 "}
                          {e.name}
                        </span>
                        {!isDir && typeof e.size === "number" ? (
                          <span className="shrink-0 text-[10px] text-muted-foreground">{e.size} B</span>
                        ) : null}
                      </button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-auto shrink-0 px-2 py-1.5 text-muted-foreground hover:text-foreground"
                        title="Rename"
                        aria-label={`Rename ${e.name}`}
                        disabled={loading}
                        onClick={(ev) => {
                          ev.stopPropagation();
                          setRenamingRel(rel);
                          setRenameDraft(e.name);
                          setError(null);
                        }}
                      >
                        <PencilIcon />
                      </Button>
                    </div>
                  )}
                </li>
              );
            })
          )}
        </ul>
      )}

      {previewLoading ? (
        <p className="mt-3 text-xs text-muted-foreground">Loading file…</p>
      ) : null}

      {preview ? (
        <div className="mt-4 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              {editing ? "Edit" : "Preview"}
            </p>
            <div className="flex flex-wrap items-center gap-2">
              {editing ? (
                <>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs"
                    disabled={savePending}
                    onClick={() => {
                      setEditing(false);
                      setEditDraft(preview.content);
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    className="h-7 text-xs"
                    disabled={savePending}
                    onClick={() => void saveFile()}
                  >
                    {savePending ? "Saving…" : "Save"}
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs"
                    disabled={previewLoading}
                    onClick={() => {
                      setEditing(true);
                      setEditDraft(preview.content);
                    }}
                  >
                    Edit
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => {
                      setPreview(null);
                      setEditing(false);
                      setEditDraft("");
                    }}
                  >
                    Close
                  </Button>
                </>
              )}
            </div>
          </div>
          {editing ? (
            <Textarea
              className="min-h-[220px] max-h-80 resize-y font-mono text-xs leading-relaxed"
              value={editDraft}
              onChange={(e) => setEditDraft(e.target.value)}
              spellCheck={false}
            />
          ) : (
            <pre className="max-h-64 overflow-auto rounded-lg border border-border bg-background p-3 text-xs leading-relaxed text-foreground/90 whitespace-pre-wrap break-words">
              {preview.content}
            </pre>
          )}
        </div>
      ) : null}
    </div>
  );
}
