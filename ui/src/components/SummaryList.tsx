import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { SummaryEntry } from "@/lib/api";

/** All stored thread summaries (the memory store), editable in place. */
export function SummaryList({
  summaries,
  loading,
  onUpdate,
  onDelete,
}: {
  summaries: SummaryEntry[];
  loading: boolean;
  onUpdate: (id: number, updated: Partial<SummaryEntry>) => Promise<void>;
  onDelete: (id: number) => void;
}) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editForm, setEditForm] = useState<Partial<SummaryEntry>>({});

  useEffect(() => {
    if (editingId !== null) {
      const s = summaries.find((x) => x.id === editingId);
      if (s) {
        setEditForm({
          summary: s.summary,
          key_info: s.key_info,
          knowledge_gaps: s.knowledge_gaps,
        });
      }
    }
  }, [editingId, summaries]);

  const save = async () => {
    if (editingId !== null) {
      await onUpdate(editingId, editForm);
      setEditingId(null);
    }
  };

  if (loading && summaries.length === 0) {
    return <p className="py-6 text-center text-sm text-muted-foreground">Loading summaries…</p>;
  }
  if (summaries.length === 0) {
    return <p className="py-10 text-center text-sm text-muted-foreground">No memory entries found.</p>;
  }

  return (
    <div className="space-y-6">
      {summaries.map((s) => (
        <div key={s.id} className="space-y-3 border-b border-border pb-6 last:border-0">
          <div className="flex items-center justify-between gap-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            <span className="truncate">Thread: {s.thread_id.split(":")[1] || s.thread_id}</span>
            <span className="shrink-0">{new Date(s.created_at).toLocaleString()}</span>
          </div>

          {editingId === s.id ? (
            <div className="space-y-4 rounded-lg border border-border bg-muted/20 p-3">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Summary</label>
                <Textarea
                  value={editForm.summary}
                  onChange={(e) => setEditForm({ ...editForm, summary: e.target.value })}
                  onKeyDown={(e) => {
                    if (e.key === "Escape") {
                      e.preventDefault();
                      setEditingId(null);
                    }
                  }}
                  className="min-h-[120px] leading-relaxed"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Key Knowledge
                </label>
                <Textarea
                  value={editForm.key_info}
                  onChange={(e) => setEditForm({ ...editForm, key_info: e.target.value })}
                  className="min-h-[80px] leading-relaxed"
                />
              </div>
              <div className="flex justify-end gap-2 pt-1">
                <Button variant="outline" size="sm" onClick={() => setEditingId(null)}>
                  Cancel
                </Button>
                <Button size="sm" onClick={save}>
                  Save
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="space-y-1">
                <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Summary</p>
                <p className="text-sm leading-relaxed text-foreground/90">{s.summary}</p>
              </div>
              {s.key_info ? (
                <div className="space-y-1">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Key Knowledge</p>
                  <p className="text-sm leading-relaxed text-foreground/90">{s.key_info}</p>
                </div>
              ) : null}
              <div className="flex justify-end gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:bg-destructive/10"
                  onClick={() => onDelete(s.id)}
                >
                  Delete
                </Button>
                <Button variant="outline" size="sm" onClick={() => setEditingId(s.id)}>
                  Edit
                </Button>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
