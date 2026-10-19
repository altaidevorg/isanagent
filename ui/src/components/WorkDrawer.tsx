import type { ReactNode } from "react";

import { SummaryList } from "@/components/SummaryList";
import { Button } from "@/components/ui/button";
import { WorkspaceFilePane } from "@/components/WorkspaceFilePane";
import {
  isActiveJob,
  isUnreadNotification,
  type BackgroundJob,
  type NotificationItem,
  type SummaryEntry,
} from "@/lib/api";
import { useDialog } from "@/lib/useDialog";
import { cn } from "@/lib/utils";

export type WorkTab = "jobs" | "files" | "agents" | "summaries";

/** Background job kinds that represent delegated agents rather than plain tool jobs. */
export function isAgentJob(job: BackgroundJob): boolean {
  return /agent|task/i.test(job.kind);
}

type WorkDrawerProps = {
  open: boolean;
  /** Below `lg` the drawer floats over the chat with a backdrop. */
  overlay: boolean;
  tab: WorkTab;
  onTabChange: (tab: WorkTab) => void;
  onClose: () => void;
  jobs: BackgroundJob[];
  notifications: NotificationItem[];
  onRefresh: () => void;
  onJumpToThread: (chatId: string) => void;
  onDismissJob: (job: BackgroundJob) => void;
  onReplyTicket: (notification: NotificationItem) => void;
  onDismissTicket: (notification: NotificationItem) => void;
  summaries: SummaryEntry[];
  summariesLoading: boolean;
  onUpdateSummary: (id: number, updated: Partial<SummaryEntry>) => Promise<void>;
  onDeleteSummary: (id: number) => void;
  /** Bumped each time the drawer opens so the file browser reloads. */
  filesNonce: number;
};

function SectionLabel({ children }: { children: ReactNode }) {
  return <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">{children}</p>;
}

function EmptyNote({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-xl border border-dashed border-[color:var(--ghost-border)] px-4 py-6 text-center text-sm text-muted-foreground">
      {children}
    </p>
  );
}

function JobCard({
  job,
  onJumpToThread,
  onDismissJob,
}: {
  job: BackgroundJob;
  onJumpToThread: (chatId: string) => void;
  onDismissJob: (job: BackgroundJob) => void;
}) {
  return (
    <div
      className={cn(
        "group cursor-pointer rounded-xl border border-border p-3 transition-colors hover:bg-muted/50",
        job.state === "waiting" ? "border-yellow-500/50 bg-yellow-500/5 shadow-sm" : "",
      )}
      role="button"
      tabIndex={0}
      onClick={() => onJumpToThread(job.chat_id)}
      onKeyDown={(e) => {
        if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          onJumpToThread(job.chat_id);
        }
      }}
    >
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="max-w-[60%] truncate font-mono text-[10px] text-muted-foreground">{job.job_id}</span>
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-tighter",
            job.state === "running"
              ? "animate-pulse bg-primary/20 text-primary"
              : job.state === "waiting"
                ? "bg-yellow-500/20 text-yellow-600 dark:text-yellow-400"
                : "bg-muted text-muted-foreground",
          )}
        >
          {job.state}
        </span>
      </div>
      <p className="text-sm font-medium text-foreground">{job.kind}</p>
      <div className="mt-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground">
            Updated {new Date(job.updated_at_ms).toLocaleTimeString()}
          </span>
          <button
            className="text-[10px] text-destructive hover:underline"
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onDismissJob(job);
            }}
          >
            Dismiss
          </button>
        </div>
        <span className="text-[10px] text-primary opacity-0 transition-opacity group-hover:opacity-100">
          View thread →
        </span>
      </div>
    </div>
  );
}

function NotificationCard({
  n,
  onJumpToThread,
  onReplyTicket,
  onDismissTicket,
}: {
  n: NotificationItem;
  onJumpToThread: (chatId: string) => void;
  onReplyTicket: (notification: NotificationItem) => void;
  onDismissTicket: (notification: NotificationItem) => void;
}) {
  const openTicket = n.kind === "clarification_ticket" && !n.resolved_at_ms;
  return (
    <div
      className={cn(
        "rounded-xl border border-border p-3",
        !n.seen_at_ms ? "border-primary/30 bg-primary/5" : "bg-card",
        openTicket ? "border-yellow-500/50 shadow-sm" : "",
      )}
    >
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
          {n.kind.replace(/_/g, " ")}
        </span>
        <span className="text-[10px] text-muted-foreground">{new Date(n.created_at_ms).toLocaleTimeString()}</span>
      </div>
      <h4 className="mb-1 text-sm font-semibold text-foreground">{n.title}</h4>
      <p className="mb-3 text-xs leading-relaxed text-muted-foreground">{n.body}</p>
      <div className="flex flex-wrap gap-2">
        {openTicket ? (
          <Button
            size="sm"
            className="h-7 border-0 bg-yellow-500 text-[10px] font-bold uppercase tracking-widest text-white hover:bg-yellow-600"
            onClick={() => onReplyTicket(n)}
          >
            Reply to request
          </Button>
        ) : null}
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-[10px] font-bold uppercase tracking-widest"
          onClick={() => onJumpToThread(n.chat_id)}
        >
          View context
        </Button>
        {openTicket ? (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 text-[10px] font-bold uppercase tracking-widest text-destructive hover:bg-destructive/10 hover:text-destructive"
            onClick={() => onDismissTicket(n)}
          >
            Dismiss
          </Button>
        ) : null}
      </div>
    </div>
  );
}

/**
 * Right-hand "Work" drawer: background jobs and notifications, workspace files, delegated
 * agents and memory summaries. Sits beside the chat on desktop and slides over it below `lg`.
 */
export function WorkDrawer(props: WorkDrawerProps) {
  const { open, overlay, tab, onTabChange, onClose, jobs, notifications } = props;
  const dialogRef = useDialog<HTMLElement>(open && overlay, onClose);

  const plainJobs = jobs.filter((job) => !isAgentJob(job));
  const agentJobs = jobs.filter(isAgentJob);
  const tabs: { id: WorkTab; label: string; count: number }[] = [
    {
      id: "jobs",
      label: "Jobs",
      count: plainJobs.filter(isActiveJob).length + notifications.filter(isUnreadNotification).length,
    },
    { id: "files", label: "Files", count: 0 },
    { id: "agents", label: "Agents", count: agentJobs.filter(isActiveJob).length },
    { id: "summaries", label: "Summaries", count: 0 },
  ];

  return (
    <>
      {overlay && open ? (
        <div aria-hidden className="fixed inset-0 z-40 bg-black/40 backdrop-blur-[1px] lg:hidden" onClick={onClose} />
      ) : null}
      <aside
        aria-label="Work"
        aria-modal={overlay && open ? true : undefined}
        className={cn(
          "flex min-h-0 flex-col overflow-hidden border-border bg-card outline-none",
          // Off-canvas below lg.
          "fixed inset-y-0 right-0 z-50 w-full max-w-[420px] border-l shadow-2xl transition-transform duration-200 ease-out",
          open ? "visible translate-x-0" : "invisible translate-x-full",
          // Docked column beside the chat on lg+.
          "lg:static lg:z-auto lg:my-4 lg:mr-4 lg:w-[380px] lg:max-w-none lg:translate-x-0 lg:rounded-xl lg:border lg:shadow-panel lg:transition-none xl:w-[420px]",
          open ? "lg:flex" : "lg:hidden",
        )}
        id="work-drawer"
        ref={dialogRef}
        role={overlay ? "dialog" : "complementary"}
      >
        <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
          <h2 className="text-base font-semibold tracking-[-0.01em] text-foreground">Work</h2>
          <div className="flex items-center gap-1">
            <Button size="sm" variant="ghost" className="h-8 text-xs" onClick={props.onRefresh}>
              Refresh
            </Button>
            <Button
              aria-label="Close work panel"
              className="h-8 w-8 p-0 text-lg leading-none"
              size="sm"
              variant="ghost"
              onClick={onClose}
            >
              ×
            </Button>
          </div>
        </div>
        <div className="flex gap-1 border-b border-border px-2 py-1.5" role="tablist">
          {tabs.map((t) => (
            <button
              aria-controls={`work-panel-${t.id}`}
              aria-selected={tab === t.id}
              className={cn(
                "flex flex-1 items-center justify-center gap-1.5 rounded-lg px-2 py-1.5 text-xs font-medium transition-colors",
                tab === t.id
                  ? "bg-[color:var(--ghost-fill)] text-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
              id={`work-tab-${t.id}`}
              key={t.id}
              role="tab"
              type="button"
              onClick={() => onTabChange(t.id)}
            >
              {t.label}
              {t.count > 0 ? (
                <span className="min-w-[1.1rem] rounded-full bg-primary px-1 text-[10px] font-semibold leading-4 text-primary-foreground">
                  {t.count}
                </span>
              ) : null}
            </button>
          ))}
        </div>
        <div
          aria-labelledby={`work-tab-${tab}`}
          className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-4"
          id={`work-panel-${tab}`}
          role="tabpanel"
        >
          {tab === "jobs" ? (
            <div className="space-y-6">
              <section>
                <SectionLabel>Notifications</SectionLabel>
                {notifications.length === 0 ? (
                  <EmptyNote>No notifications</EmptyNote>
                ) : (
                  <div className="space-y-3">
                    {notifications.map((n) => (
                      <NotificationCard
                        key={n.notification_id}
                        n={n}
                        onDismissTicket={props.onDismissTicket}
                        onJumpToThread={props.onJumpToThread}
                        onReplyTicket={props.onReplyTicket}
                      />
                    ))}
                  </div>
                )}
              </section>
              <section>
                <SectionLabel>Background jobs</SectionLabel>
                {plainJobs.length === 0 ? (
                  <EmptyNote>No active background jobs</EmptyNote>
                ) : (
                  <div className="space-y-3">
                    {plainJobs.map((job) => (
                      <JobCard
                        job={job}
                        key={job.job_id}
                        onDismissJob={props.onDismissJob}
                        onJumpToThread={props.onJumpToThread}
                      />
                    ))}
                  </div>
                )}
              </section>
            </div>
          ) : null}
          {tab === "files" ? <WorkspaceFilePane key={props.filesNonce} /> : null}
          {tab === "agents" ? (
            agentJobs.length === 0 ? (
              <EmptyNote>
                No delegated agents running in the background. Sub-agents started during a chat turn show up in
                that turn's tool calls.
              </EmptyNote>
            ) : (
              <div className="space-y-3">
                {agentJobs.map((job) => (
                  <JobCard
                    job={job}
                    key={job.job_id}
                    onDismissJob={props.onDismissJob}
                    onJumpToThread={props.onJumpToThread}
                  />
                ))}
              </div>
            )
          ) : null}
          {tab === "summaries" ? (
            <SummaryList
              loading={props.summariesLoading}
              summaries={props.summaries}
              onDelete={props.onDeleteSummary}
              onUpdate={props.onUpdateSummary}
            />
          ) : null}
        </div>
      </aside>
    </>
  );
}
