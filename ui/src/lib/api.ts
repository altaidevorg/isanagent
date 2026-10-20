/** Shapes and helpers shared by the chat view and the Work drawer. */

export type ApiErrorPayload = { error?: { code?: string; message?: string } } | null;

export type SummaryEntry = {
  id: number;
  thread_id: string;
  summary: string;
  key_info: string;
  knowledge_gaps: string;
  created_at: string;
};

export type BackgroundJob = {
  job_id: string;
  kind: string;
  chat_id: string;
  state: string;
  updated_at_ms: number;
};

export type NotificationItem = {
  notification_id: string;
  kind: string;
  title: string;
  body: string;
  chat_id: string;
  action_kind?: string | null;
  action_payload?: string | null;
  seen_at_ms?: number | null;
  resolved_at_ms?: number | null;
  created_at_ms: number;
};

export function buildErrorMessage(error: unknown) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Request failed. Try again.";
}

/** Background jobs that still need attention (counted on the Work badge). */
export function isActiveJob(job: BackgroundJob): boolean {
  return !["completed", "failed", "cancelled", "canceled", "dismissed", "done"].includes(job.state);
}

/** A notification that is unread or still waits for a reply. */
export function isUnreadNotification(n: NotificationItem): boolean {
  return !n.seen_at_ms || (n.kind === "clarification_ticket" && !n.resolved_at_ms);
}
