import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useTransition,
  type MouseEvent,
} from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const THREAD_ID_STORAGE_KEY = "isanagent_thread_id";
const SESSION_RESPONSE_KEY = "isanagent_latest_response_id";
/** Stable API `user` / sender_id: localStorage (shared across tabs). Session list keys off this. */
const SESSION_USER_KEY = "isanagent_api_user_id";
const THEME_STORAGE_KEY = "isanagent-theme";
const SIDEBAR_HINTS_KEY = "isanagent_sidebar_hints";

type ThemeMode = "light" | "dark";

function readThemeFromDocument(): ThemeMode {
  if (typeof document === "undefined") {
    return "dark";
  }
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

function applyTheme(mode: ThemeMode) {
  if (typeof document === "undefined") {
    return;
  }
  document.documentElement.classList.toggle("dark", mode === "dark");
  try {
    localStorage.setItem(THEME_STORAGE_KEY, mode);
  } catch {
    /* private mode */
  }
}

function loadSidebarHints(): Record<string, string> {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const raw = localStorage.getItem(SIDEBAR_HINTS_KEY);
    if (!raw) {
      return {};
    }
    const o = JSON.parse(raw) as Record<string, unknown>;
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(o)) {
      if (typeof v === "string" && v.trim().length > 0) {
        out[k] = v.trim().slice(0, 72);
      }
    }
    return out;
  } catch {
    return {};
  }
}

function persistSidebarHints(next: Record<string, string>) {
  try {
    localStorage.setItem(SIDEBAR_HINTS_KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
}

/** First line, truncated like the server preview. */
function truncateSidebarTitle(text: string, maxChars = 56): string {
  const line = text.split("\n")[0]?.trim() ?? "";
  if (!line) {
    return "";
  }
  const chars = [...line];
  if (chars.length <= maxChars) {
    return line;
  }
  return `${chars.slice(0, maxChars).join("")}…`;
}

function threadSidebarLabel(entry: ThreadListEntry, hints: Record<string, string>): string {
  const fromServer = entry.preview?.trim() ?? "";
  if (fromServer.length > 0) {
    return fromServer;
  }
  const fromHint = hints[entry.thread_id]?.trim() ?? "";
  if (fromHint.length > 0) {
    return fromHint;
  }
  return "…";
}

type Message = {
  id: string;
  role: string;
  content: string;
  imageUrls?: string[];
  toolCalls?: ToolCall[];
};

type ToolCall = {
  tool_name: string;
  args: string;
  result?: string;
};

type StreamEvent =
  | { type: "tool_call_started"; tool_name: string; args: string }
  | { type: "tool_progress"; tool_name: string; message: string }
  | { type: "tool_call_finished"; tool_name: string; result: string }
  | { type: "agent_thought"; thought: string }
  | { type: "completion"; content: string; thread_id: string; response_id: string }
  | { type: "error"; message: string };

type HistoryRow = {
  role: string;
  content: string;
  image_urls?: string[];
};

type ThreadListEntry = {
  thread_id: string;
  updated_at: number;
  latest_response_id: string;
  /** From API; older servers may omit. */
  preview?: string;
};

type SummaryEntry = {
  id: number;
  thread_id: string;
  summary: string;
  key_info: string;
  knowledge_gaps: string;
  created_at: string;
};

type BackgroundJob = {
  job_id: string;
  kind: string;
  chat_id: string;
  state: string;
  updated_at_ms: number;
};

type NotificationItem = {
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

type ApiErrorPayload = { error?: { code?: string; message?: string } } | null;

function pickResponseId(raw: unknown): string | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const id = (raw as { id?: unknown }).id;
  return typeof id === "string" && id.length > 0 ? id : null;
}

function extractOutputText(raw: unknown): string {
  if (!raw || typeof raw !== "object") {
    return "(No assistant text in response.)";
  }
  const output = (raw as { output?: unknown }).output;
  if (!Array.isArray(output) || output.length === 0) {
    return "(No assistant text in response.)";
  }
  const first = output[0] as { content?: unknown };
  const parts = first?.content;
  if (!Array.isArray(parts)) {
    return "(No assistant text in response.)";
  }
  const textPart = parts.find(
    (c: unknown) =>
      typeof c === "object" &&
      c !== null &&
      (c as { type?: string }).type === "output_text" &&
      typeof (c as { text?: string }).text === "string",
  ) as { text?: string } | undefined;
  return textPart?.text?.trim() || "(No assistant text in response.)";
}

function createId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `id_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

/** UUID for `thread_id` on new API turns (cancel/stop must work before SSE arrives). */
function newThreadUuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0"));
    return (
      hex.slice(0, 4).join("") +
      "-" +
      hex.slice(4, 6).join("") +
      "-" +
      hex.slice(6, 8).join("") +
      "-" +
      hex.slice(8, 10).join("") +
      "-" +
      hex.slice(10, 16).join("")
    );
  }
  throw new Error("Secure random generator unavailable for thread UUID.");
}

function readPersistedThreadId(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  const v = sessionStorage.getItem(THREAD_ID_STORAGE_KEY);
  return v && v.length > 0 ? v : null;
}

function readSessionResponseId(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  const v = sessionStorage.getItem(SESSION_RESPONSE_KEY);
  return v && v.length > 0 ? v : null;
}

function persistSessionPointers(threadId: string, latestResponseId: string) {
  sessionStorage.setItem(THREAD_ID_STORAGE_KEY, threadId);
  sessionStorage.setItem(SESSION_RESPONSE_KEY, latestResponseId);
}

function clearSessionPointers() {
  sessionStorage.removeItem(THREAD_ID_STORAGE_KEY);
  sessionStorage.removeItem(SESSION_RESPONSE_KEY);
}

function apiUserId(): string {
  if (typeof window === "undefined") {
    return "ui_anon";
  }
  const key = SESSION_USER_KEY;

  const readLocal = (): string | null => {
    try {
      const v = localStorage.getItem(key);
      return v && v.length > 0 ? v : null;
    } catch {
      return null;
    }
  };

  const writeLocal = (value: string) => {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* private mode / blocked */
    }
  };

  let id = readLocal();
  if (id) {
    return id;
  }

  // Migrate from older builds that stored the API user only in sessionStorage (per-tab).
  try {
    const legacy = sessionStorage.getItem(key);
    if (legacy && legacy.length > 0) {
      writeLocal(legacy);
      sessionStorage.removeItem(key);
      return legacy;
    }
  } catch {
    /* ignore */
  }

  id = `ui_${createId()}`;
  writeLocal(id);
  if (readLocal() === id) {
    return id;
  }

  // localStorage unavailable: same-tab-only fallback (session list will not match other tabs).
  try {
    sessionStorage.setItem(key, id);
  } catch {
    /* ignore */
  }
  return id;
}

function buildErrorMessage(error: unknown) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Request failed. Try again.";
}

/** Server no longer has this id (deleted chat, wiped DB, or different workspace). */
function isStalePreviousResponseError(status: number, payload: ApiErrorPayload): boolean {
  if (status !== 404 || !payload?.error) {
    return false;
  }
  const { code, message } = payload.error;
  return (
    code === "previous_response_not_found" ||
    (typeof message === "string" && message.includes("Unknown previous_response_id"))
  );
}

function historyRowsToMessages(rows: HistoryRow[]): Message[] {
  return rows.map((row, i) => ({
    id: `srv-${i}-${row.role}`,
    role: row.role,
    content: row.content,
    imageUrls: row.image_urls && row.image_urls.length > 0 ? row.image_urls : undefined,
  }));
}

function bubbleStyle(role: string) {
  if (role === "user") {
    return "ml-auto bg-primary text-primary-foreground shadow-btn-inset";
  }
  if (role === "assistant") {
    return "mr-auto border border-border bg-card text-card-foreground shadow-none";
  }
  return "mx-auto max-w-[90%] border border-dashed border-[color:var(--ghost-border)] bg-muted text-foreground";
}

function TypingIndicator({ step }: { step: string | null }) {
  return (
    <div className="mr-auto flex max-w-[82%] flex-col gap-2">
      <div
        aria-label="Assistant is typing"
        className="flex items-center gap-3 rounded-xl border border-border bg-card px-5 py-4"
      >
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 animate-bounce rounded-full bg-muted-foreground/80"
            style={{ animationDuration: "0.55s", animationDelay: "0ms" }}
          />
          <span
            className="inline-block h-2 w-2 animate-bounce rounded-full bg-muted-foreground/80"
            style={{ animationDuration: "0.55s", animationDelay: "0.12s" }}
          />
          <span
            className="inline-block h-2 w-2 animate-bounce rounded-full bg-muted-foreground/80"
            style={{ animationDuration: "0.55s", animationDelay: "0.24s" }}
          />
        </div>
        {step && <span className="text-sm text-muted-foreground animate-pulse">{step}</span>}
      </div>
    </div>
  );
}

function ToolAccordion({ toolCalls }: { toolCalls: ToolCall[] }) {
  const [isOpen, setIsOpen] = useState(false);

  if (toolCalls.length === 0) return null;

  return (
    <div className="mt-4 border-t border-border pt-4">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
      >
        <svg
          className={cn("h-3 w-3 transition-transform", isOpen && "rotate-90")}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
        </svg>
        {isOpen ? "Hide tool calls" : `Show ${toolCalls.length} tool calls`}
      </button>
      {isOpen && (
        <div className="mt-3 flex flex-col gap-3">
          {toolCalls.map((tc, i) => (
            <div key={i} className="rounded-lg border border-border bg-muted/30 p-3 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-yellow-600 dark:text-yellow-400">{tc.tool_name}</span>
              </div>
              <div className="mt-2 font-mono text-[11px] opacity-70 break-all">
                <span className="font-semibold">Args:</span> {tc.args}
              </div>
              {tc.result && (
                <div className="mt-2 font-mono text-[11px] opacity-70 break-all">
                  <span className="font-semibold">Result:</span> {tc.result.length > 500 ? `${tc.result.slice(0, 500)}...` : tc.result}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("read failed"));
    reader.readAsDataURL(file);
  });
}

type ComposerProps = {
  disabled: boolean;
  pending: boolean;
  onStop: () => void;
  onSubmit: (payload: { text: string; imageDataUrls: string[] }) => Promise<void>;
};

function Composer({ disabled: streamingResponse, pending, onStop, onSubmit }: ComposerProps) {
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<{ id: string; url: string; name: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const submit = async () => {
    const trimmed = draft.trim();
    if ((!trimmed && attachments.length === 0) || streamingResponse) {
      return;
    }
    const urls = attachments.map((a) => a.url);
    setDraft("");
    setAttachments([]);
    await onSubmit({ text: trimmed, imageDataUrls: urls });
  };

  const onPickFiles = async (list: FileList | null) => {
    if (!list?.length || streamingResponse) {
      return;
    }
    const imageFiles = Array.from(list).filter((f) => f.type.startsWith("image/"));
    const next: { id: string; url: string; name: string }[] = [];
    for (const file of imageFiles.slice(0, 8)) {
      try {
        const url = await fileToDataUrl(file);
        next.push({ id: createId(), url, name: file.name });
      } catch {
        /* skip */
      }
    }
    setAttachments((prev) => [...prev, ...next].slice(0, 8));
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="rounded-xl border border-border bg-card p-4 shadow-focus-soft">
      {attachments.length > 0 ? (
        <div className="mb-3 flex flex-wrap gap-2">
          {attachments.map((a) => (
            <div
              className="relative h-20 w-20 overflow-hidden rounded-lg border border-border bg-muted"
              key={a.id}
            >
              <img alt="" className="h-full w-full object-cover" src={a.url} />
              <button
                className="absolute right-0.5 top-0.5 rounded bg-background/90 px-1 text-[10px] font-medium text-foreground shadow"
                disabled={streamingResponse}
                type="button"
                onClick={() => setAttachments((prev) => prev.filter((x) => x.id !== a.id))}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ) : null}
      <Textarea
        className="min-h-[116px] resize-none border-0 bg-transparent p-0 text-[15px] shadow-none focus-visible:ring-0"
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void submit();
          }
        }}
        placeholder="Message isanagent (images: attach below). Shift+Enter for newline."
        value={draft}
      />
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <input
            accept="image/jpeg,image/png,image/gif,image/webp"
            className="hidden"
            ref={fileInputRef}
            type="file"
            multiple
            onChange={(e) => void onPickFiles(e.target.files)}
          />
          <Button
            disabled={streamingResponse}
            size="sm"
            type="button"
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
          >
            Add images
          </Button>
          <p className="text-xs text-muted-foreground">
            {streamingResponse ? "Waiting for response…" : "JPEG, PNG, GIF, WebP · up to 8"}
          </p>
        </div>
        <div className="flex gap-2">
          {pending && (
            <Button
              variant="outline"
              className="text-destructive border-destructive hover:bg-destructive/10"
              onClick={onStop}
            >
              Stop
            </Button>
          )}
          <Button
            disabled={streamingResponse || (draft.trim().length === 0 && attachments.length === 0)}
            onClick={() => void submit()}
          >
            {streamingResponse ? "Working…" : "Send"}
          </Button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [internalChatId, setInternalChatId] = useState<string | null>(() => readPersistedThreadId());
  const [latestResponseId, setLatestResponseId] = useState<string | null>(() => readSessionResponseId());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [currentToolCalls, setCurrentToolCalls] = useState<ToolCall[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessions, setSessions] = useState<ThreadListEntry[]>([]);
  const [sidebarHints, setSidebarHints] = useState<Record<string, string>>(loadSidebarHints);
  const [sessionToDelete, setSessionToDelete] = useState<ThreadListEntry | null>(null);
  const [showSummaries, setShowSummaries] = useState(false);
  const [showWorkspaceModal, setShowWorkspaceModal] = useState(false);
  const [showBackgroundPanel, setShowBackgroundPanel] = useState(false);
  const [workspacePaneNonce, setWorkspacePaneNonce] = useState(0);

  // Custom dialog state to replace window.confirm/prompt
  const [dialogConfig, setDialogConfig] = useState<{
    type: 'confirm' | 'prompt';
    title: string;
    message: string;
    onConfirm: (value?: string) => void;
    onCancel: () => void;
    defaultValue?: string;
  } | null>(null);

  const dialogInputRef = useRef<HTMLInputElement>(null);

  const showConfirm = (title: string, message: string, onConfirm: () => void) => {
    setDialogConfig({
      type: 'confirm',
      title,
      message,
      onConfirm: () => {
        onConfirm();
        setDialogConfig(null);
      },
      onCancel: () => setDialogConfig(null),
    });
  };

  const showPrompt = (title: string, message: string, defaultValue: string, onConfirm: (val: string) => void) => {
    setDialogConfig({
      type: 'prompt',
      title,
      message,
      defaultValue,
      onConfirm: (val) => {
        if (val !== undefined) onConfirm(val);
        setDialogConfig(null);
      },
      onCancel: () => setDialogConfig(null),
    });
  };
  const [jobs, setJobs] = useState<BackgroundJob[]>([]);
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [summaries, setSummaries] = useState<SummaryEntry[]>([]);
  const [summariesLoading, setSummariesLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  /** Active streaming turn chat id (matches server); set before fetch so Stop can cancel immediately. */
  const cancelChatIdRef = useRef<string | null>(null);
  const [, startTransition] = useTransition();
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);

  const requestUserId = useMemo(() => apiUserId(), []);

  const loadSummaries = useCallback(async (sessionId: string | null) => {
    setSummariesLoading(true);
    try {
      const url = sessionId 
        ? `/v1/threads/${encodeURIComponent(sessionId)}/summaries`
        : `/v1/summaries`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to load summaries (${response.status})`);
      }
      const data = (await response.json()) as SummaryEntry[];
      setSummaries(data);
    } catch (error) {
      console.error(error);
      setErrorMessage(buildErrorMessage(error));
    } finally {
      setSummariesLoading(false);
    }
  }, []);

  const loadHistory = useCallback(async (sessionId: string) => {
    setHistoryLoading(true);
    setErrorMessage(null);
    try {
      const response = await fetch(`/v1/threads/${encodeURIComponent(sessionId)}/messages`);
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as
          | { error?: { message?: string } }
          | null;
        throw new Error(payload?.error?.message || `History request failed (${response.status}).`);
      }
      const rows = (await response.json()) as HistoryRow[];
      setMessages(historyRowsToMessages(rows));
    } catch (error) {
      setErrorMessage(buildErrorMessage(error));
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    try {
      const q = new URLSearchParams({ user: requestUserId, limit: "100" });
      const response = await fetch(`/v1/threads?${q.toString()}`);
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as
          | { error?: { message?: string } }
          | null;
        throw new Error(payload?.error?.message || `Thread list failed (${response.status}).`);
      }
      const rows = (await response.json()) as ThreadListEntry[];
      setSessions(rows);
    } catch {
      /* sidebar is optional if API older */
      setSessions([]);
    } finally {
      setSessionsLoading(false);
    }
  }, [requestUserId]);

  const openSession = (entry: ThreadListEntry) => {
    startTransition(() => {
      setErrorMessage(null);
      persistSessionPointers(entry.thread_id, entry.latest_response_id);
      setInternalChatId(entry.thread_id);
      setLatestResponseId(entry.latest_response_id);
      void loadHistory(entry.thread_id);
      void loadSummaries(entry.thread_id);
    });
  };

  const jumpToThread = useCallback((threadId: string) => {
    // API thread_id format is "api:<chat_id>" while internal items use plain chat_id
    const targetId = threadId.startsWith("api:") ? threadId : `api:${threadId}`;
    const entry = sessions.find(s => s.thread_id === targetId);
    if (entry) {
      openSession(entry);
    } else {
      setInternalChatId(targetId);
      setLatestResponseId(null);
      void loadHistory(targetId);
      void loadSummaries(targetId);
    }
    setShowBackgroundPanel(false);
  }, [sessions, loadHistory, loadSummaries]);

  const loadBackgroundData = useCallback(async () => {
    try {
      const [jobsRes, notifRes] = await Promise.all([
        fetch("/v1/background-jobs?limit=100"),
        fetch("/v1/notifications?limit=100"),
      ]);
      if (jobsRes.ok) {
        setJobs((await jobsRes.json()) as BackgroundJob[]);
      }
      if (notifRes.ok) {
        setNotifications((await notifRes.json()) as NotificationItem[]);
      }
    } catch {
      // keep UI best-effort
    }
  }, []);

  const replyToTicket = async (ticketId: string, response: string) => {
    try {
      const res = await fetch(`/v1/clarification-tickets/${encodeURIComponent(ticketId)}/reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ response }),
      });
      if (res.ok) {
        void loadBackgroundData();
      } else {
        const payload = (await res.json().catch(() => null)) as ApiErrorPayload;
        setErrorMessage(payload?.error?.message || "Failed to send reply");
      }
    } catch (err) {
      console.error(err);
      setErrorMessage(buildErrorMessage(err));
    }
  };

  const dismissTicket = async (ticketId: string) => {
    try {
      const res = await fetch(`/v1/clarification-tickets/${encodeURIComponent(ticketId)}/dismiss`, {
        method: "POST",
      });
      if (res.ok) {
        void loadBackgroundData();
      } else {
        const payload = (await res.json().catch(() => null)) as ApiErrorPayload;
        setErrorMessage(payload?.error?.message || "Failed to dismiss request");
      }
    } catch (err) {
      console.error(err);
      setErrorMessage(buildErrorMessage(err));
    }
  };

  const dismissBackgroundJob = async (jobId: string) => {
    try {
      const res = await fetch(`/v1/background-jobs/${encodeURIComponent(jobId)}/dismiss`, {
        method: "POST",
      });
      if (res.ok) {
        void loadBackgroundData();
      } else {
        const payload = (await res.json().catch(() => null)) as ApiErrorPayload;
        setErrorMessage(payload?.error?.message || "Failed to dismiss job");
      }
    } catch (err) {
      console.error(err);
      setErrorMessage(buildErrorMessage(err));
    }
  };

  const updateSummary = async (id: number, updated: Partial<SummaryEntry>) => {
    try {
      const response = await fetch(`/v1/summaries/${id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updated),
      });
      if (!response.ok) {
        throw new Error(`Failed to update summary (${response.status})`);
      }
      setSummaries((prev) => prev.map((s) => (s.id === id ? { ...s, ...updated } : s)));
    } catch (error) {
      console.error(error);
      setErrorMessage(buildErrorMessage(error));
    }
  };

  const deleteSummary = (id: number) => {
    showConfirm("Delete summary", "Are you sure you want to delete this summary?", async () => {
      try {
        const response = await fetch(`/v1/summaries/${id}`, {
          method: "DELETE",
        });
        if (!response.ok) {
          throw new Error(`Failed to delete summary (${response.status})`);
        }
        setSummaries((prev) => prev.filter((s) => s.id !== id));
      } catch (error) {
        console.error(error);
        setErrorMessage(buildErrorMessage(error));
      }
    });
  };


  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  // loadBackgroundData is now defined above

  useEffect(() => {
    const id = readPersistedThreadId();
    if (id) {
      void loadHistory(id);
      void loadSummaries(id);
    }
  }, [loadHistory, loadSummaries]);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  const startNewConversation = () => {
    startTransition(() => {
      clearSessionPointers();
      setInternalChatId(null);
      setLatestResponseId(null);
      setMessages([]);
      setSummaries([]);
      setErrorMessage(null);
    });
  };


  const requestDeleteSession = (entry: ThreadListEntry, event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setSessionToDelete(entry);
  };

  const confirmDeleteSession = async () => {
    const entry = sessionToDelete;
    if (!entry) {
      return;
    }
    setSessionToDelete(null);
    setErrorMessage(null);
    try {
      const q = new URLSearchParams({ user: requestUserId });
      const response = await fetch(
        `/v1/threads/${encodeURIComponent(entry.thread_id)}?${q.toString()}`,
        { method: "DELETE" },
      );
      const payload = (await response.json().catch(() => null)) as
        | { deleted?: boolean; error?: { message?: string } }
        | null;
      if (!response.ok) {
        throw new Error(payload?.error?.message || `Delete failed (${response.status}).`);
      }
      if (!payload?.deleted) {
        throw new Error(
          "This conversation was not deleted (not found or not allowed for this user).",
        );
      }
      setSessions((prev) => prev.filter((s) => s.thread_id !== entry.thread_id));
      setSidebarHints((prev) => {
        if (!(entry.thread_id in prev)) {
          return prev;
        }
        const next = { ...prev };
        delete next[entry.thread_id];
        persistSidebarHints(next);
        return next;
      });
      if (internalChatId === entry.thread_id) {
        startNewConversation();
      }
    } catch (error) {
      setErrorMessage(buildErrorMessage(error));
    }
  };

  const buildResponsesInput = (text: string, imageDataUrls: string[]) => {
    if (imageDataUrls.length === 0) {
      return text;
    }
    const parts: object[] = [];
    if (text.trim().length > 0) {
      parts.push({ type: "text", text: text.trim() });
    }
    for (const url of imageDataUrls) {
      parts.push({
        type: "image_url",
        image_url: { url },
      });
    }
    return parts;
  };

  const stopGeneration = async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setPending(false);
    setCurrentStep(null);
    setCurrentToolCalls([]);

    const chatToCancel = cancelChatIdRef.current ?? internalChatId;
    if (!chatToCancel) {
      return;
    }

    try {
      await fetch(`/v1/chat/cancel/${encodeURIComponent(chatToCancel)}`, {
        method: "POST",
      });
    } catch (error) {
      console.error("Failed to notify server about cancellation:", error);
    }
  };

  const submitMessage = async ({ text, imageDataUrls }: { text: string; imageDataUrls: string[] }) => {
    if (pending) {
      return;
    }
    if (!text.trim() && imageDataUrls.length === 0) {
      return;
    }

    const userDisplayText = text.trim() || (imageDataUrls.length ? "(image)" : "");
    const optimisticId = createId();
    const optimisticUser: Message = {
      id: optimisticId,
      role: "user",
      content: userDisplayText,
      imageUrls: imageDataUrls.length > 0 ? [...imageDataUrls] : undefined,
    };

    const hadSessionBefore = internalChatId !== null;
    const turnChatId = internalChatId ?? newThreadUuid();

    setErrorMessage(null);
    setMessages((prev) => [...prev, optimisticUser]);
    setPending(true);
    setCurrentStep(null);
    setCurrentToolCalls([]);

    cancelChatIdRef.current = turnChatId;
    if (!hadSessionBefore) {
      setInternalChatId(turnChatId);
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let previousResponseId: string | undefined = latestResponseId ?? undefined;

    try {
      const response = await fetch("/v1/responses", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: buildResponsesInput(text, imageDataUrls),
          previous_response_id: previousResponseId,
          store: true,
          user: requestUserId,
          stream: true,
          thread_id: turnChatId,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as ApiErrorPayload;
        throw new Error(payload?.error?.message || `Request failed with ${response.status}.`);
      }

      const hdrChat = response.headers.get("X-Thread-Id");
      if (hdrChat?.trim()) {
        cancelChatIdRef.current = hdrChat.trim();
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Response body is not readable.");
      }

      let decoder = new TextDecoder();
      let assistantContent = "";
      let toolCalls: ToolCall[] = [];
      let currentToolCall: ToolCall | null = null;
      let finalChatId: string | null = null;
      let finalResponseId: string | null = null;
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // Keep the last partial line in the buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine || !trimmedLine.startsWith("data: ")) continue;

          const data = trimmedLine.slice(6);
          try {
            const event = JSON.parse(data) as StreamEvent;
            switch (event.type) {
              case "agent_thought":
                setCurrentStep(`Thinking: ${event.thought}`);
                break;
              case "tool_call_started":
                currentToolCall = { tool_name: event.tool_name, args: event.args };
                setCurrentStep(`Using tool: ${event.tool_name}`);
                break;
              case "tool_progress":
                setCurrentStep(`${event.tool_name}: ${event.message}`);
                break;
              case "tool_call_finished":
                if (currentToolCall && currentToolCall.tool_name === event.tool_name) {
                  currentToolCall.result = event.result;
                  toolCalls.push({ ...currentToolCall });
                  setCurrentToolCalls([...toolCalls]);
                  currentToolCall = null;
                }
                // Don't set currentStep to null here, let the next event (thought or next tool) replace it
                // This prevents the indicator from flickering or disappearing between steps
                break;
              case "completion":
                if (event.content === "" && event.response_id === "") {
                  // Initial ID hint (older servers); client already knows id via header / body.
                  if (event.thread_id) {
                    cancelChatIdRef.current = event.thread_id;
                  }
                  break;
                }
                assistantContent = event.content;
                finalChatId = event.thread_id;
                finalResponseId = event.response_id;
                setCurrentStep(null); // Clear step when completion arrives
                break;
              case "error":
                throw new Error(event.message);
            }
          } catch (e) {
            console.error("Failed to parse SSE event", e, "Line:", trimmedLine);
          }
        }
      }

      const assistantId = createId();
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: assistantContent,
          toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        },
      ]);

      if (finalChatId && finalResponseId) {
        persistSessionPointers(finalChatId, finalResponseId);
        setInternalChatId(finalChatId);
        setLatestResponseId(finalResponseId);

        const titleHint = truncateSidebarTitle(userDisplayText);
        if (titleHint.length > 0) {
          setSidebarHints((prev) => {
            const next = { ...prev, [finalChatId!]: titleHint };
            persistSidebarHints(next);
            return next;
          });
        }
      }

      void loadSessions();
    } catch (error) {
      if (error instanceof Error && (error.name === "AbortError" || error.message.includes("aborted"))) {
        return;
      }
      setMessages((prev) => prev.filter((m) => m.id !== optimisticId));
      if (!hadSessionBefore) {
        setInternalChatId(null);
        clearSessionPointers();
      }
      setErrorMessage(buildErrorMessage(error));
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      cancelChatIdRef.current = null;
      setPending(false);
      setCurrentStep(null);
      setCurrentToolCalls([]);
    }
  };

  return (
    <div className="flex h-dvh overflow-hidden bg-background font-sans">
      {sessionToDelete ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="delete-session-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setSessionToDelete(null);
            }
          }}
        >
          <div className="w-full max-w-md rounded-xl border border-border bg-card p-5 shadow-lg">
            <h2 id="delete-session-title" className="text-lg font-semibold tracking-[-0.02em] text-foreground">
              Delete conversation?
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              This removes the conversation from the server. It cannot be undone.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <Button type="button" variant="outline" size="sm" onClick={() => setSessionToDelete(null)}>
                Cancel
              </Button>
              <Button
                type="button"
                size="sm"
                className="bg-destructive text-destructive-foreground hover:opacity-90"
                onClick={() => void confirmDeleteSession()}
              >
                Delete
              </Button>
            </div>
          </div>
        </div>
      ) : null}
      {showBackgroundPanel ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="background-runtime-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowBackgroundPanel(false);
            }
          }}
        >
          <div className="flex h-full max-h-[85vh] w-full max-w-4xl flex-col rounded-2xl border border-border bg-card shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
            <div className="flex items-center justify-between border-b border-border px-6 py-4 bg-muted/30">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                <h2 id="background-runtime-title" className="text-lg font-semibold tracking-tight text-foreground">Background Runtime</h2>
              </div>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0 rounded-full" onClick={() => setShowBackgroundPanel(false)}>
                ×
              </Button>
            </div>
            <div className="flex flex-1 overflow-hidden">
              {/* Jobs column */}
              <div className="w-1/2 flex flex-col border-r border-border">
                <div className="p-4 border-b border-border bg-muted/10">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Active Jobs</p>
                </div>
                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                  {jobs.length === 0 ? (
                    <div className="flex h-full items-center justify-center text-muted-foreground text-sm italic">
                      No active background jobs
                    </div>
                  ) : (
                    jobs.map((job) => (
                      <div 
                        key={job.job_id} 
                        className={cn(
                          "group rounded-xl border border-border p-4 transition-all hover:bg-muted/50 cursor-pointer",
                          job.state === "waiting" ? "border-yellow-500/50 bg-yellow-500/5 shadow-sm" : ""
                        )}
                        onClick={() => jumpToThread(job.chat_id)}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-mono text-[10px] text-muted-foreground truncate max-w-[150px]">{job.job_id}</span>
                          <span className={cn(
                            "px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-tighter",
                            job.state === "running" ? "bg-primary/20 text-primary animate-pulse" : 
                            job.state === "waiting" ? "bg-yellow-500/20 text-yellow-600 dark:text-yellow-400" :
                            "bg-muted text-muted-foreground"
                          )}>
                            {job.state}
                          </span>
                        </div>
                        <p className="text-sm font-medium text-foreground">{job.kind}</p>
                        <div className="mt-3 flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-muted-foreground">Updated {new Date(job.updated_at_ms).toLocaleTimeString()}</span>
                            <button 
                              className="text-[10px] text-destructive hover:underline"
                              onClick={(e) => {
                                e.stopPropagation();
                                showConfirm("Dismiss Job", "Dismiss this background job and all associated requests?", () => {
                                  void dismissBackgroundJob(job.job_id);
                                });
                              }}
                            >
                              Dismiss
                            </button>
                          </div>
                          <span className="text-[10px] text-primary opacity-0 group-hover:opacity-100 transition-opacity">View thread →</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Notifications column */}
              <div className="w-1/2 flex flex-col">
                <div className="p-4 border-b border-border bg-muted/10">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Notifications</p>
                </div>
                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                  {notifications.length === 0 ? (
                    <div className="flex h-full items-center justify-center text-muted-foreground text-sm italic">
                      No notifications
                    </div>
                  ) : (
                    notifications.map((n) => (
                      <div 
                        key={n.notification_id} 
                        className={cn(
                          "rounded-xl border border-border p-4 transition-all",
                          !n.seen_at_ms ? "bg-primary/5 border-primary/30" : "bg-card",
                          n.kind === "clarification_ticket" && !n.resolved_at_ms ? "border-yellow-500/50 shadow-sm" : ""
                        )}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                            {n.kind.replace(/_/g, ' ')}
                          </span>
                          <span className="text-[10px] text-muted-foreground">{new Date(n.created_at_ms).toLocaleTimeString()}</span>
                        </div>
                        <h4 className="text-sm font-semibold text-foreground mb-1">{n.title}</h4>
                        <p className="text-xs text-muted-foreground leading-relaxed mb-4">{n.body}</p>
                        
                        <div className="flex flex-wrap gap-2">
                          {n.kind === "clarification_ticket" && !n.resolved_at_ms ? (
                            <Button 
                              size="sm" 
                              className="h-7 text-[10px] font-bold uppercase tracking-widest bg-yellow-500 hover:bg-yellow-600 text-white border-0"
                              onClick={() => {
                                showPrompt("Reply to request", "Enter your response:", "", (reply) => {
                                  if (reply.trim()) {
                                    void replyToTicket(n.action_payload || "", reply.trim());
                                  }
                                });
                              }}
                            >
                              Reply to request
                            </Button>
                          ) : null}
                          <Button 
                            variant="outline" 
                            size="sm" 
                            className="h-7 text-[10px] font-bold uppercase tracking-widest"
                            onClick={() => jumpToThread(n.chat_id)}
                          >
                            View context
                          </Button>
                          {n.kind === "clarification_ticket" && !n.resolved_at_ms && (
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              className="h-7 text-[10px] font-bold uppercase tracking-widest text-destructive hover:text-destructive hover:bg-destructive/10"
                              onClick={() => {
                                showConfirm("Dismiss Request", "Dismiss this request? This will also mark the background job as completed.", () => {
                                  void dismissTicket(n.action_payload || "");
                                });
                              }}
                            >
                              Dismiss
                            </Button>
                          )}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
      {showWorkspaceModal ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="workspace-modal-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowWorkspaceModal(false);
            }
          }}
        >
          <div className="flex h-full max-h-[90vh] w-full max-w-2xl flex-col rounded-xl border border-border bg-card shadow-lg overflow-hidden">
            <div className="flex items-center justify-between border-b border-border p-4">
              <h2 id="workspace-modal-title" className="text-lg font-semibold text-foreground">
                Workspace
              </h2>
              <Button variant="ghost" size="sm" onClick={() => setShowWorkspaceModal(false)}>
                Close
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto p-6">
              <WorkspaceFilePane key={workspacePaneNonce} />
            </div>
          </div>
        </div>
      ) : null}

      {dialogConfig ? (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              dialogConfig.onCancel();
            }
          }}
        >
          <div className="w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-xl">
            <h2 className="text-lg font-semibold text-foreground mb-2">{dialogConfig.title}</h2>
            <p className="text-sm text-muted-foreground mb-6">{dialogConfig.message}</p>
            
            {dialogConfig.type === 'prompt' && (
              <input
                ref={dialogInputRef}
                autoFocus
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm mb-6 focus:outline-none focus:ring-2 focus:ring-primary"
                defaultValue={dialogConfig.defaultValue}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    dialogConfig.onConfirm(dialogInputRef.current?.value || '');
                  } else if (e.key === 'Escape') {
                    dialogConfig.onCancel();
                  }
                }}
              />
            )}
            
            <div className="flex justify-end gap-3">
              <Button variant="ghost" onClick={dialogConfig.onCancel}>
                Cancel
              </Button>
              <Button
                onClick={() => {
                  if (dialogConfig.type === 'prompt') {
                    dialogConfig.onConfirm(dialogInputRef.current?.value || '');
                  } else {
                    dialogConfig.onConfirm();
                  }
                }}
              >
                {dialogConfig.type === 'confirm' ? 'Confirm' : 'OK'}
              </Button>
            </div>
          </div>
        </div>
      ) : null}
      <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-background/90 backdrop-blur-md">
        <div className="border-b border-border p-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-semibold tracking-[-0.02em] text-foreground">isanagent</p>
            <ThemeToggle />
          </div>
          <Button className="mt-2 w-full" size="sm" onClick={startNewConversation}>
            New chat
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="mt-2 w-full"
            onClick={() => {
              setShowSummaries(true);
              void loadSummaries(null); // Load all summaries
            }}
            disabled={summariesLoading}
          >
            {summariesLoading ? "Loading…" : "Summaries"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="mt-2 w-full relative"
            onClick={() => {
              void loadBackgroundData();
              setShowBackgroundPanel(true);
            }}
          >
            Background
            {notifications.some(n => !n.seen_at_ms || (n.kind === "clarification_ticket" && !n.resolved_at_ms)) && (
              <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-primary border-2 border-background animate-pulse" />
            )}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="mt-2 w-full"
            onClick={() => {
              setWorkspacePaneNonce((n) => n + 1);
              setShowWorkspaceModal(true);
            }}
          >
            Workspace Files
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          <p className="px-2 pb-1 text-xs font-medium uppercase tracking-[0.05em] text-muted-foreground">
            Threads
          </p>
          {sessionsLoading && sessions.length === 0 ? (
            <p className="px-2 text-xs text-muted-foreground">Loading…</p>
          ) : sessions.length === 0 ? (
            <p className="px-2 text-xs text-muted-foreground">No saved chats yet.</p>
          ) : (
            <ul className="space-y-1">
              {sessions.map((s) => {
                const active = internalChatId === s.thread_id;
                return (
                  <li key={s.thread_id}>
                    <div
                      className={cn(
                        "group flex cursor-pointer items-center gap-1 rounded-xl border border-transparent px-2 py-2 text-left text-sm transition-colors",
                        active
                          ? "border-[color:var(--ghost-border)] bg-[color:var(--ghost-fill)] text-foreground"
                          : "hover:border-border hover:bg-muted",
                      )}
                      role="button"
                      tabIndex={0}
                      title={s.thread_id}
                      onClick={() => openSession(s)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          openSession(s);
                        }
                      }}
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate leading-snug text-foreground">
                          {threadSidebarLabel(s, sidebarHints)}
                        </p>
                      </div>
                      <button
                        className="shrink-0 rounded-lg p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/15 hover:text-destructive group-hover:opacity-100"
                        title="Delete conversation"
                        type="button"
                        onClick={(e) => requestDeleteSession(s, e)}
                      >
                        <span className="sr-only">Delete</span>
                        <TrashIcon />
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden px-3 py-4 sm:px-5">
        <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-panel">
          <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-4 sm:px-6">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.05em] text-muted-foreground">
                Chat
              </p>
              <h1 className="mt-1 text-xl font-semibold tracking-[-0.02em] text-foreground">isanagent</h1>
              <p className="mt-1 max-w-xl text-sm leading-relaxed text-muted-foreground">
                Workspace-backed memory, multimodal input, and a thread list for this browser profile.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="shrink-0 rounded-full border border-[color:var(--ghost-border)] bg-muted px-3 py-1 text-xs font-medium text-muted-foreground">
                {pending || historyLoading ? "Syncing…" : internalChatId ? "In thread" : "New thread"}
              </span>
            </div>
          </div>

          {showSummaries && (
            <SummaryList
              summaries={summaries}
              onUpdate={updateSummary}
              onDelete={deleteSummary}
              onClose={() => setShowSummaries(false)}
            />
          )}

          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-4 sm:px-6">
            {(messages.length > 0 || summaries.length > 0 || pending) ? (
              <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col gap-4">
                {summaries.length > 0 && (
                  <div className="space-y-4">
                    {/* Only show the most recent summary (which is now the ONLY summary) */}
                    {(() => {
                      const s = summaries[0];
                      return (
                        <div key={`msg-sum-${s.id}`} className="rounded-xl border border-border bg-muted/10 p-4 shadow-sm">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Thread summary</span>
                            <div className="h-px flex-1 bg-border/50"></div>
                          </div>
                          <p className="text-sm leading-relaxed text-foreground/80">{s.summary}</p>
                          {s.key_info && (
                            <div className="mt-3 pt-3 border-t border-border/30">
                              <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1">Key Knowledge</p>
                              <p className="text-xs text-foreground/70 italic">{s.key_info}</p>
                            </div>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}
                {messages.map((message) => (
                  <article
                    className={cn(
                      "max-w-[82%] rounded-xl px-4 py-3",
                      bubbleStyle(message.role),
                    )}
                    key={message.id}
                  >
                    <div className="flex items-center gap-4">
                      <span className="text-[11px] font-semibold uppercase tracking-[0.2em] opacity-70">
                        {message.role}
                      </span>
                    </div>
                    {message.content ? (
                      <p className="mt-2 whitespace-pre-wrap text-sm leading-7">{message.content}</p>
                    ) : null}
                    {message.toolCalls && <ToolAccordion toolCalls={message.toolCalls} />}
                    {message.imageUrls?.length ? (
                      <div className="mt-2 flex flex-col gap-2">
                        {message.imageUrls.map((url, idx) => (
                          <img
                            alt=""
                            className="max-h-64 max-w-full rounded-lg object-contain"
                            key={`${message.id}-img-${idx}`}
                            src={url}
                          />
                        ))}
                      </div>
                    ) : null}
                  </article>
                ))}

                {pending ? <TypingIndicator step={currentStep} /> : null}

                <div ref={endOfMessagesRef} />
              </div>
            ) : (
              <div className="mx-auto flex min-h-full max-w-2xl flex-col justify-center">
                <div className="rounded-xl border border-dashed border-[color:var(--ghost-border)] bg-[color:var(--ghost-fill-strong)] p-8 text-center">
                  <p className="text-xs font-medium uppercase tracking-[0.05em] text-muted-foreground">
                    Workspace memory
                  </p>
                  <h2 className="mt-3 text-2xl font-semibold tracking-[-0.02em] text-foreground">
                    Start a conversation
                  </h2>
                  <Button className="mt-6" onClick={startNewConversation} variant="secondary">
                    New chat
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-border/70 px-4 py-4 sm:px-6">
            <div className="mx-auto max-w-3xl">
              {errorMessage ? (
                <div className="mb-3 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                  {errorMessage}
                </div>
              ) : null}
              <Composer 
                disabled={pending || historyLoading} 
                pending={pending}
                onStop={stopGeneration}
                onSubmit={submitMessage} 
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ThemeToggle() {
  const [mode, setMode] = useState<ThemeMode>(() => readThemeFromDocument());

  const toggle = () => {
    const next: ThemeMode = mode === "dark" ? "light" : "dark";
    setMode(next);
    applyTheme(next);
  };

  return (
    <Button
      className="h-9 w-9 shrink-0"
      size="icon"
      title={mode === "dark" ? "Light mode" : "Dark mode"}
      type="button"
      variant="ghost"
      onClick={toggle}
    >
      <span className="sr-only">Toggle color theme</span>
      {mode === "dark" ? <SunIcon /> : <MoonIcon />}
    </Button>
  );
}

function SunIcon() {
  return (
    <svg aria-hidden className="h-[18px] w-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg aria-hidden className="h-[18px] w-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
      />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg aria-hidden className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
      />
    </svg>
  );
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

function WorkspaceFilePane() {
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
    <div className="rounded-xl border border-border bg-muted/10 p-4">
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

function SummaryList({
  summaries,
  onUpdate,
  onDelete,
  onClose,
}: {
  summaries: SummaryEntry[];
  onUpdate: (id: number, updated: Partial<SummaryEntry>) => Promise<void>;
  onDelete: (id: number) => void;
  onClose: () => void;
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

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="memory-store-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="flex h-full max-h-[90vh] w-full max-w-2xl flex-col rounded-xl border border-border bg-card shadow-lg overflow-hidden">
        <div className="flex items-center justify-between border-b border-border p-4">
          <h2 id="memory-store-title" className="text-lg font-semibold text-foreground">Memory Store</h2>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {summaries.length === 0 ? (
            <p className="text-center text-muted-foreground py-10">No memory entries found.</p>
          ) : (
            <div className="space-y-8">
              {summaries.map((s) => (
                <div key={s.id} className="space-y-4 border-b border-border pb-8 last:border-0">
                  <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    <span>Thread: {s.thread_id.split(":")[1] || s.thread_id}</span>
                    <span>{new Date(s.created_at).toLocaleString()}</span>
                  </div>

                  {editingId === s.id ? (
                    <div className="space-y-4 rounded-lg bg-muted/20 p-4 border border-border">
                      <div className="space-y-2">
                        <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                          Summary
                        </label>
                        <Textarea
                          value={editForm.summary}
                          onChange={(e) => setEditForm({ ...editForm, summary: e.target.value })}
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
                      <div className="flex justify-end gap-2 pt-2">
                        <Button variant="outline" size="sm" onClick={() => setEditingId(null)}>
                          Cancel
                        </Button>
                        <Button size="sm" onClick={save}>
                          Save
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <p className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Summary</p>
                        <p className="text-sm leading-relaxed text-foreground/90">{s.summary}</p>
                      </div>
                      <div className="space-y-2">
                        <p className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Key Knowledge</p>
                        <p className="text-sm leading-relaxed text-foreground/90">{s.key_info}</p>
                      </div>
                      <div className="flex justify-end gap-2 pt-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:bg-destructive/10"
                          onClick={() => void onDelete(s.id)}
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
          )}
        </div>
      </div>
    </div>
  );
}
