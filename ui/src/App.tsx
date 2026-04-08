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

const SESSION_CHAT_KEY = "isanagent_internal_chat_id";
const SESSION_RESPONSE_KEY = "isanagent_latest_response_id";
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

function sessionSidebarLabel(entry: SessionListEntry, hints: Record<string, string>): string {
  const fromServer = entry.preview?.trim() ?? "";
  if (fromServer.length > 0) {
    return fromServer;
  }
  const fromHint = hints[entry.internal_chat_id]?.trim() ?? "";
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
};

type HistoryRow = {
  role: string;
  content: string;
  image_urls?: string[];
};

type SessionListEntry = {
  internal_chat_id: string;
  updated_at: number;
  latest_response_id: string;
  /** From API; older servers may omit. */
  preview?: string;
};

type ApiErrorPayload = { error?: { code?: string; message?: string } } | null;

function pickResponseId(raw: unknown): string | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const id = (raw as { id?: unknown }).id;
  return typeof id === "string" && id.length > 0 ? id : null;
}

function pickInternalChatId(raw: unknown): string | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const o = raw as Record<string, unknown>;
  const snake = o.internal_chat_id;
  const camel = o.internalChatId;
  if (typeof snake === "string" && snake.length > 0) {
    return snake;
  }
  if (typeof camel === "string" && camel.length > 0) {
    return camel;
  }
  return null;
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

function readSessionChatId(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  const v = sessionStorage.getItem(SESSION_CHAT_KEY);
  return v && v.length > 0 ? v : null;
}

function readSessionResponseId(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  const v = sessionStorage.getItem(SESSION_RESPONSE_KEY);
  return v && v.length > 0 ? v : null;
}

function persistSessionPointers(internalChatId: string, latestResponseId: string) {
  sessionStorage.setItem(SESSION_CHAT_KEY, internalChatId);
  sessionStorage.setItem(SESSION_RESPONSE_KEY, latestResponseId);
}

function clearSessionPointers() {
  sessionStorage.removeItem(SESSION_CHAT_KEY);
  sessionStorage.removeItem(SESSION_RESPONSE_KEY);
}

function apiUserId(): string {
  if (typeof window === "undefined") {
    return "ui_anon";
  }
  let id = sessionStorage.getItem(SESSION_USER_KEY);
  if (!id || id.length === 0) {
    id = `ui_${createId()}`;
    sessionStorage.setItem(SESSION_USER_KEY, id);
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

function TypingIndicator() {
  return (
    <div
      aria-label="Assistant is typing"
      className="mr-auto flex max-w-[82%] items-center rounded-xl border border-border bg-card px-5 py-4"
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
  onSubmit: (payload: { text: string; imageDataUrls: string[] }) => Promise<void>;
};

function Composer({ disabled, onSubmit }: ComposerProps) {
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<{ id: string; url: string; name: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const submit = async () => {
    const trimmed = draft.trim();
    if ((!trimmed && attachments.length === 0) || disabled) {
      return;
    }
    const urls = attachments.map((a) => a.url);
    setDraft("");
    setAttachments([]);
    await onSubmit({ text: trimmed, imageDataUrls: urls });
  };

  const onPickFiles = async (list: FileList | null) => {
    if (!list?.length) {
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
        disabled={disabled}
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
            disabled={disabled}
            size="sm"
            type="button"
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
          >
            Add images
          </Button>
          <p className="text-xs text-muted-foreground">
            {disabled ? "Waiting for response…" : "JPEG, PNG, GIF, WebP · up to 8"}
          </p>
        </div>
        <Button
          disabled={disabled || (draft.trim().length === 0 && attachments.length === 0)}
          onClick={() => void submit()}
        >
          {disabled ? "Working…" : "Send"}
        </Button>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [internalChatId, setInternalChatId] = useState<string | null>(() => readSessionChatId());
  const [latestResponseId, setLatestResponseId] = useState<string | null>(() => readSessionResponseId());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessions, setSessions] = useState<SessionListEntry[]>([]);
  const [sidebarHints, setSidebarHints] = useState<Record<string, string>>(loadSidebarHints);
  const [sessionToDelete, setSessionToDelete] = useState<SessionListEntry | null>(null);
  const [, startTransition] = useTransition();
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);
  const requestUserId = useMemo(() => apiUserId(), []);

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    try {
      const q = new URLSearchParams({ user: requestUserId, limit: "100" });
      const response = await fetch(`/v1/sessions?${q.toString()}`);
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as
          | { error?: { message?: string } }
          | null;
        throw new Error(payload?.error?.message || `Sessions list failed (${response.status}).`);
      }
      const rows = (await response.json()) as SessionListEntry[];
      setSessions(rows);
    } catch {
      /* sidebar is optional if API older */
      setSessions([]);
    } finally {
      setSessionsLoading(false);
    }
  }, [requestUserId]);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  const loadHistory = useCallback(async (sessionId: string) => {
    setHistoryLoading(true);
    setErrorMessage(null);
    try {
      const response = await fetch(`/v1/sessions/${encodeURIComponent(sessionId)}/messages`);
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

  useEffect(() => {
    const id = readSessionChatId();
    if (id) {
      void loadHistory(id);
    }
  }, [loadHistory]);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  const startNewConversation = () => {
    startTransition(() => {
      clearSessionPointers();
      setInternalChatId(null);
      setLatestResponseId(null);
      setMessages([]);
      setErrorMessage(null);
    });
  };

  const openSession = (entry: SessionListEntry) => {
    startTransition(() => {
      setErrorMessage(null);
      persistSessionPointers(entry.internal_chat_id, entry.latest_response_id);
      setInternalChatId(entry.internal_chat_id);
      setLatestResponseId(entry.latest_response_id);
      void loadHistory(entry.internal_chat_id);
    });
  };

  const requestDeleteSession = (entry: SessionListEntry, event: MouseEvent<HTMLButtonElement>) => {
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
        `/v1/sessions/${encodeURIComponent(entry.internal_chat_id)}?${q.toString()}`,
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
      setSessions((prev) => prev.filter((s) => s.internal_chat_id !== entry.internal_chat_id));
      setSidebarHints((prev) => {
        if (!(entry.internal_chat_id in prev)) {
          return prev;
        }
        const next = { ...prev };
        delete next[entry.internal_chat_id];
        persistSidebarHints(next);
        return next;
      });
      if (internalChatId === entry.internal_chat_id) {
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

    setErrorMessage(null);
    setMessages((prev) => [...prev, optimisticUser]);
    setPending(true);

    let previousResponseId: string | undefined = latestResponseId ?? undefined;
    let usedStaleRecovery = false;

    try {
      let responseBody: unknown;

      for (let attempt = 0; attempt < 2; attempt++) {
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
          }),
        });

        const bodyText = await response.text();
        let parsedBody: unknown = null;
        let bodyParseOk = false;
        if (bodyText.length > 0) {
          try {
            parsedBody = JSON.parse(bodyText) as unknown;
            bodyParseOk = true;
          } catch {
            parsedBody = null;
            bodyParseOk = false;
          }
        }

        if (!response.ok) {
          const payload = parsedBody as ApiErrorPayload;
          if (
            attempt === 0 &&
            previousResponseId &&
            isStalePreviousResponseError(response.status, payload)
          ) {
            previousResponseId = undefined;
            usedStaleRecovery = true;
            clearSessionPointers();
            setLatestResponseId(null);
            setInternalChatId(null);
            continue;
          }
          throw new Error(
            payload?.error?.message || `Request failed with ${response.status}.`,
          );
        }

        if (!bodyParseOk) {
          throw new Error(
            bodyText.length === 0
              ? "Empty response body."
              : "Invalid JSON in response body.",
          );
        }
        responseBody = parsedBody;
        break;
      }

      const raw = responseBody as unknown;
      const responseId = pickResponseId(raw);
      if (!responseId) {
        throw new Error("Invalid API response: missing id.");
      }

      const chatId = pickInternalChatId(raw);
      const assistantText = extractOutputText(raw);

      if (chatId) {
        persistSessionPointers(chatId, responseId);
        setInternalChatId(chatId);
        setLatestResponseId(responseId);
        if (usedStaleRecovery) {
          await loadHistory(chatId);
        } else {
          setMessages((prev) => [
            ...prev,
            { id: createId(), role: "assistant", content: assistantText },
          ]);
        }
        const titleHint = truncateSidebarTitle(userDisplayText);
        if (titleHint.length > 0) {
          setSidebarHints((prev) => {
            const next = { ...prev, [chatId]: titleHint };
            persistSidebarHints(next);
            return next;
          });
        }
      } else {
        sessionStorage.setItem(SESSION_RESPONSE_KEY, responseId);
        sessionStorage.removeItem(SESSION_CHAT_KEY);
        setLatestResponseId(responseId);
        setInternalChatId(null);
        setMessages((prev) => [
          ...prev,
          { id: createId(), role: "assistant", content: assistantText },
        ]);
      }

      void loadSessions();
    } catch (error) {
      setMessages((prev) => prev.filter((m) => m.id !== optimisticId));
      setErrorMessage(buildErrorMessage(error));
    } finally {
      setPending(false);
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
      <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-background/90 backdrop-blur-md">
        <div className="border-b border-border p-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-semibold tracking-[-0.02em] text-foreground">isanagent</p>
            <ThemeToggle />
          </div>
          <Button className="mt-2 w-full" size="sm" onClick={startNewConversation}>
            New chat
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          <p className="px-2 pb-1 text-xs font-medium uppercase tracking-[0.05em] text-muted-foreground">
            Conversations
          </p>
          {sessionsLoading && sessions.length === 0 ? (
            <p className="px-2 text-xs text-muted-foreground">Loading…</p>
          ) : sessions.length === 0 ? (
            <p className="px-2 text-xs text-muted-foreground">No saved chats yet.</p>
          ) : (
            <ul className="space-y-1">
              {sessions.map((s) => {
                const active = internalChatId === s.internal_chat_id;
                return (
                  <li key={s.internal_chat_id}>
                    <div
                      className={cn(
                        "group flex cursor-pointer items-center gap-1 rounded-xl border border-transparent px-2 py-2 text-left text-sm transition-colors",
                        active
                          ? "border-[color:var(--ghost-border)] bg-[color:var(--ghost-fill)] text-foreground"
                          : "hover:border-border hover:bg-muted",
                      )}
                      role="button"
                      tabIndex={0}
                      title={s.internal_chat_id}
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
                          {sessionSidebarLabel(s, sidebarHints)}
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
                Workspace-backed memory, multimodal input, and session list for this browser profile.
              </p>
            </div>
            <span className="shrink-0 rounded-full border border-[color:var(--ghost-border)] bg-muted px-3 py-1 text-xs font-medium text-muted-foreground">
              {pending || historyLoading ? "Syncing…" : internalChatId ? "In session" : "New session"}
            </span>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-4 sm:px-6">
            {messages.length > 0 || pending ? (
              <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col gap-4">
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

                {pending ? <TypingIndicator /> : null}

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
                  <p className="mx-auto mt-3 text-sm leading-relaxed text-muted-foreground">
                    Messages are stored in the workspace database. This tab keeps session ids in{" "}
                    <span className="font-mono text-xs">sessionStorage</span>. Use the sidebar to switch or
                    delete past chats.
                  </p>
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
              <Composer disabled={pending || historyLoading} onSubmit={submitMessage} />
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
