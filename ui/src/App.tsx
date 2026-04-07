import { useCallback, useEffect, useMemo, useRef, useState, useTransition } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const SESSION_CHAT_KEY = "isanagent_internal_chat_id";
const SESSION_RESPONSE_KEY = "isanagent_latest_response_id";
const SESSION_USER_KEY = "isanagent_api_user_id";

type Message = {
  id: string;
  role: string;
  content: string;
};

type HistoryRow = {
  role: string;
  content: string;
};

function pickResponseId(raw: unknown): string | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const id = (raw as { id?: unknown }).id;
  return typeof id === "string" && id.length > 0 ? id : null;
}

/** Present on current isanagent; older binaries omit it. */
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

/** Used when the API does not return internal_chat_id (legacy server). */
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

function historyRowsToMessages(rows: HistoryRow[]): Message[] {
  return rows.map((row, i) => ({
    id: `srv-${i}-${row.role}`,
    role: row.role,
    content: row.content,
  }));
}

function bubbleStyle(role: string) {
  if (role === "user") {
    return "ml-auto bg-primary text-primary-foreground";
  }
  if (role === "assistant") {
    return "mr-auto border border-border/70 bg-background/90 text-foreground";
  }
  return "mx-auto max-w-[90%] border border-dashed border-border/80 bg-muted/40 text-foreground";
}

type ComposerProps = {
  disabled: boolean;
  onSubmit: (value: string) => Promise<void>;
};

function Composer({ disabled, onSubmit }: ComposerProps) {
  const [draft, setDraft] = useState("");

  const submit = async () => {
    const trimmed = draft.trim();
    if (!trimmed || disabled) {
      return;
    }
    setDraft("");
    await onSubmit(trimmed);
  };

  return (
    <div className="rounded-[1.5rem] border border-border/80 bg-card/95 p-4 shadow-panel">
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
        placeholder="Message the agent (same session store as terminal). Shift+Enter for newline."
        value={draft}
      />
      <div className="mt-4 flex items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">
          {disabled
            ? "Waiting for the current response to finish."
            : "Transcript lives in workspace SQLite; this tab only keeps session ids in sessionStorage."}
        </p>
        <Button disabled={disabled || draft.trim().length === 0} onClick={() => void submit()}>
          {disabled ? "Working..." : "Send"}
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
  const [, startTransition] = useTransition();
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);
  const requestUserId = useMemo(() => apiUserId(), []);

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
      clearSessionPointers();
      setInternalChatId(null);
      setLatestResponseId(null);
      setMessages([]);
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

  const submitMessage = async (content: string) => {
    if (pending) {
      return;
    }

    setErrorMessage(null);
    setPending(true);

    try {
      const response = await fetch("/v1/responses", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: content,
          previous_response_id: latestResponseId ?? undefined,
          store: true,
          user: requestUserId,
        }),
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as
          | { error?: { message?: string } }
          | null;
        throw new Error(payload?.error?.message || `Request failed with ${response.status}.`);
      }

      const raw: unknown = await response.json();
      const responseId = pickResponseId(raw);
      if (!responseId) {
        throw new Error("Invalid API response: missing id.");
      }

      const chatId = pickInternalChatId(raw);
      if (chatId) {
        persistSessionPointers(chatId, responseId);
        setInternalChatId(chatId);
        setLatestResponseId(responseId);
        const assistantText = extractOutputText(raw);
        setMessages((prev) => [
          ...prev,
          { id: createId(), role: "user", content },
          { id: createId(), role: "assistant", content: assistantText },
        ]);
      } else {
        // Older isanagent without internal_chat_id (and no GET /v1/sessions/.../messages).
        sessionStorage.setItem(SESSION_RESPONSE_KEY, responseId);
        sessionStorage.removeItem(SESSION_CHAT_KEY);
        setLatestResponseId(responseId);
        setInternalChatId(null);
        const assistantText = extractOutputText(raw);
        setMessages((prev) => [
          ...prev,
          {
            id: createId(),
            role: "user",
            content,
          },
          {
            id: createId(),
            role: "assistant",
            content: assistantText,
          },
        ]);
      }
    } catch (error) {
      setErrorMessage(buildErrorMessage(error));
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="h-dvh overflow-hidden px-4 py-5 sm:px-6 lg:px-8">
      <div className="mx-auto flex h-[calc(100dvh-2.5rem)] w-full max-w-4xl flex-col overflow-hidden rounded-[2rem] border border-border/70 bg-card/85 shadow-panel backdrop-blur">
        <div className="flex items-center justify-between gap-3 border-b border-border/70 px-5 py-4 sm:px-6">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-muted-foreground">
              Agent-RS
            </p>
            <h1 className="mt-1 text-xl font-semibold tracking-tight">Chat (server-backed)</h1>
            <p className="mt-1 text-xs text-muted-foreground">
              Same <span className="font-mono">session_id</span> memory as the terminal channel; transcript
              from <span className="font-mono">GET /v1/sessions/…/messages</span>.
            </p>
          </div>
          <div className="flex flex-col items-end gap-2 sm:flex-row sm:items-center">
            <Button onClick={startNewConversation} size="sm" variant="outline">
              New chat
            </Button>
            <span className="rounded-full bg-accent px-3 py-1 text-xs font-medium text-accent-foreground">
              {pending || historyLoading ? "Syncing…" : internalChatId ? "In session" : "New session"}
            </span>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-4 sm:px-6">
          {messages.length > 0 ? (
            <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col gap-4">
              {messages.map((message) => (
                <article
                  className={cn("max-w-[82%] rounded-[1.5rem] px-4 py-3 shadow-sm", bubbleStyle(message.role))}
                  key={message.id}
                >
                  <div className="flex items-center gap-4">
                    <span className="text-[11px] font-semibold uppercase tracking-[0.2em] opacity-70">
                      {message.role}
                    </span>
                  </div>
                  <p className="mt-2 whitespace-pre-wrap text-sm leading-7">{message.content}</p>
                </article>
              ))}

              {pending ? (
                <div className="mr-auto max-w-[82%] rounded-[1.5rem] border border-dashed border-border bg-background/80 px-4 py-3 text-sm text-muted-foreground">
                  Agent-RS is generating a response…
                </div>
              ) : null}

              <div ref={endOfMessagesRef} />
            </div>
          ) : (
            <div className="mx-auto flex min-h-full max-w-2xl flex-col justify-center">
              <div className="rounded-[2rem] border border-dashed border-border/80 bg-background/70 p-8 text-center">
                <p className="text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground">
                  Terminal-style persistence
                </p>
                <h2 className="mt-3 text-2xl font-semibold tracking-tight">No browser transcript cache</h2>
                <p className="mx-auto mt-3 text-sm leading-7 text-muted-foreground">
                  Messages are stored in the workspace database under the current session id (like the CLI).
                  This page only keeps the session and last response id in{" "}
                  <span className="font-mono text-xs">sessionStorage</span> so the tab can continue after
                  refresh.
                </p>
                <Button className="mt-6" onClick={startNewConversation} variant="secondary">
                  Clear tab session pointers
                </Button>
              </div>
            </div>
          )}
        </div>

        <div className="border-t border-border/70 px-4 py-4 sm:px-6">
          <div className="mx-auto max-w-3xl">
            {internalChatId ? (
              <p className="mb-2 break-all font-mono text-[10px] text-muted-foreground">
                session: {internalChatId}
              </p>
            ) : null}
            {errorMessage ? (
              <div className="mb-3 rounded-2xl border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {errorMessage}
              </div>
            ) : null}
            <Composer disabled={pending || historyLoading} onSubmit={submitMessage} />
          </div>
        </div>
      </div>
    </div>
  );
}
