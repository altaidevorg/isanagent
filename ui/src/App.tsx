import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useTransition,
  type MouseEvent,
} from "react";

import { MarkdownText, preloadMarkdown } from "@/components/MarkdownText";
import { McpAppFrame, type McpAppRef } from "@/components/McpAppFrame";
import { SettingsDialog } from "@/components/SettingsDialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { WorkDrawer, type WorkTab } from "@/components/WorkDrawer";
import {
  buildErrorMessage,
  isActiveJob,
  isUnreadNotification,
  type ApiErrorPayload,
  type BackgroundJob,
  type NotificationItem,
  type SummaryEntry,
} from "@/lib/api";
import { useDialog } from "@/lib/useDialog";
import { useMediaQuery } from "@/lib/useMediaQuery";
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
  clarification?: Clarification;
};

/** The agent paused on a question (ask_user or an approval); the reply is the next message. */
type Clarification = {
  choices: string[];
  edit_diff?: { file: string; diff: string; truncated: boolean };
};

type ToolCall = {
  tool_name: string;
  args: string;
  result?: string;
  isError?: boolean;
  mcpApp?: McpAppRef;
};

type StreamEvent =
  | { type: "tool_call_started"; tool_name: string; args: string; tool_call_id?: string }
  | { type: "tool_progress"; tool_name: string; message: string; tool_call_id?: string }
  | {
      type: "tool_call_finished";
      tool_name: string;
      result: string;
      is_error?: boolean;
      tool_call_id?: string;
      mcp_app?: McpAppRef;
    }
  | { type: "agent_thought"; thought: string }
  /** Incremental assistant text for the current model call. */
  | { type: "text_delta"; text: string }
  /** The agent discarded the text streamed so far (tool calls follow, or a retry). */
  | { type: "text_reset" }
  | {
      type: "completion";
      content: string;
      thread_id: string;
      response_id: string;
      clarification?: Clarification;
    }
  | { type: "error"; message: string };

type HistoryRow = {
  role: string;
  content: string;
  image_urls?: string[];
  mcp_apps?: McpAppRef[];
};

type ThreadListEntry = {
  thread_id: string;
  updated_at: number;
  latest_response_id: string;
  /** From API; older servers may omit. */
  preview?: string;
};

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
    toolCalls: row.mcp_apps?.length
      ? row.mcp_apps.map((app) => ({
          tool_name: "",
          args: "",
          mcpApp: app,
        }))
      : undefined,
  }));
}

function bubbleStyle(role: string) {
  if (role === "user") {
    return "ml-auto bg-primary text-primary-foreground shadow-btn-inset";
  }
  if (role === "assistant") {
    return "mr-auto border border-border bg-card text-card-foreground shadow-none max-sm:!max-w-full";
  }
  return "mx-auto max-w-[90%] border border-dashed border-[color:var(--ghost-border)] bg-muted text-foreground";
}

function TypingDots() {
  return (
    <span aria-hidden className="flex items-center gap-1.5">
      {[0, 0.12, 0.24].map((delay) => (
        <span
          className="inline-block h-2 w-2 animate-bounce rounded-full bg-muted-foreground/80"
          key={delay}
          style={{ animationDuration: "0.55s", animationDelay: `${delay}s` }}
        />
      ))}
    </span>
  );
}

/**
 * The assistant reply being produced: streamed markdown text, MCP App panels and the current
 * step. Replaced by the final message (whose `completion` text is authoritative) when the turn ends.
 */
function PendingAssistantBubble({
  text,
  step,
  toolCalls,
  onChatMessage,
}: {
  text: string;
  step: string | null;
  toolCalls: ToolCall[];
  onChatMessage: (text: string) => void;
}) {
  const hasApps = toolCalls.some((toolCall) => toolCall.mcpApp);
  const finished = toolCalls.filter((toolCall) => toolCall.tool_name).length;
  return (
    <article
      aria-busy="true"
      aria-label="Assistant is responding"
      className={cn(
        "rounded-xl px-4 py-3",
        bubbleStyle("assistant"),
        hasApps ? "w-full min-w-0 max-w-full" : text ? "max-w-[82%]" : "max-w-[82%] min-w-[12rem]",
      )}
    >
      <span className="text-[11px] font-semibold uppercase tracking-[0.2em] opacity-70">assistant</span>
      {text ? <MarkdownText className="mt-2" content={text} /> : null}
      <McpAppList onChatMessage={onChatMessage} toolCalls={toolCalls} />
      <div aria-live="polite" className={cn("flex items-center gap-3", text || hasApps ? "mt-3" : "mt-2")}>
        <TypingDots />
        <span className="min-w-0 truncate text-sm text-muted-foreground">
          {step ?? (text ? "Writing…" : "Thinking…")}
        </span>
        {finished > 0 ? (
          <span className="ml-auto shrink-0 text-[11px] text-muted-foreground">
            {finished} tool call{finished === 1 ? "" : "s"}
          </span>
        ) : null}
      </div>
    </article>
  );
}

/** One finished chat message. Memoized so streaming tokens never re-render (or re-parse) history. */
const MessageBubble = memo(function MessageBubble({
  message,
  choicesEnabled,
  onChatMessage,
  onChoose,
}: {
  message: Message;
  choicesEnabled: boolean;
  onChatMessage: (text: string) => void;
  onChoose: (choice: string) => void;
}) {
  return (
    <article
      className={cn(
        "rounded-xl px-4 py-3",
        bubbleStyle(message.role),
        message.toolCalls?.some((toolCall) => toolCall.mcpApp) ? "w-full min-w-0 max-w-full" : "max-w-[82%]",
      )}
    >
      <div className="flex items-center gap-4">
        <span className="text-[11px] font-semibold uppercase tracking-[0.2em] opacity-70">{message.role}</span>
      </div>
      {message.content ? (
        message.role === "assistant" ? (
          <MarkdownText className="mt-2" content={message.content} />
        ) : (
          <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-7">{message.content}</p>
        )
      ) : null}
      <McpAppList onChatMessage={onChatMessage} toolCalls={message.toolCalls} />
      {message.clarification && (
        <ClarificationCard clarification={message.clarification} disabled={!choicesEnabled} onChoose={onChoose} />
      )}
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
  );
});

/** Stable function identity that always calls the latest `fn` (for memoized children). */
function useStableCallback<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R {
  const ref = useRef(fn);
  ref.current = fn;
  return useCallback((...args: A) => ref.current(...args), []);
}

function diffLineClass(line: string) {
  if (line.startsWith("+++") || line.startsWith("---")) return "text-muted-foreground";
  if (line.startsWith("+")) return "bg-green-500/10 text-green-700 dark:text-green-400";
  if (line.startsWith("-")) return "bg-red-500/10 text-red-700 dark:text-red-400";
  if (line.startsWith("@@")) return "text-sky-700 dark:text-sky-400";
  return "";
}

/** Shows the proposed edit and the reply buttons; a choice is sent as the next message. */
function ClarificationCard({
  clarification,
  disabled,
  onChoose,
}: {
  clarification: Clarification;
  disabled: boolean;
  onChoose: (choice: string) => void;
}) {
  const { choices, edit_diff: editDiff } = clarification;
  return (
    <div className="mt-3 flex flex-col gap-3">
      {editDiff && (
        <div className="overflow-hidden rounded-lg border border-border">
          <div className="border-b border-border bg-muted/40 px-3 py-1.5 font-mono text-xs">{editDiff.file}</div>
          <pre className="max-h-80 overflow-auto py-2 font-mono text-[11px] leading-5">
            {editDiff.diff.split("\n").map((line, i) => (
              <div className={cn("px-3", diffLineClass(line))} key={i}>
                {line || " "}
              </div>
            ))}
          </pre>
          {editDiff.truncated && (
            <div className="border-t border-border px-3 py-1.5 text-xs text-muted-foreground">Diff truncated.</div>
          )}
        </div>
      )}
      {choices.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {choices.map((choice) => (
            <Button disabled={disabled} key={choice} onClick={() => onChoose(choice)} size="sm" variant="outline">
              {choice}
            </Button>
          ))}
        </div>
      )}
    </div>
  );
}

function McpAppList({
  toolCalls,
  onChatMessage,
}: {
  toolCalls?: ToolCall[];
  onChatMessage: (text: string) => void;
}) {
  const apps = toolCalls?.filter((toolCall) => toolCall.mcpApp) ?? [];
  if (apps.length === 0) {
    return null;
  }
  return (
    <>
      {apps.map((toolCall) => (
        <McpAppFrame
          app={toolCall.mcpApp!}
          key={toolCall.mcpApp!.view_id}
          onChatMessage={onChatMessage}
        />
      ))}
    </>
  );
}

function ToolAccordion({ toolCalls }: { toolCalls: ToolCall[] }) {
  const [isOpen, setIsOpen] = useState(false);
  const listed = toolCalls.filter((toolCall) => toolCall.tool_name);
  const failed = listed.filter((toolCall) => toolCall.isError).length;

  if (listed.length === 0) return null;

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
        {isOpen ? "Hide tool calls" : `Show ${listed.length} tool calls`}
        {!isOpen && failed > 0 ? <span className="text-destructive">({failed} failed)</span> : null}
      </button>
      {isOpen && (
        <div className="mt-3 flex flex-col gap-3">
          {listed.map((tc, i) => (
            <div
              key={i}
              className={cn(
                "rounded-lg border bg-muted/30 p-3 text-xs",
                tc.isError ? "border-destructive/40" : "border-border",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-yellow-600 dark:text-yellow-400">{tc.tool_name}</span>
                {tc.isError ? (
                  <span className="rounded-full bg-destructive/15 px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest text-destructive">
                    Failed
                  </span>
                ) : null}
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

type ComposerPayload = { text: string; imageDataUrls: string[] };

/** A message typed while a turn was running; sent automatically when the turn ends. */
type QueuedMessage = ComposerPayload & { id: string };

type ComposerProps = {
  /** Thread history is loading: nothing can be typed or sent. */
  locked: boolean;
  /** A turn is running: Send queues the message instead. */
  pending: boolean;
  onStop: () => void;
  onSubmit: (payload: ComposerPayload) => void;
};

function Composer({ locked, pending, onStop, onSubmit }: ComposerProps) {
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<{ id: string; url: string; name: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const submit = () => {
    const trimmed = draft.trim();
    if ((!trimmed && attachments.length === 0) || locked) {
      return;
    }
    const urls = attachments.map((a) => a.url);
    setDraft("");
    setAttachments([]);
    onSubmit({ text: trimmed, imageDataUrls: urls });
  };

  const onPickFiles = async (list: FileList | null) => {
    if (!list?.length || locked) {
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

  const empty = draft.trim().length === 0 && attachments.length === 0;

  return (
    <div className="rounded-xl border border-border bg-card p-3 shadow-focus-soft sm:p-4">
      {attachments.length > 0 ? (
        <div className="mb-3 flex flex-wrap gap-2">
          {attachments.map((a) => (
            <div
              className="relative h-20 w-20 overflow-hidden rounded-lg border border-border bg-muted"
              key={a.id}
            >
              <img alt="" className="h-full w-full object-cover" src={a.url} />
              <button
                aria-label={`Remove ${a.name}`}
                className="absolute right-0.5 top-0.5 rounded bg-background/90 px-1 text-[10px] font-medium text-foreground shadow"
                disabled={locked}
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
        aria-label="Message"
        className="min-h-[88px] resize-none border-0 bg-transparent p-0 text-[15px] shadow-none focus-visible:ring-0 sm:min-h-[116px]"
        disabled={locked}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            submit();
          }
        }}
        placeholder={
          pending
            ? "Type your next message — it is sent when this turn ends."
            : "Message isanagent (images: attach below). Shift+Enter for newline."
        }
        value={draft}
      />
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 sm:mt-4">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <input
            accept="image/jpeg,image/png,image/gif,image/webp"
            className="hidden"
            ref={fileInputRef}
            type="file"
            multiple
            onChange={(e) => void onPickFiles(e.target.files)}
          />
          <Button
            disabled={locked}
            size="sm"
            type="button"
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
          >
            Add images
          </Button>
          <p className="hidden text-xs text-muted-foreground sm:block">
            {locked
              ? "Loading conversation…"
              : pending
                ? "Agent is working · Enter queues"
                : "JPEG, PNG, GIF, WebP · up to 8"}
          </p>
        </div>
        <div className="flex gap-2">
          {pending && (
            <Button
              variant="outline"
              className="border-destructive text-destructive hover:bg-destructive/10"
              type="button"
              onClick={onStop}
            >
              Stop
            </Button>
          )}
          <Button disabled={locked || empty} type="button" onClick={submit}>
            {pending ? "Queue" : "Send"}
          </Button>
        </div>
      </div>
    </div>
  );
}

function QueuedMessages({
  items,
  paused,
  canResume,
  waitingForModel,
  onRemove,
  onResume,
  onChooseModel,
}: {
  items: QueuedMessage[];
  paused: boolean;
  canResume: boolean;
  /** No model is selected in this session yet: nothing is sent until one is. */
  waitingForModel: boolean;
  onRemove: (id: string) => void;
  onResume: () => void;
  onChooseModel: () => void;
}) {
  if (items.length === 0) {
    return null;
  }
  return (
    <div aria-label="Queued messages" className="mb-3 rounded-xl border border-dashed border-[color:var(--ghost-border)] bg-[color:var(--ghost-fill-strong)] p-2" role="region">
      <div className="flex items-center justify-between gap-2 px-1.5 pb-1.5">
        <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          Queued · {items.length}
          {waitingForModel ? (
            <span className="ml-1.5 normal-case tracking-normal">(waiting for a model)</span>
          ) : paused ? (
            <span className="ml-1.5 normal-case tracking-normal">(paused)</span>
          ) : null}
        </p>
        {waitingForModel ? (
          <Button className="h-6 px-2 text-xs" size="sm" type="button" variant="outline" onClick={onChooseModel}>
            Choose model
          </Button>
        ) : paused && canResume ? (
          <Button className="h-6 px-2 text-xs" size="sm" type="button" variant="outline" onClick={onResume}>
            Send next
          </Button>
        ) : null}
      </div>
      <ul className="max-h-36 space-y-1 overflow-y-auto">
        {items.map((item, index) => (
          <li className="flex items-center gap-2 rounded-lg bg-card px-2.5 py-1.5 text-sm" key={item.id}>
            <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">{index + 1}</span>
            <span className="min-w-0 flex-1 truncate text-foreground" title={item.text}>
              {item.text || "(image)"}
              {item.imageDataUrls.length > 0 && item.text ? (
                <span className="ml-1.5 text-xs text-muted-foreground">
                  +{item.imageDataUrls.length} image{item.imageDataUrls.length === 1 ? "" : "s"}
                </span>
              ) : null}
            </span>
            <button
              aria-label={`Remove queued message ${index + 1}`}
              className="shrink-0 rounded-md px-1.5 text-base leading-6 text-muted-foreground hover:bg-destructive/15 hover:text-destructive"
              title="Remove"
              type="button"
              onClick={() => onRemove(item.id)}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
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
  const [sessionsError, setSessionsError] = useState<string | null>(null);
  const [sidebarHints, setSidebarHints] = useState<Record<string, string>>(loadSidebarHints);
  const [sessionToDelete, setSessionToDelete] = useState<ThreadListEntry | null>(null);
  const [workOpen, setWorkOpen] = useState(false);
  const [workTab, setWorkTab] = useState<WorkTab>("jobs");
  /** Off-canvas thread list below `md`. */
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  /** Shown in the Models tab when the dialog was opened by a send without a model. */
  const [settingsNotice, setSettingsNotice] = useState<string | null>(null);
  /** Model chosen in the web settings during this server process (`selected_key`). */
  const [selectedModelLabel, setSelectedModelLabel] = useState<string | null>(null);
  /**
   * `loading`: not known yet. `none`: the server has models but none was chosen in this
   * session, so web sends wait. `selected`: one was chosen. `ungated`: older server or no
   * model catalog; sends go straight through to the runtime default.
   */
  const [modelState, setModelState] = useState<"loading" | "none" | "selected" | "ungated">("loading");
  const [workspacePaneNonce, setWorkspacePaneNonce] = useState(0);
  /** Assistant text streamed for the running turn (cleared on `text_reset` and at turn end). */
  const [streamingText, setStreamingText] = useState("");
  /** Messages typed (or sent by MCP Apps) while a turn runs; the next one goes out when it ends. */
  const [queue, setQueue] = useState<QueuedMessage[]>([]);
  /** Stop, an error or switching threads pauses the queue instead of firing the next item. */
  const [queuePaused, setQueuePaused] = useState(false);
  const isDesktopDrawer = useMediaQuery("(min-width: 1024px)");
  const isWideSidebar = useMediaQuery("(min-width: 768px)");

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
  const pendingRef = useRef(false);

  const deleteDialogRef = useDialog(sessionToDelete !== null, () => setSessionToDelete(null));
  const sidebarRef = useDialog<HTMLElement>(sidebarOpen && !isWideSidebar, () => setSidebarOpen(false));
  const appDialogRef = useDialog(dialogConfig !== null, () => dialogConfig?.onCancel());

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
  /** All summaries, for the Memory Store panel. */
  const [summaries, setSummaries] = useState<SummaryEntry[]>([]);
  const [summariesLoading, setSummariesLoading] = useState(false);
  /** Summaries of the open thread, shown inline above the chat. */
  const [threadSummaries, setThreadSummaries] = useState<SummaryEntry[]>([]);
  const threadSummariesForRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  /** Active streaming turn chat id (matches server); set before fetch so Stop can cancel immediately. */
  const cancelChatIdRef = useRef<string | null>(null);
  const [, startTransition] = useTransition();
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  /** True while the user is at (or near) the bottom of the chat; new content only scrolls then. */
  const stickToBottomRef = useRef(true);
  const lastScrollTopRef = useRef(0);

  const requestUserId = useMemo(() => apiUserId(), []);

  useEffect(() => {
    preloadMarkdown();
  }, []);

  const loadAllSummaries = useCallback(async () => {
    setSummariesLoading(true);
    try {
      const response = await fetch(`/v1/summaries`);
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

  const loadSummaries = useCallback(async (sessionId: string) => {
    threadSummariesForRef.current = sessionId;
    setThreadSummaries([]);
    try {
      const response = await fetch(`/v1/threads/${encodeURIComponent(sessionId)}/summaries`);
      if (!response.ok) {
        throw new Error(`Failed to load summaries (${response.status})`);
      }
      const data = (await response.json()) as SummaryEntry[];
      if (threadSummariesForRef.current === sessionId) {
        setThreadSummaries(data);
      }
    } catch (error) {
      console.error(error);
      if (threadSummariesForRef.current === sessionId) {
        setErrorMessage(buildErrorMessage(error));
      }
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
      if (response.status === 404) {
        /* sidebar is optional if API older */
        setSessions([]);
        setSessionsError(null);
        return;
      }
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as
          | { error?: { message?: string } }
          | null;
        throw new Error(payload?.error?.message || `Thread list failed (${response.status}).`);
      }
      const rows = (await response.json()) as ThreadListEntry[];
      setSessions(rows);
      setSessionsError(null);
    } catch (error) {
      console.error("Failed to load threads:", error);
      setSessionsError("Can't reach the isanagent server.");
    } finally {
      setSessionsLoading(false);
    }
  }, [requestUserId]);

  const openSession = (entry: ThreadListEntry) => {
    stickToBottomRef.current = true;
    setSidebarOpen(false);
    // Queued text was written for the previous thread; hold it until the user sends it on.
    setQueuePaused(true);
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
      stickToBottomRef.current = true;
      setInternalChatId(targetId);
      setLatestResponseId(null);
      void loadHistory(targetId);
      void loadSummaries(targetId);
    }
    // The chat stays visible next to the docked drawer; the overlay drawer would cover it.
    if (!isDesktopDrawer) {
      setWorkOpen(false);
    }
    setSidebarOpen(false);
  }, [sessions, loadHistory, loadSummaries, isDesktopDrawer]);

  const loadBackgroundData = useCallback(async (): Promise<NotificationItem[] | null> => {
    try {
      const [jobsRes, notifRes] = await Promise.all([
        fetch("/v1/background-jobs?limit=100"),
        fetch("/v1/notifications?limit=100"),
      ]);
      if (jobsRes.ok) {
        setJobs((await jobsRes.json()) as BackgroundJob[]);
      }
      if (notifRes.ok) {
        const items = (await notifRes.json()) as NotificationItem[];
        setNotifications(items);
        return items;
      }
    } catch {
      // keep UI best-effort
    }
    return null;
  }, []);

  /** Mark unseen notifications as seen on the server (the Work drawer is showing them). */
  const markNotificationsSeen = useCallback(async (items: NotificationItem[]) => {
    const unseen = items.filter((n) => !n.seen_at_ms);
    if (unseen.length === 0) {
      return;
    }
    const results = await Promise.all(
      unseen.map(async (n) => {
        try {
          const res = await fetch(`/v1/notifications/${encodeURIComponent(n.notification_id)}/seen`, {
            method: "POST",
          });
          return res.ok ? n.notification_id : null;
        } catch {
          return null;
        }
      }),
    );
    const marked = new Set(results.filter((id): id is string => id !== null));
    if (marked.size === 0) {
      return;
    }
    const now = Date.now();
    setNotifications((prev) =>
      prev.map((n) => (marked.has(n.notification_id) && !n.seen_at_ms ? { ...n, seen_at_ms: now } : n)),
    );
  }, []);

  // Jobs and notifications feed the Work badge, so poll both (paused while the tab is hidden).
  useEffect(() => {
    void loadBackgroundData();
    const timer = window.setInterval(() => {
      if (!document.hidden) {
        void loadBackgroundData();
      }
    }, 20_000);
    const onVisibility = () => {
      if (!document.hidden) {
        void loadBackgroundData();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [loadBackgroundData]);

  // Notifications count as seen once the Jobs tab of the open drawer shows them.
  useEffect(() => {
    if (workOpen && workTab === "jobs") {
      void markNotificationsSeen(notifications);
    }
  }, [workOpen, workTab, notifications, markNotificationsSeen]);

  const openWork = (tab?: WorkTab) => {
    const nextTab = tab ?? workTab;
    setWorkTab(nextTab);
    setWorkspacePaneNonce((n) => n + 1);
    setWorkOpen(true);
    setSidebarOpen(false);
    void loadBackgroundData();
    if (nextTab === "summaries") {
      void loadAllSummaries();
    }
  };

  const toggleWork = () => {
    if (workOpen) {
      setWorkOpen(false);
    } else {
      openWork();
    }
  };

  const changeWorkTab = (tab: WorkTab) => {
    setWorkTab(tab);
    if (tab === "summaries") {
      void loadAllSummaries();
    } else if (tab === "jobs" || tab === "agents") {
      void loadBackgroundData();
    }
  };

  const workBadge =
    jobs.filter(isActiveJob).length + notifications.filter(isUnreadNotification).length;

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
      setThreadSummaries((prev) => prev.map((s) => (s.id === id ? { ...s, ...updated } : s)));
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
        setThreadSummaries((prev) => prev.filter((s) => s.id !== id));
      } catch (error) {
        console.error(error);
        setErrorMessage(buildErrorMessage(error));
      }
    });
  };


  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const response = await fetch("/v1/settings/models");
        if (!response.ok) {
          return;
        }
        const body = (await response.json()) as {
          active_key: string | null;
          selected_key?: string | null;
          models: Array<{ key: string; model_name: string }>;
        };
        if (cancelled) {
          return;
        }
        if (!("selected_key" in body) || body.models.length === 0) {
          // Older server or no model catalog: nothing to choose, keep sending as before.
          setModelState("ungated");
          return;
        }
        const selected = body.models.find((model) => model.key === body.selected_key);
        if (selected) {
          setSelectedModelLabel(selected.model_name);
          setModelState("selected");
        } else {
          setModelState("none");
        }
      } catch {
        if (!cancelled) {
          setModelState("ungated");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // loadBackgroundData is now defined above

  useEffect(() => {
    const id = readPersistedThreadId();
    if (id) {
      void loadHistory(id);
      void loadSummaries(id);
    }
  }, [loadHistory, loadSummaries]);

  const onMessagesScroll = () => {
    const el = scrollContainerRef.current;
    if (!el) {
      return;
    }
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= 120;
    if (nearBottom) {
      stickToBottomRef.current = true;
    } else if (el.scrollTop < lastScrollTopRef.current) {
      // Only an upward scroll unpins; a smooth scroll toward the bottom never does.
      stickToBottomRef.current = false;
    }
    lastScrollTopRef.current = el.scrollTop;
  };

  useEffect(() => {
    if (!stickToBottomRef.current) {
      return;
    }
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending, currentStep, currentToolCalls.length, threadSummaries.length]);

  // Follow streamed text without smooth-scroll jitter on every token.
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!streamingText || !el || !stickToBottomRef.current) {
      return;
    }
    el.scrollTop = el.scrollHeight;
    lastScrollTopRef.current = el.scrollTop;
  }, [streamingText]);

  /** Coalesce streamed text updates to one render per animation frame. */
  const streamFrameRef = useRef<number | null>(null);
  const streamTextRef = useRef("");
  const publishStreamingText = useCallback((text: string, immediate = false) => {
    streamTextRef.current = text;
    if (immediate) {
      if (streamFrameRef.current !== null) {
        cancelAnimationFrame(streamFrameRef.current);
        streamFrameRef.current = null;
      }
      setStreamingText(text);
      return;
    }
    if (streamFrameRef.current === null) {
      streamFrameRef.current = requestAnimationFrame(() => {
        streamFrameRef.current = null;
        setStreamingText(streamTextRef.current);
      });
    }
  }, []);

  const startNewConversation = () => {
    // Drop the turn that is still waiting on the model. Otherwise the new
    // conversation sits behind that request and looks frozen.
    void stopGeneration();
    setSidebarOpen(false);
    startTransition(() => {
      clearSessionPointers();
      setInternalChatId(null);
      setLatestResponseId(null);
      setMessages([]);
      threadSummariesForRef.current = null;
      setThreadSummaries([]);
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

    pendingRef.current = false;
    // Queued messages wait for an explicit "Send next" after a stop.
    setQueuePaused(true);
    setPending(false);
    setCurrentStep(null);
    setCurrentToolCalls([]);
    publishStreamingText("", true);

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
    if (pendingRef.current) {
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
    stickToBottomRef.current = true;
    setMessages((prev) => [...prev, optimisticUser]);
    pendingRef.current = true;
    setPending(true);
    setCurrentStep(null);
    setCurrentToolCalls([]);
    publishStreamingText("", true);

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
      /** Text streamed for the current model call; the `completion` content replaces it. */
      let streamedText = "";
      let toolCalls: ToolCall[] = [];
      /** In-flight calls keyed by tool_call_id (or `name:<tool>` from older servers); FIFO per key. */
      const inFlightToolCalls = new Map<string, ToolCall[]>();
      const toolCallKey = (toolName: string, toolCallId?: string) =>
        toolCallId ? `id:${toolCallId}` : `name:${toolName}`;
      let streamError: string | null = null;
      let finalChatId: string | null = null;
      let finalResponseId: string | null = null;
      let clarification: Clarification | undefined;
      let buffer = "";

      while (streamError === null) {
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
          let event: StreamEvent;
          try {
            event = JSON.parse(data) as StreamEvent;
          } catch (e) {
            console.error("Failed to parse SSE event", e, "Line:", trimmedLine);
            continue;
          }
          switch (event.type) {
            case "agent_thought":
              setCurrentStep(`Thinking: ${event.thought}`);
              break;
            case "text_delta":
              if (event.text) {
                streamedText += event.text;
                publishStreamingText(streamedText);
                // A tool/thought step is over once the model writes text again.
                setCurrentStep(null);
              }
              break;
            case "text_reset":
              streamedText = "";
              publishStreamingText("", true);
              break;
            case "tool_call_started": {
              const key = toolCallKey(event.tool_name, event.tool_call_id);
              const queue = inFlightToolCalls.get(key) ?? [];
              queue.push({ tool_name: event.tool_name, args: event.args });
              inFlightToolCalls.set(key, queue);
              setCurrentStep(`Using tool: ${event.tool_name}`);
              break;
            }
            case "tool_progress":
              setCurrentStep(`${event.tool_name}: ${event.message}`);
              break;
            case "tool_call_finished": {
              const key = toolCallKey(event.tool_name, event.tool_call_id);
              const queue = inFlightToolCalls.get(key);
              const started = queue?.shift();
              if (queue && queue.length === 0) {
                inFlightToolCalls.delete(key);
              }
              toolCalls.push({
                tool_name: event.tool_name,
                args: started?.args ?? "",
                result: event.result,
                isError: event.is_error === true,
                mcpApp: event.mcp_app,
              });
              setCurrentToolCalls([...toolCalls]);
              // Don't set currentStep to null here, let the next event (thought or next tool) replace it
              // This prevents the indicator from flickering or disappearing between steps
              break;
            }
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
              clarification = event.clarification;
              setCurrentStep(null); // Clear step when completion arrives
              break;
            case "error":
              streamError = event.message || "The agent reported an error.";
              break;
          }
          if (streamError !== null) {
            break;
          }
        }
      }

      if (streamError !== null) {
        void reader.cancel().catch(() => undefined);
      } else if (!finalResponseId && !assistantContent) {
        streamError = "The response ended before the assistant replied.";
      }
      // `completion` is authoritative; without one (stream error) keep the partial streamed text.
      if (!finalResponseId && !assistantContent && streamedText.trim()) {
        assistantContent = streamedText;
      }

      // Keep whatever the turn produced; skip an empty bubble when there is nothing to show.
      if (assistantContent || toolCalls.length > 0 || clarification) {
        const assistantId = createId();
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            role: "assistant",
            content: assistantContent,
            toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
            clarification,
          },
        ]);
      }

      if (streamError !== null) {
        // The server accepted the turn, so the user message stays; only surface the failure.
        setQueuePaused(true);
        setErrorMessage(streamError);
        void loadSessions();
        return;
      }

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
      setQueuePaused(true);
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
      pendingRef.current = false;
      setPending(false);
      setCurrentStep(null);
      setCurrentToolCalls([]);
      publishStreamingText("", true);
    }
  };

  const enqueueMessage = (payload: ComposerPayload) => {
    if (queue.length === 0) {
      // A pause only holds back messages that were waiting when the turn stopped.
      setQueuePaused(false);
    }
    setQueue((prev) => [...prev, { ...payload, id: createId() }]);
  };

  const openModelSettings = (notice: string | null) => {
    setSidebarOpen(false);
    setSettingsNotice(notice);
    setShowSettings(true);
  };

  const modelReady = modelState === "selected" || modelState === "ungated";

  /**
   * Composer and MCP App messages: send now, or queue behind the running turn. Without a
   * model chosen in this session the message waits in the queue and the Models tab opens;
   * it goes out once a model is selected.
   */
  const sendOrQueue = (payload: ComposerPayload) => {
    if (!payload.text.trim() && payload.imageDataUrls.length === 0) {
      return;
    }
    if (pendingRef.current || !modelReady) {
      enqueueMessage(payload);
      if (modelState === "none") {
        openModelSettings("Choose a model to send your message.");
      }
      return;
    }
    void submitMessage(payload);
  };

  // A message typed before the model list loaded: ask for a model once we know none is chosen.
  useEffect(() => {
    if (modelState === "none" && queue.length > 0 && !showSettings) {
      openModelSettings("Choose a model to send your message.");
    }
    // Only the model state resolving should open the dialog, not later queue edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelState]);

  const onModelSelected = (label: string) => {
    setSelectedModelLabel(label);
    setModelState("selected");
    if (settingsNotice && queue.length > 0) {
      // The dialog was opened to send a message; close it so the user sees it go out.
      setShowSettings(false);
      setSettingsNotice(null);
    }
  };

  // When a turn ends, send the next queued message (unless the queue is paused or no model
  // is selected yet).
  useEffect(() => {
    if (
      !modelReady ||
      pending ||
      historyLoading ||
      queuePaused ||
      queue.length === 0 ||
      pendingRef.current
    ) {
      return;
    }
    const [next, ...rest] = queue;
    setQueue(rest);
    void submitMessage({ text: next.text, imageDataUrls: next.imageDataUrls });
    // submitMessage is recreated each render; this effect only reacts to queue/turn state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelReady, pending, historyLoading, queuePaused, queue]);

  const resumeQueue = () => {
    setQueuePaused(false);
  };

  const deliverUiMessage = useStableCallback((text: string) => {
    const trimmed = text.trim();
    if (trimmed) {
      sendOrQueue({ text: trimmed, imageDataUrls: [] });
    }
  });

  const chooseClarification = useStableCallback((choice: string) => {
    void submitMessage({ text: choice, imageDataUrls: [] });
  });

  return (
    <div className="flex h-dvh overflow-hidden bg-background font-sans">
      {sessionToDelete ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 outline-none"
          ref={deleteDialogRef}
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
      {showSettings ? (
        <SettingsDialog
          notice={settingsNotice}
          onClose={() => {
            setShowSettings(false);
            setSettingsNotice(null);
          }}
          onActiveModel={onModelSelected}
        />
      ) : null}
      {dialogConfig ? (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4 outline-none"
          ref={appDialogRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby="app-dialog-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              dialogConfig.onCancel();
            }
          }}
        >
          <div className="w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-xl">
            <h2 id="app-dialog-title" className="text-lg font-semibold text-foreground mb-2">{dialogConfig.title}</h2>
            <p className="text-sm text-muted-foreground mb-6">{dialogConfig.message}</p>
            
            {dialogConfig.type === 'prompt' && (
              <input
                ref={dialogInputRef}
                data-autofocus
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm mb-6 focus:outline-none focus:ring-2 focus:ring-primary"
                defaultValue={dialogConfig.defaultValue}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    dialogConfig.onConfirm(dialogInputRef.current?.value || '');
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
      {sidebarOpen && !isWideSidebar ? (
        <div aria-hidden className="fixed inset-0 z-40 bg-black/40 md:hidden" onClick={() => setSidebarOpen(false)} />
      ) : null}
      <aside
        aria-label="Threads"
        aria-modal={sidebarOpen && !isWideSidebar ? true : undefined}
        className={cn(
          "flex w-72 shrink-0 flex-col border-r border-border bg-background outline-none",
          // Off-canvas menu below md; docked column from md up.
          "fixed inset-y-0 left-0 z-50 shadow-2xl transition-transform duration-200 ease-out",
          sidebarOpen ? "visible translate-x-0" : "invisible -translate-x-full",
          "md:visible md:static md:z-auto md:w-64 md:translate-x-0 md:bg-background/90 md:shadow-none md:backdrop-blur-md md:transition-none",
        )}
        id="thread-sidebar"
        ref={sidebarRef}
        role={sidebarOpen && !isWideSidebar ? "dialog" : undefined}
      >
        <div className="border-b border-border p-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-semibold tracking-[-0.02em] text-foreground">isanagent</p>
            <div className="flex items-center gap-1">
              <ThemeToggle />
              <Button
                aria-label="Close menu"
                className="h-9 w-9 text-lg md:hidden"
                size="icon"
                type="button"
                variant="ghost"
                onClick={() => setSidebarOpen(false)}
              >
                ×
              </Button>
            </div>
          </div>
          {selectedModelLabel ? (
            <p className="mt-1 truncate text-xs text-muted-foreground" title={selectedModelLabel}>
              {selectedModelLabel}
            </p>
          ) : modelState === "none" ? (
            <button
              className="mt-1 block max-w-full truncate rounded text-left text-xs text-muted-foreground/80 hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
              title="Choose a model in Settings"
              type="button"
              onClick={() => openModelSettings(null)}
            >
              No model selected
            </button>
          ) : null}
          <Button className="mt-2 w-full" size="sm" onClick={startNewConversation}>
            New chat
          </Button>
          <div className="mt-2 flex gap-2">
            <Button
              aria-controls="work-drawer"
              aria-expanded={workOpen}
              className={cn("relative flex-1", workOpen && "bg-[color:var(--ghost-fill)]")}
              size="sm"
              variant="outline"
              onClick={toggleWork}
            >
              Work
              {workBadge > 0 ? (
                <span
                  aria-label={`${workBadge} items need attention`}
                  className="ml-1.5 min-w-[1.25rem] rounded-full bg-primary px-1.5 text-[10px] font-semibold leading-[1.15rem] text-primary-foreground"
                >
                  {workBadge}
                </span>
              ) : null}
            </Button>
            <Button
              className="flex-1"
              size="sm"
              variant="outline"
              onClick={() => {
                // On narrow screens the sidebar overlays the page; close it so it does not cover the dialog.
                openModelSettings(null);
              }}
            >
              Settings
            </Button>
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          <p className="px-2 pb-1 text-xs font-medium uppercase tracking-[0.05em] text-muted-foreground">
            Threads
          </p>
          {sessionsError ? (
            <div className="mx-2 mb-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive" role="alert">
              <p>{sessionsError}</p>
              <Button
                className="mt-2 h-7 text-xs"
                disabled={sessionsLoading}
                size="sm"
                type="button"
                variant="outline"
                onClick={() => void loadSessions()}
              >
                {sessionsLoading ? "Retrying…" : "Retry"}
              </Button>
            </div>
          ) : null}
          {sessionsLoading && sessions.length === 0 && !sessionsError ? (
            <p className="px-2 text-xs text-muted-foreground">Loading…</p>
          ) : sessions.length === 0 ? (
            sessionsError ? null : (
              <p className="px-2 text-xs text-muted-foreground">No saved chats yet.</p>
            )
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
                        // Let the nested delete button handle its own keys.
                        if (e.target !== e.currentTarget) {
                          return;
                        }
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
                        className="shrink-0 rounded-lg p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/15 hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100 group-focus-within:opacity-100"
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

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden px-2 py-2 sm:px-5 sm:py-4">
        <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-panel">
          <div className="flex items-center justify-between gap-3 border-b border-border px-3 py-3 sm:px-6 sm:py-4">
            <div className="flex min-w-0 items-center gap-2">
              <Button
                aria-controls="thread-sidebar"
                aria-expanded={sidebarOpen}
                aria-label="Open menu"
                className="h-9 w-9 shrink-0 md:hidden"
                size="icon"
                type="button"
                variant="ghost"
                onClick={() => setSidebarOpen(true)}
              >
                <MenuIcon />
              </Button>
              <div className="min-w-0">
                <p className="hidden text-xs font-medium uppercase tracking-[0.05em] text-muted-foreground sm:block">
                  Chat
                </p>
                <h1 className="truncate text-lg font-semibold tracking-[-0.02em] text-foreground sm:mt-1 sm:text-xl">
                  isanagent
                </h1>
                <p className="mt-1 hidden max-w-xl text-sm leading-relaxed text-muted-foreground lg:block">
                  Workspace-backed memory, multimodal input, and a thread list for this browser profile.
                </p>
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <span className="hidden shrink-0 rounded-full border border-[color:var(--ghost-border)] bg-muted px-3 py-1 text-xs font-medium text-muted-foreground sm:inline">
                {pending || historyLoading ? "Syncing…" : internalChatId ? "In thread" : "New thread"}
              </span>
              <Button
                aria-controls="work-drawer"
                aria-expanded={workOpen}
                aria-label={workBadge > 0 ? `Work (${workBadge})` : "Work"}
                className="relative h-9 md:hidden"
                size="sm"
                type="button"
                variant="outline"
                onClick={toggleWork}
              >
                Work
                {workBadge > 0 ? (
                  <span className="ml-1.5 min-w-[1.25rem] rounded-full bg-primary px-1.5 text-[10px] font-semibold leading-[1.15rem] text-primary-foreground">
                    {workBadge}
                  </span>
                ) : null}
              </Button>
            </div>
          </div>

          <div
            className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-4 sm:px-6"
            ref={scrollContainerRef}
            onScroll={onMessagesScroll}
          >
            {(messages.length > 0 || threadSummaries.length > 0 || pending) ? (
              <div className="mx-auto flex min-h-full w-full min-w-0 max-w-3xl flex-col gap-4">
                {threadSummaries.length > 0 && (
                  <div className="space-y-4">
                    {/* Only show the most recent summary (which is now the ONLY summary) */}
                    {(() => {
                      const s = threadSummaries[0];
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
                {messages.map((message, index) => (
                  <MessageBubble
                    choicesEnabled={!pending && index === messages.length - 1}
                    key={message.id}
                    message={message}
                    onChatMessage={deliverUiMessage}
                    onChoose={chooseClarification}
                  />
                ))}

                {pending ? (
                  <PendingAssistantBubble
                    onChatMessage={deliverUiMessage}
                    step={currentStep}
                    text={streamingText}
                    toolCalls={currentToolCalls}
                  />
                ) : null}

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

          <div className="border-t border-border/70 px-3 py-3 sm:px-6 sm:py-4">
            <div className="mx-auto max-w-3xl">
              {errorMessage ? (
                <div
                  className="mb-3 flex items-start justify-between gap-3 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
                  role="alert"
                >
                  <p className="min-w-0 flex-1 whitespace-pre-wrap break-words">{errorMessage}</p>
                  <button
                    aria-label="Dismiss error"
                    className="-my-1 shrink-0 rounded-md px-1.5 text-base leading-6 hover:bg-destructive/15"
                    title="Dismiss"
                    type="button"
                    onClick={() => setErrorMessage(null)}
                  >
                    ×
                  </button>
                </div>
              ) : null}
              <QueuedMessages
                canResume={!pending && !historyLoading}
                items={queue}
                paused={queuePaused}
                waitingForModel={modelState === "none"}
                onRemove={(id) => setQueue((prev) => prev.filter((item) => item.id !== id))}
                onResume={resumeQueue}
                onChooseModel={() => openModelSettings("Choose a model to send your message.")}
              />
              <Composer
                locked={historyLoading}
                pending={pending}
                onStop={() => void stopGeneration()}
                onSubmit={sendOrQueue}
              />
            </div>
          </div>
        </div>
      </div>

      <WorkDrawer
        filesNonce={workspacePaneNonce}
        jobs={jobs}
        notifications={notifications}
        open={workOpen}
        overlay={!isDesktopDrawer}
        summaries={summaries}
        summariesLoading={summariesLoading}
        tab={workTab}
        onClose={() => setWorkOpen(false)}
        onDeleteSummary={deleteSummary}
        onDismissJob={(job) =>
          showConfirm("Dismiss Job", "Dismiss this background job and all associated requests?", () => {
            void dismissBackgroundJob(job.job_id);
          })
        }
        onDismissTicket={(n) =>
          showConfirm(
            "Dismiss Request",
            "Dismiss this request? This will also mark the background job as completed.",
            () => {
              void dismissTicket(n.action_payload || "");
            },
          )
        }
        onJumpToThread={jumpToThread}
        onRefresh={() => {
          void loadBackgroundData();
          if (workTab === "summaries") {
            void loadAllSummaries();
          }
          if (workTab === "files") {
            setWorkspacePaneNonce((n) => n + 1);
          }
        }}
        onReplyTicket={(n) =>
          showPrompt("Reply to request", "Enter your response:", "", (reply) => {
            if (reply.trim()) {
              void replyToTicket(n.action_payload || "", reply.trim());
            }
          })
        }
        onTabChange={changeWorkTab}
        onUpdateSummary={updateSummary}
      />
    </div>
  );
}

function MenuIcon() {
  return (
    <svg aria-hidden className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path d="M4 6h16M4 12h16M4 18h16" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
    </svg>
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

