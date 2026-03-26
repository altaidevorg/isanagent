import { useEffect, useMemo, useRef, useState, useTransition } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

type MessageRole = "user" | "assistant";

type Message = {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
};

type Conversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  latestResponseId: string | null;
  messages: Message[];
};

type StoredState = {
  userId: string;
  activeConversationId: string | null;
  conversations: Conversation[];
};

type ResponsesOutputContent = {
  type: string;
  text?: string;
};

type ResponsesOutputItem = {
  content?: ResponsesOutputContent[];
};

type ResponsesResponse = {
  id: string;
  output?: ResponsesOutputItem[];
};

const STORAGE_KEY = "agent_rs_ui_state_v1";
const DEFAULT_TITLE = "New conversation";

function createId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `id_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function buildDefaultState(): StoredState {
  return {
    userId: `ui_${createId()}`,
    activeConversationId: null,
    conversations: [],
  };
}

function loadStoredState(): StoredState {
  if (typeof window === "undefined") {
    return buildDefaultState();
  }

  const rawValue = window.localStorage.getItem(STORAGE_KEY);
  if (!rawValue) {
    return buildDefaultState();
  }

  try {
    const parsed = JSON.parse(rawValue) as Partial<StoredState>;
    const userId =
      typeof parsed.userId === "string" && parsed.userId.length > 0
        ? parsed.userId
        : `ui_${createId()}`;
    const conversations = Array.isArray(parsed.conversations)
      ? parsed.conversations.filter(isConversation)
      : [];
    const storedActiveConversationId =
      typeof parsed.activeConversationId === "string"
        ? parsed.activeConversationId
        : null;
    const activeConversationId = conversations.some(
      (conversation) => conversation.id === storedActiveConversationId,
    )
      ? storedActiveConversationId
      : conversations[0]?.id ?? null;

    return {
      userId,
      activeConversationId,
      conversations,
    };
  } catch {
    return buildDefaultState();
  }
}

function isMessage(value: unknown): value is Message {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Partial<Message>;
  return (
    typeof candidate.id === "string" &&
    (candidate.role === "user" || candidate.role === "assistant") &&
    typeof candidate.content === "string" &&
    typeof candidate.createdAt === "number"
  );
}

function isConversation(value: unknown): value is Conversation {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Partial<Conversation>;
  return (
    typeof candidate.id === "string" &&
    typeof candidate.title === "string" &&
    typeof candidate.createdAt === "number" &&
    typeof candidate.updatedAt === "number" &&
    (typeof candidate.latestResponseId === "string" ||
      candidate.latestResponseId === null) &&
    Array.isArray(candidate.messages) &&
    candidate.messages.every(isMessage)
  );
}

function deriveTitle(content: string) {
  const trimmed = content.trim().replace(/\s+/g, " ");
  if (!trimmed) {
    return DEFAULT_TITLE;
  }
  if (trimmed.length <= 42) {
    return trimmed;
  }
  return `${trimmed.slice(0, 39)}...`;
}

function createConversation(): Conversation {
  const now = Date.now();
  return {
    id: createId(),
    title: DEFAULT_TITLE,
    createdAt: now,
    updatedAt: now,
    latestResponseId: null,
    messages: [],
  };
}

function extractAssistantText(response: ResponsesResponse) {
  const content = response.output
    ?.flatMap((item) => item.content ?? [])
    .find((entry) => entry.type === "output_text" && typeof entry.text === "string");

  return content?.text?.trim() || "The agent returned an empty response.";
}

function formatTimestamp(value: number) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
    day: "numeric",
  }).format(new Date(value));
}

function buildErrorMessage(error: unknown) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Request failed. Try again.";
}

function rollbackOptimisticMessage(
  conversation: Conversation,
  userMessageId: string,
): Conversation {
  const messages = conversation.messages.filter((message) => message.id !== userMessageId);

  return {
    ...conversation,
    title: messages.length === 0 ? DEFAULT_TITLE : conversation.title,
    updatedAt: messages[messages.length - 1]?.createdAt ?? conversation.createdAt,
    messages,
  };
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
        placeholder="Ask Agent-RS anything. Shift+Enter for a new line."
        value={draft}
      />
      <div className="mt-4 flex items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">
          {disabled
            ? "Waiting for the current response to finish."
            : "Messages are stored in this browser and continued via /v1/responses."}
        </p>
        <Button disabled={disabled || draft.trim().length === 0} onClick={() => void submit()}>
          {disabled ? "Working..." : "Send"}
        </Button>
      </div>
    </div>
  );
}

export default function App() {
  const [state, setState] = useState<StoredState>(() => loadStoredState());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pendingConversationId, setPendingConversationId] = useState<string | null>(null);
  const [isNavPending, startTransition] = useTransition();
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // Ignore local persistence failures and keep the in-memory session usable.
    }
  }, [state]);

  const sortedConversations = useMemo(
    () =>
      [...state.conversations].sort((left, right) => right.updatedAt - left.updatedAt),
    [state.conversations],
  );

  const activeConversation = useMemo(() => {
    return (
      state.conversations.find(
        (conversation) => conversation.id === state.activeConversationId,
      ) ?? null
    );
  }, [state.activeConversationId, state.conversations]);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [activeConversation?.messages.length, pendingConversationId]);

  const selectConversation = (conversationId: string) => {
    startTransition(() => {
      setState((currentState) => ({
        ...currentState,
        activeConversationId: conversationId,
      }));
    });
  };

  const startNewConversation = () => {
    startTransition(() => {
      setState((currentState) => {
        const conversation = createConversation();
        return {
          ...currentState,
          activeConversationId: conversation.id,
          conversations: [conversation, ...currentState.conversations],
        };
      });
      setErrorMessage(null);
    });
  };

  const submitMessage = async (content: string) => {
    if (pendingConversationId) {
      return;
    }

    setErrorMessage(null);

    const requestUserId = state.userId;
    let conversationId = state.activeConversationId;
    let latestResponseId: string | null = null;
    const userMessage: Message = {
      id: createId(),
      role: "user",
      content,
      createdAt: Date.now(),
    };

    setState((currentState) => {
      const conversations = [...currentState.conversations];
      const now = Date.now();

      let conversationIndex = conversationId
        ? conversations.findIndex((conversation) => conversation.id === conversationId)
        : -1;

      if (conversationIndex === -1) {
        const conversation = createConversation();
        conversationId = conversation.id;
        conversations.unshift(conversation);
        conversationIndex = 0;
      }

      const currentConversation = conversations[conversationIndex];
      latestResponseId = currentConversation.latestResponseId;

      const nextConversation: Conversation = {
        ...currentConversation,
        title:
          currentConversation.messages.length === 0 ||
          currentConversation.title === DEFAULT_TITLE
            ? deriveTitle(content)
            : currentConversation.title,
        updatedAt: now,
        messages: [...currentConversation.messages, userMessage],
      };

      conversations[conversationIndex] = nextConversation;

      return {
        ...currentState,
        activeConversationId: conversationId,
        conversations,
      };
    });

    if (!conversationId) {
      throw new Error("Conversation state could not be created.");
    }

    setPendingConversationId(conversationId);

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

      const data = (await response.json()) as ResponsesResponse;
      const assistantMessage: Message = {
        id: createId(),
        role: "assistant",
        content: extractAssistantText(data),
        createdAt: Date.now(),
      };

      setState((currentState) => ({
        ...currentState,
        conversations: currentState.conversations.map((conversation) =>
          conversation.id === conversationId
            ? {
                ...conversation,
                latestResponseId: data.id,
                updatedAt: assistantMessage.createdAt,
                messages: [...conversation.messages, assistantMessage],
              }
            : conversation,
        ),
      }));
    } catch (error) {
      setState((currentState) => ({
        ...currentState,
        conversations: currentState.conversations.map((conversation) =>
          conversation.id === conversationId
            ? rollbackOptimisticMessage(conversation, userMessage.id)
            : conversation,
        ),
      }));
      setErrorMessage(buildErrorMessage(error));
    } finally {
      setPendingConversationId(null);
    }
  };

  return (
    <div className="h-dvh overflow-hidden px-4 py-5 sm:px-6 lg:px-8">
      <div className="mx-auto flex h-[calc(100dvh-2.5rem)] w-full max-w-7xl gap-4 overflow-hidden lg:gap-6">
        <aside className="hidden min-h-0 w-80 shrink-0 rounded-[2rem] border border-border/70 bg-card/90 p-5 shadow-panel backdrop-blur lg:flex lg:flex-col">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-muted-foreground">
                Agent-RS
              </p>
              <h1 className="mt-2 text-2xl font-semibold tracking-tight">
                Built-in UI
              </h1>
            </div>
            <Button onClick={startNewConversation} size="sm">
              New chat
            </Button>
          </div>

          <div className="mt-4 rounded-2xl bg-secondary/65 p-4 text-sm text-secondary-foreground">
            <p className="font-medium">Local-first mode</p>
            <p className="mt-1 text-xs leading-5 text-secondary-foreground/80">
              Browser state persists conversations. The server only stores response chain ids.
            </p>
          </div>

          <div className="mt-5 flex-1 overflow-y-auto pr-1">
            <div className="space-y-2">
              {sortedConversations.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-border bg-background/80 p-4 text-sm text-muted-foreground">
                  Start a chat to create your first local conversation.
                </div>
              ) : (
                sortedConversations.map((conversation) => {
                  const isActive = conversation.id === state.activeConversationId;
                  return (
                    <button
                      className={cn(
                        "w-full rounded-2xl border px-4 py-3 text-left transition-colors",
                        isActive
                          ? "border-primary/50 bg-primary/10 text-foreground"
                          : "border-transparent bg-background/70 hover:border-border hover:bg-accent/50",
                      )}
                      key={conversation.id}
                      onClick={() => selectConversation(conversation.id)}
                      type="button"
                    >
                      <div className="flex items-center justify-between gap-4">
                        <p className="line-clamp-1 font-medium">{conversation.title}</p>
                        <span className="text-[11px] text-muted-foreground">
                          {formatTimestamp(conversation.updatedAt)}
                        </span>
                      </div>
                      <p className="mt-2 line-clamp-2 text-xs leading-5 text-muted-foreground">
                        {conversation.messages[conversation.messages.length - 1]?.content ||
                          "No messages yet."}
                      </p>
                    </button>
                  );
                })
              )}
            </div>
          </div>

          <div className="mt-4 rounded-2xl border border-border/70 bg-background/80 p-4 text-xs text-muted-foreground">
            <p>User id</p>
            <p className="mt-1 break-all font-mono text-[11px] text-foreground/80">
              {state.userId}
            </p>
          </div>
        </aside>

        <main className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-[2rem] border border-border/70 bg-card/85 shadow-panel backdrop-blur">
          <div className="flex items-center justify-between gap-3 border-b border-border/70 px-5 py-4 sm:px-6">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-muted-foreground">
                Chat
              </p>
              <h2 className="mt-1 text-xl font-semibold tracking-tight">
                {activeConversation?.title || "New conversation"}
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <Button className="lg:hidden" onClick={startNewConversation} size="sm" variant="outline">
                New chat
              </Button>
              <span className="rounded-full bg-accent px-3 py-1 text-xs font-medium text-accent-foreground">
                {pendingConversationId ? "Waiting for response" : "Ready"}
              </span>
            </div>
          </div>

          <div className="border-b border-border/70 px-4 py-3 lg:hidden">
            <div className="flex gap-2 overflow-x-auto pb-1">
              {sortedConversations.length === 0 ? (
                <div className="rounded-full border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">
                  No saved conversations yet
                </div>
              ) : (
                sortedConversations.map((conversation) => (
                  <button
                    className={cn(
                      "shrink-0 rounded-full border px-3 py-2 text-xs font-medium transition-colors",
                      conversation.id === state.activeConversationId
                        ? "border-primary/40 bg-primary/10 text-foreground"
                        : "border-border bg-background/80 text-muted-foreground",
                    )}
                    key={conversation.id}
                    onClick={() => selectConversation(conversation.id)}
                    type="button"
                  >
                    {conversation.title}
                  </button>
                ))
              )}
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-4 sm:px-6">
            {activeConversation?.messages.length ? (
              <div className="mx-auto flex min-h-full w-full max-w-4xl flex-col gap-4">
                {activeConversation.messages.map((message) => (
                  <article
                    className={cn(
                      "max-w-[82%] rounded-[1.5rem] px-4 py-3 shadow-sm",
                      message.role === "user"
                        ? "ml-auto bg-primary text-primary-foreground"
                        : "mr-auto border border-border/70 bg-background/90 text-foreground",
                    )}
                    key={message.id}
                  >
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-[11px] font-semibold uppercase tracking-[0.2em] opacity-70">
                        {message.role}
                      </span>
                      <span className="text-[11px] opacity-70">
                        {formatTimestamp(message.createdAt)}
                      </span>
                    </div>
                    <p className="mt-2 whitespace-pre-wrap text-sm leading-7">
                      {message.content}
                    </p>
                  </article>
                ))}

                {pendingConversationId === activeConversation.id ? (
                  <div className="mr-auto max-w-[82%] rounded-[1.5rem] border border-dashed border-border bg-background/80 px-4 py-3 text-sm text-muted-foreground">
                    Agent-RS is generating a response...
                  </div>
                ) : null}

                <div ref={endOfMessagesRef} />
              </div>
            ) : (
              <div className="mx-auto flex min-h-full max-w-3xl flex-col justify-center">
                <div className="rounded-[2rem] border border-dashed border-border/80 bg-background/70 p-8 text-center">
                  <p className="text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground">
                    Embedded UI
                  </p>
                  <h3 className="mt-3 text-3xl font-semibold tracking-tight">
                    Talk to Agent-RS without Slack
                  </h3>
                  <p className="mx-auto mt-3 max-w-2xl text-sm leading-7 text-muted-foreground">
                    This UI keeps conversations in your browser and continues them by sending the
                    last stored response id back to the existing responses API.
                  </p>
                  <div className="mt-6 flex flex-wrap items-center justify-center gap-2 text-xs text-muted-foreground">
                    <span className="rounded-full bg-secondary px-3 py-1">Same-origin API</span>
                    <span className="rounded-full bg-secondary px-3 py-1">Local persistence</span>
                    <span className="rounded-full bg-secondary px-3 py-1">Final-only responses</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-border/70 px-4 py-4 sm:px-6">
            <div className="mx-auto max-w-4xl">
              {errorMessage ? (
                <div className="mb-3 rounded-2xl border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                  {errorMessage}
                </div>
              ) : null}
              <Composer
                disabled={pendingConversationId !== null || isNavPending}
                onSubmit={submitMessage}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
