import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";

export type McpAppRef = {
  view_id: string;
  resource_uri: string;
};

type McpAppCsp = {
  connectDomains?: string[];
  resourceDomains?: string[];
  frameDomains?: string[];
  baseUriDomains?: string[];
};

type McpAppSnapshot = {
  html: string;
  csp?: McpAppCsp;
  tool_args?: Record<string, unknown>;
  tool_result?: unknown;
};

function originList(values: string[] | undefined): string {
  return (values ?? []).filter((value) => value.trim().length > 0).join(" ");
}

function cspMeta(csp: McpAppCsp | undefined): string {
  const resources = originList(csp?.resourceDomains);
  const connect = originList(csp?.connectDomains);
  const frames = originList(csp?.frameDomains);
  const bases = originList(csp?.baseUriDomains);
  const policy = [
    "default-src 'none'",
    "script-src 'unsafe-inline'",
    resources ? `style-src 'unsafe-inline' ${resources}` : "style-src 'unsafe-inline'",
    resources ? `img-src data: blob: ${resources}` : "img-src data: blob:",
    resources ? `font-src data: ${resources}` : "font-src data:",
    resources ? `media-src ${resources}` : "media-src 'none'",
    connect ? `connect-src ${connect}` : "connect-src 'none'",
    frames ? `frame-src ${frames}` : "frame-src 'none'",
    bases ? `base-uri ${bases}` : "base-uri 'none'",
    "form-action 'none'",
  ].join("; ");
  return `<meta http-equiv="Content-Security-Policy" content="${policy.replace(/"/g, "")}">`;
}

const fitStyle = `<style id="isanagent-fit">
html, body { max-width: 100% !important; box-sizing: border-box; }
*, *::before, *::after { box-sizing: border-box; }
body { margin: 0; overflow-x: hidden; }
img, svg, video, canvas { max-width: 100%; }
.container { max-width: 100% !important; }
.header, .header h1, .item, .add-form, .chat-sync-bar { flex-wrap: wrap; min-width: 0; }
.item-content { flex: 1 1 140px; min-width: 0; }
.item-title { min-width: 0; overflow-wrap: anywhere; }
input, textarea, select, button { max-width: 100%; }
</style>`;

function withCsp(html: string, csp: McpAppCsp | undefined): string {
  const fit = html.includes("isanagent-fit") ? "" : fitStyle;
  if (/content-security-policy/i.test(html)) {
    return html.includes("<head>") ? html.replace("<head>", `<head>${fit}`) : `${fit}${html}`;
  }
  const meta = cspMeta(csp);
  if (html.includes("<head>")) {
    return html.replace("<head>", `<head>${meta}${fit}`);
  }
  return `${meta}${fit}${html}`;
}

function frameHeight(current: number, reported: number): number {
  const next = Math.min(1400, Math.max(220, Math.ceil(reported)));
  // The widget measures the iframe viewport, so each extra pixel we add comes
  // back as a taller page and the frame grows without new content.
  if (next <= current + 2) {
    return current;
  }
  return next;
}

type DisplayMode = "inline" | "fullscreen";

function themeMode(): "dark" | "light" {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

function hostContext(theme: "dark" | "light", displayMode: DisplayMode) {
  return { theme, displayMode };
}

function grantedDisplayMode(requested: unknown): DisplayMode {
  return requested === "fullscreen" ? "fullscreen" : "inline";
}

function messageText(params: { content?: { text?: unknown }; text?: unknown } | undefined): string {
  if (typeof params?.content?.text === "string") {
    return params.content.text;
  }
  if (typeof params?.text === "string") {
    return params.text;
  }
  return "";
}

export function McpAppFrame({
  app,
  onChatMessage,
}: {
  app: McpAppRef;
  onChatMessage: (text: string) => void;
}) {
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const snapshotRef = useRef<McpAppSnapshot | null>(null);
  const appReadyRef = useRef(false);
  const displayModeRef = useRef<DisplayMode>("inline");
  const onChatMessageRef = useRef(onChatMessage);
  onChatMessageRef.current = onChatMessage;
  const [srcDoc, setSrcDoc] = useState<string | null>(null);
  const [height, setHeight] = useState(360);
  const [error, setError] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<DisplayMode>("inline");

  const postHostContext = useCallback((mode: DisplayMode) => {
    if (!appReadyRef.current) {
      return;
    }
    frameRef.current?.contentWindow?.postMessage(
      {
        jsonrpc: "2.0",
        method: "ui/notifications/host-context-changed",
        params: hostContext(themeMode(), mode),
      },
      "*",
    );
  }, []);

  const setMode = useCallback(
    (next: DisplayMode) => {
      displayModeRef.current = next;
      setDisplayMode(next);
      postHostContext(next);
    },
    [postHostContext],
  );

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const response = await fetch(`/v1/mcp/apps/${encodeURIComponent(app.view_id)}`);
        if (!response.ok) {
          throw new Error(`MCP App view failed (${response.status})`);
        }
        const snapshot = (await response.json()) as McpAppSnapshot;
        if (cancelled) {
          return;
        }
        snapshotRef.current = snapshot;
        appReadyRef.current = false;
        setSrcDoc(withCsp(snapshot.html, snapshot.csp));
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "MCP App failed to load");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [app.view_id]);

  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      const frame = frameRef.current;
      if (!frame || event.source !== frame.contentWindow) {
        return;
      }
      const data = event.data as {
        jsonrpc?: string;
        id?: number | string;
        method?: string;
        params?: {
          height?: number;
          url?: string;
          name?: string;
          mode?: string;
          arguments?: unknown;
          content?: { text?: unknown };
          text?: unknown;
        };
      };
      if (!data || data.jsonrpc !== "2.0") {
        return;
      }
      const post = (payload: object) => {
        frame.contentWindow?.postMessage(payload, "*");
      };

      if (data.method === "ui/initialize" && data.id != null) {
        post({
          jsonrpc: "2.0",
          id: data.id,
          result: {
            protocolVersion: "2026-01-26",
            hostInfo: { name: "isanagent", version: "0.14.0" },
            hostCapabilities: { openLinks: {}, serverTools: {}, logging: {} },
            hostContext: hostContext(themeMode(), displayModeRef.current),
          },
        });
        appReadyRef.current = true;
        return;
      }

      if (data.method === "ui/notifications/initialized") {
        const snapshot = snapshotRef.current;
        post({
          jsonrpc: "2.0",
          method: "ui/notifications/tool-input",
          params: { arguments: snapshot?.tool_args ?? {} },
        });
        if (snapshot?.tool_result) {
          post({
            jsonrpc: "2.0",
            method: "ui/notifications/tool-result",
            params: snapshot.tool_result,
          });
        }
        return;
      }

      if (data.method === "ui/notifications/size-changed") {
        const next = Number(data.params?.height);
        if (Number.isFinite(next)) {
          setHeight((current) => frameHeight(current, next));
        }
        return;
      }

      if (data.method === "ui/message" && data.id != null) {
        const text = messageText(data.params);
        post({ jsonrpc: "2.0", id: data.id, result: {} });
        if (text.trim()) {
          onChatMessageRef.current(text);
        }
        return;
      }

      if (data.method === "ui/open-link" && data.id != null) {
        const url = data.params?.url ?? "";
        if (url.startsWith("https://") || url.startsWith("http://")) {
          window.open(url, "_blank", "noopener,noreferrer");
        }
        post({ jsonrpc: "2.0", id: data.id, result: {} });
        return;
      }

      if (data.method === "ui/request-display-mode" && data.id != null) {
        const mode = grantedDisplayMode(data.params?.mode);
        post({ jsonrpc: "2.0", id: data.id, result: { mode } });
        setMode(mode);
        return;
      }

      if (data.method === "tools/call" && data.id != null) {
        const requestId = data.id;
        const name = data.params?.name;
        const args = data.params?.arguments ?? {};
        void (async () => {
          try {
            const response = await fetch(`/v1/mcp/apps/${encodeURIComponent(app.view_id)}/tools/call`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name, arguments: args }),
            });
            const payload = (await response.json()) as { error?: { message?: string } };
            if (!response.ok) {
              throw new Error(payload.error?.message || `Tool call failed (${response.status})`);
            }
            post({ jsonrpc: "2.0", id: requestId, result: payload });
          } catch (err) {
            post({
              jsonrpc: "2.0",
              id: requestId,
              error: {
                code: -32603,
                message: err instanceof Error ? err.message : "Tool call failed",
              },
            });
          }
        })();
      }
    };

    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [app.view_id, setMode]);

  useEffect(() => {
    const root = document.documentElement;
    let current = themeMode();
    const observer = new MutationObserver(() => {
      const next = themeMode();
      if (next === current || !appReadyRef.current) {
        current = next;
        return;
      }
      current = next;
      frameRef.current?.contentWindow?.postMessage(
        {
          jsonrpc: "2.0",
          method: "ui/notifications/host-context-changed",
          params: hostContext(next, displayModeRef.current),
        },
        "*",
      );
    });
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (displayMode !== "fullscreen") {
      return;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMode("inline");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [displayMode, setMode]);

  if (error) {
    return <p className="mt-3 text-xs text-destructive">{error}</p>;
  }
  if (!srcDoc) {
    return <p className="mt-3 text-xs text-muted-foreground">Loading app…</p>;
  }
  const fullscreen = displayMode === "fullscreen";
  return (
    <div
      className={
        fullscreen
          ? "fixed inset-0 z-[70] flex flex-col bg-background p-3 sm:p-5"
          : "mt-3"
      }
    >
      <div className={fullscreen ? "mb-3 flex justify-end" : "mb-2 flex justify-end"}>
        <Button
          aria-pressed={fullscreen}
          size="sm"
          title={fullscreen ? "Back to chat" : "Full screen"}
          type="button"
          variant="outline"
          onClick={() => setMode(fullscreen ? "inline" : "fullscreen")}
        >
          {fullscreen ? "Back to chat" : "Full screen"}
        </Button>
      </div>
      <iframe
        className={
          fullscreen
            ? "block min-h-0 w-full min-w-0 flex-1 rounded-lg border border-border bg-card"
            : "block w-full max-w-full min-w-0 rounded-lg border border-border bg-card"
        }
        ref={frameRef}
        sandbox="allow-scripts allow-forms"
        srcDoc={srcDoc}
        style={fullscreen ? undefined : { height }}
        title="Interactive app"
      />
    </div>
  );
}
