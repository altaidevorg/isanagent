import { memo, useEffect, useMemo, useRef, useState, type ComponentPropsWithoutRef, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import diff from "highlight.js/lib/languages/diff";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import python from "highlight.js/lib/languages/python";
import rust from "highlight.js/lib/languages/rust";
import typescript from "highlight.js/lib/languages/typescript";
import ini from "highlight.js/lib/languages/ini";
import yaml from "highlight.js/lib/languages/yaml";

import { cn } from "@/lib/utils";

hljs.registerLanguage("bash", bash);
hljs.registerLanguage("diff", diff);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("json", json);
hljs.registerLanguage("python", python);
hljs.registerLanguage("rust", rust);
hljs.registerLanguage("typescript", typescript);
// highlight.js has no separate TOML grammar; `ini` covers it (and registers the `toml` alias).
hljs.registerLanguage("ini", ini);
hljs.registerLanguage("yaml", yaml);
hljs.registerAliases(["sh", "shell", "zsh", "console"], { languageName: "bash" });
hljs.registerAliases(["py"], { languageName: "python" });
hljs.registerAliases(["rs"], { languageName: "rust" });
hljs.registerAliases(["ts", "tsx"], { languageName: "typescript" });
hljs.registerAliases(["js", "jsx", "mjs", "cjs"], { languageName: "javascript" });
hljs.registerAliases(["yml"], { languageName: "yaml" });
hljs.registerAliases(["patch"], { languageName: "diff" });

function textOf(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (typeof node === "object" && "props" in node) {
    return textOf((node as { props: { children?: ReactNode } }).props.children);
  }
  return "";
}

async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the legacy path (insecure origin, denied permission) */
  }
  try {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  } catch {
    return false;
  }
}

function CopyButton({ text }: { text: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return (
    <button
      aria-label={state === "copied" ? "Copied" : "Copy code"}
      className="rounded-md px-2 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
      type="button"
      onClick={async () => {
        const ok = await copyText(text);
        setState(ok ? "copied" : "failed");
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setState("idle"), 1500);
      }}
    >
      {state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : "Copy"}
    </button>
  );
}

const CodeBlock = memo(function CodeBlock({ code, language }: { code: string; language: string | null }) {
  const known = language !== null && hljs.getLanguage(language) !== undefined;
  const html = useMemo(() => {
    if (!known || !language) return null;
    try {
      return hljs.highlight(code, { language, ignoreIllegals: true }).value;
    } catch {
      return null;
    }
  }, [code, language, known]);
  return (
    <div className="md-code group/code my-3 overflow-hidden rounded-lg border border-border bg-muted/40">
      <div className="flex items-center justify-between border-b border-border/70 px-3 py-1">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          {language || "text"}
        </span>
        <CopyButton text={code} />
      </div>
      <pre className="overflow-x-auto px-3 py-2.5 text-[12.5px] leading-relaxed">
        {html !== null ? (
          <code className="hljs font-mono" dangerouslySetInnerHTML={{ __html: html }} />
        ) : (
          <code className="font-mono">{code}</code>
        )}
      </pre>
    </div>
  );
});

const components: Components = {
  pre({ children }) {
    // `pre > code.language-x` for fenced blocks; take the language and raw text from the code child.
    const child = Array.isArray(children) ? children[0] : children;
    const props = (child as { props?: { className?: string; children?: ReactNode } } | null)?.props;
    const match = /language-([\w+#.-]+)/.exec(props?.className ?? "");
    const code = textOf(props?.children ?? children).replace(/\n$/, "");
    return <CodeBlock code={code} language={match ? match[1].toLowerCase() : null} />;
  },
  code({ className, children, ...rest }: ComponentPropsWithoutRef<"code">) {
    return (
      <code
        className={cn("rounded-[5px] border border-border/70 bg-muted px-1 py-[1px] font-mono text-[0.85em]", className)}
        {...rest}
      >
        {children}
      </code>
    );
  },
  a({ children, href, ...rest }) {
    return (
      <a
        {...rest}
        className="font-medium text-[hsl(var(--ring))] underline underline-offset-2 hover:opacity-80"
        href={href}
        rel="noopener noreferrer"
        target="_blank"
      >
        {children}
      </a>
    );
  },
  table({ children }) {
    return (
      <div className="my-3 overflow-x-auto rounded-lg border border-border">
        <table className="w-full border-collapse text-left text-[13px]">{children}</table>
      </div>
    );
  },
  th({ children, style }) {
    return (
      <th className="border-b border-border bg-muted/60 px-3 py-1.5 font-semibold" style={style}>
        {children}
      </th>
    );
  },
  td({ children, style }) {
    return (
      <td className="border-b border-border/60 px-3 py-1.5 align-top" style={style}>
        {children}
      </td>
    );
  },
};

const remarkPlugins = [remarkGfm];

/** GFM markdown for assistant text. Memoized on `content`, so finished messages never re-parse. */
export const Markdown = memo(function Markdown({ content, className }: { content: string; className?: string }) {
  return (
    <div className={cn("md-body text-sm leading-7", className)}>
      <ReactMarkdown components={components} remarkPlugins={remarkPlugins}>
        {content}
      </ReactMarkdown>
    </div>
  );
});
