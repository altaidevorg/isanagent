import { useState } from "react";

type Block =
  | { type: "heading"; level: number; text: string }
  | { type: "paragraph"; text: string }
  | { type: "list"; items: string[] }
  | { type: "code"; lang: string; code: string };

function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i] ?? "";
    const fence = /^```(.*)$/.exec(line);
    if (fence) {
      const lang = fence[1]?.trim() ?? "";
      const code: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i]?.startsWith("```")) {
        code.push(lines[i] ?? "");
        i += 1;
      }
      if (i < lines.length) {
        i += 1;
      }
      blocks.push({ type: "code", lang, code: code.join("\n") });
      continue;
    }
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      blocks.push({
        type: "heading",
        level: heading[1]?.length ?? 1,
        text: heading[2] ?? "",
      });
      i += 1;
      continue;
    }
    if (/^\s*(?:[-*]|\d+\.)\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*(?:[-*]|\d+\.)\s+/.test(lines[i] ?? "")) {
        items.push((lines[i] ?? "").replace(/^\s*(?:[-*]|\d+\.)\s+/, ""));
        i += 1;
      }
      blocks.push({ type: "list", items });
      continue;
    }
    if (line.trim() === "") {
      i += 1;
      continue;
    }
    const para = [line];
    i += 1;
    while (i < lines.length) {
      const next = lines[i] ?? "";
      if (
        next.trim() === "" ||
        next.startsWith("```") ||
        /^(#{1,6})\s+/.test(next) ||
        /^\s*(?:[-*]|\d+\.)\s+/.test(next)
      ) {
        break;
      }
      para.push(next);
      i += 1;
    }
    blocks.push({ type: "paragraph", text: para.join("\n") });
  }
  return blocks;
}

function Inline({ text }: { text: string }) {
  const parts: { key: string; node: string; kind: "text" | "code" | "strong" }[] = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let n = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) {
      parts.push({ key: `t${n}`, node: text.slice(last, match.index), kind: "text" });
      n += 1;
    }
    const token = match[0];
    if (token.startsWith("`")) {
      parts.push({ key: `c${n}`, node: token.slice(1, -1), kind: "code" });
    } else {
      parts.push({ key: `b${n}`, node: token.slice(2, -2), kind: "strong" });
    }
    n += 1;
    last = match.index + token.length;
  }
  if (last < text.length) {
    parts.push({ key: `t${n}`, node: text.slice(last), kind: "text" });
  }
  return (
    <>
      {parts.map((part) => {
        if (part.kind === "code") {
          return (
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]" key={part.key}>
              {part.node}
            </code>
          );
        }
        if (part.kind === "strong") {
          return (
            <strong className="font-semibold" key={part.key}>
              {part.node}
            </strong>
          );
        }
        return <span key={part.key}>{part.node}</span>;
      })}
    </>
  );
}

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  };
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-muted/40">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-1.5">
        <span className="font-mono text-[10px] uppercase tracking-[0.08em] text-muted-foreground">
          {lang || "code"}
        </span>
        <button
          className="text-[11px] font-medium text-muted-foreground hover:text-foreground"
          type="button"
          onClick={() => void copy()}
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 font-mono text-[13px] leading-6">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function MarkdownMessage({ text }: { text: string }) {
  const blocks = parseMarkdown(text);
  if (blocks.length === 0) {
    return null;
  }
  return (
    <div className="mt-2 space-y-2 text-sm leading-7">
      {blocks.map((block, index) => {
        if (block.type === "code") {
          return <CodeBlock code={block.code} key={index} lang={block.lang} />;
        }
        if (block.type === "heading") {
          const Tag = block.level <= 2 ? "h3" : "h4";
          return (
            <Tag className="font-semibold tracking-[-0.02em] text-foreground" key={index}>
              <Inline text={block.text} />
            </Tag>
          );
        }
        if (block.type === "list") {
          return (
            <ul className="list-disc space-y-1 pl-5" key={index}>
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>
                  <Inline text={item} />
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p className="whitespace-pre-wrap" key={index}>
            <Inline text={block.text} />
          </p>
        );
      })}
    </div>
  );
}
