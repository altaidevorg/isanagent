import { lazy, memo, Suspense } from "react";

import { cn } from "@/lib/utils";

const loadMarkdown = () => import("@/components/Markdown");
const LazyMarkdown = lazy(() => loadMarkdown().then((m) => ({ default: m.Markdown })));

/** Start fetching the markdown renderer (its own chunk) before the first reply needs it. */
export function preloadMarkdown() {
  void loadMarkdown();
}

/**
 * Assistant markdown, rendered by a lazily loaded chunk (react-markdown, GFM, highlight.js) so the
 * first paint does not wait for it. Plain text stands in until the chunk arrives.
 */
export const MarkdownText = memo(function MarkdownText({ content, className }: { content: string; className?: string }) {
  return (
    <Suspense
      fallback={<p className={cn("whitespace-pre-wrap break-words text-sm leading-7", className)}>{content}</p>}
    >
      <LazyMarkdown className={className} content={content} />
    </Suspense>
  );
});
