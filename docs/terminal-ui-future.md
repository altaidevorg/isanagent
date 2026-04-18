# Terminal UI (Ratatui) — follow-up work

This document tracks **larger or optional** improvements that are intentionally **out of scope** for the current terminal polish PRs. Implement when product priorities align (several items also benefit the **HTTP API** and **embedded web UI**, not only the TUI).

## High impact / cross-channel

- **Streaming assistant output** — Incremental tokens from the provider through the bus to all UIs (TUI transcript cell updates, API SSE/WebSocket, web client). Requires a clear contract for partial vs final messages and back-pressure.
- **Plain-text copy variant** — Optional `/copy plain` or modifier to copy **rendered or stripped** text instead of raw markdown.
- **Bracketed paste & long paste UX** — Safe handling of multiline / large pastes in the compose line; optional external `$EDITOR` for long input (heavy).

## Transcript & navigation

- **In-transcript search** — e.g. `Ctrl+F` or `/find` with highlight and next/prev match over flattened transcript text.
- **Explicit jump keys** — “Jump to newest” / “oldest” beyond scroll + follow-tail (e.g. `G` / `g` vi-style, if desired).
- **True horizontal pan** — Today, **horizontal mouse wheel** maps to **vertical** scroll only. A separate horizontal offset would require layout changes for wrapped lines.

## Clipboard & remote

- **OSC 52** — Clipboard over SSH for terminals that support it (complements `arboard`).
- **Linux Wayland/X11** — If `arboard` with `default-features = false` is insufficient on some desktops, document or add optional `wl-clipboard` / X11 integration.

## Polish & accessibility

- **Stricter `NO_COLOR`** — Optionally strip bold/italic as well as color for maximum plain output.
- **Light / high-contrast themes** — Detect `COLORFGBG` / `TERM` / user preference for palettes other than the current dark default.
- **Toast for other actions** — Reuse the status-strip toast for non-transcript feedback (e.g. “session reset”) where a full system cell is noisy.

## Testing

- **Integration tests for the TUI** — Hard to automate; consider snapshot tests for `build_status_line` / `outbound_to_cell` with more fixtures, or a headless smoke script.

When picking up an item, prefer a **short design note** in this file (one paragraph) before large refactors so API and TUI stay aligned.
