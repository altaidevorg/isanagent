//! Compaction helpers used by the in-loop auto-compaction path in [`super::AgentLogic`].
//!
//! - **PR-1** (pre-summarization stripping): [`preprocess_transcript_for_compaction`].
//! - **PR-2** (sectional summary template): [`SummarySections`], [`build_sectional_prompt`].
//!
//! Subsequent PRs (tool-result swap, etc.) will extend this module.

use crate::utils::{ChatMessage, ContentPart, MessageContent};
use serde::{Deserialize, Serialize};

/// Default token cap for tool-role messages during compaction preprocessing.
/// Matches the PR-1 plan default of 10_000 tokens.
///
/// **Wiring status.** Currently hardcoded at the call site in
/// `AgentLogic::run_reasoning_loop`; not yet configurable via `MemoryConfig`.
/// Tracked as the PR-1.1 follow-up — plumbing through `AgentLogicParams` +
/// `ReasoningSpawnArgs` + `ReasoningLoopCtx` is disproportionate for two
/// scalar knobs, deferred until tuning is actually requested.
pub const PREPROCESS_TOOL_RESULT_MAX_TOKENS_DEFAULT: usize = 10_000;

/// Default for image-stripping during compaction preprocessing. Same wiring caveat
/// as [`PREPROCESS_TOOL_RESULT_MAX_TOKENS_DEFAULT`].
pub const PREPROCESS_STRIP_IMAGES_DEFAULT: bool = true;

/// PR-3: fraction of the model's context window at which to trigger auto-compaction.
/// Used by [`effective_compaction_threshold`]. Hardcoded — `MemoryConfig` plumbing
/// for caller override is tracked as a PR-3.1 follow-up (same pattern as PR-1.1).
pub const TRIGGER_AT_PERCENTAGE_DEFAULT: f32 = 0.85;

/// PR-3: tokens of headroom left for the response and immediate tool outputs after
/// compaction. Trigger fires no later than `window - reserve`.
pub const RESERVE_TOKENS_DEFAULT: usize = 16_384;

/// PR-3: compute the effective token threshold for auto-compaction.
///
/// Returns the smallest of:
/// - `absolute` — the user-configured `short_term_threshold_tokens` floor.
/// - `(window * percentage) as usize` — percentage of the model's context
///   window (when the window is known). With `percentage=0.85`, Sonnet (200k)
///   compacts at ~170k and Opus (1M) would compact at ~850k from a single
///   configured percentage. The previous hardcoded 100k tripped at very
///   different fractions of those windows.
/// - `window.saturating_sub(reserve)` — leave `reserve` tokens of headroom for
///   the assistant response and immediate tool output that follow.
///
/// When `window` is `None`, returns `absolute` unchanged — provider doesn't
/// know its own context size, so we can't safely use a relative threshold.
///
/// **Degenerate-config defense.** When the caller misconfigures (`reserve >= window`,
/// `percentage <= 0`, or `percentage > 1`), naive computation produces a 0-or-near-0
/// threshold; `approx_tokens >= 0` is always true, so compaction would fire every
/// turn — an infinite-loop trap. In those cases the function falls back to
/// `absolute`. A final `.max(1)` guarantees the return is never literally zero.
pub fn effective_compaction_threshold(
    absolute: usize,
    window: Option<usize>,
    percentage: f32,
    reserve: usize,
) -> usize {
    let Some(w) = window else {
        return absolute.max(1);
    };
    if reserve >= w || !(0.0 < percentage && percentage <= 1.0) {
        return absolute.max(1);
    }
    let pct_bound = ((w as f32) * percentage) as usize;
    let reserve_bound = w - reserve; // safe: reserve < w by the guard above
    absolute.min(pct_bound).min(reserve_bound).max(1)
}

/// Rough token estimate per byte. Same heuristic used by the existing
/// auto-compaction threshold check (`char_count / 4`).
const BYTES_PER_TOKEN_HEURISTIC: usize = 4;

/// Build a transcript for compaction summarization, applying preprocessing rules.
///
/// - When `strip_images` is true, image content parts are dropped entirely;
///   when false, they are rendered as the placeholder `[image]` so the summarizer
///   at least knows one was present.
/// - Tool-role messages (`role == "tool"`) whose rendered text exceeds
///   `tool_result_max_tokens` (in token units) are truncated at the byte
///   boundary equivalent of that token count, with a `…[truncated]` suffix.
///   Truncation uses [`crate::utils::truncate_utf8_safe`] so it never splits
///   a multi-byte codepoint.
/// - System messages are skipped (mirrors the pre-PR-1 transcript loop).
///
/// Returns `(transcript, approx_tokens_after_preprocess)`. The token estimate
/// uses the same byte/4 heuristic the trigger check uses, so it can be compared
/// directly to `tokens_before` in [`crate::bus::TelemetryEvent::CompactionTriggered`].
pub fn preprocess_transcript_for_compaction(
    context: &[ChatMessage],
    strip_images: bool,
    tool_result_max_tokens: usize,
) -> (String, usize) {
    let mut transcript = String::new();
    for msg in context {
        if msg.role == "system" {
            continue;
        }
        if let Some(content) = &msg.content {
            let mut rendered = render_content(content, strip_images);
            if msg.role == "tool" {
                let max_bytes = tool_result_max_tokens.saturating_mul(BYTES_PER_TOKEN_HEURISTIC);
                if rendered.len() > max_bytes {
                    crate::utils::truncate_utf8_safe(&mut rendered, max_bytes, "…[truncated]");
                }
            }
            transcript.push_str(&format!("{}: {}\n\n", msg.role, rendered));
        } else if msg.tool_calls.is_some() {
            transcript.push_str(&format!("{}: [Invoked Tools]\n\n", msg.role));
        }
    }
    let approx_tokens = transcript.len() / BYTES_PER_TOKEN_HEURISTIC;
    (transcript, approx_tokens)
}

// === PR-2: sectional summary template ===

/// LLM-facing prompt asking for the 8-slot sectional summary as JSON.
///
/// Slot list matches the v2 plan PR-2 schema. Each slot is non-optional in the
/// LLM's output (the model is told to use `null` for unknown strings and `[]`
/// for empty arrays), so [`SummarySections::completeness`] computes a faithful
/// 0..1 score over which slots the model actually populated.
pub const SECTIONAL_PROMPT: &str = "\
You are compacting a long conversation transcript. Produce a structured summary \
as a single JSON object with EXACTLY these 8 fields — every field must be present, \
use `null` for unknown strings and `[]` for empty arrays:

  \"task_overview\": string | null,   // 1-2 sentences naming the user's high-level goal
  \"current_state\": string | null,   // where the work stands at the most recent turn
  \"files_touched\": string[],         // file paths the agent read, wrote, or edited
  \"key_decisions\": string[],         // architectural or design choices the agent made
  \"discoveries\": string[],           // facts learned from tool calls (bugs found, APIs confirmed)
  \"next_steps\": string[],            // pending or planned work items
  \"open_questions\": string[],        // unresolved questions for future turns
  \"external_refs\": string[]          // URLs, docs, or external resources referenced

Respond with the JSON object only — no prose, no markdown code fence.";

/// Parsed 8-slot summary, the structured form behind compaction. Stored verbatim
/// in `session_summaries.sections_json` (see `MemoryMessage::WriteSectionsJson`)
/// and rendered to Markdown for the legacy `summary` column.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SummarySections {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_overview: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_state: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files_touched: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub key_decisions: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub discoveries: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub next_steps: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_questions: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub external_refs: Vec<String>,
}

impl SummarySections {
    /// Parse the LLM's JSON response into structured slots. Missing keys default
    /// to `None` / `vec![]` — completeness reflects what actually arrived.
    /// Non-string array entries are dropped silently rather than rejected.
    pub fn from_json(value: &serde_json::Value) -> Self {
        let get_str = |k: &str| -> Option<String> {
            value
                .get(k)
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        let get_arr = |k: &str| -> Vec<String> {
            value
                .get(k)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default()
        };
        Self {
            task_overview: get_str("task_overview"),
            current_state: get_str("current_state"),
            files_touched: get_arr("files_touched"),
            key_decisions: get_arr("key_decisions"),
            discoveries: get_arr("discoveries"),
            next_steps: get_arr("next_steps"),
            open_questions: get_arr("open_questions"),
            external_refs: get_arr("external_refs"),
        }
    }

    /// Fraction of the 8 slots that contain at least one non-empty entry.
    /// Drives `TelemetryEvent::CompactionCompleted.section_completeness`.
    pub fn completeness(&self) -> f32 {
        let mut filled = 0u32;
        if self.task_overview.is_some() {
            filled += 1;
        }
        if self.current_state.is_some() {
            filled += 1;
        }
        if !self.files_touched.is_empty() {
            filled += 1;
        }
        if !self.key_decisions.is_empty() {
            filled += 1;
        }
        if !self.discoveries.is_empty() {
            filled += 1;
        }
        if !self.next_steps.is_empty() {
            filled += 1;
        }
        if !self.open_questions.is_empty() {
            filled += 1;
        }
        if !self.external_refs.is_empty() {
            filled += 1;
        }
        f32::from(filled as u16) / 8.0
    }

    /// Render to the Markdown form stored in the legacy `summary` column.
    /// Empty sections are omitted so the rendered text stays compact.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        if let Some(t) = &self.task_overview {
            out.push_str("## Task overview\n");
            out.push_str(t);
            out.push_str("\n\n");
        }
        if let Some(s) = &self.current_state {
            out.push_str("## Current state\n");
            out.push_str(s);
            out.push_str("\n\n");
        }
        fn list_section(out: &mut String, heading: &str, items: &[String]) {
            if items.is_empty() {
                return;
            }
            out.push_str("## ");
            out.push_str(heading);
            out.push('\n');
            for item in items {
                out.push_str("- ");
                out.push_str(item);
                out.push('\n');
            }
            out.push('\n');
        }
        list_section(&mut out, "Files touched", &self.files_touched);
        list_section(&mut out, "Key decisions", &self.key_decisions);
        list_section(&mut out, "Discoveries", &self.discoveries);
        list_section(&mut out, "Next steps", &self.next_steps);
        list_section(&mut out, "Open questions", &self.open_questions);
        list_section(&mut out, "External refs", &self.external_refs);
        out.trim_end().to_string()
    }
}

/// Build the full prompt sent to the summarizer LLM. Optionally injects a
/// previously generated summary so the model can revise rather than restart,
/// and an optional caller-supplied `FOCUS:` block (PR-5) that biases which
/// content the model keeps versus drops.
pub fn build_sectional_prompt(
    existing_summary: Option<&str>,
    transcript: &str,
    focus_instructions: Option<&str>,
) -> String {
    let mut prompt = String::from(SECTIONAL_PROMPT);
    if let Some(prev) = existing_summary {
        let trimmed = prev.trim();
        if !trimmed.is_empty() {
            prompt.push_str("\n\nEXISTING SUMMARY TO UPDATE:\n");
            prompt.push_str(trimmed);
        }
    }
    if let Some(focus) = focus_instructions {
        let trimmed = focus.trim();
        if !trimmed.is_empty() {
            prompt.push_str("\n\nFOCUS:\n");
            prompt.push_str(trimmed);
        }
    }
    prompt.push_str("\n\nNEW TRANSCRIPT:\n");
    prompt.push_str(transcript);
    prompt
}

// === PR-7: tool-result cache helpers ===

/// Maximum byte length of the head excerpt embedded in a tool result's compact
/// summary. Long enough to give the model a hint of what was cached without
/// re-bloating the transcript.
const COMPACT_SUMMARY_HEAD_BYTES: usize = 80;

/// Build the compact placeholder text that replaces a tool result's content
/// during a compaction swap. The text includes the `tool_call_id` so the LLM
/// can pass it to `recall_tool_result` to retrieve the original.
///
/// Format: `[Tool result archived. Recall: recall_tool_result(tool_call_id="…"). Original: tool=… bytes=… head="…"]`.
pub fn build_compact_placeholder(
    tool_call_id: &str,
    tool_name: &str,
    full_content: &str,
) -> String {
    let bytes = full_content.len();
    // UTF-8-safe head excerpt. Single-line so the placeholder stays on one line
    // in the conversation log.
    let mut head = String::new();
    for c in full_content.chars() {
        if head.len() + c.len_utf8() > COMPACT_SUMMARY_HEAD_BYTES {
            break;
        }
        if c == '\n' || c == '\r' {
            head.push(' ');
        } else {
            head.push(c);
        }
    }
    if head.len() < bytes {
        head.push('…');
    }
    format!(
        "[Tool result archived. Recall: recall_tool_result(tool_call_id=\"{}\"). Original: tool={} bytes={} head=\"{}\"]",
        tool_call_id, tool_name, bytes, head
    )
}

/// PR-7.2: number of most-recent user turns whose tool results stay in full
/// form. Tool results older than this many user turns from the latest become
/// candidates for the per-iteration staleness swap.
pub const KEEP_RECENT_USER_TURNS_DEFAULT: usize = 3;

/// PR-7.2: identify tool messages that are *stale* — older than
/// `keep_recent_user_turns` user turns from the latest — and produce the
/// payloads needed to swap them. Walks `messages_with_ids` from newest to
/// oldest, counts user turns, and once the count exceeds the threshold all
/// subsequent tool messages with a `tool_call_id` are eligible. Idempotent:
/// already-swapped messages (placeholder prefix detected) are skipped.
///
/// Returns `(db_id, tool_call_id, tool_name, full_content, placeholder)` per
/// eligible swap, so the caller can fire `CacheToolResult` + `UpdateMessageContent`
/// in pairs.
pub fn identify_stale_tool_swaps(
    messages_with_ids: &[(i64, ChatMessage)],
    keep_recent_user_turns: usize,
) -> Vec<(i64, String, String, String, String)> {
    let mut user_turns_seen = 0usize;
    let mut results: Vec<(i64, String, String, String, String)> = Vec::new();
    for (db_id, msg) in messages_with_ids.iter().rev() {
        if msg.role == "user" {
            user_turns_seen += 1;
            continue;
        }
        // A tool we encounter at user_turns_seen=K belongs to the user turn
        // `K` slots back from the latest (0 = latest). Keep tools whose turn
        // is within the last `keep_recent_user_turns` — that's the range
        // `user_turns_seen ∈ [0, keep_recent_user_turns - 1]`. Strict `<`.
        if user_turns_seen < keep_recent_user_turns {
            continue;
        }
        // Same swap predicate as the transient + at-compaction paths
        // (`swap_all_tool_results_in_place` and `do_compaction`).
        let Some(payload) = try_build_tool_swap(msg) else {
            continue;
        };
        results.push((
            *db_id,
            payload.tool_call_id,
            payload.tool_name,
            payload.full_content,
            payload.placeholder,
        ));
    }
    results
}

/// PR-7: parsed payload for a single tool-result swap. Carries everything the
/// caller needs to (a) write the original to `tool_result_cache` via
/// `MemoryMessage::CacheToolResult` and (b) overwrite the message content
/// (in memory, in the DB, or both — caller's choice).
#[derive(Debug)]
pub struct ToolSwapPayload {
    pub tool_call_id: String,
    pub tool_name: String,
    pub full_content: String,
    pub placeholder: String,
}

/// PR-7: classify a single chat message — is it a swappable tool result?
/// Returns `Some(payload)` when the message is a tool result with a
/// `tool_call_id` whose content is not already a placeholder; `None` otherwise.
///
/// Single source of truth for the swap predicate. All three swap sites
/// (`swap_all_tool_results_in_place` at the transient summarizer step,
/// `identify_stale_tool_swaps` at the per-iteration staleness check, and
/// `do_compaction`'s in-line ID-bearing swap) call this to decide eligibility.
pub fn try_build_tool_swap(msg: &ChatMessage) -> Option<ToolSwapPayload> {
    if msg.role != "tool" {
        return None;
    }
    let tool_call_id = msg.tool_call_id.clone()?;
    let tool_name = msg.name.clone().unwrap_or_else(|| "unknown".to_string());
    let full_content = msg.content.as_ref()?.text_content();
    if full_content.starts_with("[Tool result archived.") {
        return None;
    }
    let placeholder = build_compact_placeholder(&tool_call_id, &tool_name, &full_content);
    Some(ToolSwapPayload {
        tool_call_id,
        tool_name,
        full_content,
        placeholder,
    })
}

/// PR-7: walk a transcript and replace **every** swappable tool-role message
/// with its compact placeholder, in place. Only messages whose `tool_call_id`
/// is `Some(…)` are swapped (orphans can't be recalled, so swapping them would
/// be lossy). Caller controls "staleness" — by definition this swaps anything
/// it can swap. For position-based staleness, use [`identify_stale_tool_swaps`].
///
/// Returns the count of messages swapped and `(tool_call_id, full_content, tool_name)`
/// triples for the caller to forward to `MemoryMessage::CacheToolResult`.
pub fn swap_all_tool_results_in_place(
    context: &mut [ChatMessage],
) -> (usize, Vec<(String, String, String)>) {
    let mut to_cache: Vec<(String, String, String)> = Vec::new();
    let mut swapped: usize = 0;
    for msg in context.iter_mut() {
        let Some(payload) = try_build_tool_swap(msg) else {
            continue;
        };
        let ToolSwapPayload {
            tool_call_id,
            tool_name,
            full_content,
            placeholder,
        } = payload;
        to_cache.push((tool_call_id, full_content, tool_name));
        msg.content = Some(crate::utils::MessageContent::Text(placeholder));
        swapped += 1;
    }
    (swapped, to_cache)
}

// === PR-4.1: shared compaction runner ===

/// Outcome of [`do_compaction`]. The matched `CompactionFailed` / `CompactionCompleted`
/// telemetry event is emitted internally before this returns, so callers don't have to.
#[derive(Debug)]
#[non_exhaustive]
pub enum CompactionOutcome {
    /// Summary persisted, reflection cursor advanced, `CompactionCompleted` emitted.
    Succeeded,
    /// LLM call returned an error, or response was unparseable JSON.
    /// `CompactionFailed` already emitted.
    Failed,
    /// Cancellation token fired during the summarizer call.
    /// `CompactionFailed { reason: "cancelled" }` already emitted. The caller is
    /// expected to invoke its own cancel handler (e.g. the `persist_and_cancel!`
    /// macro inside the reasoning loop).
    Cancelled,
}

/// Inputs for [`do_compaction`], bundled so the call site doesn't expand to a
/// 12-arg tuple. All references are borrowed for the duration of the call.
pub struct DoCompactionArgs<'a> {
    pub chat_id: &'a str,
    pub session_key: &'a str,
    pub trigger_reason: crate::bus::CompactionTrigger,
    pub tokens_before: u32,
    pub turns_before: u32,
    pub current_context: &'a [ChatMessage],
    /// Most recent existing summary text (Markdown), if any — fed back into the
    /// sectional prompt so the model can revise rather than restart.
    pub existing_summary: Option<&'a str>,
    /// PR-5: caller-supplied focus block appended to the summarizer prompt as
    /// `FOCUS: …`. `None` for auto-triggered compactions.
    pub focus_instructions: Option<&'a str>,
    pub provider: &'a dyn crate::traits::Provider,
    pub memory_node: &'a crate::NodeHandle<crate::memory::MemoryMessage>,
    pub outbound_tx: &'a tokio::sync::mpsc::Sender<crate::bus::BusMessage>,
    pub cancel_token: &'a tokio_util::sync::CancellationToken,
}

/// Run one compaction cycle: preprocess → summarize → persist (legacy 3-slot row
/// via `AddSummary` + structured JSON via `WriteSectionsJson` + reflection cursor
/// advance via `UpdateThreadMetadata`). Emits the full matched telemetry pair
/// (`CompactionTriggered` + one of `CompactionCompleted`/`CompactionFailed`).
///
/// Used by both the in-loop auto-compaction threshold trigger and the PR-4.1
/// emergency-recovery path that fires on `LLMError::ContextOverflow`.
pub async fn do_compaction(args: DoCompactionArgs<'_>) -> CompactionOutcome {
    use crate::bus::{BusMessage, TelemetryEvent};
    use crate::memory::{MemoryMessage, SharedReply};

    let started = std::time::Instant::now();

    // PR-7 / PR-7.1: tool-result swap. We fetch the messages-since-reflection
    // *with their DB ids* so we can persist the swap to the `messages` table
    // (subsequent iterations see the compact form). Originals go to the
    // `tool_result_cache` table; `recall_tool_result` recovers them on demand.
    //
    // If the ID-bearing fetch fails for any reason, we fall back to swapping
    // a clone of the caller's `current_context` — that preserves the PR-7 v0
    // transient-only behavior (summarizer sees the smaller input, but the DB
    // is untouched) rather than failing the whole compaction.
    let messages_with_ids: Vec<(i64, ChatMessage)> = {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = args
            .memory_node
            .send_packet(crate::memory::MemoryMessage::GetMessagesSinceReflection {
                thread_id: args.session_key.to_string(),
                reply: crate::memory::SharedReply::new(tx),
            })
            .await;
        rx.await
            .ok()
            .and_then(|r| r.ok())
            .map(|(rows, _)| rows)
            .unwrap_or_default()
    };

    // Capture the last message id now, before `messages_with_ids` is moved
    // into the swap below. No new messages are appended to the thread during
    // compaction, so this is the same id a fresh query would return at the
    // end — reused there to advance the reflection cursor without re-querying.
    let last_msg_id = messages_with_ids.last().map(|(id, _)| *id);

    let mut swapped_context: Vec<ChatMessage>;
    let mut cache_entries: Vec<(String, String, String)>;
    let mut id_updates: Vec<(i64, String)> = Vec::new();

    if messages_with_ids.is_empty() {
        // Fallback path — no IDs available, transient swap only.
        swapped_context = args.current_context.to_vec();
        let (_, ce) = swap_all_tool_results_in_place(&mut swapped_context);
        cache_entries = ce;
    } else {
        // Preferred path — swap on the ID-bearing tuples, harvest both the
        // cache entries and the per-row UPDATE payloads in one pass. Uses
        // the same `try_build_tool_swap` predicate as the transient and
        // staleness paths so behavior stays consistent across call sites.
        let mut working: Vec<(i64, ChatMessage)> = messages_with_ids;
        cache_entries = Vec::new();
        for (db_id, msg) in working.iter_mut() {
            let Some(payload) = try_build_tool_swap(msg) else {
                continue;
            };
            let ToolSwapPayload {
                tool_call_id,
                tool_name,
                full_content,
                placeholder,
            } = payload;
            cache_entries.push((tool_call_id, full_content, tool_name));
            id_updates.push((*db_id, placeholder.clone()));
            msg.content = Some(crate::utils::MessageContent::Text(placeholder));
        }
        swapped_context = working.into_iter().map(|(_, m)| m).collect();
    }

    // Cache writes — originals are recoverable via `recall_tool_result`.
    for (tool_call_id, full_content, tool_name) in &cache_entries {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let compact = build_compact_placeholder(tool_call_id, tool_name, full_content);
        let _ = args
            .memory_node
            .send_packet(crate::memory::MemoryMessage::CacheToolResult {
                tool_call_id: tool_call_id.clone(),
                chat_id: args.chat_id.to_string(),
                session_key: args.session_key.to_string(),
                tool_name: tool_name.clone(),
                full_content: full_content.clone(),
                compact_summary: compact,
                reply: crate::memory::SharedReply::new(tx),
            })
            .await;
        let _ = rx.await;
    }

    // PR-7.1: persist the swap into the `messages` table. UPDATE statements
    // that miss (thread cleared mid-compaction) are no-ops — the cache write
    // above already preserved the original content for recall.
    for (message_id, new_content) in &id_updates {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = args
            .memory_node
            .send_packet(crate::memory::MemoryMessage::UpdateMessageContent {
                message_id: *message_id,
                new_content: new_content.clone(),
                reply: crate::memory::SharedReply::new(tx),
            })
            .await;
        let _ = rx.await;
    }

    // 1. PR-1: preprocess the transcript.
    let (transcript, tokens_after_preprocess) = preprocess_transcript_for_compaction(
        &swapped_context,
        PREPROCESS_STRIP_IMAGES_DEFAULT,
        PREPROCESS_TOOL_RESULT_MAX_TOKENS_DEFAULT,
    );
    let tokens_after_preprocess_u32 = tokens_after_preprocess.min(u32::MAX as usize) as u32;

    // 2. Emit CompactionTriggered (matched by Completed/Failed below).
    let _ = args
        .outbound_tx
        .send(BusMessage::Telemetry(TelemetryEvent::CompactionTriggered {
            chat_id: args.chat_id.to_string(),
            reason: args.trigger_reason.clone(),
            tokens_before: args.tokens_before,
            turns_before: args.turns_before,
            tokens_after_preprocess: tokens_after_preprocess_u32,
        }))
        .await;

    // 3. PR-2: sectional prompt (with PR-5 focus block if provided).
    let prompt =
        build_sectional_prompt(args.existing_summary, &transcript, args.focus_instructions);
    let summary_context = vec![
        ChatMessage::system(
            "You are a helpful assistant that summarizes conversations into structured JSON.",
        ),
        ChatMessage::user(&prompt),
    ];

    // 4. Provider call with cancel select.
    let response = tokio::select! {
        res = args.provider.chat(&summary_context, None) => res,
        _ = args.cancel_token.cancelled() => {
            let _ = args.outbound_tx
                .send(BusMessage::Telemetry(TelemetryEvent::CompactionFailed {
                    chat_id: args.chat_id.to_string(),
                    reason: "cancelled".to_string(),
                    tokens_at_failure: args.tokens_before,
                }))
                .await;
            return CompactionOutcome::Cancelled;
        }
    };

    let resp = match response {
        Ok(r) => r,
        Err(e) => {
            let _ = args
                .outbound_tx
                .send(BusMessage::Telemetry(TelemetryEvent::CompactionFailed {
                    chat_id: args.chat_id.to_string(),
                    reason: format!("provider error: {}", e),
                    tokens_at_failure: args.tokens_before,
                }))
                .await;
            return CompactionOutcome::Failed;
        }
    };

    let parsed = match crate::utils::extract_json_from_llm_response(&resp.content) {
        Some(v) => v,
        None => {
            let _ = args
                .outbound_tx
                .send(BusMessage::Telemetry(TelemetryEvent::CompactionFailed {
                    chat_id: args.chat_id.to_string(),
                    reason: "summary response was not parseable JSON".to_string(),
                    tokens_at_failure: args.tokens_before,
                }))
                .await;
            return CompactionOutcome::Failed;
        }
    };

    // 5. Parse + persist.
    let sections = SummarySections::from_json(&parsed);
    let summary_md = sections.to_markdown();
    let section_completeness = sections.completeness();
    let summary_bytes = summary_md.len().min(u32::MAX as usize) as u32;
    let sections_json = serde_json::to_string(&sections).unwrap_or_else(|_| "{}".to_string());

    // AddSummary (legacy row).
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = args
        .memory_node
        .send_packet(MemoryMessage::AddSummary {
            thread_id: args.session_key.to_string(),
            summary: summary_md,
            key_info: String::new(),
            knowledge_gaps: String::new(),
            reply: SharedReply::new(tx),
        })
        .await;
    let _ = rx.await;

    // WriteSectionsJson (PR-2 structured column).
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = args
        .memory_node
        .send_packet(MemoryMessage::WriteSectionsJson {
            thread_id: args.session_key.to_string(),
            sections_json,
            reply: SharedReply::new(tx),
        })
        .await;
    let _ = rx.await;

    // Advance reflection cursor — reuse the id captured before the swap
    // instead of issuing another GetMessagesSinceReflection round-trip.
    if let Some(last_id) = last_msg_id {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = args
            .memory_node
            .send_packet(MemoryMessage::UpdateThreadMetadata {
                thread_id: args.session_key.to_string(),
                last_reflection_msg_id: Some(last_id),
                reply: SharedReply::new(tx),
            })
            .await;
        let _ = rx.await;
    }

    let _ = args
        .outbound_tx
        .send(BusMessage::Telemetry(TelemetryEvent::CompactionCompleted {
            chat_id: args.chat_id.to_string(),
            tokens_before: args.tokens_before,
            tokens_after: summary_bytes / 4,
            wall_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            summary_bytes,
            section_completeness,
        }))
        .await;

    CompactionOutcome::Succeeded
}

fn render_content(content: &MessageContent, strip_images: bool) -> String {
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.clone()),
                ContentPart::ImageUrl { .. } if strip_images => None,
                ContentPart::ImageUrl { .. } => Some("[image]".to_string()),
                ContentPart::Document { document } => Some(format!(
                    "[document: {}]",
                    document.name.as_deref().unwrap_or("document")
                )),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{ChatMessage, ContentPart, ImageUrl, MessageContent};

    fn user_with_parts(parts: Vec<ContentPart>) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(parts)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            is_error: None,
        }
    }

    fn tool_message(text: &str) -> ChatMessage {
        ChatMessage {
            role: "tool".to_string(),
            content: Some(MessageContent::Text(text.to_string())),
            name: Some("some_tool".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_0".to_string()),
            reasoning_content: None,
            is_error: None,
        }
    }

    #[test]
    fn image_parts_are_stripped_when_enabled() {
        let parts = vec![
            ContentPart::Text {
                text: "look here".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,XXXX".to_string(),
                    detail: None,
                },
            },
        ];
        let ctx = vec![user_with_parts(parts)];
        let (out, _) = preprocess_transcript_for_compaction(&ctx, true, 10_000);
        assert!(out.contains("look here"));
        assert!(
            !out.contains("XXXX") && !out.contains("[image]"),
            "image must be stripped"
        );
    }

    #[test]
    fn image_parts_become_placeholder_when_disabled() {
        let parts = vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,XXXX".to_string(),
                    detail: None,
                },
            },
        ];
        let ctx = vec![user_with_parts(parts)];
        let (out, _) = preprocess_transcript_for_compaction(&ctx, false, 10_000);
        assert!(out.contains("describe"));
        assert!(
            out.contains("[image]"),
            "should keep placeholder when not stripping"
        );
    }

    #[test]
    fn tool_results_truncated_to_token_cap() {
        let huge = "x".repeat(200_000); // 200k bytes ≈ 50k tokens
        let ctx = vec![tool_message(&huge)];
        let (out, _) = preprocess_transcript_for_compaction(&ctx, true, 100); // cap = 100 tokens = 400 bytes
                                                                              // The transcript line is "tool: " + truncated + "…[truncated]" + "\n\n".
                                                                              // Truncated body must be at most 400 bytes; full line is bounded around it.
        assert!(out.contains("…[truncated]"), "must mark truncation");
        let body_len = out.len();
        assert!(
            body_len < 1_000,
            "huge tool result must be truncated; got {} bytes",
            body_len
        );
    }

    #[test]
    fn small_tool_results_pass_through_unchanged() {
        let ctx = vec![tool_message("small output")];
        let (out, _) = preprocess_transcript_for_compaction(&ctx, true, 10_000);
        assert!(out.contains("small output"));
        assert!(!out.contains("…[truncated]"));
    }

    #[test]
    fn system_messages_are_skipped() {
        let ctx = vec![
            ChatMessage::system("system prompt content"),
            ChatMessage::user("user query"),
        ];
        let (out, _) = preprocess_transcript_for_compaction(&ctx, true, 10_000);
        assert!(out.contains("user query"));
        assert!(!out.contains("system prompt content"));
    }

    // === PR-2: sectional summary tests ===

    #[test]
    fn from_json_extracts_all_slots() {
        let raw = serde_json::json!({
            "task_overview": "Investigate failing CI build.",
            "current_state": "Identified root cause; fix in progress.",
            "files_touched": ["src/foo.rs", "  src/bar.rs  "],
            "key_decisions": ["Pin reqwest 0.12"],
            "discoveries": ["TLS handshake fails on macOS only"],
            "next_steps": ["Write regression test", "Bump version"],
            "open_questions": ["Should we cache the root certs?"],
            "external_refs": ["https://github.com/seanmonstar/reqwest/issues/2024"],
        });
        let s = SummarySections::from_json(&raw);
        assert_eq!(
            s.task_overview.as_deref(),
            Some("Investigate failing CI build.")
        );
        assert_eq!(s.files_touched, vec!["src/foo.rs", "src/bar.rs"]); // trimmed
        assert_eq!(s.next_steps.len(), 2);
        assert!(s.external_refs[0].starts_with("https://"));
    }

    #[test]
    fn from_json_treats_missing_fields_as_empty() {
        let raw = serde_json::json!({
            "task_overview": "Foo",
            "files_touched": ["a"]
            // current_state missing entirely; arrays absent
        });
        let s = SummarySections::from_json(&raw);
        assert!(s.task_overview.is_some());
        assert!(s.current_state.is_none());
        assert!(s.files_touched.len() == 1);
        assert!(s.key_decisions.is_empty());
        assert!(s.discoveries.is_empty());
    }

    #[test]
    fn from_json_drops_empty_strings_and_non_string_array_entries() {
        let raw = serde_json::json!({
            "task_overview": "   ", // whitespace only → None
            "current_state": null,
            "files_touched": ["a", "", 42, null, "  b  "],
        });
        let s = SummarySections::from_json(&raw);
        assert!(
            s.task_overview.is_none(),
            "whitespace-only string drops to None"
        );
        assert!(s.current_state.is_none());
        assert_eq!(s.files_touched, vec!["a", "b"]);
    }

    #[test]
    fn completeness_counts_filled_slots() {
        let empty = SummarySections::default();
        assert!((empty.completeness() - 0.0).abs() < f32::EPSILON);

        let all_filled = SummarySections {
            task_overview: Some("a".to_string()),
            current_state: Some("b".to_string()),
            files_touched: vec!["x".to_string()],
            key_decisions: vec!["x".to_string()],
            discoveries: vec!["x".to_string()],
            next_steps: vec!["x".to_string()],
            open_questions: vec!["x".to_string()],
            external_refs: vec!["x".to_string()],
        };
        assert!((all_filled.completeness() - 1.0).abs() < f32::EPSILON);

        let half = SummarySections {
            task_overview: Some("a".to_string()),
            current_state: Some("b".to_string()),
            files_touched: vec!["x".to_string()],
            key_decisions: vec!["x".to_string()],
            ..Default::default()
        };
        assert!((half.completeness() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn to_markdown_omits_empty_sections() {
        let s = SummarySections {
            task_overview: Some("Fix CI".to_string()),
            files_touched: vec!["src/foo.rs".to_string()],
            ..Default::default()
        };
        let md = s.to_markdown();
        assert!(md.contains("## Task overview"));
        assert!(md.contains("Fix CI"));
        assert!(md.contains("## Files touched"));
        assert!(md.contains("- src/foo.rs"));
        // Sections with no content should not appear at all.
        assert!(!md.contains("## Current state"));
        assert!(!md.contains("## Discoveries"));
        assert!(!md.contains("## External refs"));
    }

    #[test]
    fn to_markdown_round_trips_via_from_json() {
        let original = SummarySections {
            task_overview: Some("research".to_string()),
            current_state: Some("blocked on cred refresh".to_string()),
            files_touched: vec!["src/auth.rs".to_string()],
            next_steps: vec!["wire token refresh".to_string()],
            ..Default::default()
        };
        let raw_json = serde_json::to_value(&original).expect("serialize");
        let parsed = SummarySections::from_json(&raw_json);
        assert_eq!(original, parsed);
    }

    #[test]
    fn build_sectional_prompt_includes_existing_summary_when_present() {
        let p1 = build_sectional_prompt(None, "user: do X", None);
        assert!(p1.contains("NEW TRANSCRIPT:"));
        assert!(!p1.contains("EXISTING SUMMARY"));

        let p2 = build_sectional_prompt(Some("prev summary text"), "user: continue", None);
        assert!(p2.contains("EXISTING SUMMARY TO UPDATE:"));
        assert!(p2.contains("prev summary text"));
        assert!(p2.contains("NEW TRANSCRIPT:"));
        assert!(p2.find("EXISTING SUMMARY").unwrap() < p2.find("NEW TRANSCRIPT:").unwrap());
    }

    #[test]
    fn build_sectional_prompt_treats_blank_existing_summary_as_absent() {
        let p = build_sectional_prompt(Some("   \n  "), "user: hi", None);
        assert!(!p.contains("EXISTING SUMMARY"));
    }

    #[test]
    fn build_sectional_prompt_appends_focus_block_when_provided() {
        let p = build_sectional_prompt(None, "user: continue", Some("keep the API design talk"));
        assert!(p.contains("FOCUS:"));
        assert!(p.contains("keep the API design talk"));
        // Focus must appear before transcript so the model reads it first.
        assert!(p.find("FOCUS:").unwrap() < p.find("NEW TRANSCRIPT:").unwrap());
    }

    #[test]
    fn build_sectional_prompt_treats_blank_focus_as_absent() {
        let p = build_sectional_prompt(None, "user: hi", Some("   \n   "));
        assert!(!p.contains("FOCUS:"));
    }

    // === PR-3 effective_compaction_threshold tests ===

    #[test]
    fn effective_threshold_falls_back_to_absolute_when_window_unknown() {
        let t = effective_compaction_threshold(100_000, None, 0.85, 16_384);
        assert_eq!(t, 100_000);
    }

    #[test]
    fn effective_threshold_uses_percentage_when_tighter_than_absolute() {
        // 200k window * 0.85 = 170k → below 200k absolute → percentage wins
        let t = effective_compaction_threshold(200_000, Some(200_000), 0.85, 16_384);
        // Percentage = 170k; reserve = 200k - 16k = 184k; absolute = 200k → 170k.
        assert_eq!(t, 170_000);
    }

    #[test]
    fn effective_threshold_uses_reserve_when_tighter_than_percentage() {
        // With a generous percentage (0.99) and a 16k reserve, reserve binds first.
        let t = effective_compaction_threshold(1_000_000, Some(200_000), 0.99, 16_384);
        // pct = 198k; reserve = 184k; absolute = 1M → 184k (reserve wins).
        assert_eq!(t, 200_000_usize.saturating_sub(16_384));
    }

    #[test]
    fn effective_threshold_honors_absolute_as_floor() {
        // Tiny window with a tight absolute — absolute wins.
        let t = effective_compaction_threshold(5_000, Some(200_000), 0.85, 16_384);
        assert_eq!(t, 5_000);
    }

    // === PR-7 tool-result swap tests ===

    fn tool_msg_with_id(id: &str, content: &str, name: &str) -> ChatMessage {
        ChatMessage {
            role: "tool".to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: Some(name.to_string()),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            reasoning_content: None,
            is_error: None,
        }
    }

    #[test]
    fn build_compact_placeholder_embeds_id_and_head() {
        let body = "Found 17 matching files:\n/foo/bar.rs\n/foo/baz.rs\n…";
        let s = build_compact_placeholder("call_42", "search_text", body);
        assert!(s.contains("call_42"));
        assert!(s.contains("search_text"));
        // Bytes count matches the original
        assert!(s.contains(&format!("bytes={}", body.len())));
        // Head excerpt is single-line (newlines flattened to spaces)
        assert!(!s.split("head=\"").nth(1).unwrap().contains('\n'));
    }

    #[test]
    fn build_compact_placeholder_handles_multibyte_safely() {
        let body = "αβγδ ".repeat(200);
        let s = build_compact_placeholder("c1", "fetch", &body);
        // Must not panic; head trimmed to byte budget with ellipsis.
        assert!(s.contains('…'));
    }

    #[test]
    fn swap_replaces_tool_messages_with_placeholders() {
        let mut ctx = vec![
            ChatMessage::user("find foo"),
            ChatMessage::assistant("I'll search."),
            tool_msg_with_id("c1", "huge result body".repeat(100).as_str(), "search_text"),
            ChatMessage::assistant("Found it."),
        ];
        let (swapped, cached) = swap_all_tool_results_in_place(&mut ctx);
        assert_eq!(swapped, 1);
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].0, "c1");
        let tool_after = ctx
            .iter()
            .find(|m| m.role == "tool")
            .and_then(|m| m.content.as_ref())
            .map(|c| c.text_content())
            .unwrap();
        assert!(tool_after.starts_with("[Tool result archived."));
        assert!(tool_after.contains("c1"));
    }

    #[test]
    fn swap_skips_tool_messages_without_id() {
        let mut ctx = vec![ChatMessage {
            role: "tool".to_string(),
            content: Some(MessageContent::Text("orphan tool result".to_string())),
            name: Some("legacy_tool".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            is_error: None,
        }];
        let (swapped, cached) = swap_all_tool_results_in_place(&mut ctx);
        assert_eq!(swapped, 0);
        assert!(cached.is_empty());
    }

    // === PR-7.2 staleness swap tests ===

    fn user_msg(text: &str) -> ChatMessage {
        ChatMessage::user(text)
    }
    fn assistant_msg(text: &str) -> ChatMessage {
        ChatMessage::assistant(text)
    }

    #[test]
    fn identify_stale_swaps_keeps_recent_user_turns_tools_intact() {
        // Three user turns; default keep_recent = 3 means NONE of these tool
        // results are stale (all within the keep window).
        let messages = vec![
            (1, user_msg("first")),
            (2, assistant_msg("a1")),
            (3, tool_msg_with_id("c1", "result 1", "t")),
            (4, user_msg("second")),
            (5, assistant_msg("a2")),
            (6, tool_msg_with_id("c2", "result 2", "t")),
            (7, user_msg("third")),
            (8, assistant_msg("a3")),
            (9, tool_msg_with_id("c3", "result 3", "t")),
        ];
        let stale = identify_stale_tool_swaps(&messages, KEEP_RECENT_USER_TURNS_DEFAULT);
        assert!(
            stale.is_empty(),
            "with 3 user turns and keep=3, nothing is stale; got {:?}",
            stale
        );
    }

    #[test]
    fn identify_stale_swaps_marks_old_tool_results() {
        // Five user turns, keep_recent = 2: tools BEFORE the last 2 user turns
        // are stale.
        let messages = vec![
            (1, user_msg("first")),
            (2, tool_msg_with_id("c1", "first tool", "t")), // stale
            (3, user_msg("second")),
            (4, tool_msg_with_id("c2", "second tool", "t")), // stale
            (5, user_msg("third")),
            (6, tool_msg_with_id("c3", "third tool", "t")), // stale
            (7, user_msg("fourth")),
            (8, tool_msg_with_id("c4", "fourth tool", "t")), // keep
            (9, user_msg("fifth")),
            (10, tool_msg_with_id("c5", "fifth tool", "t")), // keep
        ];
        let stale = identify_stale_tool_swaps(&messages, 2);
        let ids: Vec<&str> = stale.iter().map(|t| t.1.as_str()).collect();
        assert_eq!(ids, vec!["c3", "c2", "c1"]); // newest-stale-first, then older
    }

    #[test]
    fn identify_stale_swaps_skips_id_less_tool_messages() {
        let messages = vec![
            (1, user_msg("u1")),
            (
                2,
                ChatMessage {
                    role: "tool".to_string(),
                    content: Some(MessageContent::Text("orphan".to_string())),
                    name: Some("legacy".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                    is_error: None,
                },
            ),
            (3, user_msg("u2")),
            (4, tool_msg_with_id("c_kept", "kept tool", "t")),
            (5, user_msg("u3")),
            (6, user_msg("u4")),
            (7, user_msg("u5")),
            (8, user_msg("u6")), // many user turns past the orphan
        ];
        let stale = identify_stale_tool_swaps(&messages, 2);
        // The orphan would be eligible by position, but has no tool_call_id —
        // we can't recall it, so we don't compact it.
        assert!(
            stale.iter().all(|t| !t.1.is_empty()),
            "no orphans expected; got {:?}",
            stale
        );
    }

    #[test]
    fn identify_stale_swaps_skips_already_swapped() {
        let placeholder = build_compact_placeholder("c1", "t", "original body");
        let messages = vec![
            (1, user_msg("first")),
            (
                2,
                ChatMessage {
                    role: "tool".to_string(),
                    content: Some(MessageContent::Text(placeholder)),
                    name: Some("t".to_string()),
                    tool_calls: None,
                    tool_call_id: Some("c1".to_string()),
                    reasoning_content: None,
                    is_error: None,
                },
            ),
            (3, user_msg("u2")),
            (4, user_msg("u3")),
            (5, user_msg("u4")),
        ];
        let stale = identify_stale_tool_swaps(&messages, 2);
        assert!(
            stale.is_empty(),
            "already-swapped messages must not be re-swapped"
        );
    }

    #[test]
    fn swap_is_idempotent_on_already_swapped_messages() {
        let mut ctx = vec![tool_msg_with_id("c2", "first body", "x")];
        let _ = swap_all_tool_results_in_place(&mut ctx);
        let (swapped_again, cached_again) = swap_all_tool_results_in_place(&mut ctx);
        assert_eq!(swapped_again, 0, "re-swap must be a no-op");
        assert!(cached_again.is_empty());
    }

    #[test]
    fn effective_threshold_falls_back_to_absolute_when_reserve_exceeds_window() {
        // Degenerate config (reserve >= window). Without the guard, the naive
        // computation would return 0 and trigger compaction every turn. With
        // the guard, we fall back to `absolute` (the user's explicit ceiling).
        let t = effective_compaction_threshold(80_000, Some(50_000), 0.85, 100_000);
        assert_eq!(t, 80_000);
    }

    #[test]
    fn effective_threshold_falls_back_to_absolute_for_invalid_percentage() {
        // percentage <= 0 or > 1: drop the window-aware path entirely.
        assert_eq!(
            effective_compaction_threshold(80_000, Some(200_000), 0.0, 16_384),
            80_000
        );
        assert_eq!(
            effective_compaction_threshold(80_000, Some(200_000), -0.5, 16_384),
            80_000
        );
        assert_eq!(
            effective_compaction_threshold(80_000, Some(200_000), 1.5, 16_384),
            80_000
        );
    }

    #[test]
    fn effective_threshold_never_returns_zero() {
        // Final `.max(1)` floor — even if a future bound regresses to zero,
        // callers never face `approx_tokens >= 0` always-true.
        assert!(effective_compaction_threshold(0, None, 0.85, 16_384) >= 1);
        assert!(effective_compaction_threshold(0, Some(200_000), 0.85, 16_384) >= 1);
    }

    #[test]
    fn preprocessing_reduces_tokens_meaningfully_on_image_heavy_input() {
        // PR-1 acceptance criterion: ≥30% input-token reduction on a synthetic chat
        // with multiple image parts. We use 3 images and a large tool result.
        let big_image = ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: format!("data:image/png;base64,{}", "A".repeat(50_000)),
                detail: None,
            },
        };
        let ctx = vec![
            user_with_parts(vec![
                ContentPart::Text {
                    text: "image #1 description".to_string(),
                },
                big_image.clone(),
            ]),
            user_with_parts(vec![big_image.clone()]),
            user_with_parts(vec![big_image]),
            tool_message(&"y".repeat(80_000)), // 80kb ≈ 20k tokens, capped to 10k tokens
        ];
        // Baseline: render with no stripping and no truncation cap to estimate
        // the un-preprocessed token count.
        let (baseline, baseline_tokens) =
            preprocess_transcript_for_compaction(&ctx, false, usize::MAX);
        let (stripped, stripped_tokens) = preprocess_transcript_for_compaction(&ctx, true, 10_000);
        assert!(baseline.len() > stripped.len());
        let reduction = 1.0 - (stripped_tokens as f64 / baseline_tokens as f64);
        assert!(
            reduction >= 0.30,
            "expected ≥30% token reduction; baseline={} stripped={} reduction={:.3}",
            baseline_tokens,
            stripped_tokens,
            reduction
        );
    }
}
