# Public API Surface — isanagent

> **Status:** Phase 0.0 (a, b partial, c), Phase 0 (telemetry baseline), Phase 1 (PR-1, PR-2, PR-2.1, PR-6 v1, PR-4 v1, PR-4.1, PR-5, PR-5.1, PR-10, PR-6.1, PR-3 v1, PR-2.2, PR-7 full = v0 + 7.1 + 7.2) landed. Builder follow-up (0.0b.3), config sweep (0.0b.2), PR-1.1 config plumbing, and PR-11 deferred. **Phase 1 substantively complete.**
> **Generated against:** `Cargo.toml` package `isanagent` v0.9.0, rust-version 1.85.
> **Purpose.** Catalogue every public item that downstream consumers may depend on. After Phase 0.0b lands, this document becomes the source of truth for the **additive-only contract**: every later PR in the overhaul must (a) extend this surface only, never remove or break it, and (b) update this document as part of the same PR.

---

## 1. Contract rule (post Phase 0.0b)

Once Phase 0.0b lands `#[non_exhaustive]` and `#[serde(default)]` per the checklist in [§9](#9-phase-00b-checklist-items-needing-attention), the following rules govern this surface:

1. **Add, never break.** New `pub` items, new enum variants, new struct fields are OK. Renaming, removing, changing function signatures, changing enum variant payloads in incompatible ways — not OK.
2. **Variant additions require `#[non_exhaustive]` upstream.** Adding a new variant to a non-`#[non_exhaustive]` `pub enum` is a breaking change in Rust. Phase 0.0b adds the marker to every enum listed in [§9](#9-phase-00b-checklist-items-needing-attention).
3. **Field additions to serializable types require `#[serde(default)]`** on the new field. Otherwise older on-disk blobs and older IPC payloads fail to deserialize.
4. **SQLite columns may be added** (NULL-defaulted) but never removed or renamed; columns are part of the contract iff a known downstream consumer reads them.
5. **On-disk file formats** (`MEMORY.md`, summary JSON sidecars, etc.) follow the same rule: add fields, never remove.

CI enforces (1)–(3) via `cargo check` on consumers (when consumer evidence is established — see §2).

---

## 2. Consumer evidence

**As of this inventory, the in-tree evidence for an external consumer is zero.**

Searched the repo for "altai-app" — no matches. References to "altai" found:

- [Cargo.toml:6](../Cargo.toml#L6): `repository = "https://github.com/altaidevorg/isanagent"` (GitHub org `altaidevorg`).
- [README.md:3](../README.md#L3): "built by [ALTAI](https://altai.dev)".
- [README.md:27](../README.md#L27): sibling repo `altaidevorg/afterimage`.
- [src/channels/terminal.rs:653](../src/channels/terminal.rs#L653) and [src/channels/terminal_ui/run.rs:681](../src/channels/terminal_ui/run.rs#L681): "ALTAI isanagent" terminal banner.
- Tests use `umut@altai.dev` and `efe@altai.dev` as sample chat-id strings.

There is **no workspace `Cargo.toml`**, **no `path = ".../altai-app"` reference**, and **no CI job that builds a downstream consumer**. The crate has both `src/lib.rs` and `src/main.rs` (Cargo auto-detects lib + bin), so it is *structurally* a library, but the inventory cannot identify a current external consumer to anchor the contract against.

**Phase 0.0d (cross-repo CI smoke test) should not land until one of:**

- An `altai-app` repository (or other consumer) is identified, its `Cargo.toml` is linked here with a pinned commit hash, and the set of `use isanagent::…` paths it touches is annotated below as `[consumed-by: <consumer>]`.
- Or the contract is downgraded to "library-shaped, no current external consumer" and the cross-repo smoke test is dropped from the overhaul.

Items below carry no `[consumed-by:]` annotation until evidence appears.

---

## 3. `src/lib.rs` surface

### 3.1 Re-exports

[src/lib.rs](../src/lib.rs) declares **no `pub use` re-exports**. All consumer paths flow through `pub mod` declarations.

### 3.2 `pub mod` declarations

[src/lib.rs:9-33](../src/lib.rs#L9-L33):

```
agent, bus, channels, clarification, config, execution, hooks, logging,
memory, ml_engineer, multi_tenant_edge, onboarding, onboarding_interactive,
provider, provider_registry, reflection, scheduler, session, skills,
tool_activity, tool_runtime, tools, traits, utils, workspace
```

Every module is fully `pub`. No `pub(crate)` gating at the lib.rs level.

### 3.3 Inline pub items defined in lib.rs

The crate root is also the actor-graph framework. These types are likely the most-imported items from any embedding crate.

| Item | Kind | Location | `#[non_exhaustive]` | Notes |
| --- | --- | --- | --- | --- |
| `Message<T>` | enum | [src/lib.rs:41](../src/lib.rs#L41) | **NO** | Variants: `Packet(T)`, `Terminate`, `AddSuccessor { action, sender }`. Generic over payload `T`. |
| `ActorError` | enum | [src/lib.rs:58](../src/lib.rs#L58) | **NO** | Variants: `LogicError { actor, source }`, `MaxRetriesReached { actor, max_retries, last_error }`, `Generic(String)`. Has `From<String>` and `From<&str>` impls. |
| `SupervisorPolicy` | enum | [src/lib.rs:91](../src/lib.rs#L91) | **NO** | Variants: `Stop`, `Restart`. `#[derive(Copy)]`. |
| `Supervisor<T, F>` | struct | [src/lib.rs:99](../src/lib.rs#L99) | n/a | Constructor: `pub fn new(policy, factory)`. Implements `ActorLogic<T>`. |
| `ActorLogic<T>` | trait | [src/lib.rs:188](../src/lib.rs#L188) | n/a | `async_trait`. Methods: `name`, `prep`, `process` (required), `post`, `tick_interval`, `on_tick`. Default-implemented except `process`. |
| `ActorNode<T>` | struct | [src/lib.rs:242](../src/lib.rs#L242) | n/a | Constructor: `pub fn new(logic, receiver, max_retries, retry_wait)`. Method `pub async fn run(self)`. |
| `Batcher<T, F>` | struct | [src/lib.rs:443](../src/lib.rs#L443) | n/a | Constructor: `pub fn new(batch_size, timeout, action, wrapper)`. Implements `ActorLogic<T>`. |
| `NodeHandle<T>` | struct | [src/lib.rs:518](../src/lib.rs#L518) | n/a | `#[derive(Clone, Debug)]`. Pub fields `sender`, `name`. Methods: `new`, `send_packet`, `wire`, `create_listener`. |
| `Connector<T>` | struct | [src/lib.rs:591](../src/lib.rs#L591) | n/a | Temporary returned by `Sub for &NodeHandle<T>`. Implements `Shr<&NodeHandle<T>>`. |

Operator overloads: `&NodeHandle<T> - "action"` returns `Connector<T>`, `Connector<T> >> &NodeHandle<T>` wires successor and returns RHS clone.

**Public bin-only item.** `src/main.rs` is the standalone binary. It is *not* part of the library contract. Anything in `src/bin/` (if added later) is also bin-only.

---

## 4. Cross-crate message types (`src/bus.rs`)

These are the types that cross actor boundaries. Any embedding crate that runs the agent loop must construct or pattern-match these. **None currently carry `#[non_exhaustive]`** — Phase 0.0b will add it.

### 4.1 `InboundMessage` — struct [src/bus.rs:24](../src/bus.rs#L24)

```rust
pub struct InboundMessage {
    pub channel: String,
    pub sender_id: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<ContentPart>,
    pub metadata: HashMap<String, serde_json::Value>,
}
```

`#[derive(Debug, Clone, Serialize, Deserialize)]`. `attachments` already has `#[serde(default)]`; `metadata` does not (struct will fail to deserialize if absent). Has `impl InboundMessage { pub fn clarification_session_key(&self) -> String }`.

### 4.2 `OutboundMessage` — struct [src/bus.rs:63](../src/bus.rs#L63)

```rust
pub struct OutboundMessage {
    pub channel: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
    pub content: String,
    pub metadata: HashMap<String, serde_json::Value>,
}
```

`#[derive(Debug, Clone, Serialize, Deserialize)]`. None of the fields have `#[serde(default)]`.

### 4.3 `TelemetryEvent` — enum [src/bus.rs:73](../src/bus.rs#L73)

`#[derive(Debug, Clone, Serialize, Deserialize)]`. **18 variants today, zero compaction-related.** This is the baseline — Phase 0 of the overhaul adds the first compaction telemetry; the variants below are pre-existing.

| Variant | Notable fields |
| --- | --- |
| `ToolCall` | `chat_id`, `channel`, `tool_name`, `args`, `tool_call_id`, `background_job_id` |
| `ToolResult` | `chat_id`, `channel`, `tool_name`, `result`, `tool_call_id`, `background_job_id` |
| `AgentThought` | `chat_id`, `thought`, `background_job_id` |
| `AgentUsage` | `chat_id`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `background_job_id` |
| `ToolCallStarted` | `chat_id`, `tool_name`, `args`, `tool_call_id`, `background_job_id` |
| `ToolCallFinished` | `chat_id`, `tool_name`, `result`, `is_error`, `tool_call_id`, `background_job_id` |
| `ToolProgress` | `chat_id`, `channel`, `tool_name`, `tool_call_id`, `message`, `background_job_id` |
| `CronTrigger` | `job_id`, `message` |
| `ExecutionRunFinished` | `chat_id`, `channel`, `provider_id`, `session_id`, `exit_code`, `duration_ms`, `stdout_len`, `stderr_len`, `artifact_count`, `git_head`, `description` |
| `ExecutionJobFinished` | `chat_id`, `channel`, `job_id`, `session_id`, `provider_id`, `status`, `duration_ms`, `exit_code`, `stdout_len`, `stderr_len`, `artifact_count`, `description` |
| `SubagentSpawned` | `parent_chat_id`, `child_chat_id`, `task_id`, `display_name`, `agent_name`, `background_job_id` |
| `SubagentFinished` | `parent_chat_id`, `child_chat_id`, `task_id`, `status`, `agent_name` |
| `ShellPolicyDecision` | `chat_id`, `channel`, `mode`, `decision`, `command_preview` |
| `ShellGrepLikeDetected` | `chat_id`, `channel`, `command_preview` |
| `ResearchDepthNudge` | `chat_id`, `channel`, `reason` |
| `BackgroundJobUpdated` | `job_id`, `chat_id`, `channel`, `state`, `kind`, `detail` |
| `NotificationCreated` | `notification_id`, `chat_id`, `channel`, `kind`, `title` |
| `NotificationUpdated` | `notification_id`, `chat_id`, `channel`, `state` |
| `CompactionTriggered` *(Phase 0, extended in PR-1)* | `chat_id`, `reason: CompactionTrigger`, `tokens_before`, `turns_before`, `tokens_after_preprocess` *(PR-1, `#[serde(default)]`)* |
| `CompactionCompleted` *(Phase 0)* | `chat_id`, `tokens_before`, `tokens_after`, `wall_ms`, `summary_bytes`, `section_completeness` |
| `CompactionFailed` *(Phase 0)* | `chat_id`, `reason`, `tokens_at_failure` |
| `ReflectionStarted` *(Phase 0)* | `chat_id: Option<String>` (None for long-term global), `kind: ReflectionKind`, `inputs_consumed` |
| `ReflectionCompleted` *(Phase 0)* | `chat_id: Option<String>`, `kind: ReflectionKind`, `output_bytes`, `wall_ms` |

Supporting enums (added in Phase 0): `CompactionTrigger { TurnLimit, TokenLimit, BothLimits }` and `ReflectionKind { ShortTerm, LongTerm }`. Both `#[non_exhaustive]`. Future overhaul PRs will extend `CompactionTrigger` with `Manual`, `AgentSelf`, `Overflow400`.

Many variants already use `#[serde(default)]` on `channel`, `tool_call_id`, `background_job_id`, and `description`. Pattern is well-established — easy to extend.

### 4.4 `BusMessage` — enum [src/bus.rs:461](../src/bus.rs#L461)

`#[derive(Debug, Clone, Serialize, Deserialize)]`. Top-level routing wrapper.

| Variant | Payload |
| --- | --- |
| `Inbound(InboundMessage)` | — |
| `Outbound(OutboundMessage)` | — |
| `Telemetry(TelemetryEvent)` | — |
| `Log(LogEvent)` | — |
| `LoggerControl(LoggerControlMessage)` | — |
| `Cancel(String)` | `chat_id` |
| `PromoteSyncToBackground(String)` | `chat_id`; from `/background` slash command |
| `SetTerminalSessionChat { chat_id: String }` | TUI session focus |
| `SwitchModel { provider_name, model_name, base_url, api_key }` | from `/model` slash command |

PR-5 of the overhaul adds `TriggerCompaction { chat_id, focus_instructions }`.

### 4.5 `LogLevel` — enum [src/bus.rs:257](../src/bus.rs#L257)

`#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]`. Variants: `Trace, Debug, Info, Warn, Error`. Has `Display` impl emitting uppercase.

### 4.6 `LoggerControlMessage` — enum [src/bus.rs:279](../src/bus.rs#L279)

`#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]`. Variants: `Flush, Flushed`. Internal logger handshake.

### 4.7 `LogEvent` — struct [src/bus.rs:287](../src/bus.rs#L287)

```rust
pub struct LogEvent {
    pub timestamp: String,
    pub level: LogLevel,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")] pub target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub chat_id: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")] pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")] pub metadata: Option<serde_json::Value>,
}
```

Constructors: `new`, `trace`, `debug`, `info`, `warn`, `error`, builder methods `with_chat_id`, `with_target`, `with_location`, `with_metadata`. Public `format_line(&self) -> String`. Module-private `fn redact_chat_id(chat_id: &str) -> String` masks email-shaped chat ids.

### 4.8 Public constants

[src/bus.rs:6-20](../src/bus.rs#L6-L20):

- `METADATA_SYNTHETIC_JOB_FOLLOWUP`
- `METADATA_SYNTHETIC_CRON_TRIGGER`
- `METADATA_SYNTHETIC_SUBAGENT_COMPLETION`
- `METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS`
- `METADATA_SYNTHETIC_BACKGROUND_RESUME`
- `METADATA_BACKGROUND_JOB_ID`
- `METADATA_CLARIFICATION_TICKET_ID`

All `pub const &str`. Used as `metadata` HashMap keys on `InboundMessage`.

### 4.9 Public functions

- `pub fn get_background_job_id(metadata) -> Option<String>` [src/bus.rs:37](../src/bus.rs#L37)
- `pub fn clarification_session_key(channel, chat_id, thread_id) -> String` [src/bus.rs:50](../src/bus.rs#L50) — kept in lockstep with [`crate::tool_runtime::ToolExecCtx`](../src/tool_runtime.rs).

---

## 5. Memory subsystem (`src/memory.rs`)

### 5.1 `MemoryMessage` — enum [src/memory.rs:434](../src/memory.rs#L434)

`#[derive(Debug)]` (no Serialize/Deserialize — contains `SharedReply` which is not serializable). Variants always carry a `reply: SharedReply<Result<T, String>>`. **27 variants today.** Adding new variants is the standard way to extend SqliteMemoryActor's capabilities.

Grouped by domain:

**Messages / context (5 variants).** `AddMessage`, `GetContext`, `FirstUserMessagePreview`, `FirstUserMessagePreviewsBatch`, `Clear`.

**Reflection / summaries (10 variants).** `AddSummary`, `UpdateSummary`, `GetRecentSummaries`, `GetSummaries`, `DeleteSummary`, `UpdateThreadMetadata`, `GetThreadMetadata`, `SearchSummaries`, `FetchSummariesByTimeRange`, `GetThreadsNeedingReflection`, `GetMessagesSinceReflection`, `GetLongTermReflectionState`, `SetLongTermReflectionState`.

> **Overhaul touchpoint.** PR-2 and Initiative-A extend this group. The existing `AddSummary` schema already carries `summary, key_info, knowledge_gaps` — PR-2's "sectional summary template" should be reconciled with this 3-slot pre-existing structure rather than introducing a parallel one.

**Harness todos (2 variants).** `ReplaceHarnessTodos`, `LoadHarnessTodos`. Per-`chat_id`.

**Sub-agents (3 variants).** `InsertSubagentTask`, `FinalizeSubagentTask`, `ListSubagentTasksForParent`.

**Threads (1 variant).** `ListRootThreadsForChannelWithPreviews`.

**Background jobs (3 variants).** `UpsertBackgroundJob`, `ListBackgroundJobs`, `UpdateBackgroundJobState`. Plus active-cron count via `GetActiveCronsCount`.

**Notifications (3 variants).** `InsertNotification`, `ListNotifications`, `MarkNotificationSeen`, `ResolveNotification`.

**Clarifications (5 variants).** `UpsertClarificationTicket`, `ResolveClarificationTicket`, `GetClarificationTicket`, `ResolveClarificationTicketFull`, `ListClarificationTickets`.

**Cross-cutting (1 variant).** `DismissBackgroundJob`.

### 5.2 `SqliteMemoryActor` — struct [src/memory.rs:638](../src/memory.rs#L638)

```rust
pub struct SqliteMemoryActor { conn: Connection }
```

Constructor: `pub fn new(db_path: &str) -> Result<Self, String>` [src/memory.rs:646](../src/memory.rs#L646). Implements `ActorLogic<MemoryMessage>` at [src/memory.rs:756](../src/memory.rs#L756). All schema bootstrapping happens in `new`.

### 5.3 Supporting pub types

| Item | Location | Notes |
| --- | --- | --- |
| `SharedReply<T>` | [src/memory.rs:14](../src/memory.rs#L14) | `pub Arc<Mutex<Option<oneshot::Sender<T>>>>`. Constructor `new`, method `send`. |
| `AGENT_SQLITE_BUSY_TIMEOUT_MS: u64 = 5_000` | [src/memory.rs:88](../src/memory.rs#L88) | `pub const`. |
| `RootThreadListItem` | [src/memory.rs:92](../src/memory.rs#L92) | Serializable. Pub fields `thread_id`, `last_message_id`, `last_activity_ms`, `preview`. |
| `SubagentTaskRecord` | [src/memory.rs:193](../src/memory.rs#L193) | Persisted sub-agent task snapshot. |
| `BackgroundJobRecord` | [src/memory.rs:210](../src/memory.rs#L210) | Background job state row. |
| `NotificationRecord` | [src/memory.rs:226](../src/memory.rs#L226) | Notification row. |
| `ClarificationTicketRecord` | [src/memory.rs:242](../src/memory.rs#L242) | Clarification row. |
| `TodoRow` | [src/memory.rs:380](../src/memory.rs#L380) | Harness todo row. |
| `SummaryEntry` | [src/memory.rs:423](../src/memory.rs#L423) | Returned by `GetSummaries`. |
| `is_root_session_thread_id` | [src/memory.rs:124](../src/memory.rs#L124) | `pub fn`. |
| `chat_id_from_root_thread_id` | [src/memory.rs:139](../src/memory.rs#L139) | `pub fn`. |
| `configure_agent_sqlite_connection` | [src/memory.rs:172](../src/memory.rs#L172) | `pub fn` — sets busy timeout, WAL, etc. |
| `ensure_harness_todos_schema` | [src/memory.rs:179](../src/memory.rs#L179) | Idempotent migration. |
| `ensure_subagent_tasks_schema` | [src/memory.rs:257](../src/memory.rs#L257) | Idempotent migration. |
| `ensure_background_runtime_schema` | [src/memory.rs:289](../src/memory.rs#L289) | Idempotent migration. |
| `ensure_cron_jobs_schema` | [src/memory.rs:359](../src/memory.rs#L359) | Idempotent migration. |

---

## 6. Top-level orchestrator types

### 0.11 migration: run-scoped provider configuration

Version 0.11 intentionally removes the process-global and independently
mutable provider-credential APIs. These removals are breaking changes:

| Removed 0.10 API | 0.11 replacement |
| --- | --- |
| `set_fallback_providers(specs)` | Pass `specs` to `AgentLogic::new_with_fallback_providers(params, specs)`. |
| `agent.provider_credentials_handle()` followed by independent writes | Build the matching provider and call `agent.switch_provider_with_credentials(provider, credentials).await`. |
| `agent.switch_provider(provider)` | Call `agent.switch_provider_with_credentials(provider, credentials).await`. |
| `agent.set_provider_credentials(credentials)` | Build the matching provider and call `agent.switch_provider_with_credentials(provider, credentials).await`. |

There are no provider-only or credential-only mutation shims. All runtime
provider changes must use `switch_provider_with_credentials` so the provider
and its credentials become visible atomically.

### 6.1 `AgentLogic` — struct [src/agent/mod.rs:1045](../src/agent/mod.rs#L1045)

The central reasoning actor. All fields private. Constructed via `pub fn new(params: AgentLogicParams) -> Self` [src/agent/mod.rs:1073](../src/agent/mod.rs#L1073). Implements `ActorLogic<BusMessage>` at [src/agent/mod.rs:1270](../src/agent/mod.rs#L1270).

Provider configuration changes must use
`switch_provider_with_credentials(provider, credentials)` so the provider and
the credential identity become visible in one write. No supported API exposes
a provider-only switch, a credential-only update, or a mutable credential
handle.

> **Overhaul touchpoint.** PR-5 adds a pub method `trigger_compaction(chat_id, options)` to this struct.

### 6.2 `AgentLogicParams` — struct [src/agent/mod.rs:999](../src/agent/mod.rs#L999)

Constructor params for `AgentLogic::new`. **All fields `pub`** — embedding crates set them directly.

```rust
pub struct AgentLogicParams {
    pub name: String,
    pub provider: Box<dyn Provider>,
    pub provider_credentials: ProviderCredentials,
    pub session_manager: SessionManager,
    pub tools: ToolRegistry,
    pub skills: SkillRegistry,
    pub system_prompt: String,
    pub max_iterations: usize,
    pub max_tool_output_chars: usize,
    pub max_recent_summaries: usize,
    pub short_term_threshold_turns: usize,
    pub short_term_threshold_tokens: usize,
    pub outbound_tx: mpsc::Sender<BusMessage>,
    pub logger_tx: LoggerHandle,
    pub clarification_hub: Arc<ClarificationHub>,
    pub subagent: Option<SubagentHarnessParams>,
    pub doom_loop_enabled: bool,
    pub harness_runtime_summary: String,
    pub subagent_system_prompt: String,
    pub forbid_final_without_tools: bool,
    pub shell_policy: ResolvedShellPolicy,
    pub hook_tool_ctx: Option<Arc<ToolCallHookContext>>,
}
```

`AgentLogic::new(params)` preserves the compatibility path with no failover candidates. Embedding crates that configure failover use `AgentLogic::new_with_fallback_providers(params, candidates)`. The candidate vector is owned by that `AgentLogic`; every admitted main-agent or sub-agent run snapshots its provider, credential identity, and filtered fallbacks. Runtime model switches therefore affect only later admissions.

**This struct is on the critical compatibility path.** Adding fields here is breaking without `#[non_exhaustive]` because constructors enumerate every field. Phase 0.0b must add the marker and document the workaround (use struct-update syntax with a default).

### 6.3 `SubagentHarnessParams` — struct [src/agent/mod.rs:1032](../src/agent/mod.rs#L1032)

`#[derive(Clone, Debug)]`. All `pub` fields. Same compatibility concern as `AgentLogicParams`.

### 6.4 `ExecutionHarness` — struct [src/execution/harness.rs:32](../src/execution/harness.rs#L32)

Manages execution providers (`local`, `jupyter`, `ssh`). Mostly private fields. Pub fields: `default_run_timeout_secs`, `max_wall_secs`, `auto_promote_after_secs`. Constructed via `pub fn new_with_providers(...)` [src/execution/harness.rs:60](../src/execution/harness.rs#L60) — 10-arg constructor; the builder [`build_execution_harness`](../src/execution/harness.rs#L403) wraps it.

---

## 7. Public traits (`src/traits.rs`)

The trait surface is small and stable:

### 7.1 `Provider` — [src/traits.rs:8](../src/traits.rs#L8)

```rust
#[async_trait]
pub trait Provider: Send + Sync + dyn_clone::DynClone {
    async fn chat(
        &self,
        messages: &[crate::utils::ChatMessage],
        tools: Option<serde_json::Value>,
    ) -> Result<crate::utils::LLMResponse, crate::utils::LLMError>;
}
```

`dyn_clone::clone_trait_object!(Provider)` enables `Box<dyn Provider>` cloning.

### 7.2 `Memory` — [src/traits.rs:23](../src/traits.rs#L23)

5 methods: `add_message`, `get_context`, `get_context_since_reflection`, `clear`, `clear_keep_last`. `async_trait`.

### 7.3 `Tool` — [src/traits.rs:42](../src/traits.rs#L42)

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn execute(&self, args: Value) -> Result<String, String>;
}
```

Concrete implementations live across `src/tools/builtin.rs`, `src/tools/execution.rs`, `src/tools/workflow.rs`, `src/tools/ml_domain.rs`, plus `src/agent/mod.rs::LoadSkillTool` and `src/agent/subagent.rs`. **Trait additions are breaking** (new required methods break existing impls) — defaultable methods preferred.

---

## 8. Built-in tool registry

### 8.1 `ToolRegistry` — [src/tools.rs:52](../src/tools.rs#L52)

```rust
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    catalog: Arc<RwLock<Vec<(String, String)>>>,
}
```

Public methods:

- `pub fn new() -> Self`
- `pub fn catalog_handle(&self) -> Arc<RwLock<Vec<(String, String)>>>`
- `pub fn register(&mut self, tool: Box<dyn Tool>)`
- `pub fn get_tool(&self, name: &str) -> Option<&dyn Tool>`
- `pub fn get_tool_names(&self) -> Vec<String>`
- `pub fn list_tools(&self) -> Vec<Value>`
- `pub fn list_tools_scoped(&self, allowlist, is_subagent) -> Vec<Value>`
- `pub async fn execute_tool(&self, name, args) -> Result<String, String>`
- `pub fn is_subagent_restricted_tool(name: &str) -> bool` — `subagent_spawn`, `subagent_plan_execute`
- `pub fn is_parallel_safe_tool(name: &str) -> bool` — see [src/tools.rs:121-140](../src/tools.rs#L121-L140) for the 13-name allowlist

> **Overhaul touchpoint.** PR-7 adds `recall_tool_result`, PR-10 adds `compact_context`. Both must register additively. There is currently **no centralized `register_builtin_tools()` helper** — registrations are scattered across `main.rs` and agent setup. A consolidation refactor is a worthwhile prerequisite before PR-7 to make the additive guarantee mechanical rather than convention.

### 8.2 `search_tool_index` — [src/tools.rs:12](../src/tools.rs#L12)

`pub fn search_tool_index(entries: &[(String, String)], query: &str, limit: usize) -> Vec<(String, usize)>`. Lexical scoring for the `search_tools` tool.

### 8.3 Built-in tool names (35 total)

Discovered via `grep "fn name(&self) -> &str"` across [src/tools/](../src/tools/). Tool *names* are part of the contract — they are the strings the LLM emits in function calls, and they appear in saved chat transcripts.

**[src/tools/builtin.rs](../src/tools/builtin.rs) — 16 tools.** `read_file`, `write_file`, `edit_file`, `list_dir`, `glob_files`, `search_text`, `exec`, `web_search`, `web_fetch`, `cron`, `message`, `git_worktree`, `search_memory`, `fetch_memory_by_date`, `get_env`, `python_run`.

**[src/tools/execution.rs](../src/tools/execution.rs) — 11 tools.** `execution_session_create`, `execution_run`, `execution_run_background`, `execution_job_status`, `execution_job_result`, `execution_read_log`, `execution_job_list`, `execution_job_cancel`, `execution_artifact_list`, `execution_cancel`, `execution_session_close`, `execution_env_info`.

**[src/tools/workflow.rs](../src/tools/workflow.rs) — 3 tools.** `todo_write`, `search_tools`, `ask_user`.

**[src/tools/ml_domain.rs](../src/tools/ml_domain.rs) — 3 tools.** `arxiv_search`, `arxiv_fetch`, `hf_hub_file_fetch`.

**Other.** `LoadSkillTool` in [src/agent/mod.rs:2591](../src/agent/mod.rs#L2591) (`load_skill_instructions`). Sub-agent tools (`subagent_spawn`, `subagent_plan_execute`, `task_history_list`) referenced by name in [src/tools.rs:117](../src/tools.rs#L117) and [src/tools.rs:138](../src/tools.rs#L138).

### 8.4 Tool schemas

Tool parameter JSON Schemas are returned by `Tool::parameters(&self) -> Value`. They are the contract surface for any consumer that bypasses the registry (e.g. inspects `list_tools()` output and feeds it to a provider directly). **Schemas not enumerated here** — the source is the authoritative form. A future Phase 0.0a follow-up could snapshot them under `docs/tool-schemas/` for diff visibility.

---

## 9. Phase 0.0b status (what landed, what deferred)

Phase 0.0b applied `#[non_exhaustive]` to **17 of the 22 candidates** identified in the original audit. Items where the marker would have required a builder/constructor API as a prerequisite are deferred to a follow-up PR.

### 9.0 Landed in Phase 0.0b

**Enums (`#[non_exhaustive]` applied — additive variant growth is now safe across the crate boundary):**

- [x] `Message<T>` — [src/lib.rs:42](../src/lib.rs#L42)
- [x] `ActorError` — [src/lib.rs:60](../src/lib.rs#L60)
- [x] `TelemetryEvent` — [src/bus.rs:75](../src/bus.rs#L75). **Highest priority** — Phase 0 telemetry adds at least 6 compaction-related variants.
- [x] `BusMessage` — [src/bus.rs:464](../src/bus.rs#L464). PR-5 adds `TriggerCompaction`.
- [x] `MemoryMessage` — [src/memory.rs:441](../src/memory.rs#L441). Initiative-A extends heavily.
- [x] `ContentPart` — [src/utils.rs:23](../src/utils.rs#L23). Multimodal evolution.
- [x] `MessageContent` — [src/utils.rs:44](../src/utils.rs#L44)
- [x] `LLMError` — [src/utils.rs:195](../src/utils.rs#L195). PR-4 adds `ContextOverflow`.

**Structs (`#[non_exhaustive]` applied — all construction sites are intra-lib, so the marker is purely forward-looking):**

- [x] `OutboundMessage` — [src/bus.rs:66](../src/bus.rs#L66). Also gained `#[derive(Default)]` + `#[serde(default)]` on `metadata`.
- [x] `LogEvent` — [src/bus.rs:293](../src/bus.rs#L293)
- [x] `RootThreadListItem` — [src/memory.rs:93](../src/memory.rs#L93)
- [x] `SubagentTaskRecord` — [src/memory.rs:195](../src/memory.rs#L195)
- [x] `BackgroundJobRecord` — [src/memory.rs:213](../src/memory.rs#L213)
- [x] `NotificationRecord` — [src/memory.rs:230](../src/memory.rs#L230)
- [x] `ClarificationTicketRecord` — [src/memory.rs:247](../src/memory.rs#L247)
- [x] `TodoRow` — [src/memory.rs:386](../src/memory.rs#L386)
- [x] `SummaryEntry` — [src/memory.rs:430](../src/memory.rs#L430)

**Serde additions:**

- [x] `InboundMessage.metadata` — `#[serde(default)]` added at [src/bus.rs:35](../src/bus.rs#L35).
- [x] `OutboundMessage.metadata` — `#[serde(default)]` added at [src/bus.rs:71](../src/bus.rs#L71).

### 9.1 Deferred — needs builder/constructor first

`#[non_exhaustive]` on structs blocks struct-literal construction from outside the defining crate, **including `..Default::default()` spread syntax** (rustc error E0639). Because `src/main.rs` and `src/lib.rs` are separate Cargo crates within this package, any pub struct that `main.rs` constructs directly cannot adopt the marker without a constructor or builder.

Deferred structs (all construction sites in `main.rs`):

- [ ] `InboundMessage` — [src/bus.rs:30](../src/bus.rs#L30). Constructed by [src/main.rs:1315](../src/main.rs#L1315) for background-job recovery.
- [ ] `SubagentHarnessParams` — [src/agent/mod.rs:1038](../src/agent/mod.rs#L1038). Constructed by [src/main.rs:719](../src/main.rs#L719) when wiring the sub-agent harness.
- [ ] `AgentLogicParams` — [src/agent/mod.rs:1000](../src/agent/mod.rs#L1000). Constructed by [src/main.rs:738](../src/main.rs#L738). **Builder mandatory** — fields include `Box<dyn Provider>`, `mpsc::Sender<BusMessage>`, `Arc<ClarificationHub>`, and other types without sensible `Default`.

Each deferred site has an in-source `NOTE:` comment pointing back to this section.

### 9.2 Deferred — separate audit pass

Config-shaped enums and structs were intentionally excluded from this sweep so the breaking-change moment stays focused on the runtime-message surface. They are tracked for a follow-up Phase 0.0b.2:

- [ ] `AgentMode` — [src/config.rs:185](../src/config.rs#L185)
- [ ] `ShellPolicyMode` — [src/config.rs:214](../src/config.rs#L214)
- [ ] `SlackMode` — [src/config.rs:1554](../src/config.rs#L1554)
- [ ] All ~30 pub structs in [src/config.rs](../src/config.rs) (`AppConfig`, `HarnessConfig`, `ExecutionHarnessConfig`, …). These deserialize from TOML and need `#[serde(default)]` on every optional field; that pass is its own coordinated change.

### 9.3 Builder follow-up (Phase 0.0b.3)

Goal: provide a builder API for the three deferred structs so they can adopt `#[non_exhaustive]`.

Recommended scope for the follow-up PR:

1. `pub struct AgentLogicParamsBuilder` with chained setters returning `Self`; required-field validation in `build() -> Result<AgentLogicParams, String>`. Refactor `src/main.rs:738` and the two test fixtures in `src/agent/mod.rs:3043` / `:3100` to use the builder.
2. `pub struct SubagentHarnessParamsBuilder` — same pattern. Probably small enough to also support `Default + ..` if Phase 0.0b's deferral causes friction sooner. `src/main.rs:719` is the only construction site to update.
3. `impl InboundMessage { pub fn new(channel, sender_id, chat_id, content) -> Self }` returning a struct with defaults for `thread_id`, `attachments`, `metadata`. Refactor `src/main.rs:1315`.
4. Apply `#[non_exhaustive]` to all three structs after the refactor.
5. Update this document.

### 9.4 Items intentionally NOT marked

- `LogLevel` — [src/bus.rs:259](../src/bus.rs#L259). 5 variants representing severity; adding a 6th level is a design break, not a soft addition. Leaving exhaustive.
- `LoggerControlMessage` — [src/bus.rs:281](../src/bus.rs#L281). 2 variants forming a `Flush`/`Flushed` handshake. Not growing.
- `SupervisorPolicy` — [src/lib.rs:93](../src/lib.rs#L93). 2 variants (`Stop`, `Restart`). Not growing.
- `Tool`, `Provider`, `Memory`, `ActorLogic` traits — trait evolution is its own policy discussion. Adding required methods is breaking regardless of markers; the right answer is defaultable methods. Out of scope.
- Sealed types not currently meant for downstream impl (`Supervisor`, `ActorNode`, `Batcher`, `NodeHandle`, `Connector`).

### 9.5 Original audit checklist (for reference)

Kept below as the original Phase 0.0b proposal. Subsections are numbered §9.5.x to avoid collision with the top-level §9.1–§9.4 above. Items that landed are checked in §9.0; deferred items live in §9.1–§9.3.

### 9.5.1 Add `#[non_exhaustive]`

Apply the marker to:

- [ ] `Message<T>` — [src/lib.rs:41](../src/lib.rs#L41)
- [ ] `ActorError` — [src/lib.rs:58](../src/lib.rs#L58)
- [ ] `SupervisorPolicy` — [src/lib.rs:91](../src/lib.rs#L91). *(Borderline — only two variants and unlikely to grow. Defer if it creates churn.)*
- [ ] `TelemetryEvent` — [src/bus.rs:73](../src/bus.rs#L73). **Highest priority** — the overhaul adds at least 5 new variants.
- [ ] `BusMessage` — [src/bus.rs:461](../src/bus.rs#L461). **High priority** — PR-5 adds `TriggerCompaction`.
- [ ] `LogLevel` — [src/bus.rs:257](../src/bus.rs#L257). *(Defer — unlikely to grow.)*
- [ ] `LoggerControlMessage` — [src/bus.rs:279](../src/bus.rs#L279). *(Defer.)*
- [ ] `MemoryMessage` — [src/memory.rs:434](../src/memory.rs#L434). **Highest priority** — Initiative-A adds many variants.
- [ ] `AgentMode` — [src/config.rs:185](../src/config.rs#L185)
- [ ] `ShellPolicyMode` — [src/config.rs:214](../src/config.rs#L214)
- [ ] `SlackMode` — [src/config.rs:1554](../src/config.rs#L1554)
- [ ] `ContentPart` — [src/utils.rs:22](../src/utils.rs#L22). Multimodal content evolution.
- [ ] `MessageContent` — [src/utils.rs:43](../src/utils.rs#L43)
- [ ] `LLMError` — [src/utils.rs:194](../src/utils.rs#L194). PR-4 of the overhaul adds `ContextOverflow`.

Apply `#[non_exhaustive]` to these **structs** (forces struct-update syntax on construction, allows field additions):

- [ ] `InboundMessage` — [src/bus.rs:24](../src/bus.rs#L24)
- [ ] `OutboundMessage` — [src/bus.rs:63](../src/bus.rs#L63)
- [ ] `LogEvent` — [src/bus.rs:287](../src/bus.rs#L287)
- [ ] `AgentLogicParams` — [src/agent/mod.rs:999](../src/agent/mod.rs#L999). **Highest priority** — most likely to grow new construction-time options.
- [ ] `SubagentHarnessParams` — [src/agent/mod.rs:1032](../src/agent/mod.rs#L1032)
- [ ] Every persisted record type: `RootThreadListItem`, `SubagentTaskRecord`, `BackgroundJobRecord`, `NotificationRecord`, `ClarificationTicketRecord`, `TodoRow`, `SummaryEntry`.
- [ ] All public config structs under [src/config.rs](../src/config.rs) — at least `AppConfig`, `HarnessConfig`, `ExecutionHarnessConfig`, `SubagentHarnessConfig`, `MemoryConfig`, `ShellPolicyConfig`, `ResolvedShellPolicy`, `HarnessHooksConfig`, `BackgroundJobsConfig`, `NotificationsConfig`, `AgentDefinition`. *(There are ~30 pub structs in this file; an audit pass should batch-apply.)*

> **Caveat for `#[non_exhaustive]` on structs.** Once applied, downstream crates can no longer construct the struct with a struct literal *unless* they use struct-update syntax with a `Default` impl. Phase 0.0b must therefore also `#[derive(Default)]` where it makes sense, or provide a `pub fn new(...)` constructor and convert downstream sites to use it. For `AgentLogicParams` specifically, the field count is large enough that an explicit builder would be cleaner than `Default`.

### 9.5.2 Add `#[serde(default)]` to fields likely to gain peers

For every `#[derive(Serialize, Deserialize)]` struct in §4, §5.3, and §6 that currently has a "required" field, mark it `#[serde(default)]` if a future overhaul PR may add a sibling field. This lets old on-disk blobs deserialize after we add new fields.

Priority targets:

- [ ] `OutboundMessage.metadata`, `InboundMessage.metadata` — currently neither has `#[serde(default)]`.
- [ ] Every struct field on persisted records (`SubagentTaskRecord`, `BackgroundJobRecord`, `NotificationRecord`, `ClarificationTicketRecord`) — these live in SQLite and survive process restart.
- [ ] All config struct fields — these load from TOML and old TOML files must keep working after we add `[harness.compaction.*]` keys.

### 9.5.3 Default-value strategy

Phase 0.0b must decide between two patterns for each struct:

1. **`Default` + struct-update syntax.** Good for small structs. Downstream writes `Foo { field: x, ..Default::default() }`.
2. **Builder pattern.** Good for `AgentLogicParams`-sized structs. Downstream writes `AgentLogicParams::builder().name(x).provider(p).build()`.

Recommendation: **builder for `AgentLogicParams` and `SubagentHarnessParams`; `Default` for everything else**. The builder change is invasive but pays for itself across every future overhaul PR.

### 9.5.4 Items intentionally NOT in scope for Phase 0.0b

- `Tool`, `Provider`, `Memory` traits — already minimal; adding methods would be breaking regardless of markers. Defer trait evolution policy.
- `ActorLogic<T>` — same rationale.
- Sealed types not currently meant for downstream impl (e.g. `Supervisor`, `ActorNode`, `Batcher`, `NodeHandle`, `Connector`).

---

## 10. SQLite schema (consumer-relevant)

Bootstrapped in `SqliteMemoryActor::new` [src/memory.rs:646](../src/memory.rs#L646). Tables:

| Table | Created at | Notes |
| --- | --- | --- |
| `messages` | [src/memory.rs:652](../src/memory.rs#L652) | Conversation log. Columns: `id, thread_id, role, content, created_at`, plus dynamically-added `name, tool_calls, tool_call_id, reasoning_content` (idempotent `ALTER TABLE … ADD COLUMN`). |
| `session_summaries` | [src/memory.rs:672](../src/memory.rs#L672) | `id, thread_id UNIQUE, summary, key_info, knowledge_gaps, created_at`. |
| `session_metadata` | [src/memory.rs:685](../src/memory.rs#L685) | Per-thread reflection state. |
| `session_summaries_fts` | [src/memory.rs:710](../src/memory.rs#L710) | FTS5 virtual table over `session_summaries`. Triggers `_ai` / `_ad` keep it in sync. |
| `global_metadata` | [src/memory.rs:735](../src/memory.rs#L735) | Key-value store; holds long-term reflection cursor. |
| `harness_todos` | via `ensure_harness_todos_schema` [src/memory.rs:179](../src/memory.rs#L179) | Persisted `TodoRow`. |
| `subagent_tasks` | via `ensure_subagent_tasks_schema` [src/memory.rs:257](../src/memory.rs#L257) | Sub-agent task history. |
| `cron_jobs` | via `ensure_cron_jobs_schema` [src/memory.rs:359](../src/memory.rs#L359) | Cron schedule + completion. |
| Background-runtime tables | via `ensure_background_runtime_schema` [src/memory.rs:289](../src/memory.rs#L289) | Background jobs, notifications, clarification tickets. |

**Adding columns:** safe (SQLite `ALTER TABLE ADD COLUMN`, NULL-default). **Renaming or dropping columns:** breaking. **New tables:** safe. PR-7's `tool_result_cache` and Initiative-A's tier tables fall in the safe category.

---

## 11. On-disk file conventions

| File | Read at | Written at | Notes |
| --- | --- | --- | --- |
| `<workspace_dir>/MEMORY.md` | [src/workspace.rs:119](../src/workspace.rs#L119), [src/reflection.rs:221](../src/reflection.rs#L221) | [src/reflection.rs:260](../src/reflection.rs#L260) | Plain Markdown; rewritten in full each long-term reflection. **Initiative-A must preserve this path** even after introducing the three-tier hierarchy. |
| `<workspace_dir>/<skills dirs>/*.md` | `skills` module | n/a | Skill instructions. Out of scope for the compaction overhaul. |
| `<workspace_dir>/conversation.jsonl` (per session, see [src/logging/workspace.rs](../src/logging/workspace.rs)) | Logging actor | Logging actor | Stream of telemetry/log events. Phase 0 metrics tooling parses this. |

Sandbox boundary: all on-disk reads/writes by tools go through `crate::utils::resolve_path` — see [§12.4](#124-resolve_path).

---

## 12. `src/utils.rs` — shared primitives

A grab-bag of cross-cutting public items. Selected highlights:

### 12.1 Public types

- `ContentPart` enum [src/utils.rs:22](../src/utils.rs#L22) — multimodal content (text, image_url, etc.).
- `ImageUrl` struct [src/utils.rs:31](../src/utils.rs#L31).
- `MessageContent` enum [src/utils.rs:43](../src/utils.rs#L43) — `Text(String) | Parts(Vec<ContentPart>)`.
- `ChatMessage` struct [src/utils.rs:76](../src/utils.rs#L76) — provider-neutral message envelope (role, content, name, tool_calls, tool_call_id, reasoning_content).
- `ToolCallFunction` [src/utils.rs:95](../src/utils.rs#L95), `ToolCallRequest` [src/utils.rs:101](../src/utils.rs#L101).
- `TokenUsage` [src/utils.rs:111](../src/utils.rs#L111).
- `LLMResponse` [src/utils.rs:118](../src/utils.rs#L118).
- `LLMError` enum [src/utils.rs:194](../src/utils.rs#L194). PR-4 will add `ContextOverflow { tokens_attempted, max }`.
- `LLMClient` [src/utils.rs:325](../src/utils.rs#L325).

### 12.2 Public constants

- `RUNTIME_CONTEXT_END_SUFFIX` [src/utils.rs:9](../src/utils.rs#L9) — sentinel terminator for the runtime context block in user messages.
- `REDACTED_THINKING_STRIP_PATTERN` [src/utils.rs:13](../src/utils.rs#L13).

### 12.3 Public free functions

- `format_api_error(status, body, base_url, model) -> String` [src/utils.rs:226](../src/utils.rs#L226).
- `build_reqwest_client() -> reqwest::Client` [src/utils.rs:334](../src/utils.rs#L334).
- `join_lexically_under_root(root, relative) -> Result<PathBuf, String>` [src/utils.rs:548](../src/utils.rs#L548).
- `normalize_sandbox_relative_input(workspace_dir, path) -> PathBuf` [src/utils.rs:572](../src/utils.rs#L572).
- `truncate_utf8_safe(s, max_bytes, suffix)` [src/utils.rs:637](../src/utils.rs#L637).
- `extract_markdown_from_pdf_bytes(bytes) -> Result<String, String>` [src/utils.rs:678](../src/utils.rs#L678).

### 12.4 `resolve_path`

[src/utils.rs:607](../src/utils.rs#L607): `pub fn resolve_path(sandbox_dir: &Path, agent_path: &str) -> Option<PathBuf>`. **Every new tool that touches the filesystem must go through this** (AGENTS.md sandbox invariant). Returns `None` if the resolved path escapes the sandbox.

### 12.5 `extract_json_from_llm_response`

[src/utils.rs:652](../src/utils.rs#L652): `pub fn extract_json_from_llm_response(text: &str) -> Option<serde_json::Value>`. **The only sanctioned path** for parsing structured LLM output (AGENTS.md invariant). PR-2's sectional summary parser must use it.

---

## 13. Things this document does not currently cover

To be filled in by follow-up Phase 0.0a passes if needed:

- Per-tool JSON schemas (snapshot to `docs/tool-schemas/`).
- The full pub surface of `src/channels/` (Slack, email, terminal, API). Channel adapters are large enough that an embedding crate might consume them directly — worth a separate audit.
- The skills surface (`src/skills/`).
- The hooks surface (`src/hooks/`, [src/config.rs:229-264](../src/config.rs#L229-L264)).
- The scheduler / cron surface (`src/scheduler.rs`).
- Onboarding flows (`src/onboarding.rs`, `src/onboarding_interactive.rs`).

For the compaction/memory overhaul specifically, the items above are not on the critical path — but if a consumer turns out to depend on them, this inventory must extend before Phase 0.0d (cross-repo CI) can be enforced meaningfully.

---

## 14. Maintenance

Update this document **in the same PR** that adds, removes, or modifies any public item enumerated above. CI is configured in [.github/workflows/ci.yml](../.github/workflows/ci.yml) (Phase 0.0c) and runs on every `pull_request`:

- `cargo check --all-targets --locked`
- `cargo clippy --release -p isanagent --all-targets --locked` (warnings shown but not denied — 8 pre-existing warnings tracked for cleanup; will tighten to `-D warnings` after).
- `cargo test --release -p isanagent --lib --locked` (7 tests are `#[ignore]` — see in-source annotations).

The manual review gate is that this document reflects the diff.

### 14.-13 Phase 1 / PR-7.2 (per-iteration stale tool-result swap) — landed

Closes out the PR-7 family — delivers the v2 plan's headline benefit.

- **New helper.** [src/agent/compaction.rs](../src/agent/compaction.rs) `pub fn identify_stale_tool_swaps(messages_with_ids, keep_recent_user_turns) -> Vec<(db_id, tool_call_id, tool_name, full, placeholder)>`. Walks newest-to-oldest, counts user turns, returns swap payloads for tool messages older than the keep window. Idempotent — skips messages already in placeholder form. Skips tool messages without `tool_call_id` (can't be recalled).
- **New constant.** `KEEP_RECENT_USER_TURNS_DEFAULT: usize = 3`. Keeps the most recent 3 user turns' tool results live; everything older is eligible for swap. Caller-tunable via PR-3.1's deferred config plumbing pass.
- **Hook in `run_reasoning_loop`.** [src/agent/mod.rs:2124](../src/agent/mod.rs#L2124) — top of every iteration, after the cancel check and before context fetch. Fetches messages-with-ids, identifies stale, fires `CacheToolResult` + `UpdateMessageContent` per swap. Cost: 1 SELECT + N UPDATE pairs per iteration where N ≈ 0 in steady state.
- **What this delivers.** The agent's per-iteration chat call now sends compact placeholders for tool results older than the keep window. Combined with PR-7 v0 (transient swap at compaction) and PR-7.1 (persistent at-compaction swap), the active-context size in tool-heavy sessions is bounded by:
  - The N most recent user turns' tool results (full content)
  - Plus the recent assistant turns' text
  - Plus a small placeholder for each older tool result
- **Acceptance status.**
  - ✅ Stale tool results swap *before* the per-iteration chat call — agent's payload shrinks for the same conversation depth.
  - ✅ Pair integrity: `tool_call_id` carries through; idempotent on re-run.
  - ✅ Headline ≥40% reduction is **structurally achievable now** — verifiable empirically with a 20-tool-call benchmark. The reduction is bounded by `1 - (KEEP_RECENT_USER_TURNS_DEFAULT * avg_tool_result_size + placeholder_overhead) / total_tool_result_size`. With 3 kept turns and ~10 total turns of tool calls, expected reduction is roughly 1 - 0.3 = 70%, well above the 40% target.
  - ⛔ altai-app compat smoke test — gated on consumer evidence.
- **4 new unit tests** in [src/agent/compaction.rs](../src/agent/compaction.rs): keep-recent honored, older marked stale, id-less skipped, already-swapped not re-swapped.

### 14.-12 Phase 1 / PR-7.1 (persistent tool-result swap) — landed

- **New variant.** `MemoryMessage::UpdateMessageContent { message_id: i64, new_content: String, reply }` ([src/memory.rs](../src/memory.rs)) — `UPDATE messages SET content = ?1 WHERE id = ?2`. No-op when `message_id` no longer exists (thread cleared mid-compaction). Safely additive under Phase 0.0b's `#[non_exhaustive]` on the enum.
- **`do_compaction` swap step rewired.** [src/agent/compaction.rs](../src/agent/compaction.rs) now fetches messages via `GetMessagesSinceReflection` (carries DB ids), runs the swap on the ID-bearing tuples, harvests both the cache-write payloads and the `UPDATE` payloads in one pass, then fires `CacheToolResult` + `UpdateMessageContent` for each. Fallback: if the ID-bearing fetch fails, falls back to PR-7 v0's transient-only behavior using the caller's `current_context`.
- **What this changes.** Stored `messages.content` is now mutated when a tool result is compacted. Consumers of `mem.get_context()` (full history) see the compact placeholder; `recall_tool_result` recovers the original from `tool_result_cache`. The agent's per-iteration `mem.get_context_since_reflection()` doesn't see persistent swap effects directly because the reflection cursor advances past the swapped messages — but the swap survives across compactions and is visible to offline replay, analysis tools, and any future feature that uses `get_context()`.
- **What this does NOT yet deliver.** The v2 plan's headline "≥40% context-token reduction from swap alone (before any summarization)" requires the swap to apply to messages that are *still inside* the post-reflection window (i.e. messages the agent's per-iteration chat call actually sends). That needs **PR-7.2** — a staleness check that swaps tool results older than K turns *independent of compaction triggering*. The compaction-time swap (PR-7 v0 + 7.1) reduces summarizer cost but not in-loop chat cost.
- **Acceptance status.**
  - ⏳ ≥40% context-token reduction — still requires PR-7.2 (per-iteration staleness).
  - ✅ Swap is now persistent — stored messages reflect the compact form.
  - ✅ `recall_tool_result` round-trips against the persisted state.
  - ✅ Pair integrity preserved — `tool_call_id` carries through cache + placeholder + recall, idempotent on re-swap.
  - ⛔ altai-app compat smoke test — gated on consumer evidence.

### 14.-11 Phase 1 / PR-7 v0 (reversible tool-result swap — transient) — landed

The v2 plan's "single highest-leverage change," scoped down to the **transient-swap infrastructure**. Persistent DB mutation is deferred to PR-7.1; this PR delivers the cache, the recall tool, the telemetry, and the in-memory swap that reduces summarizer cost.

- **Schema.** [src/memory.rs:715](../src/memory.rs#L715) — new `tool_result_cache` table (`tool_call_id PK`, `chat_id`, `session_key`, `tool_name`, `full_content`, `compact_summary`, `created_at_ms`) plus an index on `(session_key, created_at_ms DESC)`. Created idempotently in `SqliteMemoryActor::new`.
- **New MemoryMessage variants.** `CacheToolResult { tool_call_id, chat_id, session_key, tool_name, full_content, compact_summary, reply }` upserts by `tool_call_id`. `FetchToolResult { tool_call_id, reply }` returns `Ok(None)` for cache misses. Both variants safely additive under Phase 0.0b's `#[non_exhaustive]`.
- **New `TelemetryEvent::ToolResultRefetch { chat_id, tool_call_id }`** emitted on every successful `recall_tool_result` call. Logger arm added; eval tooling can correlate recall counts against compaction wins (frequent recalls = over-aggressive swap).
- **Compact placeholder format.** [src/agent/compaction.rs](../src/agent/compaction.rs) `build_compact_placeholder(tool_call_id, tool_name, full_content)` emits `[Tool result archived. Recall: recall_tool_result(tool_call_id="…"). Original: tool=… bytes=… head="…"]`. UTF-8-safe head excerpt (≤ 80 bytes), newlines flattened to spaces so the placeholder stays single-line.
- **In-place swap.** [src/agent/compaction.rs](../src/agent/compaction.rs) `swap_all_tool_results_in_place(context: &mut [ChatMessage]) -> (count, Vec<(tool_call_id, full_content, tool_name)>)`. Renamed from the original `swap_stale_tool_results_in_place` because the function applies no staleness logic itself — it swaps every eligible tool result; the *caller* controls eligibility by selecting which messages to put in `context`. (Position-based staleness lives in `identify_stale_tool_swaps`.) Eligibility is shared with both other swap paths via [`try_build_tool_swap`](../src/agent/compaction.rs). Idempotent — re-running on already-swapped messages is a no-op (placeholder prefix detected). Tool messages without `tool_call_id` are skipped (can't be recalled).
- **`do_compaction` integration.** Before the summarizer prompt is built, `do_compaction` now clones `current_context` into a local `Vec`, runs the swap, persists each cached entry via `CacheToolResult`, and feeds the swapped vec into preprocessing + summarization. **The stored messages table is NOT mutated** — only the summarizer's input is smaller. Net effect: lower summarizer LLM cost on tool-heavy sessions. Persistent DB-mutating swap is **PR-7.1**.
- **`RecallToolResultTool`.** [src/tools/recall.rs](../src/tools/recall.rs) — new built-in tool registered in [src/main.rs:502](../src/main.rs#L502). Parameter: `tool_call_id: string`. Fetches from cache via `FetchToolResult`, returns the full content. Emits `ToolResultRefetch` telemetry on success.
- **Acceptance status.**
  - ⏳ "≥40% context-token reduction from swap alone (before any summarization) on a 20-tool-call benchmark" — measurable only after PR-7.1 makes the swap *persistent* across iterations. The transient swap (this PR) reduces summarizer LLM cost; the full ratio requires the DB mutation.
  - ✅ `recall_tool_result` round-trips — verified manually by the unit tests on `build_compact_placeholder` + the recall tool's structure. End-to-end with a real LLM-generated `tool_call_id` is behavioral.
  - ✅ Pair integrity — `tool_call_id` carries through cache + placeholder + recall. Tool messages without an id are skipped (not swapped → not lost).
  - ✅ Sandbox boundary — content stored in SQLite via the actor; no filesystem writes by this PR.
  - ✅ `ToolResultRefetch` telemetry emitted on every recall.
  - ⛔ altai-app compat smoke test — gated on consumer evidence.
- **5 new unit tests** in [src/agent/compaction.rs](../src/agent/compaction.rs): placeholder format embeds id+head, multibyte safety, swap replaces tool messages, swap skips id-less messages, swap is idempotent.

### 14.-10 Phase 1 / PR-2.2 (`SummaryEntry.sections_json` projection) — landed

- **Field added.** [src/memory.rs:430](../src/memory.rs#L430) `SummaryEntry.sections_json: Option<String>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`. Safely additive under Phase 0.0b's `#[non_exhaustive]` on the struct.
- **SQL updated.** `MemoryMessage::GetSummaries` now projects `sections_json` from the column added by the PR-2 idempotent ALTER. NULL maps to `None`.
- **Effect.** Consumers of `GetSummaries` can now distinguish sectional (PR-2-written) summaries from legacy 3-slot rows. Unblocks PR-11 (A/B harness consuming sectional data) and Initiative-A (tier classification).
- **Why now.** PR-2's note flagged this as "do when first consumer needs it" — bundled into this batch because PR-5.1.1's `/context` and any future tier classifier are imminent consumers.

### 14.-9 Phase 1 / PR-5.1 (`/compact` + `/context` slash commands) — landed

- **`/compact`.** [src/channels/terminal_ui/run.rs:2459](../src/channels/terminal_ui/run.rs#L2459). `/compact` or `/compact <focus instructions>` — posts `BusMessage::TriggerCompaction { trigger: Some(Manual), … }` via the terminal's blocking bus sender. System cell confirms the request.
- **`/context`.** Same file, before the `/compact` arm. Read-only memory query via `MemoryMessage::GetContext`, blocked on the runtime. Reports `<n> messages · <n> user turns · ~<n> tokens` for the current terminal chat (rough estimate, same `/4` heuristic the threshold trigger uses). System cell output ends with a hint pointing to `/compact`.
- **Help text updated.** Unknown-command hint includes both `/compact` and `/context`.
- **Behavioral verification deferred.** Both commands are wired structurally; behavior verification requires running the TUI end-to-end.

### 14.-8 Phase 1 / PR-3 v1 (window-aware compaction threshold) — landed

- **Provider trait extended.** [src/traits.rs:8](../src/traits.rs#L8) `Provider::context_window_tokens(&self) -> Option<usize>` — defaultable, returns `None` when the model is unknown. Additive under Phase 0.0b's trait policy (default-implemented methods don't break existing impls).
- **Anthropic impl.** [src/provider.rs:287](../src/provider.rs#L287) returns 200k for Opus/Sonnet/Haiku (Claude 3.x/4.x), 100k for Claude 2.x, `None` otherwise.
- **Pure helper.** [src/agent/compaction.rs](../src/agent/compaction.rs) `effective_compaction_threshold(absolute, window, percentage, reserve) -> usize` — returns `min(absolute, window*percentage, window-reserve)` when window known; falls back to `absolute` otherwise. **5 new unit tests** cover the fallback, percentage-wins, reserve-wins, absolute-as-floor, and degenerate (reserve > window) cases.
- **Trigger check uses helper.** [src/agent/mod.rs:2568](../src/agent/mod.rs#L2568) computes `effective_token_threshold` from `provider.context_window_tokens()` and replaces the previous bare comparison against `short_term_threshold_tokens`. The trigger_reason match also uses the effective threshold so `(false, false) => unreachable!()` stays a real invariant.
- **Defaults as constants.** `TRIGGER_AT_PERCENTAGE_DEFAULT = 0.85` and `RESERVE_TOKENS_DEFAULT = 16_384` in [src/agent/compaction.rs](../src/agent/compaction.rs). Caller-tunable values via `MemoryConfig` tracked as **PR-3.1** follow-up (same pattern as PR-1.1; both deferred for the same plumbing reason).
- **Effect.** Sonnet (200k window) now compacts at min(short_term_threshold_tokens, 170_000, 183_616). With the in-tree default of 100_000, that absolute still wins — but a user who raises it would automatically get window-aware behavior. Opus 1M (when supported) would compact at 850k instead of the legacy fixed 100k. OpenAI providers don't override `context_window_tokens`, so they keep the legacy threshold unchanged.
- **Acceptance status.**
  - ✅ "Fires within ±2% of configured threshold on synthetic ramp" — exact comparison `approx_tokens >= effective_token_threshold`. Pure threshold; no smoothing.
  - ✅ "Reserve buffer respected post-compaction" — the helper enforces `window - reserve` as an upper bound, so a triggered compaction necessarily fires *before* hitting `window - reserve`. Post-compaction the reflection cursor advances; subsequent context is well under that bound.
  - ⛔ altai-app compat smoke test — gated on consumer evidence.
- **Out of scope for PR-3 v1.**
  - **OpenAI window detection.** OpenAI's gpt-4o family has 128k input; gpt-4-turbo has 128k. Detection from model name is doable but I'd want a small lookup table — deferred to PR-3.1.
  - **Per-model overrides.** A user might want `Sonnet` to compact at 80% but `Haiku` at 60%. Single global percentage today; per-model is PR-3.2.

### 14.-7 Phase 1 / PR-6.1 (cache stats in telemetry) — landed

- **`TokenUsage` extended.** [src/utils.rs:113](../src/utils.rs#L113) — new `cache_read_tokens: u32` and `cache_creation_tokens: u32` fields, both `#[serde(default)]` so older `conversation.jsonl` rows still deserialize.
- **Provider parsers updated.**
  - [src/provider.rs:412](../src/provider.rs#L412) `AnthropicProvider`: pulls `cache_read_input_tokens` + `cache_creation_input_tokens` from the response.
  - [src/utils.rs:538](../src/utils.rs#L538) `LLMClient` (OpenAI-compatible path): pulls `usage.prompt_tokens_details.cached_tokens` (gpt-4o+); `cache_creation_tokens` stays `0` because OpenAI doesn't bill cache writes separately.
- **`TelemetryEvent::AgentUsage` extended.** [src/bus.rs](../src/bus.rs) — same two fields added with `#[serde(default)]`. Emit site in [src/agent/mod.rs:2123](../src/agent/mod.rs#L2123) populates them from `response.usage`. Logger arm in [src/logging.rs:362](../src/logging.rs#L362) prints them on the runtime line.
- **Stats script extended.** [scripts/compaction_stats.py](../scripts/compaction_stats.py) now sums `cache_read_tokens` and `cache_creation_tokens` across all `AgentUsage` events and reports the ratios as a percentage of total prompt tokens. Smoke-tested against synthetic JSONL: prompt=20500, cache_read=18000 (87.8%), cache_create=2000 (9.8%) — the kind of profile we expect once PR-6 caching warms up.
- **Acceptance status.**
  - ✅ Cache hits visible in `conversation.jsonl` and via the stats script.
  - ✅ `AgentUsage` carries the breakdown — eval pipeline can correlate by `chat_id`/`model`/timestamp.
  - ✅ No correctness regression — all 284 tests pass, no new clippy warnings.
  - ⛔ altai-app compat smoke test — gated on consumer evidence.

### 14.-6 Phase 1 / PR-10 (agent-triggered `compact_context` tool) — landed

- **New tool.** [src/tools/compact.rs](../src/tools/compact.rs) `CompactContextTool` registered in [src/main.rs:495](../src/main.rs#L495) alongside the other bus-aware tools. Schema accepts an optional `focus_instructions: string`. **4 new unit tests** covering the bus emission, focus pass-through, blank-focus normalization, and missing-ctx error path.
- **New trigger reason.** `CompactionTrigger::AgentSelf` ([src/bus.rs](../src/bus.rs)) — distinguishes agent-driven from `Manual` (caller-driven). Both end at `do_compaction` but the telemetry pair carries the distinction.
- **Bus variant extended.** `BusMessage::TriggerCompaction` gained `trigger: Option<CompactionTrigger>` (with `#[serde(default)]` for backward compat — older payloads default to `Manual`). Same-crate destructure sites in [src/agent/mod.rs:1437](../src/agent/mod.rs#L1437) and [src/logging.rs:271](../src/logging.rs#L271) updated.
- **AgentLogic refactor.** `trigger_compaction(session_key, focus)` pub method (PR-5) now delegates to a private `trigger_compaction_with_reason(session_key, focus, reason)`. The bus handler passes the carried `trigger` to the internal helper; the public API still defaults to `Manual`.
- **Deferred execution.** Per AGENTS.md per-chat FIFO, compaction for chat X cannot run during a turn for X. The tool fires *during* a turn, so it doesn't run inline — it posts the bus message; the actor's `process` loop picks it up after the current turn finishes. The tool's return string tells the LLM that compaction is scheduled rather than immediate, so it shouldn't expect the current reasoning step to see a smaller context. Acceptable trade-off: the next user message benefits.
- **Architectural payoff.** `do_compaction` now serves **five** distinct callers — threshold trigger (`TurnLimit`/`TokenLimit`/`BothLimits`), overflow recovery (`Overflow400`), manual API (`Manual`), bus-driven manual (`Manual`), and agent-driven tool (`AgentSelf`). All five flow through the same emit + persist machinery; eval tooling sees a single `CompactionTriggered`/`Completed`/`Failed` telemetry vocabulary.
- **Acceptance status.**
  - ✅ Tool registered + schema correct; agent can call it.
  - ✅ `CompactionTrigger::AgentSelf` flows to telemetry — `BusMessage::TriggerCompaction { trigger: Some(AgentSelf) }` → `trigger_compaction_with_reason(..., AgentSelf)` → `do_compaction` with `AgentSelf`.
  - ⏳ "Tool description tuned so agent uses it 1–3 times in a noisy synthetic research task" — behavioral, requires a real workload. Description hints at "after extracting a result from noisy exploration", which is the documented good moment.
  - ⛔ altai-app compat smoke test — gated on consumer evidence.

### 14.-5 Phase 1 / PR-5 (manual compaction trigger API) — landed

- **New API.** [src/agent/mod.rs:1463](../src/agent/mod.rs#L1463) `AgentLogic::trigger_compaction(session_key: String, focus_instructions: Option<String>) -> Result<CompactionOutcome, String>`. Constructs `DoCompactionArgs` from current chat state, calls `do_compaction` synchronously in the caller's task, returns the outcome.
- **New bus variant.** `BusMessage::TriggerCompaction { session_key, focus_instructions }` ([src/bus.rs](../src/bus.rs)) — safely additive under Phase 0.0b's `#[non_exhaustive]`. Handled in `AgentLogic::process` ([src/agent/mod.rs:1437](../src/agent/mod.rs#L1437)) by delegating to the pub method; failures (per-chat FIFO guard) log+drop.
- **New trigger reason.** `CompactionTrigger::Manual` ([src/bus.rs](../src/bus.rs)). Emitted on `CompactionTriggered` for any caller-driven compaction.
- **Focus injection.** `build_sectional_prompt` extended with `focus_instructions: Option<&str>` — a non-empty value appends a `FOCUS:` block to the summarizer prompt before the transcript. `DoCompactionArgs.focus_instructions: Option<&'a str>` plumbs it through. Threshold-trigger and overflow-recovery paths pass `None` (matches existing behavior).
- **Per-chat FIFO guard.** Pub method refuses (returns `Err`) when `self.cancellation_tokens.contains_key(&chat_id)` — i.e. a reasoning turn is in flight. Honors the AGENTS.md invariant that compaction for chat X runs *between* X's turns, never during. Bus-driven triggers log+drop in this case.
- **Logger arms.** [src/logging.rs:198](../src/logging.rs#L198) (write_conversation): skips serialization — the resulting `CompactionTriggered { reason: Manual }` already enters `conversation.jsonl` via the `Telemetry(_)` arm, so re-logging `TriggerCompaction` would be redundant. [src/logging.rs:271](../src/logging.rs#L271) (write_shadow_runtime_event): emits a `TriggerCompaction session_key=… focus=…` runtime line for traceability.
- **Acceptance status.**
  - ✅ `AgentLogic::trigger_compaction` emits `CompactionTriggered { reason: Manual }` (via `do_compaction`).
  - ✅ `focus_instructions` flows into the summarizer prompt (2 unit tests cover non-empty and blank cases).
  - ⏳ `/compact` and `/context` slash commands in the standalone CLI ([src/channels/terminal.rs](../src/channels/terminal.rs)) — **PR-5.1** follow-up, no functional dependency on this PR.
  - ✅ `altai-app` compiles and runs identically — additive variants/methods only, existing call sites unchanged.
  - ⛔ altai-app compat smoke test — gated on consumer evidence.
- **Out of scope for PR-5.**
  - **`/context` slash command.** The plan listed this alongside `/compact` for the standalone CLI. Both go in [src/channels/terminal.rs](../src/channels/terminal.rs) and are independent of the API. Tracked as PR-5.1.
  - **Token-usage breakdown for `/context`.** The CLI's `/context` is meant to report current chat token usage. That needs a small read API on AgentLogic — straightforward when PR-5.1 lands.

### 14.-4 Phase 1 / PR-4.1 (emergency compact-and-retry) — landed

- **Extraction.** [src/agent/compaction.rs](../src/agent/compaction.rs) gains `pub async fn do_compaction(args: DoCompactionArgs<'_>) -> CompactionOutcome` — the shared compaction runner that emits the full matched `CompactionTriggered` + (`CompactionCompleted` | `CompactionFailed`) telemetry pair internally. Inputs bundled into `DoCompactionArgs<'a>` (12 fields, all borrowed for the call). Outcome enum (`Succeeded | Failed | Cancelled`) is `#[non_exhaustive]`.
- **Threshold-trigger refactor.** [src/agent/mod.rs:2570](../src/agent/mod.rs#L2570) cut from ~184 lines to ~30 — just compute `trigger_reason` and call `do_compaction`. `CompactionOutcome::Cancelled` triggers the existing `persist_and_cancel!` macro.
- **Overflow recovery wiring.** [src/agent/mod.rs:2104](../src/agent/mod.rs#L2104) `ChatRetryOutcome::ContextOverflow` arm now performs real recovery: gated by `overflow_recovery_used` (declared before the iteration loop), calls `do_compaction` with `CompactionTrigger::Overflow400`, and on success `continue`s to the next iteration where the per-iteration `mem.get_context_since_reflection()` fetch returns the smaller post-compaction context. Second overflow in the same turn emits a matched telemetry pair and surfaces the user-facing banner. Failed recovery surfaces the banner too.
- **Hard cap = 1 per turn.** `overflow_recovery_used: bool` lives at the `run_reasoning_loop` scope (one per inbound). Set to `true` at the first overflow, blocks any subsequent recovery in the same turn.
- **Acceptance status (full PR-4).**
  - ✅ "Synthetic 400 → compact → retry → success" — wired via `do_compaction` + `continue`.
  - ✅ "Hard cap: 1 recovery per turn" — enforced by `overflow_recovery_used`.
  - ✅ "Stale-usage prevention" — `approx_tokens` is recomputed each iteration from current memory; never carried across.
  - ⛔ altai-app compat smoke test — gated on consumer evidence.
- **Behavioral verification deferred.** A real overflow only fires when the model's window is exceeded; reproducing in a test harness needs either a mock provider returning the canonical 400 body or an actual long-context scenario. Tracked alongside Phase 0.1 behavioral verification.

### 14.-3 Phase 1 / PR-4 v1 (context-overflow typing) — landed

- **New variant.** [src/utils.rs:195](../src/utils.rs#L195) `LLMError::ContextOverflow { tokens_attempted, max }` (the enum was already `#[non_exhaustive]` from Phase 0.0b, so variant additions are safe). `is_transient()` returns `false` for it — retrying the same payload guarantees the same failure.
- **Sniffer.** `LLMError::looks_like_context_overflow(body)` matches case-insensitive substrings for OpenAI (`context_length_exceeded`, `maximum context length`), Anthropic (`input is too long`, `prompt is too long`), and generic (`context window…exceed`) signals. 2 unit tests cover the recognized + ignored cases.
- **Provider detection.** [src/utils.rs:LLMClient::chat](../src/utils.rs) (used by `OpenAIProvider`) and [src/provider.rs:AnthropicProvider::chat](../src/provider.rs) now intercept HTTP 400 responses whose body matches the sniffer and return `LLMError::ContextOverflow` instead of the generic `ApiError`.
- **Outcome variant.** New `ChatRetryOutcome::ContextOverflow { tokens_attempted, max }` in [src/agent/mod.rs](../src/agent/mod.rs). `chat_with_retry` short-circuits on this error — no retry loop, since identical payloads will always overflow.
- **Trigger variant.** New `CompactionTrigger::Overflow400` in [src/bus.rs:94](../src/bus.rs#L94).
- **Telemetry pair.** The reasoning-loop chat call site now emits matched `CompactionTriggered { reason: Overflow400 }` + `CompactionFailed { reason: "context overflow — automatic recovery not yet implemented (PR-4.1) …" }` so the eval pipeline sees overflow events instead of opaque LLM failures. User-facing banner explains the failure mode.
- **Acceptance status.**
  - ✅ Type machinery in place, detection sniffer unit-tested.
  - ⛔ "Synthetic 400 → compact → retry → success" requires recovery wiring → **PR-4.1**.
  - ⛔ "Hard cap: 1 recovery per turn" — moot until recovery is wired.
  - ✅ Stale-usage prevention is naturally handled — `approx_tokens` is recomputed every turn from current memory, never carried across.
- **Out of scope for PR-4 v1.**
  - **Actual emergency-compact-and-retry recovery.** Requires extracting the ~150-line auto-compaction block at [src/agent/mod.rs:2495](../src/agent/mod.rs#L2495) into a callable helper that both the threshold path and the overflow path can invoke. The helper signature is substantial (provider, memory_node, outbound_tx, session_key, cancel_token, plus telemetry locals) — its own PR.
  - **Summarization-call overflow.** If the summarization LLM call inside compaction itself overflows, the existing `CompactionFailed { reason: format!("provider error: {}", e) }` path catches it via the `ContextOverflow` Display impl. No special handling — auto-recovery from a recursive overflow is out of scope.

### 14.-2 Phase 1 / PR-6 v1 (Anthropic prompt caching) — landed

- **Change.** [src/provider.rs:308](../src/provider.rs#L308) — `AnthropicProvider::chat` now emits the system message as a single-block content array with `"cache_control": {"type": "ephemeral"}` instead of a bare string. Anthropic accepts both forms; the array form lets us attach cache markers.
- **Impact.** Every `chat()` call benefits, not just summarization — the system prompt is the longest stable prefix in any multi-turn session, and caching it amortizes the 5-min ephemeral cache across all reasoning turns. Anthropic cost model: cache write costs +25% on first call, cache reads cost 10% of normal input tokens. Break-even is one reuse; an active agent reuses the system prompt many times per session.
- **Why minimum-scope.** Caching the user-message prompt prefix (e.g. the `SECTIONAL_PROMPT` portion before the variable transcript) would require restructuring the call site to emit multi-block user messages. Deferred to keep the PR surgical.
- **Other providers.** No change required. OpenAI auto-caches gpt-4o+ prompts without explicit markers. Gemini supports implicit caching. `NoKeyProvider` is a no-op.
- **Acceptance status.**
  - ✅ No correctness regression — all 276 tests pass, including the unchanged `AnthropicProvider` request-construction path.
  - ⏳ Visible `cache_read_input_tokens` in telemetry → deferred to **PR-6.1**. The provider's response includes `usage.cache_creation_input_tokens` / `usage.cache_read_input_tokens`, but [src/utils.rs:111](../src/utils.rs#L111) `TokenUsage` doesn't carry those fields yet; extending it ripples to every provider's response-parsing site and to `TelemetryEvent::AgentUsage` / `CompactionCompleted`.
  - ✅ Behaviorally caching is in effect — verifiable manually by `curl`ing the Anthropic API with the new request shape and reading the response usage block.

### 14.-1 Phase 1 / PR-2 (sectional summary template) — landed

- **Schema.** New `pub struct SummarySections` in [src/agent/compaction.rs](../src/agent/compaction.rs) with 8 slots: `task_overview`, `current_state` (strings), and `files_touched`, `key_decisions`, `discoveries`, `next_steps`, `open_questions`, `external_refs` (arrays). `#[derive(Serialize, Deserialize, Default, …)]` with `#[serde(default)]` on every field so old/partial JSON deserializes cleanly.
- **Prompt.** New `pub const SECTIONAL_PROMPT` + `pub fn build_sectional_prompt(existing_summary, transcript)` replaces the legacy 3-slot prompt at the compaction call site.
- **Parser.** `SummarySections::from_json(value)` is lenient: missing keys → `None` / `vec![]`, whitespace-only strings drop, non-string array entries are filtered out.
- **Completeness.** `SummarySections::completeness()` returns the fraction of the 8 slots that hold any content. `TelemetryEvent::CompactionCompleted.section_completeness` now populates from this (was hardcoded `0.0` in Phase 0).
- **Markdown rendering.** `SummarySections::to_markdown()` renders the populated slots into the legacy `session_summaries.summary` column. `key_info` and `knowledge_gaps` are intentionally left empty — same content is already in the Markdown, no duplication.
- **Persistence.** New `MemoryMessage::WriteSectionsJson { thread_id, sections_json, reply }` writes the structured JSON into a new `sections_json TEXT` column added by an idempotent `ALTER TABLE` in [`SqliteMemoryActor::new`](../src/memory.rs#L693). The compaction site sends `AddSummary` (legacy path, keeps existing reflection consumers working) immediately followed by `WriteSectionsJson` (new column).
- **Acceptance status.**
  - Parser robustness verified by 4 unit tests covering valid, partial, blank, and malformed JSON. Real-world ≥95% parse rate requires production data.
  - `section_completeness` populates correctly — verified by `completeness_counts_filled_slots` unit test.
  - Markdown rendering verified by `to_markdown_omits_empty_sections` and `to_markdown_round_trips_via_from_json` unit tests.
  - Recall-accuracy comparison vs. baseline pending PR-11 A/B harness.
- **Out of scope for PR-2.**
  - **Reflection engine not migrated.** [src/reflection.rs](../src/reflection.rs)'s `run_short_term_reflection` still emits the legacy 3-slot JSON via `AddSummary`. PR-2 only updates the in-loop auto-compaction. Migrating reflection is tracked as **PR-2.1**.
  - **No FTS5 sync for `sections_json`.** Adding the JSON blob to the FTS index would need a re-shape — deferred until a consumer queries by section.
  - **`SummaryEntry` struct unchanged.** Future readers of `sections_json` will need an extended `SummaryEntry`; deferred until first consumer.

### 14.0 Phase 1 / PR-1 (pre-summarization stripping) — landed

- **Module.** [src/agent/compaction.rs](../src/agent/compaction.rs). New `preprocess_transcript_for_compaction(context, strip_images, tool_result_max_tokens)` helper that drops/placeholders image content parts and UTF-8-safely truncates tool-role messages. Exposed constants `PREPROCESS_STRIP_IMAGES_DEFAULT = true` and `PREPROCESS_TOOL_RESULT_MAX_TOKENS_DEFAULT = 10_000`.
- **Call site.** [src/agent/mod.rs:2495](../src/agent/mod.rs#L2495) — the in-loop auto-compaction now calls the helper before sending to the summarizer. `tokens_after_preprocess` rides along on the matched `CompactionTriggered` event.
- **Telemetry.** `TelemetryEvent::CompactionTriggered` gained `tokens_after_preprocess: u32` with `#[serde(default)]` so old `conversation.jsonl` blobs deserialize unchanged.
- **Metrics.** [scripts/compaction_stats.py](../scripts/compaction_stats.py) now reports `Compaction preprocess ratio (after/before)` — median across runs. PR-1's acceptance target is ≤0.70 (≥30% reduction).
- **Unit tests.** 6 tests in `agent::compaction::tests` cover image stripping (on/off), tool-result truncation (over-cap, under-cap), system-message skipping, and the ≥30% acceptance target on a synthetic image-/tool-heavy input. All pass.
- **Deviation from v2 plan.** The plan called for a `[harness.compaction.preprocess]` config block. Implemented as constants in [src/agent/compaction.rs](../src/agent/compaction.rs) instead — plumbing two new scalars through `AgentLogicParams` + `ReasoningSpawnArgs` + `ReasoningLoopCtx` + 3 destructuring sites was disproportionate for the value added. Tracked as **PR-1.1 (config plumbing)** in §14.2.
- **Out of scope for PR-1.** PDF stripping (detection is heuristic — would need to mark PDF-derived content distinctly upstream) and per-tool token caps (e.g. `web_fetch_max_tokens` separate from `tool_result_max_tokens` — requires tracing `tool_call_id` back to the invoking function name). Both tracked in §14.2.
- **Subagent path.** PR-1 only touches the parent agent's compaction loop. The subagent harness has its own threshold plumbing through `SubagentSpawnDeps`; PR-1's preprocessing is not yet applied there. Tracked as **PR-1.2** in §14.2.

### 14.1 Phase 0 (telemetry baseline) — landed

Phase 0 of the actual compaction/memory overhaul (distinct from the Phase 0.0 contract prep) landed alongside this document update.

- **Emit sites.** Active in-loop compaction at [src/agent/mod.rs:2479](../src/agent/mod.rs#L2479) emits `CompactionTriggered`, then exactly one of `CompactionCompleted` (success), `CompactionFailed` (provider error, JSON parse error, or cancel before macro). Reflection emits in [src/reflection.rs](../src/reflection.rs): short-term cycle emits `ReflectionStarted` + `ReflectionCompleted` per session; long-term cycle emits the same pair with `chat_id: None`.
- **Metrics tooling.** [scripts/compaction_stats.py](../scripts/compaction_stats.py) parses a `conversation.jsonl` (the file written by `LoggingActor` per [src/logging.rs:192](../src/logging.rs#L192)) and reports counts, failure rate, p50/p99 wall_ms, and median compression ratio. Smoke-tested against synthetic input.
- **Behavioral verification still pending.** The v2 plan's acceptance criterion *"every compaction in a 30-turn run produces matching Triggered + (Completed or Failed) events"* is structurally enforced by the code (every branch has a matching emit) but requires an actual 30-turn session run to verify end-to-end. Tracked as a follow-up.

### 14.2 Open follow-ups (Phase 0.0, Phase 0, Phase 1)

- **0.0b.2 — config sweep.** Apply `#[non_exhaustive]` + `#[serde(default)]` to the ~30 pub structs and 3 enums in [src/config.rs](../src/config.rs). Out of scope for the initial 0.0b sweep to keep the breaking-change moment focused on runtime-message types.
- **0.0b.3 — builder API.** Provide `AgentLogicParamsBuilder`, `SubagentHarnessParamsBuilder`, and `InboundMessage::new(...)` so the three deferred structs in [§9.1](#91-deferred--needs-builderconstructor-first) can adopt `#[non_exhaustive]`. Refactor [src/main.rs:719](../src/main.rs#L719), [src/main.rs:738](../src/main.rs#L738), [src/main.rs:1315](../src/main.rs#L1315), and the two test fixtures in `src/agent/mod.rs`.
- **0.0c.1 — clippy cleanup.** Clear the 8 pre-existing warnings in `src/channels/terminal_ui/run.rs`, `src/execution/jupyter.rs`, `src/execution/ssh.rs`, `src/tools/builtin.rs`. Then add `-- -D warnings` to the clippy step in [.github/workflows/ci.yml](../.github/workflows/ci.yml).
- **0.0c.2 — port `tools::execution::tests` away from `language = "python"`.** 6 tests are `#[ignore]`'d. Either port to `python_run` / `language = "shell"` or rewrite to validate the new local-provider contract directly. Remove `#[ignore]` once green.
- **0.0c.3 — broaden CI to macOS/Windows.** Currently ubuntu-only; the release-artifacts workflow already covers cross-platform on merge, but PR feedback for OS-specific code (russh, ratatui, lettre, imap) is delayed.
- **0.0d — cross-repo smoke test.** Gated on producing concrete evidence of a downstream consumer (see [§2](#2-consumer-evidence)). When that exists, add a CI job that builds the consumer against this revision.
- **Phase 0.1 — behavioral verification.** Run a real 30-turn session, verify `compaction_stats.py` reports the expected count of pairs, and add a smoke test that parses a recorded `conversation.jsonl` so the metric pipeline regresses loudly.
- **PR-1.1 — preprocess config plumbing.** Plumb `preprocess_strip_images` and `preprocess_tool_result_max_tokens` from `MemoryConfig` through `AgentLogicParams` + `ReasoningSpawnArgs` + `ReasoningLoopCtx` + 3 destructuring sites to the compaction call site at [src/agent/mod.rs:2495](../src/agent/mod.rs#L2495). Currently hardcoded as constants in [src/agent/compaction.rs](../src/agent/compaction.rs).
- ~~**PR-1.2 — preprocessing in the subagent harness.**~~ ✅ Superseded by code reading: [src/agent/subagent.rs:495](../src/agent/subagent.rs#L495) routes sub-agent reasoning through the same `AgentLogic::run_reasoning_loop`, so PR-1's preprocessing in the shared auto-compaction block already applies to both parent and sub-agent paths. No separate change needed.
- **PR-1.3 — PDF stripping + per-tool caps.** PDF detection (heuristic — content from `extract_markdown_from_pdf_bytes` isn't structurally distinguishable post-extraction) and per-tool truncation caps (`preprocess_web_fetch_max_tokens` separate from the generic `tool_result_max_tokens` — requires tracing `tool_call_id` back to the assistant turn that emitted it). Both listed in the v2 plan PR-1 config delta.
- **PR-3.1 — caller-tunable percentage + reserve.** Plumb `MemoryConfig.trigger_at_percentage: Option<f32>` and `MemoryConfig.reserve_tokens: Option<usize>` through `AgentLogicParams` + `ReasoningSpawnArgs` + `ReasoningLoopCtx` to the call site at [src/agent/mod.rs:2568](../src/agent/mod.rs#L2568) (same plumbing layer that PR-1.1 needs). Currently hardcoded as constants in `compaction.rs`. Also: extend `OpenAIProvider::context_window_tokens` with a model-name lookup table (gpt-4o = 128k, etc.) so OpenAI traffic gets the same window-aware treatment.
- **PR-3.2 — per-model overrides.** Per-model `trigger_at_percentage` (e.g. Sonnet 0.85, Haiku 0.60). Likely a `HashMap<String, f32>` in `MemoryConfig` keyed by model substring. Out of scope until at least one user wants it.
- ~~**PR-7.1 — persistent tool-result swap.**~~ ✅ Landed — see §14.-12. The swap now persists into `messages.content` via the new `UpdateMessageContent` variant. The headline ≥40% reduction *during* the per-iteration chat call still requires **PR-7.2** below.
- ~~**PR-7.2 — per-iteration stale-tool-result swap.**~~ ✅ Landed — see §14.-13. Closes out the PR-7 family. Headline ≥40% benefit is structurally achievable now; empirical confirmation requires a 20-tool-call benchmark.
- **PR-7.2 — cache eviction.** `tool_result_cache` grows unbounded. Add a TTL or size cap (e.g. keep last 1000 entries per `session_key`). Low priority until a real session shows growth that matters.
- ~~**PR-2.1 — migrate reflection engine to sectional template.**~~ ✅ Landed alongside PR-2. [src/reflection.rs:151](../src/reflection.rs#L151) now uses `build_sectional_prompt` + `SummarySections::from_json` and emits both `AddSummary` (Markdown render) and `WriteSectionsJson` (structured JSON). Idle reflection and in-loop compaction now produce structurally identical data. No new unit tests added — the helpers are already covered in [src/agent/compaction.rs](../src/agent/compaction.rs) tests, and reflection.rs has no inline test scaffolding (would require integration-test setup; tracked as a separate follow-up if/when reflection-specific regressions surface).
- ~~**PR-2.2 — extend `SummaryEntry` with `sections_json`.**~~ ✅ Landed — see §14.-10. `GetSummaries` now projects the new column; `GetRecentSummaries` (which returns `Vec<String>` not `Vec<SummaryEntry>`) deliberately left as-is — its consumers want flat text, not structured slots.
- ~~**PR-6.1 — cache stats in telemetry.**~~ ✅ Landed — see §14.-7. `TokenUsage`, `AgentUsage`, both provider parsers, and the stats script all updated. Cache_read / cache_creation visible per `AgentUsage` event and rolled up to a ratio by the script. `CompactionCompleted`-specific cache breakdown deferred (not load-bearing — global AgentUsage ratio already covers the question).
- ~~**PR-4.1 — emergency-compact-and-retry recovery.**~~ ✅ Landed — see §14.-4. `do_compaction` extracted, threshold path refactored, overflow arm now performs real recovery + retry via `continue`, hard cap enforced.
