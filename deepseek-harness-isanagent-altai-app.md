# What Can isanagent and Altai App Learn from DeepSeek Harness?

## Executive Summary

DeepSeek Harness offers useful architectural patterns for both **isanagent** and **Altai App**, but they should be adopted selectively.

The correct division of responsibility is:

- **isanagent** owns the execution runtime: model calls, tool execution, sandboxing, approvals, sessions, compaction, subagents, and execution events.
- **Altai Agent Service** exposes that runtime consistently to Desktop, CLI, and future clients.
- **Altai App** owns the product experience: editor, terminal, permission UI, artifact viewers, replay UI, Work OS, and human interaction.
- **Altai Work OS** owns cross-run work lifecycle. It should not be moved into isanagent.

The highest-value improvements are:

1. Secret-safe subprocess execution
2. Real OS-level sandboxing
3. A shared approval protocol
4. Native LLM streaming
5. Recoverable large tool outputs
6. A canonical append-only execution log
7. Server-owned session projections
8. Metadata-driven tool scheduling

DeepSeek Harness is still a developer preview and warns about compatibility-breaking changes. Its Cordis runtime should therefore not be adopted wholesale. See the [DeepSeek Harness repository](https://github.com/deepseek-ai/deepseek-harness).

## Current Product Boundaries

| Layer | Primary responsibility |
|---|---|
| Altai App | Desktop UX, editor, PTY, notebooks, LSP, Git, artifact viewers, approvals, Work OS |
| `altai-agent-service` | Shared host-neutral interface for Desktop, CLI, and other clients |
| isanagent | LLM loop, tools, sessions, execution, subagents, memory, compaction |
| Altai Work OS | Work items, attempts, reviews, inbox, and cross-run lifecycle |
| DeepSeek Harness | Reference architecture and source of reusable patterns |

Altai's existing control-plane/execution-plane decision is correct:

- Work and review lifecycle stays in the Altai host.
- isanagent executes one authorized attempt.
- React and other renderers do not own authoritative transitions.
- Run-internal todos and subagents do not become project-management records automatically.

This boundary should be preserved.

## Recommendation Matrix

| DeepSeek Pattern | isanagent Responsibility | Altai App Responsibility |
|---|---|---|
| Process sandbox | Implement sandbox providers and policies | Expose modes and enforcement status |
| Credential isolation | Sanitize execution environments | Keep credentials in Rust/keychain |
| Approval service | Own decision protocol and audit events | Render approval UI and collect decisions |
| LLM streaming | Emit typed provider chunks | Render deltas and partial responses |
| Spill storage | Preserve complete tool output | Provide artifact viewer and open actions |
| Tool scheduler | Classify and schedule tool calls | Visualize running/barrier/background calls |
| Session event log | Own model-visible execution truth | Consume events and projections |
| Session projections | Produce authoritative snapshots | Replace frontend-derived mirror state |
| Session fork/query | Implement execution semantics | Expose branch and trajectory UI |
| LSP | Consume through a capability interface | Reuse existing managed LSP runtime |
| Persistent PTY | Consume through a terminal capability | Reuse existing Altai PTY implementation |
| Message feedback | Bind feedback to stable message IDs | Provide rating and review UI |
| Plugin composition | Provide Rust capability seams | Continue using skills, MCP, and commands |

## P0 — Security Foundation

### 1. Secret-Safe Subprocess Environments

isanagent currently forwards the host environment directly to subprocesses launched by:

- `exec`
- `python_run`
- Local execution sessions
- UV-managed environment setup

Because Altai embeds isanagent as a Rust dependency, the same behavior applies to agent-controlled execution inside Altai.

Altai also owns other process surfaces, including:

- Native shell commands
- Notebook execution
- MCP servers
- LSP servers
- Workflow hooks
- Git commands

Not every process should receive the same environment.

#### Recommendation

Introduce a shared execution environment policy:

```rust
pub struct ExecutionEnvironmentPolicy {
    pub inherited: Vec<String>,
    pub explicit: HashMap<String, String>,
    pub credential_grants: Vec<CredentialGrant>,
    pub scrub_secrets: bool,
}
```

Agent-controlled subprocesses should:

- Call `env_clear()` by default.
- Receive only an approved base environment.
- Receive credentials through explicit grants.
- Never inherit every `*_TOKEN`, `*_KEY`, or `*_PASSWORD` variable.
- Redact output before persistence as a second layer of protection.

Altai should distinguish between:

- A user-owned interactive terminal, which may inherit the normal login environment
- An agent-owned subprocess, which should use the restricted environment policy
- A managed provider such as LSP or MCP, which should receive only declared variables

#### Ownership

- Core policy: **isanagent**
- Credential storage: **Altai Rust host**
- Permission UI: **Altai App**
- Per-tool credential grants: **Altai Agent Service**

### 2. Real Operating-System Sandbox

Path containment prevents simple directory escape, but it does not confine a child process after it starts.

DeepSeek provides:

- `read-only`
- `workspace-write`
- `danger-full-access`
- Linux Bubblewrap or Landlock
- macOS Seatbelt
- Windows restricted tokens and ACLs
- Explicit `full` or `partial` enforcement reporting
- Fail-closed behavior when confinement is required but unavailable

See the [DeepSeek sandbox design](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/sandbox.md).

#### Recommended Architecture

```rust
pub trait ProcessSandbox {
    async fn confine(
        &self,
        command: CommandSpec,
        policy: SandboxExecutionPolicy,
    ) -> Result<ConfinedCommand, SandboxError>;
}
```

```rust
pub enum SandboxMode {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}
```

The resolved policy should include:

- Workspace root
- Session identity
- Filesystem mode
- Network policy
- Credential policy
- Enforcement requirements

#### Altai App Integration

The existing Altai permission modes can map to sandbox policies:

| Altai Mode | Filesystem | Shell | Sandbox |
|---|---|---|---|
| Ask | Approval required | Approval required | Read-only by default |
| Auto-edit | Workspace writes allowed | Approval required | Workspace-write for edits |
| Plan | Mutations denied | Read-only commands only | Read-only |
| Bypass | Allowed | Allowed | Danger-full-access with warning |

Altai should display whether enforcement is:

- Fully enforced
- Partially enforced
- Unavailable
- Explicitly bypassed

### 3. Shared Approval Protocol

Altai already has stronger approval UX than isanagent alone:

- Permission modes
- Diff-first review
- Edit proposals
- Plan mode
- Approval cards
- A Rust-owned security boundary

However, the durable approval protocol should be standardized in isanagent and surfaced through `altai-agent-service`.

#### Required Events

```text
approval/requested
approval/decided
approval/cancelled
approval/unavailable
```

Each approval should contain:

- Stable request ID
- Session and run ID
- Tool-call ID
- Action type
- Human-readable reason
- Requested capability
- Final outcome
- Decision timestamp

The only granting result should be explicit:

```rust
enum ApprovalOutcome {
    AllowedOnce,
    Rejected,
    Cancelled,
    Unavailable,
}
```

Missing UI, disconnected renderer, timeout, and malformed responses should fail closed.

#### Ownership

- Decision semantics and audit: **isanagent**
- Transport: **Altai Agent Service**
- Human interaction: **Altai App**
- Cross-run review disposition: **Altai Work OS**

## P1 — Responsiveness and Reliability

### 4. Native LLM Streaming

isanagent's provider contract currently returns a complete response. As a result, Altai may display event streams, but it cannot receive genuine provider token deltas from the embedded isanagent path.

DeepSeek persists:

```text
assistant/chunk*
assistant/message
```

See the [DeepSeek agent lifecycle](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/agent-lifecycle.md).

#### isanagent Changes

Add a streaming provider contract:

```rust
pub trait Provider {
    async fn stream(
        &self,
        request: LlmRequest,
        sink: LlmStreamSink,
    ) -> Result<LlmFinish, LlmError>;
}
```

Typed chunks should distinguish:

- Text delta
- Reasoning delta
- Tool-call start
- Tool-call argument delta
- Usage update
- Finish
- Provider error
- Cancellation

#### Altai Agent Service Changes

Extend the event protocol with:

```text
assistant_message_started
assistant_message_delta
assistant_message_completed
reasoning_delta
```

The existing `agent_message` event can remain for backward compatibility.

#### Altai App Changes

- Reduce deltas into the active message.
- Render partial output immediately.
- Preserve partial output after cancellation.
- Show time-to-first-token.
- Avoid replacing the whole message for every token.
- Batch UI updates to animation frames.
- Recover streamed output through journal replay.

### 5. Spill Storage for Large Outputs

Both isanagent and Altai contain output truncation paths. Truncation protects context and UI performance, but the complete output should remain recoverable.

DeepSeek persists the complete result and returns a preview plus an opaque locator. See the [DeepSeek spill storage design](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/spill.md).

#### Recommended Flow

```text
Tool completes
    ↓
Output exceeds inline limit
    ↓
Full output stored in private spill storage
    ↓
Head/tail preview returned to model
    ↓
Artifact reference emitted to Altai
    ↓
User or model can open/recall the full output
```

#### isanagent Responsibilities

- Store the full output.
- Use session-scoped private storage.
- Return an opaque locator.
- Provide retrieval instructions.
- Connect locators to `recall_tool_result`.
- Apply retention and cleanup policies.

#### Altai App Responsibilities

- Render an “Open full output” action.
- Reuse the artifact and execution browser.
- Support logs, JSON, CSV, text, images, and notebook output.
- Show byte count and truncation state.
- Avoid loading large artifacts into React state until requested.

### 6. Metadata-Driven Tool Scheduler

Tool concurrency in isanagent is currently determined through a hardcoded parallel-safe name list.

Replace this with tool-owned metadata:

```rust
pub struct ToolPolicy {
    pub effect: ToolEffect,
    pub execution_mode: ExecutionMode,
    pub approval: ApprovalRequirement,
    pub output_retention: OutputRetention,
    pub timeout: Duration,
}
```

```rust
pub enum ExecutionMode {
    Parallel,
    Serial,
    Barrier,
    Background,
}
```

#### Scheduler Rules

- Use bounded concurrency.
- Preserve result order.
- Treat writes as barriers by default.
- Re-check policy before execution.
- Stop launching calls after cancellation.
- Apply per-session limits.
- Apply stricter subagent limits.
- Register long-running calls with the job runtime.
- Ensure disposal cancels and awaits owned jobs.

#### Altai App UX

Altai can surface the scheduler state in its existing run inspector:

- Running calls
- Waiting calls
- Barrier calls
- Background jobs
- Cancellation progress
- Approval blockers
- Concurrency limits

## P1/P2 — Durable State Architecture

### 7. Canonical Append-Only Execution Log

Altai already has an append-only `EventJournal` with:

- Run IDs
- Monotonic sequence numbers
- Terminal-event protection
- Duplicate detection
- Replay APIs
- Journal-before-renderer delivery

This is a strong foundation.

However, it is currently primarily a host and presentation replay journal. isanagent separately stores model conversation history and can mutate or delete message rows during compaction.

DeepSeek's stronger invariant is:

> Every model-visible input must be reconstructable from the canonical session log.

See the [DeepSeek architecture](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md).

#### Recommended Three-Layer Model

```text
isanagent SessionEventLog
    Canonical execution and model-visible history
            ↓
altai-agent-service RunEventJournal
    Stable client protocol and delivery replay
            ↓
Altai App projections
    UI state, transcript, todos, jobs, artifacts
```

Separately:

```text
Altai Work OS / work.db
    Work items, attempts, reviews, inbox
```

#### Avoid Dual Authority

- isanagent owns execution truth.
- Altai's run journal owns client delivery history.
- Work OS owns cross-run product state.
- Zustand stores must not become independent authoritative copies.

#### Migration

1. Add `session_events` in isanagent.
2. Shadow-write alongside the current message store.
3. Generate model history from session events.
4. Map session events into Altai run events.
5. Verify replay equivalence.
6. Represent compaction as a surface replacement event.
7. Stop deleting canonical history.
8. Add session fork and trajectory export.

### 8. Server-Owned Session Projections

Altai currently derives some state by interpreting frontend events. For example, the frontend observes `todo_write` tool input and mirrors it into a Zustand todo store.

This creates multiple possible sources of truth:

- isanagent SQLite todos
- Run event journal
- React event bridge
- Zustand store
- Browser persistence
- Work OS records

DeepSeek's session projection model is a strong fit for this problem.

#### Recommended Projection API

```text
session/projection/snapshot
session/projection/changed
```

Example projection:

```json
{
  "asOfSeq": 184,
  "values": {
    "run": {
      "status": "running",
      "activeToolCalls": []
    },
    "todos": [],
    "subagents": [],
    "jobs": [],
    "approvals": [],
    "artifacts": [],
    "usage": {}
  }
}
```

#### Benefits for Altai

- React receives finished state rather than reconstructing it differently.
- Reconnect begins with a consistent snapshot.
- Event gaps can be repaired.
- Todo parsing leaves the frontend.
- New Desktop, CLI, and VS Code clients share identical state.
- Zustand becomes a cache, not an authority.
- Projection versions can invalidate stale persisted state.

This is one of the highest-value DeepSeek ideas specifically for Altai App.

## P2 — Shared Capabilities

### 9. Rust-Native Capability Seams

Cordis itself should not be ported. Instead, apply its service/provider/consumer separation to Rust.

Suggested interfaces:

- `LlmRuntime`
- `SessionEventStore`
- `FileSystemProvider`
- `SubprocessProvider`
- `ProcessSandbox`
- `ApprovalService`
- `CredentialProvider`
- `JobRegistry`
- `SpillStore`
- `SessionProjectionRegistry`
- `LspProvider`
- `TerminalProvider`

The existing `altai-agent-service` is the correct host-neutral boundary. These capabilities should be composed behind it rather than introducing another daemon or agent loop.

### 10. Reuse Altai's LSP and PTY

Unlike standalone isanagent, Altai already has:

- A real persistent PTY
- Managed language-server installation
- LSP clients
- Editor integration
- Terminal output access

Therefore, LSP and PTY should not be reimplemented separately inside Altai.

Instead:

- Define `LspProvider` and `TerminalProvider` interfaces.
- Let Altai register its existing implementations with the embedded agent service.
- Allow standalone isanagent to use local fallback implementations.
- Expose agent tools such as:
  - `lsp_definition`
  - `lsp_references`
  - `lsp_hover`
  - `lsp_diagnostics`
  - `terminal_start`
  - `terminal_write`
  - `terminal_read`
  - `terminal_stop`

Access must remain workspace-scoped and permission-aware.

### 11. Session Forking and Experiment Branches

Session forking would be especially valuable for Altai's ML workflows.

Possible uses:

- Compare two models from the same context.
- Try alternative implementations.
- Run ablation branches.
- Preserve a successful baseline.
- Branch a paper-reproduction experiment.
- Compare optimization approaches.

isanagent should implement the fork semantics. Altai should provide:

- “Branch from here”
- Parent/child trajectory view
- Model comparison
- Diff between branches
- Promote branch result to a Work OS review

### 12. Message Feedback and Evaluation

DeepSeek's message-level feedback model can strengthen Altai's evaluation system.

Store feedback against stable assistant message IDs:

- Positive or negative rating
- Optional reviewer note
- Model and provider
- Prompt and session reference
- Tool trajectory
- Run and Work Attempt IDs
- Reviewer identity when available

#### Ownership

- Stable message identity: **isanagent**
- Feedback sidecar service: **Altai Agent Service**
- Feedback UI: **Altai App**
- Evaluation aggregation: **Altai Eval Lab or Work OS**

## Features Altai Already Has

The following DeepSeek ideas already have substantial Altai equivalents:

- Append-only run event journal
- Replay and reconnect recovery
- Versioned host protocol
- Permission modes
- Diff-first edit approvals
- Persistent PTY
- Managed LSP
- Background jobs
- Agent personas
- Skills
- MCP
- Checkpoints and rewind
- Worktrees
- Multi-agent orchestration
- Run inspector
- Artifact handling
- Execution and Work OS separation

These systems should be integrated with the new isanagent capabilities rather than replaced.

## Features That Should Not Be Ported Directly

Avoid directly porting:

- Cordis runtime
- Dynamic hot-reload plugin graph
- Generated browser-side executable extensions
- DeepSeek's entire web UI composition system
- Duplicate PTY or LSP implementations
- Duplicate planning, subagent, or background-job systems
- Project-management state inside isanagent
- A new control-plane daemon

Altai's accepted host-service and Work OS architecture already defines better ownership boundaries for these concerns.

## Combined Implementation Roadmap

### Phase 1 — Security

#### isanagent

- Restricted subprocess environments
- Credential grants
- `ProcessSandbox`
- Central approval protocol
- Sandbox enforcement telemetry

#### Altai App

- Permission-mode mapping
- Sandbox status UI
- Approval answerer
- Review all agent-controlled subprocess surfaces

### Phase 2 — Output and Streaming

#### isanagent

- Streaming provider contract
- Typed LLM chunks
- Spill storage
- Bounded tool scheduler

#### Altai Agent Service

- Extend the versioned event protocol
- Journal streamed events
- Transport artifact references

#### Altai App

- Incremental message rendering
- Artifact viewer integration
- Tool scheduler visualization

### Phase 3 — Durable Execution Truth

#### isanagent

- Append-only session event log
- Model-history projector
- Compaction replacement events
- Session fork and query

#### Altai Agent Service

- Map execution events to client events
- Provide projection snapshots and deltas
- Maintain replay compatibility

#### Altai App

- Replace frontend event interpretation with projections
- Make Zustand stores disposable caches
- Add trajectory and branch views

### Phase 4 — Shared Capabilities

- Rust-native capability seams
- Altai LSP provider adapter
- Altai PTY provider adapter
- Feedback service
- Eval and support-bundle exports
- Runtime invariant registry

## Final Recommendation

DeepSeek Harness should influence the system at three different levels.

### isanagent

Adopt:

- Safe execution
- Sandbox providers
- Durable model-visible event logging
- Streaming
- Structured scheduling
- Replaceable capability interfaces

### Altai Agent Service

Adopt:

- Stable transport contracts
- Projection APIs
- Approval routing
- Journal-first delivery
- Capability negotiation

### Altai App

Adopt:

- Projection-driven UI state
- Sandbox and approval visibility
- Streaming rendering
- Full-output artifact navigation
- Session branching and trajectory inspection
- Message-level feedback

The central architectural rule should be:

> isanagent owns execution truth, Altai Agent Service owns the client boundary, Altai App owns interaction, and Work OS owns cross-run work.
