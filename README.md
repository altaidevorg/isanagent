# isanagent

**An always-on, agentic ML engineer for your workspace** — built by [ALTAI](https://altai.dev). isanagent doesn’t just answer prompts: it **pushes work toward something shippable** — research, code, runs, checks, and handoffs you can actually use.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

---

## Why people reach for it

You have a fuzzy goal (“fine-tune a model for this task”, “Research about new methods and apply them to my model”,  “speed up this model for inference”, “stand up a tiny LM in Flax”, “generate a preference dataset”, “figure out why this kernel is slow” ). isanagent behaves more like a **senior research engineer who owns the outcome** than a chat window: it reads the repo, hits the web and papers when your intuition is stale, runs code in a **controlled execution harness** (local Python, Jupyter, SSH, Colab MCP — depending on how you configure it), and **iterates with evidence** instead of guessing.

**Talk is cheap. So is code that never ran.** The point is deliverables: notebooks, trained or tuned models, working scripts, cleaned-up docs — and an honest story of what worked, what didn’t, and what to try next.

**Zero infra needed.** isanagent can make use of Colab for free!

---

## What it’s good at

| You want… | isanagent can… |
|------------|----------------|
| **End-to-end ML / JAX / PyTorch workflows** | Draft, run, measure, refactor — including long jobs via background execution and job polling so the agent doesn’t go silent for an hour. |
| **Fresh facts** | `web_search` / `web_fetch` and `arxiv_search` / `arxiv_fetch` so you’re not relying on a frozen snapshot of the world. |
| **Heavy notebooks & plots** | Jupyter-aware playbooks: large outputs land as **artifacts** you can open and reason about instead of drowning the chat. |
| **Parallel or staged research** | Subagents for forked investigation, with history you can audit. |
| **Structured habits** | Bundled **skills** (after `onboard`): execution research, long-running jobs, scientific Python debugging, synthetic datasets with [**Afterimage**](https://github.com/altaidevorg/afterimage), cron-style automation, skill authoring, and more — loaded on demand so context stays lean. |
| **Where you already work** | **Terminal** for a focused dev loop, **HTTP API + optional embedded UI** for browser chat, plus **Slack** and **email** when you wire them in. |
| **Multi-provider, hot-swappable models** | Configure multiple LLM providers (Gemini, OpenAI, Anthropic, DeepSeek, OpenRouter) in `config.toml` and switch between them at runtime with `/model`. Your last choice is remembered across restarts. |

---

## See it in the wild (real Colab runs)

These notebooks were produced **with isanagent**: you give the direction; it drives implementation, explains tradeoffs, and cites what it read — including **your exact prompt** at the top where asked.

### NanoLLM in Flax — tiny LM, full tutorial walkthrough

A compact language-model implementation in **Flax**, written as a **step-by-step tutorial** through the code — not a stub. The notebook **introduces itself at the top** and quotes the author’s prompt verbatim, as requested.

**[Open in Colab →](https://colab.research.google.com/drive/1ULFIwpen558pk_Eb-4PtJB6gS9ot4yWG?usp=sharing)**

### TurboQuant in JAX + Pallas — optimize, measure, explain

**TurboQuant** implemented in JAX with a **Pallas** kernel: about **3× faster encoding**, decoding unchanged — and an explanation of **why decoding didn’t speed up**, with pointers into **relevant XLA reading**. Several optimization attempts on the Pallas side, with sources called out. Same pattern: rich walkthrough, iterations you can follow, and the **exact user prompt** preserved at the top with a short self-introduction.

**[Open in Colab →](https://colab.research.google.com/drive/13M4Q7HfczdoqQL2P3HAhPZPF4ZKc-5lf?usp=sharing)**

If that’s the kind of “finish the thing and show your work” energy you want in **your** repo or notebook stack, you’re in the right place.

---

## Get started

**Fast path:** download a prebuilt binary from **[Releases](https://github.com/altaidevorg/isanagent/releases)** (Linux, **macOS** Apple silicon, and Windows), run it, and complete the first-run wizard. The embedded browser UI is baked into the binary.

### Prebuilt binary (recommended)

**One-liner (latest [`main-latest`](https://github.com/altaidevorg/isanagent/releases/tag/main-latest))** — same assets as on the release page; downloads the binary next to you, then runs it (same first-run / onboard behavior as below):

```bash
# Linux (x86_64)
curl -fsSL https://github.com/altaidevorg/isanagent/releases/download/main-latest/isanagent-linux-x86_64 -o isanagent && chmod +x isanagent && ./isanagent
```

```bash
# macOS (Apple silicon)
curl -fsSL https://github.com/altaidevorg/isanagent/releases/download/main-latest/isanagent-macos-aarch64 -o isanagent && chmod +x isanagent && ./isanagent
```

```powershell
# Windows (x86_64, PowerShell)
Invoke-WebRequest https://github.com/altaidevorg/isanagent/releases/download/main-latest/isanagent-windows-x86_64.exe -OutFile isanagent.exe; .\isanagent.exe
```

1. Or open **[Releases](https://github.com/altaidevorg/isanagent/releases)** and download the asset for your platform from **Latest main build** (tag [`main-latest`](https://github.com/altaidevorg/isanagent/releases/tag/main-latest)): `isanagent-linux-x86_64`, `isanagent-macos-aarch64`, or `isanagent-windows-x86_64.exe`.
2. On Linux or macOS, mark it executable (example): `chmod +x isanagent-linux-x86_64` or `chmod +x isanagent-macos-aarch64`.
3. Run the binary from a terminal (examples): `./isanagent-linux-x86_64` (Linux) or `./isanagent-macos-aarch64` (macOS); on Windows, run `isanagent-windows-x86_64.exe` from Explorer or `.\isanagent-windows-x86_64.exe` in PowerShell.

If you use the **default workspace** (`~/.isanagent` on Unix, or the equivalent on Windows) and that folder does not exist yet, **the first run starts the interactive onboard wizard** (provider, API key env var, model, and workspace layout), then continues into the agent in the same session. For a custom workspace path, run `isanagent onboard` (add `--interactive` for the full wizard) or `isanagent --workspace /path/to/workspace` once the directory and `config.toml` exist.

Set API credentials the wizard recommends (for example `GEMINI_API_KEY` or your provider's variable). Turn on **`[api] enabled = true`** and **`serve_ui = true`** in `config.toml` when you want the browser UI on `http://127.0.0.1:<port>/`. For channels, memory, harness options, and sandbox rules, see [`AGENTS.md`](./AGENTS.md).

### Multi-provider setup

Configure multiple providers in `config.toml` and switch between them at runtime:

```toml
[providers.gemini-2-5-flash]
provider_name = "gemini"
model_name = "gemini-2.5-flash"
api_key = "AIza..."

[providers.gpt-4o]
provider_name = "openai"
model_name = "gpt-4o"
# Uses $OPENAI_API_KEY automatically

[providers.claude-sonnet]
provider_name = "anthropic"
model_name = "claude-sonnet-4-6"
```

Use `/model` in the TUI to open the interactive model selector, or `/model gemini-2-5-flash` to switch directly. Your choice is remembered across restarts.

Provider selection is isolated per accepted run. Switching models does not alter an in-flight run; messages accepted after the switch use the new provider and credentials even if they wait in that chat's FIFO. Configured failover candidates are also snapshotted per run, so concurrent chats cannot overwrite one another's fallback policy.

### Skill management

isanagent supports installing specialized **skills** (structured procedures and instructions) from remote GitHub repositories. You can install an entire repository of skills or a specific one using shorthand `owner/repo` or full URLs.

**From the CLI:**
```bash
# Install all skills from a repository
isanagent skills add google-deepmind/science-skills

# Install a specific skill from a repository
isanagent skills add huggingface/sentence-transformers --skill train-sentence-transformers
```

**From the TUI (Slash Command):**
```text
/skills add google-deepmind/science-skills
/skills add huggingface/sentence-transformers train-sentence-transformers
```

Skills are installed directly to your workspace's `skills/` directory and are immediately available for the agent to load using the `load_skill_instructions` tool. Use `/skills list` (TUI) or `isanagent skills list` (CLI) to see what's installed.

### Build from source (optional)

From a clone of this repo, **`ui/dist` is already present**, so a normal Rust build is enough unless you edited `ui/`:

```bash
cargo build --release
./target/release/isanagent
```

To scaffold a workspace at a specific path without the default first-run flow:

```bash
cargo run --release -- onboard --workspace my_agent
# then:
cargo run --release -- --workspace my_agent
```

You only need `cd ui && npm ci && npm run build` if you are changing the frontend.

---

## Contributing

From the repo root:

```bash
cargo fmt
cargo clippy --release -p isanagent --all-targets
cargo test --release -p isanagent
```

On Windows, prefer **`--release`** for builds and tests if debug linking hits PDB issues.
