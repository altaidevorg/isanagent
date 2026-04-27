# isanagent

**An always-on, agentic ML engineer for your workspace** — built by [ALTAI](https://altai.dev). isanagent doesn’t just answer prompts: it **pushes work toward something shippable** — research, code, runs, checks, and handoffs you can actually use.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache 2.0-blue.svg)](LICENSE)

---

## Why people reach for it

You have a fuzzy goal (“speed up this encoder”, “stand up a tiny LM in Flax”, “generate a preference dataset”, “figure out why this kernel is slow”). isanagent behaves more like a **senior research engineer who owns the outcome** than a chat window: it reads the repo, hits the web and papers when your intuition is stale, runs code in a **controlled execution harness** (local Python, Jupyter, SSH, Colab MCP — depending on how you configure it), and **iterates with evidence** instead of guessing.

**Talk is cheap. So is code that never ran.** The point is deliverables: notebooks, trained or tuned models, working scripts, cleaned-up docs — and an honest story of what worked, what didn’t, and what to try next.

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

## Get running in a few commands

**1. Build the embedded UI** (the Rust binary embeds `ui/dist` at compile time):

```bash
cd ui && npm ci && npm run build && cd ..
```

**2. Build isanagent**

```bash
cargo build --release
```

**3. Scaffold a workspace** (config, sandbox, starter `AGENTS.md` / `SOUL.md`, and skills):

```bash
cargo run -- onboard --workspace my_agent
```

**4. Set your provider** (see `my_agent/config.toml` — e.g. Gemini/OpenAI-compatible URL + `GEMINI_API_KEY` or your provider’s env var).

**5. Run**

```bash
cargo run -- --workspace my_agent
```

Turn on **`[api] enabled = true`** and **`serve_ui = true`** when you want the browser UI on `http://127.0.0.1:<port>/`. For channel details, memory reflection, harness options, and sandbox rules, the **authoritative architecture note** is [`AGENTS.md`](./AGENTS.md) in this repo; your onboarded workspace mirrors the same ideas locally.

---

## Contributing

From the repo root:

```bash
cargo fmt
cargo clippy --release -p isanagent --all-targets
cargo test --release -p isanagent
```

On Windows, prefer **`--release`** for builds and tests if debug linking hits PDB issues.