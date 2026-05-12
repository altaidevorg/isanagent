# OpenSimula examples

These scripts exercise **OpenSimula** (`afterimage.simula`): an open implementation of ideas from Davidson et al., *Reasoning-Driven Synthetic Data Generation and Evaluation* (TMLR; [OpenReview PDF](https://openreview.net/pdf?id=NALsdGEPhB)), aligned with the mechanism-design framing in Google’s [research blog](https://research.google/blog/designing-synthetic-datasets-for-the-real-world-mechanism-design-and-reasoning-from-first-principles/). They are **not** a Google reference implementation.

## Model choice

Examples use **`gemini-2.5-flash`** as the default teacher: fast and inexpensive with acceptable quality for taxonomy expansion, strategies, meta-prompts, and critics. The paper’s downstream experiments used **Gemini 2.5 Flash** (non-thinking) as the teacher model; matching that family keeps behavior closer to the reported setup while staying cheap for local runs.

## How each knob maps to the paper

| Parameter / API | Paper role | What it controls | Sensible default |
|-------------------|------------|------------------|------------------|
| `instruction_y` | **y** | Global dataset intent: what domain, tone, and constraints the synthetic data should satisfy (§2.1–2.2). | One tight paragraph: audience, task, and non-goals. |
| `document_provider` (optional) | **S** | Reference material from the target domain so factors and taxonomies are grounded (§2.1: “and/or a sample S”). Use bounded excerpts (the library caps size in `document_context.py`). | Real policy snippets, product docs, or rubrics—not toy one-liners. |
| `target_depth_D` | **D** | Maximum taxonomy depth per factor; deeper trees sharpen **global** coverage control but cost more and risk missing branches (Fig. 1c, Appendix B.4). | **2–3** for prototypes; **4+** only when you need fine leaves and accept cost. |
| `proposal_N` | **N** (Best-of-N) | Independent child proposals per node before the critic merges them (Appendix B.4). Higher **N** widens the proposal distribution toward edge cases. | **2–4**; increase if leaves look repetitive. |
| `build_taxonomy(..., show_progress=…)` | — | When **True**, prints nested **tqdm** progress for factor proposal and per-factor BFS; demotes duplicate `afterimage.simula` INFO logs to DEBUG. Pair with `configure_example_console()`. Default **False**. | **True** in examples. |
| `build_taxonomy(..., max_factors=…)` | — | The model’s factor list is truncated to this count (each factor runs a full BFS tree). Previously uncapped lists caused **hundreds** of sequential API calls and looked like a hang. Default **4**. | **3–4** for interactive runs. |
| `max_children_per_node` | — | After the critic, at most this many children are kept per parent (limits breadth). Default **8**. | **6–10**; lower if `D` is large. |
| `max_frontier_per_depth` | — | At each depth, at most this many frontier nodes are expanded per factor (limits wide BFS waves). Default **16**. | **12–16** for local demos. |
| `OpenSimula(..., temperature=…)` | — | Stochasticity for all structured LLM steps in this facade (taxonomy, strategies, scenarios, critics). Lower = more deterministic trees and stricter judges. | **0.35–0.45** for reproducible runs; raise slightly if diversity stalls. |
| `infer_strategies` → `sample_mix` | §2.2 **sampling strategies** | Strategies define **which factors are jointly sampled** and with what weights, avoiding incompatible mixes. `sample_mix` draws one **mix** (a tuple of taxonomy nodes = “requirements”). | Always run `infer_strategies` once per bundle (or hand-author `SamplingStrategySpec` for full control). |
| `draw_meta_prompt(..., K=…, complexify_c=…, sequential=…)` | **Local diversity** + **c** | **K** distinct **meta-prompts** (scenarios) for the same mix; one is subsampled (Algorithm 2). **complexify_c** is the probability **c** of complexifying that meta-prompt—orthogonal difficulty vs. coverage (§2.2). **sequential=True** uses the paper’s large-**N/V** regime: generate scenarios one-by-one with prior attempts in context to reduce mode collapse. | **K = 4–8**. **c**: the paper’s **Local** ablation uses **c = 0.5** (Table 1); that is aggressive. Examples use **~0.25–0.35** unless you want maximum difficulty skew. |
| `generate_*_datapoint` | Alg. 2 **generate + critic** | Requirement critic and optional refine loop; MCQ path adds **double-critic** (§2.2, §3.1) after the point satisfies the meta-prompt. | Leave `max_refine_rounds` at **4** unless traces show chronic rejection. |

**Evaluation helpers** (not run in the minimal one-row scripts): `assign_datapoint_to_taxonomy`, `level_ratio_coverage`, and `elo_complexity_scores` implement §2.3 / Appendix E style signals. For Elo-style batch complexity, the paper reports a practical tradeoff around **batch size and repeat count ≈ 5** (Appendix E; “BS = N = 5”); pass `batch_size=5`, `repeats=5` into `elo_complexity_scores` when you wire evaluation.

**Scale note:** Experiments in the paper generated on the order of **512k** points per domain; the two minimal scripts generate **one** row each to keep cost predictable. [`corpus_batch_qa.py`](corpus_batch_qa.py) generates several rows and appends accepted points to JSONL.

**If the script “stalls” after `Simula created`:** it is usually inside `build_taxonomy`. The first structured Gemini call can take **15–60s**; the rest are logged at **INFO** by `afterimage.simula.taxonomy_builder`. Enable logging (as in `minimal_pipeline.py`) or watch stderr. Before cost caps, wide taxonomies issued **hundreds** of sequential calls (tens of minutes).

**Clean console output:** the examples call `configure_example_console()` from `afterimage.simula.cli_logging`, which sets root/`afterimage.simula` log levels and **mutes `httpx` / `google_genai`** INFO spam. Pass **`show_progress=True`** to `OpenSimula.build_taxonomy()` for **tqdm** bars (per-factor nested progress during BFS). Library status lines move to **DEBUG** while tqdm is active so you do not get duplicate noise.

---

## Runnable scripts

All require `GEMINI_API_KEY` (or change `LLMFactory.create` to your provider).

- [`minimal_pipeline.py`](minimal_pipeline.py) — **y** + optional **S** → taxonomy → strategies → mix → meta-prompt → **one** single-QA row (smallest linear demo).
- [`mcq_pipeline.py`](mcq_pipeline.py) — taxonomy without docs → **one** four-option MCQ with **double-critic** after the requirement loop (§3.1).
- [`corpus_batch_qa.py`](corpus_batch_qa.py) — **six** static policy excerpts, **multi-sample** single-QA generation with bounded concurrency, **`opensimula/`** checkpoint + `run_config.json`, incremental **`data/train.jsonl`** (accepted rows only), optional **`--resume`** and **`--push-hf`**.

### Checkpoints, JSONL, and Hub (batch example)

[`corpus_batch_qa.py`](corpus_batch_qa.py) writes a versioned **`manifest.json`** under `<output-dir>/opensimula/` (`producer`: `afterimage`, `format`: `opensimula`, `format_version`: `1.0`) plus taxonomy and strategy JSON. **`run_config.json`** is typed as **`OpenSimulaRunConfig`** (optional `name` / `description`, hyperparameters, paths). Older files that used the legacy **`example`** string are still loaded: that value becomes **`name`** when `name` is absent. Accepted datapoints are appended one line at a time to **`data/train.jsonl`** as each async sample completes (so a crash mid-run still keeps prior rows).

| Flag | Role |
|------|------|
| `--output-dir DIR` | Run root (default `outputs/simula_corpus_batch`). Holds `opensimula/` and `data/train.jsonl`. |
| `--num-samples N` | Number of independent (mix → meta → generate) pipelines. |
| `--resume` | Expect an existing `opensimula/` under `output-dir`; skip taxonomy and `infer_strategies`, then only run sample generation. |
| `--max-concurrency K` | Cap concurrent sample pipelines (each still does full LLM work). |
| `--push-hf REPO_ID` | After a **fresh** save (no `--resume`), upload `opensimula/` to the dataset repo. Needs `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`). Also uploads **`README.md`** at the repo root: auto-generated (YAML `tags` + intro + links) unless you pass `dataset_card=` from Python via `Checkpointer.push_to_hub`. |

```bash
python corpus_batch_qa.py --output-dir ./runs/corpus1 --num-samples 8 --max-concurrency 2

# Reuse taxonomy + strategies; append more JSONL lines
python corpus_batch_qa.py --output-dir ./runs/corpus1 --resume --num-samples 4
```

Library helpers: `Checkpointer` / `save_checkpoint`, `OpenSimulaRunConfig` (typed `run_config.json`), `OpenSimula.aiter_single_qa_samples` / `agenerate_single_qa_samples`, `append_datapoints_jsonl`, `load_checkpoint`, `pull_checkpoint_from_hub`, `push_checkpoint_to_hub`.

To download a Hub repo that only contains `opensimula/` (or that subtree) and then resume:

```python
from pathlib import Path

from afterimage.simula import load_checkpoint, pull_checkpoint_from_hub

root = Path("./from_hub")
pull_checkpoint_from_hub("username/secqa-opensimula", root)
ckpt = load_checkpoint(root)
# ckpt.bundle, ckpt.sampling_strategy, ckpt.run_config, ckpt.manifest
```

---

## Inline sketch (minimal single-row API)

```python
import asyncio
import os

from afterimage.providers import InMemoryDocumentProvider, LLMFactory
from afterimage.simula import OpenSimula


async def main():
    api_key = os.environ["GEMINI_API_KEY"]
    llm = LLMFactory.create(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key=api_key,
    )
    docs = InMemoryDocumentProvider(["""... realistic excerpt ..."""])
    sim = OpenSimula(llm, temperature=0.4)
    bundle = await sim.build_taxonomy(
        "... realistic y ...",
        document_provider=docs,
        target_depth_D=3,
        proposal_N=3,
    )
    OpenSimula.validate_taxonomy_bundle(bundle)
    spec = await sim.infer_strategies(bundle)
    mix = sim.sample_mix(bundle, spec)
    meta = await sim.draw_meta_prompt(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        K=6,
        complexify_c=0.3,
    )
    row = await sim.generate_single_qa_datapoint(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
    )
    print(row.model_dump(mode="json") if row else "rejected")


if __name__ == "__main__":
    asyncio.run(main())
```

Batch + JSONL (see [`corpus_batch_qa.py`](corpus_batch_qa.py)):

```python
# from pathlib import Path
# from afterimage.simula import Checkpointer, append_datapoints_jsonl
# with Checkpointer(out_dir) as cp:
#     bundle.save(cp); spec.save(cp); cp.write_run_config(OpenSimulaRunConfig(name="my-run"))
# async for _, rec in sim.aiter_single_qa_samples(..., n=10, max_concurrency=2):
#     if rec:
#         append_datapoints_jsonl(out_dir / "data" / "train.jsonl", [rec])
```

---

## Multi-turn (third pattern)

`SimulaInstructionGeneratorCallback` feeds **one first user message per dialog** into `ConversationGenerator` (see `conversation_generator.py`: one `acall` per dialog). Align `num_dialogs` with the number of `(instruction, metadata)` tuples.

Use **realistic** first turns: role, stakes, and what they already tried—so the respondent model gets a believable support or advisory thread.

```python
from afterimage import AsyncConversationGenerator
from afterimage.simula import SimulaInstructionGeneratorCallback

scenarios = [
    (
        "You are a SOC analyst on shift. You just escalated a possible BEC thread; "
        "the mailbox shows mixed legitimate invoices and one odd PDF request. "
        "Ask the security assistant what to verify first and why.",
        {"incident_class": "BEC", "urgency": "P2"},
    ),
    (
        "You are an HR business partner drafting parental leave for an EU employee. "
        "Ask the assistant for the minimum statutory angles you must not get wrong.",
        {"domain": "leave_policy", "region": "EU"},
    ),
]
callback = SimulaInstructionGeneratorCallback(scenarios)
# generator = AsyncConversationGenerator(
#     respondent_prompt="You are a senior security and compliance advisor...",
#     api_key=api_key,
#     model_name="gemini-2.5-flash",
#     instruction_generator_callback=callback,
# )
# await generator.generate(num_dialogs=len(scenarios), max_turns=4, max_concurrency=2)
```
