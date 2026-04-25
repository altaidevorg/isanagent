---
name: literature-to-recipe
description: Turn papers and docs into a concrete training or evaluation recipe (datasets, metrics, hyperparameters, fine-tuning code).
always: true
---

# Literature to recipe

Goal: extract **actionable** training/eval recipes from sources, not summaries of abstracts.

## Workflow

1. **Anchor** — identify 1–3 anchor papers or official docs for the task (use `arxiv_search`, `arxiv_fetch`, `web_search`, `web_fetch`, `hf_hub_file_fetch` for Hub configs).
2. **Methodology** — read sections on data, preprocessing, model, optimization, and evaluation metrics (not the abstract alone).
3. **Extract** — write down: dataset name/version, splits, preprocessing, loss, batching assumptions, LR schedule, epochs/steps, hardware used, and reported metric **with the exact benchmark definition**.
4. **Map to workspace** — translate into files or `execution_*` steps that fit this sandbox (paths, env vars, pinned package versions).
5. **Validate data** — inspect real files in the repo (`read_file`, `glob_files`, `search_text`); never assume column names.
6. **Traceability** — cite arXiv IDs or URLs for every non-obvious choice.

Output format for the user:

- **Sources** (bullets with links or ids)
- **Recipe** (numbered steps)
- **Risks / unknowns** (what was not stated clearly in the paper)
