---
name: gliner2
description: Use GLiNER2 for local or API-based entity extraction, relation extraction, structured JSON extraction, classification, combined schemas, batching, and long documents; also develop, debug, review, and validate the GLiNER2 Python repository when its source is available. Use when an agent needs to design GLiNER2 schemas, write extraction code, interpret outputs, choose local versus API inference, or change GLiNER2 implementation, training, LoRA/PEFT, packaging, compatibility, tests, tutorials, or releases.
---

# GLiNER2 Repository

## Choose the workflow

- For using GLiNER2 as an installed library or API, read [references/usage.md](references/usage.md) completely. It is self-contained and does not require this repository.
- For modifying or reviewing GLiNER2 source, follow the repository workflow below and consult the reference only when public API behavior is relevant.
- If the installed package differs from the reference, inspect its version and callable signatures. Do not assume repository-internal modules are available to a library consumer.

## Verify consumer code before presenting it

- Distinguish quick APIs from schema-builder APIs. Their accepted dictionary shapes are not interchangeable. In particular, pass `multi_label=` and `cls_threshold=` as keyword arguments to `Schema.classification`; do not pass a quick-API `{labels, multi_label}` configuration as the builder's `labels` argument.
- Keep structures flat. Call `structure(name)` with only the structure name, represent repeated records as sibling structures, and chain the next schema method directly. Never call `structure(..., dtype=...)`, nest one structure inside another, or call a nonexistent `end_structure()` method.
- Request every metadata field later consumed. Code that reads `start` or `end` must set `include_spans=True`; code that reads `confidence` must set `include_confidence=True`.
- Check capabilities on the actual extractor object. Do not call local-only long-document methods on the hosted API client unless the installed version exposes them.
- Before long-document work, record `gliner2.__file__`, `gliner2.__version__`, and the four long-method capability checks. The PyPI `1.3.2` wheel can lack methods present in repository source carrying the same version. If any required long method is absent, install the pinned repository source in an isolated environment as described in the usage reference; do not monkeypatch methods onto the class and report that as native support.
- Match implementation to scale requirements. If the request requires processing multiple long documents or explicitly asks for batching, the representative code must call `batch_extract_long` or `batch_extract_entities_long`; merely mentioning batching while calling a single-document method is insufficient.
- Treat batching as an execution property, not a method name. A wrapper that loops over documents sequentially or accepts but does not use `batch_size` is not native batch inference; report it separately as sequential fallback behavior.
- Prefer public exception classes and the client's built-in retry behavior over parsing exception strings or layering retries blindly.
- Test hosted retry behavior at the adapter/transport layer or inspect its configured retry policy. Monkeypatching `Session.post` bypasses adapter retries and cannot prove retry behavior, although it can still prove that no real request escaped.
- Inspect or build schemas without loading a model when possible. A model-free schema build should confirm task labels, `multi_label`, thresholds, structures, and choices.
- Never invent a package or model version. Report the installed version or use an explicit placeholder that the operator must replace.
- Do not assume Hugging Face-style kwargs are honored by GLiNER2. In the repository version covered by this skill, `from_pretrained(..., revision=...)` does not forward `revision` to downloads. For reproducible model pinning, use a verified local snapshot path and record its immutable revision/hash unless the installed implementation is proven to support revision forwarding.
- Call code “representative” until it has been executed. Report exactly what was and was not run; never claim “production-ready” from static reasoning alone.
- Treat span equality as offset-integrity validation, not semantic correctness. Validate extracted values and task quality separately.
- Score evaluations strictly: assign one status per scenario, require every explicit output and assertion for `PASS`, and use `PARTIAL` when plumbing works but a required field or behavior is missing. Do not double-count subchecks in totals. Report API/implementation validity separately from semantic model quality.
- Never log secrets, source documents, extracted sensitive spans, or review messages containing those values. Local inference reduces data transfer but does not by itself establish regulatory compliance.
- Keep extraction-completeness checks separate from mandatory human-review policy. If every result requires clinician, legal, or other expert review, do not set `requires_review` only when extraction is empty.

Before delivering consumer code, run a model-free preflight when the package is available:

1. Construct every schema and call `schema.build()`.
2. Assert classification labels, `multi_label`, thresholds, structure names, fields, and choices.
3. Inspect every called method with `hasattr` or `inspect.signature` when capability may vary by version or extractor type. For long-context work, also record the imported package path and source commit; a version string alone is insufficient.
4. Scan code and logs for invented versions, fabricated timestamps/results, secrets, source text, and extracted sensitive values.
5. Confirm that mocked tests inject the mock actually used by the function; avoid globals created before patching.
6. Confirm every requested task is represented by the right GLiNER2 task type: a requested relation must use `relations`, not an entity label containing the word “relation.”
7. Check empty outputs with content, not mapping truthiness: use `any(items for items in entities.values())` when requested entity keys may map to empty lists.
8. Recursively validate every span-bearing object, including entity items, relation `head`/`tail` endpoints, and structured fields.

## Work from repository evidence

- Read the affected implementation, its public re-exports, nearby tests, and the corresponding tutorial before changing behavior.
- Treat `README.md` and `tutorial/` as user-facing contracts, but verify claims against code and tests.
- Preserve unrelated work in the tree. Use `rg` for discovery and `apply_patch` for edits.
- Prefer the smallest coherent change. Update tests and documentation when public behavior changes.

## Respect the architecture

- Keep the base `gliner2` import torch-free. `gliner2/__init__.py` eagerly exposes schemas and the HTTP client but lazy-loads local model and LoRA symbols through `__getattr__`. Do not introduce eager imports of `torch`, `transformers`, `peft`, or local-model modules on the base-import path.
- Put public schema construction and validation in `gliner2/inference/schema.py` and Pydantic request models in `gliner2/inference/schema_model.py`.
- Put high-level extraction orchestration in `gliner2/inference/engine.py`, core neural-model behavior and serialization in `gliner2/model.py`, and tokenization/batch preparation in `gliner2/processor.py`.
- Keep long-document token boundaries, chunk overlap, global-offset remapping, and deduplication behavior in `gliner2/inference/chunking.py` or the matching engine entry points.
- Keep training examples and datasets in `gliner2/training/data.py`, trainer/configuration behavior in `gliner2/training/trainer.py`, and PEFT compatibility logic in `gliner2/training/lora.py`.
- Keep cloud behavior isolated in `gliner2/api_client.py`. Tests must mock HTTP and must not require credentials or live services.

## Protect public contracts

- Preserve result shapes for entity, relation, structure, and classification extraction, including `include_confidence` and `include_spans` variants.
- Preserve character-span correctness: returned text must equal `source[start:end]`. Long-context APIs return offsets into the original document, not a chunk.
- Compare optimized batch paths with single-example behavior. Account for mixed lengths, padding, masks, truncation, and per-sample span counts.
- Preserve `max_len` behavior in both inference and the training collator.
- Treat symbols exported from `gliner2/__init__.py`, method signatures, serialized model files, and adapter directories as compatibility surfaces.
- Treat legacy LoRA APIs as deprecated compatibility shims unless the task explicitly removes them. Preserve warning categories, adapter config fields, filenames, weight-key translation, round trips, and numerical parity covered by compatibility fixtures.
- When changing training, check gradient accumulation, evaluation/checkpoint cadence, rank-zero-only writes, distributed samplers, process-group handling, and wrapped versus unwrapped model state.

## Validate proportionally

Run focused tests first, then broaden when shared code changes. Use the active environment's Python/pytest command; do not assume a particular environment manager.

- Schema or extraction output: `tests/test_entity_extraction.py`, `tests/test_relation_extraction.py`, `tests/test_structure_extraction.py`.
- Batching, masks, or spans: `tests/test_batching_correctness.py`, `tests/test_batch_span_mask_trim.py`.
- Context limits: `tests/test_inference_max_len.py`, `tests/test_train_max_len.py`.
- API client: `tests/test_api_client_error_handling.py`.
- Imports, dependencies, or package surface: `tests/test_torch_free_import.py` and the public-surface checks in `tests/test_backwards_compat.py`.
- LoRA/PEFT, model loading, or serialization: `tests/test_lora_peft.py`, `tests/test_backwards_compat.py`.
- Trainer logic: `tests/test_trainer_distributed.py`; add `tests/test_trainer_distributed_integration.py` when multiprocessing/distributed behavior changes.
- End-to-end learning behavior: `tests/test_overfit_ner.py` only when the change can affect optimization or task learning.

Expect some inference and overfitting tests to download pretrained models or require substantial compute. Inspect markers and fixtures before running them, report any unrun tests explicitly, and never weaken assertions merely to avoid those requirements.

## Keep documentation and releases aligned

- Update the matching file in `tutorial/` and concise examples in `README.md` for public API changes.
- Maintain Python 3.8+ compatibility declared in `pyproject.toml` unless the task explicitly changes support.
- For releases, follow `RELEASE.md`, update `gliner2/__init__.py::__version__`, validate the build, and never upload to PyPI or create a remote release without explicit authorization.

## Finish with evidence

Summarize changed behavior, compatibility implications, tests run and their results, and any validation skipped because it requires model downloads, GPUs, distributed resources, or external credentials.
