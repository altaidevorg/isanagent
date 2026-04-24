# AfterImage — consumer reference (installed package)

This file is for agents using **afterimage as a dependency** in another project. Module paths (`afterimage.conversation_generator`, …) refer to the **installed** package layout (e.g. under `site-packages/`). They match the published API documented at [https://afterimage.altai.dev/api/index.html](https://afterimage.altai.dev/api/index.html).

**Primary docs:** [https://afterimage.altai.dev](https://afterimage.altai.dev) · **llms.txt:** [https://afterimage.altai.dev/llms.txt](https://afterimage.altai.dev/llms.txt) · **PyPI:** [https://pypi.org/project/afterimage](https://pypi.org/project/afterimage)

---

## INSTALLED-PACKAGE

| Question | Consumer action |
|----------|-----------------|
| What version is this environment? | `python -c "import afterimage; print(afterimage.__version__)"` |
| Where is the code on disk? | `python -c "import afterimage, inspect; print(inspect.getfile(afterimage))"` |
| What extras exist? | Read PyPI “Optional extras” or `llms.txt` for the version you installed |
| Entry points | `afterimage` (CLI), `afterimage-server` (with `server` extra) |

To browse source: open the file path printed by `inspect.getfile` in your editor, or use `help(module.Class)`.

---

## MODULE-CLI

The CLI is the Click application in `afterimage.cli` (console script `afterimage`).

| Subcommand | Purpose |
|------------|---------|
| `generate` | `-c/--config` YAML path; `--dry-run` prints plan only |
| `validate` | Validate YAML |
| `export` | `-i` input JSONL; `-f/--format` repeatable; `--list-formats`; `--split` |
| `analyze` | Dataset HTML report |
| `preference` | DPO-style pairs from config; `--dry-run`, `--num-pairs`, etc. |

**Export format names** depend on the installed version. List them reliably:

```bash
python -c "from afterimage.integrations import list_formats; print([x['name'] for x in list_formats()])"
```

A typical install includes: `alpaca`, `dpo`, `llama_factory`, `messages`, `openai`, `oumi`, `raw`, `sharegpt` — always confirm with the command above.

---

## MODULE-MODEL-PROVIDERS

`afterimage.types.ModelProviderName` / `MODEL_PROVIDER_NAMES`:

`gemini`, `openai`, `deepseek`, `local`, `openrouter`

---

## MODULE-CONVERSATION-GENERATOR

`afterimage.conversation_generator.ConversationGenerator` (also re-exported as `afterimage.async_conversation_generator.AsyncConversationGenerator` — same class).

### Constructor (`__init__`)

| Parameter | Type / notes |
|-----------|----------------|
| `respondent_prompt` | `str` — system prompt for the assistant-side model |
| `api_key` | `str \| SmartKeyPool` |
| `correspondent_prompt` | `str \| None` — **required** unless `instruction_generator_callback` is set |
| `model_name` | `str \| None` — package default if omitted (`afterimage.common.default_model_name`) |
| `safety_settings` | `list[dict[str, str]] \| None` |
| `auto_improve` | `bool` — builds `ConversationJudge` when `True` |
| `evaluator_model_name` | `str \| None` — judge LLM when `auto_improve` |
| `model_provider_name` | `ModelProviderName`, default `"gemini"` |
| `llm_factory_kwargs` | `dict[str, Any] \| None` — e.g. `{"base_url": "..."}` for local/OpenAI-compatible servers |
| `embedding_provider` | optional `EmbeddingProvider` |
| `embedding_provider_config` | `dict \| None` |
| `judge_config` | `ConversationJudgeConfig \| None` |
| `storage` | `BaseStorage \| None` — default `JSONLStorage()` |
| `monitor` | `GenerationMonitor \| None` |
| `instruction_generator_callback` | `BaseInstructionGeneratorCallback \| None` |
| `respondent_prompt_modifier` | `BaseRespondentPromptModifierCallback \| None` |
| `turn_hooks` | `ConversationTurnHooks \| None` |

**Hard rule:** at least one of `correspondent_prompt` or `instruction_generator_callback` must be non-`None` or `__init__` raises `ValueError`.

### `generate`

Signature and semantics: use `help(ConversationGenerator.generate)` on the installed package. Summary:

- An `instruction_generator_callback` must be available (constructor preferred over deprecated `generate(..., instruction_generator_callback=...)`).
- `max_turns`: per-dialog cap; actual assistant turns per dialog are sampled uniformly from `1` through `max_turns` inside generation logic.
- `num_dialogs`: when set, appends a fixed-count stopping callback.
- If no stopping criteria and no `num_dialogs`, the implementation supplies a default fixed count.

### Loading results

`ConversationGenerator.load_conversations(limit=None, offset=None)` delegates to `storage.load_conversations` (`afterimage.base.BaseGenerator`).

---

## MODULE-INSTRUCTION-CALLBACKS

| Class | Import location | Role |
|-------|-----------------|------|
| `SimpleInstructionGeneratorCallback` | `afterimage.callbacks` | No document context; `n_instructions` per round |
| `ContextualInstructionGeneratorCallback` | `afterimage.callbacks` | Samples `num_random_contexts` docs |
| `PersonaInstructionGeneratorCallback` | `afterimage.callbacks` | Contextual + persona sampling |

**PersonaInstructionGeneratorCallback:** `api_key`, `documents: list[str] | DocumentProvider`, optional `prompt`, `model_name`, `model_provider_name`, `num_random_contexts=1`, `n_instructions=3`, `separator_text`, `safety_settings`, `monitor`, `llm_create_extras`.

---

## MODULE-DOCUMENT-PROVIDERS

Concrete classes live under `afterimage.providers` (see `afterimage.providers.document_providers` on disk).

| Class | Key constructor arguments |
|-------|---------------------------|
| `InMemoryDocumentProvider` | `texts: list[str \| Document]`, optional `target_context_usage_count` |
| `DirectoryDocumentProvider` | `directory`, optional `file_patterns`, `encoding`, `recursive`, `min_length`, `cache`, `target_context_usage_count` |
| `FileSystemDocumentProvider` | `path_pattern` glob, `encoding`, `recursive`, `min_length`, `cache`, `target_context_usage_count` |
| `JSONLDocumentProvider` | `path_pattern`, `content_key="text"`, `encoding`, `recursive`, `cache`, `max_docs`, `target_context_usage_count` |
| `QdrantDocumentProvider` | `client`, `collection_name`, `content_key="text"`, `batch_size`, … |

**CLI/YAML:** document providers are whatever your **installed** version’s config loader supports; check [afterimage.altai.dev](https://afterimage.altai.dev) for the `documents.provider` values valid in YAML. For arbitrary in-memory strings, use **`InMemoryDocumentProvider` in Python** rather than inventing YAML keys.

---

## MODULE-PERSONA-GENERATOR

`afterimage.persona_generator.PersonaGenerator`:

`async def generate_from_documents(self, documents: DocumentProvider | list[str], max_docs=None, n_iterations=None, target_data_count=None, num_random_contexts=1)`

- `list[str]` is wrapped internally with `InMemoryDocumentProvider`.
- Mutates document objects in place with persona metadata.

---

## MODULE-STORAGE

| Class | Notes |
|-------|--------|
| `JSONLStorage` | `afterimage.storage.JSONLStorage` — `conversations_path`, `documents_path`, `encoding`, `lock_timeout`; defaults may use timestamped filenames and optional `AFTERIMAGE_JSONL_DIR` env |
| `SQLStorage` | `afterimage.storage.SQLStorage` — `__init__(url, conversations_table_name="conversations", documents_table_name="documents", metadata_fields=None, batch_size=100)` — requires `sqlalchemy` |

YAML `output.storage`: typically `jsonl` or `sql` per docs for your version; SQL URL is usually taken from `output.path` in config-driven runs.

---

## MODULE-KEY-POOL

`afterimage.key_management.SmartKeyPool`:

`__init__(api_keys: list[str], hourly_limit=None, daily_limit=None, error_threshold=1000000, cooldown_period=600)`  
`SmartKeyPool.from_single_key(api_key: str)`

---

## MODULE-EVALUATOR

`afterimage.evaluator.ConversationJudge`

- `__init__(self, llm: LLMProvider, embedding_provider: EmbeddingProvider, monitor=None, *, config=None)`
- `from_factory(cls, llm, *, key_pool, model_provider_name, embedding_provider_config=None, monitor=None, config=None)`

`ConversationJudgeConfig`: thresholds such as `min_acceptable_score`, `aggregation_mode`, `metric_weights`, grade band fields.

`default_embedding_provider_config(model_provider_name)` returns a dict for `EmbeddingProviderFactory.create` — see docstring on the installed module for provider-specific defaults.

---

## MODULE-PREFERENCE

`afterimage.preference.generator.PreferenceGenerator`

Constructor: `conversation_generator`, `judge`, `config: PreferenceConfig | None`, `secondary_llm_provider` (required for some strategies).

- `async def generate(...)` → `(list[PreferencePair], PreferenceAnalytics)`
- `def save_pairs(pairs, analytics=None)` — uses `PreferenceConfig.output_path`, `output_format`, `save_log`

`PreferenceConfig` (`afterimage.preference.types`): `num_pairs`, `num_responses`, `min_score_gap`, `strategy`, `secondary_model`, `multi_turn`, `max_concurrency`, `output_format`, `output_path`, `save_log`, `log_path`.

YAML `preference` blocks are validated by the package’s config models and merged into `PreferenceConfig` when using the CLI.

### Complete Python example (aligned with CLI preference flow)

```python
import asyncio
import os

from afterimage import ConversationGenerator
from afterimage.callbacks import SimpleInstructionGeneratorCallback
from afterimage.evaluator import ConversationJudge, default_embedding_provider_config
from afterimage.key_management import SmartKeyPool
from afterimage.preference.generator import PreferenceGenerator
from afterimage.preference.types import PreferenceConfig
from afterimage.providers import LLMFactory

async def main() -> None:
    api_key = os.environ["GEMINI_API_KEY"]
    pool = SmartKeyPool.from_single_key(api_key)

    instruction_cb = SimpleInstructionGeneratorCallback(
        api_key=pool,
        model_name="gemini-2.5-flash",
        model_provider_name="gemini",
    )
    gen = ConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        api_key=pool,
        model_name="gemini-2.5-flash",
        model_provider_name="gemini",
        instruction_generator_callback=instruction_cb,
    )

    judge_llm = LLMFactory.create(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key=pool,
    )
    embed_cfg = default_embedding_provider_config("gemini")
    judge = ConversationJudge.from_factory(
        judge_llm,
        key_pool=pool,
        model_provider_name="gemini",
        embedding_provider_config=embed_cfg,
    )

    pref = PreferenceGenerator(
        conversation_generator=gen,
        judge=judge,
        config=PreferenceConfig(
            num_pairs=10,
            output_path="./output/preferences.jsonl",
            output_format="dpo",
        ),
    )
    pairs, analytics = await pref.generate()
    pref.save_pairs(pairs, analytics)
    await judge.aclose()
    print(f"pairs: {len(pairs)}, valid: {analytics.total_valid}")

asyncio.run(main())
```

---

## MODULE-YAML-CONFIG

High-level: `afterimage.config.load_config(path)` returns a validated `AfterImageConfig`.

Cross-field rules (always confirm against docs for your version), commonly:

- `personas.enabled` requires a `documents` section.
- If `documents` is set, `context.enabled` must be compatible with grounded generation (see official configuration reference).
- Some `generation.stopping` rules require `documents` or `personas.enabled`.

Details: [https://afterimage.altai.dev](https://afterimage.altai.dev) configuration section, or `help(afterimage.config.AfterImageConfig)` if available.

---

## MODULE-CONFIG-TO-GENERATOR

`afterimage.config_to_generator`:

| Function | Role |
|----------|------|
| `build_conversation_run(config)` | `BuiltConversationRun` with `generator`, `stopping_criteria`, `num_requested` |
| `build_generator(config)` | `ConversationGenerator` only |

Useful if the user’s app loads YAML and needs the same wiring as the CLI without shelling out.

---

## Gemini Python client

Google’s Gen AI SDK for Python: [https://googleapis.github.io/python-genai/](https://googleapis.github.io/python-genai/) (AfterImage depends on this stack for Gemini; consult when debugging API usage outside AfterImage’s wrappers).

---

## Version drift

Always treat **`afterimage.__version__`** and the output of `list_formats()` as ground truth for the **current environment**. The tables above follow the public API of the PyPI package; if behavior differs between versions, prefer the docs matching the installed version or the source file shown by `inspect.getfile`.
