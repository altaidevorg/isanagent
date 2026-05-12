---
name: synthetic-dataset-with-afterimage
description: Use when you need to generate a synthetic dataset. You can use Afterimage Python package to create, conversational, tool-calling, multi-choice questions, or other types of structured output datasets for SFT, or preference data for DPO.
---

# AfterImage
 
AfterImage is a **Python 3.11+** library and a **Click CLI** entry point named `afterimage`. It simulates a **correspondent** (user side) and **respondent** (assistant side), optionally grounded on documents, then writes **JSONL** (default) or **SQL** storage. It also allows schema-based structured output generation, tool-calling generation, and the state-of-the-art Simula method for the best quality datasets.

**When running scripts created with this skill, also refer to `long-running-execution` because synthetic dataset generation may take some time. This is *critical* to manage long-running jobs correctly.**

**Authoritative sources for consumers:**

- [https://afterimage.altai.dev](https://afterimage.altai.dev) — main documentation
- [https://afterimage.altai.dev/llms.txt](https://afterimage.altai.dev/llms.txt) — install, links, and examples oriented to agents
- [https://afterimage.altai.dev/api/index.html](https://afterimage.altai.dev/api/index.html) — API reference (matches the published package)
- [https://pypi.org/project/afterimage](https://pypi.org/project/afterimage) — version, extras, metadata
- Optional: clone the GitHub repo **only** when the user needs upstream examples (`examples/`) or is contributing patches

---

## Mental model

1. **Respondent** = model you want to imitate in training data (`respondent_prompt` / YAML `respondent.system_prompt`).
2. **Correspondent** = model that plays the user; driven by a **correspondent system prompt** and/or an **instruction generator callback** that emits first user turns and metadata (context, persona).
3. Each dialog samples an integer **turn count uniformly from `1` through `max_turns`**. Unless you need to generate a multi-turn conversational dataset, keep `max_turns=1` (default). Setting `max_turns > 1` enables back-and-forth multi-turn conversations where the correspondent handles follow-ups.
4. CLI is enabled, but for most realistic dataset generation needs it's quite simple, so prefer writing Python code, instead. However, the CLI tool is quite useful for exporting datasets to a common format that is ready to feed into other packages, e.g., unsloth uses `messages` format and you can export to `messages` format after the dataset generation is complete.
5. You can find useful examples in this skill directory, i.e., `<your_sandbox_dir>/skills/synthetic-dataset-with-afterimage/examples`. Use them as the starter. Never modify the example files under that directory so they can remain **source of truth.** Instead, you can copy one or more files that you thin are relevant to your current task, and modify it outside the examples directory.

---

## Installation

Check if it's already installed first (most likely), otherwise:

```bash
pip install afterimage
# optional extras (see PyPI “Optional extras” or llms.txt for current names):
pip install "afterimage[embeddings-local]"
pip install "afterimage[server]"
pip install "afterimage[training]"
```

* Console scripts: `afterimage` (CLI). `afterimage-server` requires the `server` extra.
* Pin versions in the user’s project (`requirements.txt` / `pyproject.toml`) if reproducibility matters.
* Afterimage Has these defaults for the LLM API:
  - **Default model provider:** "gemini". Additionally, "deepseek", "openai", "openrouter", and "local" options are also available. If the user wants to use any other OpenAI-compatible API, e.g., ollama, llama.cpp, together AI etc. set `model_provider_name="ollama"` and include the base url in `llm_factory_kwargs={"base_url": "http://localhost:11434/v1"}`
  - **Default model name:** "gemini-2.5-flash". If you change `model_provider_name` to something else, ask if the user is ok with the model name in your mind because the LLM world is moving quite fast and it's quite likely that a better model released after your knowledge cutoff.

---

## CLI workflow (no Python)

Paths in the following examples commands refer to the `examples` subdirectory that sits next to this `SKILL.md` file, e.g., `<your_sandbox_dir>/skills/synthetic-dataset-with-afterimage/examples`.

```bash
export GEMINI_API_KEY=your_key
afterimage validate -c ./configs/basic.yaml
afterimage generate -c ./configs/basic.yaml
afterimage generate -c ./configs/basic.yaml --dry-run
afterimage export -i ./output/dataset.jsonl -f sharegpt -f messages --split 0.9
afterimage export --list-formats
afterimage analyze -i ./output/dataset.jsonl -o ./reports/dataset.html
afterimage preference -c ./configs/preference.yaml
```

- Some example scripts are provided in `<your_sandbox_dir>/skills/examples`, but if they are not what you're looking for, more examples might have been released in the upstream repo’s `examples` directory, e.g., https://github.com/altaidevorg/afterimage/tree/main/examples

---

## Python API — document + persona grounding

1. Build a `DocumentProvider` (e.g. `InMemoryDocumentProvider(list[str])`) or `JSONLDocumentProvider` / `DirectoryDocumentProvider` from `afterimage.providers`.
2. **Optional:** `await PersonaGenerator(...).generate_from_documents(docs)` — mutates document objects in place with personas.
3. Pass `PersonaInstructionGeneratorCallback` (or `ContextualInstructionGeneratorCallback`) as `instruction_generator_callback`.
4. For respondent answers conditioned on the same context string, set `respondent_prompt_modifier=WithContextRespondentPromptModifier()` (exported from top-level `afterimage`).

```python
import asyncio
import os

from afterimage import (
    ConversationGenerator,
    InMemoryDocumentProvider,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
    WithContextRespondentPromptModifier,
)

DOCUMENTS = [
    "Espresso is brewed under pressure and is the base for milk drinks.",
    "A pour-over uses a filter; control grind and pour rate for extraction.",
]

async def main() -> None:
    api_key = os.environ["GEMINI_API_KEY"]
    docs = InMemoryDocumentProvider(DOCUMENTS)

    persona_gen = PersonaGenerator(api_key=api_key, model_name="gemini-2.5-flash")
    await persona_gen.generate_from_documents(docs)

    instruction_cb = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        model_name="gemini-2.5-flash",
        num_random_contexts=1,
        n_instructions=3,
    )

    gen = ConversationGenerator(
        respondent_prompt="You are a coffee educator. Ground answers in the provided context.",
        api_key=api_key,
        model_name="gemini-2.5-flash",
        instruction_generator_callback=instruction_cb,
        respondent_prompt_modifier=WithContextRespondentPromptModifier(),
    )
    await gen.generate(num_dialogs=50, max_turns=3, max_concurrency=5)

asyncio.run(main())
```

`generation` in Afterimage must provide a stop signal: either `num_dialogs` or at least one `generation.stopping` rule (see configuration reference on [Afterimage docs](https://afterimage.altai.dev/api/stopping_callbacks#stopping-callbacks)
 
---

## DPO / preference pairs

Use `afterimage.preference.generator.PreferenceGenerator` with a configured `ConversationGenerator` and `ConversationJudge`.

`ConversationJudge` is constructed with an **`LLMProvider`** and an **`EmbeddingProvider`**.

```python
from afterimage import SmartKeyPool, LLMFactory, EmbeddingProviderFactory, ConversationJudge
from afterimage.preference.types import PreferenceConfig

pool = SmartKeyPool.from_single_key(api_key)
llm = LLMFactory.create(
    provider="gemini", model_name="gemini-2.5-flash", api_key=pool
)
embedding = EmbeddingProviderFactory.create(
    {"type": "gemini", "model": "gemini-embedding-001"},
    key_pool=pool,
)

judge = ConversationJudge(
    llm=llm,
    embedding_provider=embedding,
)

pref = gen.to_preference_generator(
    judge=judge,
    config=PreferenceConfig(num_pairs=100, output_path="./out/prefs.jsonl"),
)
pairs, analytics = await pref.generate()
pref.save_pairs(pairs, analytics)
await judge.aclose()
```

Full judge wiring example: [reference.md](reference.md).

---

## Structured Output

Use `AsyncStructuredGenerator` to generate datasets where the assistant's response must conform to a specific Pydantic schema.

```python
from afterimage import AsyncStructuredGenerator
from pydantic import BaseModel

class MyOutputSchema(BaseModel):
    reasoning: str
    final_answer: str

gen = AsyncStructuredGenerator(
    output_schema=MyOutputSchema,
    respondent_prompt="You are an expert.",
    api_key=api_key,
    model_name="gemini-2.5-flash",
    instruction_generator_callback=instruction_cb,
)
await gen.generate(num_samples=10)
```

---

## Tool Calling

Use `ConversationGenerator` with `ToolCallingInstructionGeneratorCallback` and pass the tools to the generator.

```python
from typing import List, Union
from pydantic import BaseModel, Field
from afterimage import AsyncStructuredGenerator, ToolCallingInstructionGeneratorCallback

class SearchFlights(BaseModel):
    """Search for flights."""
    # ... arguments ...

class AnyToolCall(BaseModel):
    function: Union[SearchFlights]

class ToolInvocation(BaseModel):
    reasoning: str = Field(description="Reasoning for selecting the tool.")
    response: str = Field(description="The final response to the user.")
    tool_calls: List[AnyToolCall] = Field(description="A list of tool calls to execute.")

tools = [SearchFlights]

instruction_cb = ToolCallingInstructionGeneratorCallback(
    api_key=api_key,
    documents=docs,
    model_name="gemini-2.5-flash",
    tools=tools,
    n_instructions=3,
)

gen = AsyncStructuredGenerator(
    output_schema=ToolInvocation,
    respondent_prompt="You are a helpful assistant.",
    api_key=api_key,
    model_name="gemini-2.5-flash",
    instruction_generator_callback=instruction_cb,
)
await gen.generate(num_samples=10)
```

---

## More examples

The following examples (and more) can be found in `<your_sandbox_dir>/skills/synthetic-dataset-with-afterimage/examples`.

- Medical Triage Dialogue (Multi-turn Conversational): `medical_triage_multiturn.py`
- Code Review Assistant (Structured Output): `code_review_structured.py`
- Flight Booking Agent (Tool Calling): `flight_booking_tool_calling.py`
- Tone Alignment Preference Pairs (DPO): `tone_alignment_dpo.py`
- Math Word Problem Solver (Chain-of-Thought with Simula): `math_reasoning_simula.py`
- Customer support data generation with `StructuredGenerator`: `customer_support_data_with_structured_output.py`
- MCQ generation with `StructuredGenerator`: `generate_chem_mcq_with_structured_output.py`
- A more advanced way to generate MCQs with the Simula method: `<your_sandbox_dir>/skills/synthetic-dataset-with-afterimage/examples/simula/mcq_pipeline.py`
- How to generate multiple samples with the Simula method: `<your_sandbox_dir>/skills/synthetic-dataset-with-afterimage/examples/simula/corpus_batch_qa.py`
- Tool-calling dataset generation with tool schemas: `tool_calling_generator.py`
- RAG-grounded synthetic dataset generation in the legal domain: read `<your_sandbox_dir>/skills/synthetic-dataset-with-afterimage/examples/caselaw_rag/README.md`
- Simula, a reasoning-driven synthetic dataset generation and evaluation: read `<your_sandbox_dir>/skills/synthetic-dataset-with-afterimage/examples/simula/README.md`. This method may take some time, but it gives state-of-the-art results with hierarachical taxonomy building, probablistic complexification, ELO-based pruning and  double-critic evaluation.

---

## Quality gate (`auto_improve`)

With `ConversationGenerator(..., auto_improve=True)`, a judge is created automatically. This actively evaluates generated conversations and filters out low-quality ones based on default or custom criteria. For `model_provider_name="local"`, local embeddings may be required; the package raises a clear `ValueError` suggesting `pip install "afterimage[embeddings-local]"` when needed.

---

## Name alias

`AsyncConversationGenerator` is the same class as `ConversationGenerator` (re-export). Prefer `ConversationGenerator` in new code.

---

## Best practices (consumers)

- **Validate before spend:** `afterimage validate -c ...` or `afterimage generate ... --dry-run`.
- **Match provider keys:** `model_provider_name` / YAML `model.provider` must be one of `gemini`, `openai`, `deepseek`, `local`, `openrouter` (`afterimage.types.MODEL_PROVIDER_NAMES`).
- **Instruction callback on the constructor:** avoids deprecation warnings and matches CLI behavior.
- **Stable output path:** pass `JSONLStorage(conversations_path=...)` into `ConversationGenerator` when you cannot rely on default timestamped files.
- **YAML `documents.provider`:** supported values for the CLI are those documented for configs (see official docs). For in-memory text corpora in **Python**, use `InMemoryDocumentProvider`; do not assume a YAML `memory` provider exists unless the docs for your installed version say so.

---

## Finding implementation details without a repo checkout

| Goal | What to run or open |
|------|---------------------|
| Installed version | `python -c "import afterimage; print(afterimage.__version__)"` |
| Package location | `python -c "import afterimage, inspect; print(inspect.getfile(afterimage))"` |
| CLI help | `afterimage --help`, `afterimage generate --help`, … |
| Symbol signatures | `help(afterimage.ConversationGenerator)`, `inspect.signature(...)` |
| Export IDs for this install | `python -c "from afterimage.integrations import list_formats; print([x['name'] for x in list_formats()])"` |
| Deep reference | [reference.md](reference.md) |