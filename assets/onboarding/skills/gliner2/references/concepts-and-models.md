# GLiNER2 concepts and model selection

## Contents

- [What GLiNER2 is](#what-gliner2-is)
- [Task model](#task-model)
- [Choose a model](#choose-a-model)
- [Language evidence](#language-evidence)
- [Choose local or hosted inference](#choose-local-or-hosted-inference)
- [Choose a device](#choose-a-device)
- [Evaluate before deployment](#evaluate-before-deployment)

## What GLiNER2 is

GLiNER2 is a compact, encoder-based, schema-conditioned information-extraction system. It extends GLiNER so that one bidirectional Transformer model can perform named-entity recognition, text classification, structured extraction, and directional relation extraction. A schema supplies the labels, field names, descriptions, choices, and task composition at inference time.

This is not free-form generation. Most GLiNER2 outputs are either:

- spans copied from the source text;
- closed-set classification or choice labels; or
- records assembled from those spans and choices.

That distinction makes offsets and deterministic validation useful. It does not make a prediction factually correct: a span can point to real text while being assigned to the wrong field.

The local implementation tokenizes the schema and text together, runs a Transformer encoder, constructs span representations for span-bearing tasks, predicts structure/instance counts, and uses classification scores for closed-label tasks. Combined schemas share one model invocation instead of requiring separate task-specific pipelines.

Sources: the [GLiNER2 paper](https://arxiv.org/abs/2507.18546), repository `README.md`, `gliner2/model.py`, and `gliner2/processor.py`.

## Task model

| Task | Schema describes | Typical result |
|---|---|---|
| Entities | entity types, optional descriptions, cardinality and thresholds | mentions grouped by type |
| Classification | task name, mutually exclusive or multi-label choices | one label or a list of labels |
| Structures | flat record types and their fields | zero or more record dictionaries |
| Relations | directional relation names, optional descriptions and thresholds | `(head, tail)` pairs grouped by relation |
| Combined | any mixture of the above | one dictionary containing all task results |

GLiNER2 is useful when the desired output schema is known but the exact entity types or fields vary between calls. Prefer deterministic parsers when the input already has a reliable machine format. Treat arithmetic, identifier validity, dates, required fields, cross-record association, and business rules as downstream validation—not as model guarantees.

## Choose a model

The following model names are published by Fastino. Parameter counts are the model-card/repository figures, not memory estimates.

| Model | Published size | Evidence and intended choice |
|---|---:|---|
| `fastino/gliner2-base-v1` | 205M | Default local model; English-tagged model card; lowest local memory/latency of the general models. |
| `fastino/gliner2-large-v1` | 340M | Larger general model; use after measuring a worthwhile quality gain. Hugging Face may round its UI size differently. |
| `fastino/gliner2-multi-v1` | about 0.3B | Multilingual-family model in Fastino's official collection; use for multilingual evaluation. The card body contains stale copied size text, so treat the Hugging Face model-file/UI estimate as approximate. |
| GLiNER XL | 1B | Hosted API model documented in the repository README; it is not a local `from_pretrained` model ID. |

Model sources: [official Fastino GLiNER2 collection](https://huggingface.co/collections/fastino/gliner2-family), [base model card](https://huggingface.co/fastino/gliner2-base-v1), [large model card](https://huggingface.co/fastino/gliner2-large-v1), and [multi model card](https://huggingface.co/fastino/gliner2-multi-v1).

Start with `fastino/gliner2-base-v1` for English smoke tests and ordinary workloads. Benchmark base and large on the same held-out domain set before paying the large model's memory/latency cost. Select `multi-v1` because the workload is multilingual, not merely because its tokenizer accepts a language.

Specialized community or task-specific fine-tunes may expose a narrower label space or different quality profile. Inspect their model cards and base-model lineage before substituting them.

## Language evidence

Do not turn tokenizer coverage into a fixed supported-language claim.

- The base model card is tagged English.
- The large model metadata lists English, French, and Spanish, but a Fastino maintainer directs multilingual users to `gliner2-multi-v1`.
- The multi model's Hugging Face UI says six languages without enumerating a validated benchmark set. In the [official discussion](https://huggingface.co/fastino/gliner2-multi-v1/discussions/1), a Fastino maintainer says it *should* support most mDeBERTa languages (100+) and explicitly says performance is use-case dependent and must be evaluated.

Therefore report language support in evidence-qualified terms. For example: “the model card tags English” or “the maintainer expects broad mDeBERTa coverage.” Do not report “100+ validated languages.” Build a held-out set for every language, domain, task, label description language, and script that matters. Mixed-language inputs deserve their own evaluation slice.

## Choose local or hosted inference

| Concern | Local `from_pretrained` | Hosted `from_api` |
|---|---|---|
| Credentials | No GLiNER2 API key; model download may require network/cache access | `PIONEER_API_KEY` required |
| Data path | Text remains on the machine running inference | Text is sent to the configured API endpoint |
| Compute | CPU, CUDA, or MPS on the caller | Provider-managed |
| Custom/fine-tuned models | Supported by loading a local/Hugging Face artifact | API uses provider models |
| `RegexValidator` | Supported | Builder warns and ignores it |
| Native long-context helpers | Present only in source/package builds that expose them | Not exposed by the current API client |
| Different schema per batch item | Native batch path | Falls back to one request per item with a warning |
| Failure modes | model availability, memory, device/runtime | authentication, validation, network, rate limit, server |

The clients intentionally share many method names, but they are not capability-identical. Check the actual object's methods before choosing a path. The bundled inference scripts use local inference and never require an API key.

## Choose a device

- **CPU:** valid and privacy-friendly; use for smoke tests, low throughput, and constrained deployments. Expect slower large-model and training workloads.
- **CUDA:** preferred for throughput, larger batches, long-document chunk batches, and training. Select a batch size from measured available memory.
- **MPS:** useful on Apple Silicon for bounded inference. Confirm that the installed PyTorch build supports the operations used by the chosen model; fall back to CPU when it does not.

Pass `map_location="cpu"`, `"cuda"`, or `"mps"` to `GLiNER2.from_pretrained`. Verify the executed device rather than trusting the requested string:

```python
devices = sorted({str(parameter.device) for parameter in extractor.parameters()})
print(devices)
```

`quantize=True` currently converts the local model to fp16; it is not generic integer quantization. Treat it as a GPU-oriented memory/performance option and validate quality and operator support. `compile=True` calls `torch.compile` on selected components; measure warm-up and steady-state latency before enabling it in a service.

## Evaluate before deployment

Keep two evaluations separate:

1. **API and structural validity:** schema builds, result shape is usable, input/output order matches, spans map exactly to the source, and closed choices are valid.
2. **Semantic quality:** required facts are present, fields and relations are correct, duplicates/overlaps are acceptable, and confidence thresholds meet the domain's precision/recall target.

Confidence is a model score, not a universally calibrated probability. Tune global and per-field thresholds on a held-out set. Record empty and negative examples; they reveal over-prediction that positive-only demos hide.
