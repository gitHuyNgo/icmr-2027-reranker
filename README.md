# ICMR 2027

This project will study cultural multimodal question answering with retrieval-augmented
VLMs: does the highest-ranked retrieved document actually help the downstream VLM
answer correctly? Batch 1 validates a no-RAG infrastructure baseline. Batch 2 adds
a real RAVENEA cVQA pilot comparing No-RAG against cached Top-1 RAG. No oracle
utility, learned evidence selector, training, or paper result tables are implemented.

**Infrastructure-only synthetic smoke data. Never use these numbers in the paper.**
The three geometric images validate execution; their metrics have no scientific meaning.
Batch 2 uses the real released RAVENEA cVQA dataset; its pilot is not a final paper result.

## Repository structure

```text
icmr-2027/
├── README.md, .gitignore, .gitattributes, requirements.txt, pytest.ini
├── configs/baseline/
│   ├── batch1_smoke.yaml
│   ├── batch2_ravenea_no_rag.yaml
│   └── batch2_ravenea_top1_rag.yaml
├── data/
│   ├── README.md
│   ├── smoke/                         # tiny tracked synthetic data
│   ├── ravenea/                       # ignored real dataset
│   └── ravenea_manifest.json          # ignored integrity/provenance
├── artifacts/ravenea/                # ignored reusable pilot/retrieval artifacts
├── docs/{README.md,ravenea_semantics.md}
├── outputs/.gitkeep                  # experiment/comparison runs are ignored
├── scripts/
│   ├── generate_smoke_data.py
│   ├── run_baseline.py
│   ├── run_batch1_smoke.sh
│   ├── prepare_ravenea.py
│   ├── prepare_batch2_models.py
│   ├── cache_ravenea_retrieval.py
│   ├── compare_batch2.py
│   └── run_batch2.sh
├── src/icmr2027/
│   ├── __init__.py, config.py, experiments.py
│   ├── datasets/{__init__.py,base.py,ravenea.py,prepare.py}
│   ├── models/{__init__.py,vlm.py}
│   ├── evaluation/{__init__.py,metrics.py,ravenea.py,comparison.py}
│   ├── retrieval/{__init__.py,base.py,corpus.py,cache.py,clip.py,build.py}
│   └── utils/{__init__.py,io.py,reproducibility.py,run_metadata.py}
└── tests/                            # Batch 1 and 2 offline CPU tests
    └── fixtures/ravenea/             # synthetic schema fixtures, never research data
```

`configs/` defines experiments. `data/` holds local inputs; only the tiny smoke data
is tracked. `src/` provides dataset/model interfaces and evaluation/utilities.
`scripts/` runs experiments. `outputs/` holds ignored artifacts. `tests/` verifies
CPU behavior without model downloads or CUDA. `docs/` holds research notes.

## Environment setup

Use Python 3.10+ (Python 3.11 or 3.12 is a practical server choice). From your Linux
GPU server repository checkout, run:

```bash
cd /path/to/icmr-2027
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Before installing requirements, select and execute the matching CUDA-enabled
PyTorch and Torchvision pip command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for your server's OS and supported CUDA build. The NVIDIA driver must support that
build; requirements intentionally do not choose a CUDA wheel index. Install Torch
and Torchvision together from that index so their compiled operators are compatible. Then:

```bash
python -m pip install -r requirements.txt
pytest -q
```

There is no editable package installation: the runner derives the repository root
from its own location, and `pytest.ini` adds `src/` to the test import path. Relative
config, dataset and output paths resolve against that root, even from a different cwd.
The repository can be moved without changing Python source.

For CPU-only development on Windows PowerShell, the current checkout uses:

```powershell
Set-Location 'D:\Documents\A_Research\ICMR\icmr-2027'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pillow pyyaml pytest numpy huggingface_hub
# Keep pytest temporary files inside the checkout when system TEMP is restricted.
$env:TEMP = Join-Path (Get-Location) ".pytest_cache/tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
pytest -q
```

The minimal CPU test dependencies above deliberately exclude model weights and the
GPU stack. Actual inference needs all of `requirements.txt` and a suitable device.
The Bash launcher targets Linux GPU servers.

## GPU Server Experiments

### Batch 1 — Baseline Infrastructure Smoke Test

#### Step 1 — Enter repo

Use the actual location of the checkout on the server:

```bash
cd /path/to/icmr-2027
```

#### Step 2 — Activate environment

```bash
source .venv/bin/activate
```

#### Step 3 — Check GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

#### Step 4 — Run tests

```bash
pytest -q
```

These tests need no internet, CUDA, Torch, Transformers, or VLM weights. Runner
wiring tests use an explicitly labelled stub only inside pytest's temporary folder;
they do not constitute real model inference.

#### Step 5 — Run Batch 1 smoke test

```bash
bash scripts/run_batch1_smoke.sh
```

The launcher prints the Python executable and `nvidia-smi`, checks CUDA, then runs:

```bash
python scripts/run_baseline.py --config configs/baseline/batch1_smoke.yaml
```

The runner prints `Run directory: ...` and, on success, `Completed run: ...` for
that exact invocation. There is no distributed inference. Override the launcher's
Python executable if needed using `PYTHON=/absolute/path/to/python bash scripts/run_batch1_smoke.sh`.

The default is [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
a relatively small 3B image/text instruct checkpoint. Its
[Transformers inference support](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
keeps this smoke pipeline simple. Requirements use `transformers>=4.57.6,<5` to stay
on the stable 4.x API supporting the implemented Qwen2.5-VL adapter. Torchvision is
required: the processor initializes `AutoVideoProcessor` even when only images are
used ([processor source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py)).
`use_fast=False` selects the image-processing implementation; it does not remove
this dependency. No qwen-vl-utils, video decoder extras, or FlashAttention installation
are needed for this image-only pipeline.

Edit the YAML to change the checkpoint ID or local checkpoint directory, revision,
device, dtype, generation limit, temperature, dataset limit, or output root.
This adapter supports Qwen2.5-VL-compatible checkpoints; other model families
require an adapter implementing the same `generate(image, question, context=None)`
interface. A local model path should be absolute. `revision: null` uses the current
Hub default; set an immutable Hub commit in `revision` for later reproducible runs.
The resolved model commit, when available, is saved in metadata.

The default uses one CUDA device, bfloat16, and greedy generation
(`temperature: 0.0`). No fallback silently moves inference to CPU. BF16 support
is checked before weights load; set `dtype: float16` on GPUs without BF16 support.
The implementation uses the standard processor, a PIL RGB image, and PyTorch SDPA.
Model/processor loading occurs only in the inference runner, never in unit tests.

#### Step 6 — Expected output structure

```text
outputs/batch1/<run_id>/
├── config.yaml
├── metrics.json
├── predictions.jsonl
└── run_meta.json
```

Run IDs contain a UTC timestamp with microseconds plus a random suffix, e.g.
`20261007T123456123456Z_a1b2c3d4`. Allocation is atomic and never reuses a directory.
`config.yaml` stores the effective config including defaults. `run_meta.json` stores
UTC times, Git commit/dirty state (null if unavailable), hostname, Python/package
versions, CUDA/GPU details, seed, model ID/revision, run status and duration.

Predictions are UTF-8 JSONL containing `id`, `question`, `ground_truth`, `prediction`,
`correct`, `inference_seconds`, and `infrastructure_only`. Images are never embedded.
Metrics contain `num_samples`, `accuracy`, metric name, run status, and a smoke-data
warning. Exact match strips outer whitespace, lowercases, and collapses whitespace;
by default it strips boundary ASCII punctuation while preserving internal punctuation
and decimals. Set `evaluation.strip_punctuation: false` to disable that last step.
This provisional metric is not the final paper metric.

If inference fails after config loading and directory allocation, artifacts are retained
with `status: failed`, an error in metadata, and `accuracy: null`. Partial predictions
remain available but are not reported as completed metrics. Launcher preflight or invalid
config failures happen before run allocation. Seeds, greedy decoding and deterministic
Torch settings improve repeatability; identical results across hardware/software are
not guaranteed. The runner records installed package versions; requirements specify
compatibility bounds rather than a full environment lock.

#### Step 7 — Inspect latest results

```bash
find outputs/batch1 -maxdepth 2 -type f | sort
latest_run="$(find outputs/batch1 -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [ -n "$latest_run" ]; then
    printf 'Latest run: %s\n' "$latest_run"
    cat "$latest_run/metrics.json"
    cat "$latest_run/run_meta.json"
    head -n 3 "$latest_run/predictions.jsonl"
fi
```

The latest allocated directory may contain a failure; inspect `status` before
interpreting anything. Use the printed `Completed run` path to inspect a specific
successful invocation.

#### Step 8 — Troubleshooting

- **`nvidia-smi` unavailable or fails:** use an NVIDIA GPU server and check its
  driver/container GPU access. The launcher stops before inference.
- **CUDA unavailable:** check the selected Python environment and install a CUDA-enabled
  PyTorch build via the official selector. The runner and launcher stop on this condition.
- **Missing Python modules:** activate `.venv` and install `requirements.txt`; the minimal
  CPU-only test environment is insufficient for inference.
- **`AutoVideoProcessor requires the Torchvision library`:** Torchvision was missing
  from the initial requirements. Install the updated requirements with a matching
  Torch/Torchvision CUDA pair. For the reported server with `torch==2.14.1+cu130`,
  use the repair commands below. `use_fast=False` does not bypass video processor
  initialization, even though Batch 1 supplies no video.
- **Torchvision import errors such as `operator torchvision::nms does not exist`:**
  check that Torch and Torchvision use compatible versions and the same CUDA wheel
  index; reinstall the matching pair using the official PyTorch selector.
- **Insufficient VRAM:** free GPU memory, reduce `max_new_tokens`, or use another
  smaller compatible checkpoint. There is no quantization or multi-GPU fallback.
- **BF16 unsupported:** change `model.dtype` to `float16` in the YAML.
- **Model download or access fails:** first inference needs Hugging Face access,
  sufficient cache storage, and any permission required by the chosen checkpoint.
  Set `HF_TOKEN` in the environment for checkpoints requiring authentication; never
  commit it. Errors are retained in failed-run metadata when allocation has occurred.
- **Missing model cache on an offline server:** use internet access for the first run
  or configure `model.name` with an absolute path to an existing complete checkpoint.
  Set `model.local_files_only: true` to require cached/local files. `HF_HOME` can point
  to a cache outside the repository.
- **Missing/corrupt smoke files:** run `python scripts/generate_smoke_data.py` and
  rerun tests. The adapter checks manifest fields, unique IDs, and local image paths.


### Repair the reported CUDA 13.0 environment

The user-reported GPU run on 2026-10-07 passed CUDA preflight on an RTX 3090
with `torch==2.14.1+cu130`, then failed while loading the processor because
Torchvision was absent. Its failed run directory was
`outputs/batch1/20261007T095033768636Z_1014aba7/`; inference did not complete.

After copying the updated repository files to that server, run:

```bash
cd /root/ICMR/icmr-2027
source .venv/bin/activate
python -m pip install "torch==2.14.1+cu130" torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, torchvision; from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
pytest -q
bash scripts/run_batch1_smoke.sh
```

The Torch constraint keeps the installed build while pip selects a compatible
Torchvision release. The wheel index follows the installed `+cu130` build,
not the maximum CUDA version printed by `nvidia-smi`. Other servers should use
the matching pair/index from the official installation selector. Each retry
allocates a new directory; the failed run remains intact. These repair commands
have not been executed on the remote server by this agent.


### Batch 2 — RAVENEA cVQA Real-Data Baseline

Batch 2 validates real cultural multiple-choice QA with paired No-RAG and Top-1
RAG conditions. RAVENEA supplies culturally grounded questions, Wikipedia documents,
and the official RAVENEA-CLIP checkpoint. The same seeded 50-question pilot, image,
Qwen checkpoint/revision, prompt/option formatting, generation parameters and metric
are used in both conditions. Only insertion of cached Top-1 evidence changes.

**Batch 2 pilot results are not final paper results.** A higher or lower pilot
accuracy is pipeline-validation evidence, not a research conclusion. Batch 3 will
run the proper baseline experiment.

#### Source authority and actual schema

Official RAVENEA source commit inspected:
[`b7b3c290b3ad2c9caf17f28fefebfd9b912170ec`](https://github.com/yfyuan01/RAVENEA/tree/b7b3c290b3ad2c9caf17f28fefebfd9b912170ec).
Dataset source: `jaagli/ravenea`; pinned release:
`ff5a212a9bfa1515f82e0930b37b7d64e3e9ee2e`.
Retriever: `jaagli/ravenea-clip-vit-large-patch14` at
`890d85d3539a21fab2bb349d4874d5dfef5dd3ec`.
VLM: `Qwen/Qwen2.5-VL-3B-Instruct` at
`66285546d2b821cf421d4f5eb2576359d3770cd3`.

Actual released cVQA rows contain `file_name`, lists of `questions`, `options`,
`answers`, available `country`/`category`, and image-level `enwiki_ids` annotations.
Each options list has four strings labelled `A. ...` through `D. ...`; answers
are single uppercase letters. Flattened IDs follow the official helper exactly:
`<original file_name>_<question_index>`. IDs retain the original `./ravenea/images/`
prefix; disk paths resolve under `data/ravenea/images/`.

Local verified counts: **112 cVQA image rows, 209 questions, 112 unique cVQA images,
11,396 documents, zero missing images and duplicate document IDs**. Metadata rows:
train 1,505; validation 74; test 161; all 1,868. All cVQA images belong to the test
split. The manifest reports critical JSONL sizes/hashes, image hashes, counts,
source revision and example keys without dumping articles or absolute paths.

**Retrieval scope:** official `inference.py` ranks each image's `enwiki_ids`, not
all Wikipedia articles. The released cVQA pool has exactly ten candidates per
image. Batch 2 faithfully uses `candidate_scope: official_enwiki_ids` and saves
all ten, with an image-only query. It does not concatenate question text, answers,
options, generated captions or culture labels into retrieval inputs. All 11,396
documents can be encoded once for reusable embeddings, but scoring remains within
the official image pool. This is not an unrestricted full-corpus RAG experiment.

The checkpoint lists `LocalCLIPModel`, but official inference uses `AutoModel`
with `model_type: clip` and no `auto_map`. We use that same standard CLIP inference
path; no custom training code is required. Scores retain L2 normalization and
learned logit scaling; they are raw CLIP logits, not probabilities. Ties retain
annotation order. Cache ranks start at 1 (official TREC output starts at 0).

See [the semantic audit](docs/ravenea_semantics.md) for all inspected files,
released fields, parsing edge cases, context processing and deviations.

#### Step 1 — Environment and explicit preparation

On the GPU server:

```bash
cd /root/ICMR/icmr-2027
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi
pytest -q
```

Ensure matching CUDA-enabled Torch/Torchvision builds as documented above.
Requirements now require `transformers>=4.57.6,<5`, matching the inspected
retriever release's supported 4.x API. The only added package is Hugging Face Hub.

Explicitly prepare the dataset and pinned model snapshots:

```bash
bash scripts/run_batch2.sh prepare
```

This downloads the official 783 MB dataset archive if needed, safely extracts it
into `data/ravenea/`, validates cVQA/images/documents, writes
`data/ravenea_manifest.json`, and explicitly caches the retriever and Qwen model.
Allow enough disk space for archive, extracted images/corpus and both model
snapshots. Model caches follow `HF_HOME`; they are not put into experiment runs.
A complete dataset is validated/reused without re-download. To redownload/reextract
explicitly, use `python scripts/prepare_ravenea.py --force`; previous local data is
preserved in an ignored timestamped sibling directory.

Equivalent separate preparation commands:

```bash
python scripts/prepare_ravenea.py
python scripts/prepare_batch2_models.py
```

An already downloaded archive can be supplied via
`python scripts/prepare_ravenea.py --archive /absolute/path/to/ravenea.zip`.

#### Step 2 — Validate the real dataset

```bash
python scripts/prepare_ravenea.py --validate-only
```

Or `bash scripts/run_batch2.sh validate`. Output includes example/document keys,
question/image/document counts, missing images, duplicate IDs and metadata counts.
Existing arbitrary data without known download provenance receives a null source
revision; hashes are still recorded. Nothing automatically invents a dataset revision.

#### Step 3 — Retrieve and validate reusable Top-10 candidates

```bash
bash scripts/run_batch2.sh retrieval
python scripts/cache_ravenea_retrieval.py --validate-only
```

Retrieval loads the Top-1 config, persists a stable seeded selection of 50 question
IDs, prepares normalized FP32 document embeddings with 77-token text truncation,
then ranks each query image's ten candidates. Questions sharing an image reuse the
same ranking. Selected IDs are stored at
`artifacts/ravenea/pilot_seed42_ids.json`, with seed/limit/data fingerprint.
A changed pilot/data fingerprint fails rather than silently replacing this file;
choose a new `dataset.pilot_path` when intentionally changing the pilot.

The cache is reused after fingerprint/checksum and invariant validation. It verifies
all pilot IDs, image joins, corpus membership, candidate-pool membership, unique doc
IDs, finite descending scores and contiguous ranks. Use
`bash scripts/run_batch2.sh retrieval --force` to explicitly replace mismatched
embedding/retrieval artifacts after intentional settings/software changes.

```text
artifacts/ravenea/
├── pilot_seed42_ids.json
└── retrieval/
    ├── corpus_embeddings.npy
    ├── corpus_index.json
    ├── pilot_seed42_top10.jsonl
    └── retrieval_meta.json
```

The index preserves document order, corpus hash, model/revision, encoding settings
and embedding checksum. Retrieval metadata records timestamps, versions, seed,
device/dtype, query modality, scope, source commit, data/pilot/cache hashes and status.
JSONL entries store question ID, original image filename, retriever ID, ranked document
IDs and raw scores. They never duplicate full Wikipedia text.

Inspect candidates and metadata:

```bash
head -n 1 artifacts/ravenea/retrieval/pilot_seed42_top10.jsonl
python -m json.tool artifacts/ravenea/retrieval/retrieval_meta.json
python -m json.tool artifacts/ravenea/pilot_seed42_ids.json
```

#### Step 4 — Run No-RAG and Top-1 RAG on identical questions

```bash
bash scripts/run_batch2.sh no-rag
bash scripts/run_batch2.sh top1-rag
```

Equivalent direct commands:

```bash
python scripts/run_baseline.py --config configs/baseline/batch2_ravenea_no_rag.yaml
python scripts/run_baseline.py --config configs/baseline/batch2_ravenea_top1_rag.yaml
```

Both configs pin the same VLM revision and set `local_files_only: true`.
Inference never downloads the dataset, retriever or VLM weights. Top-1 generation
only resolves cached rank 1 by document ID and calls the existing
`generate(image, question, context=document_text)` API; it never recomputes retrieval.
No-RAG calls the same API with `context=None`.

The official multiple-choice prompt requests `Answer: $LETTER`; original labelled
options are preserved. Batch 1's generic open-ended prompt remains unchanged.
The official context helper removes Markdown heading lines, takes 256 words,
removes text after the last `. ` separator and appends a period. No extra evidence
summary or undocumented truncation is applied.

Scoring is case-sensitive and reproduces the official first `Answer:` A-D match,
then start-of-string `A)` style fallback, then a single stripped uppercase letter.
The official permissive regex can accept `Answer: APPLE` as A; we preserve that.
Lowercase labels and `A. text` are malformed/incorrect. All examples stay in the
accuracy denominator. Predictions store both raw output and parsed label (null if
malformed); metrics record malformed count and optional country accuracies.

RAG predictions additionally store `retrieved_doc_id`, `retrieval_rank`, raw
`retrieval_score`, and `context_chars`; no article is duplicated into predictions.
Each run stores the selected IDs and a comparison identity covering data/image
hashes, VLM settings, seed, prompt and scoring. Model revisions are also recorded.

```text
outputs/batch2/
├── no_rag/<run_id>/
│   ├── config.yaml
│   ├── metrics.json
│   ├── predictions.jsonl
│   └── run_meta.json
└── top1_rag/<run_id>/
    ├── config.yaml
    ├── metrics.json
    ├── predictions.jsonl
    └── run_meta.json
```

#### Step 5 — Compare completed paired runs

```bash
bash scripts/run_batch2.sh compare
```

This selects the latest **completed** run in each condition. To choose exact runs:

```bash
python scripts/compare_batch2.py \
  --no-rag-run outputs/batch2/no_rag/<run_id> \
  --top1-rag-run outputs/batch2/top1_rag/<run_id>
```

Substitute real run IDs in the explicit command. Comparison refuses different ID
sets, images/questions/options/truths, VLM revisions, seeds, prompt/generation/data
fingerprints, unfinished runs and inconsistent stored parsing/correctness.
It prints pilot accuracies/delta, malformed counts and transition counts, then writes:

```text
outputs/batch2/comparison/<comparison_id>/
├── summary.json
├── comparisons.jsonl
└── pair_meta.json
```

Per-example transitions are `wrong_to_correct`, `correct_to_wrong`,
`correct_to_correct` and `wrong_to_wrong`, joined by official question ID.
This is a small pilot comparison, not oracle utility analysis or significance testing.

#### Step 6 — Inspect results and independently check ID equality

```bash
find outputs/batch2 -maxdepth 3 -type f | sort
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from icmr2027.evaluation.comparison import latest_completed
from icmr2027.utils.io import read_jsonl
left = latest_completed(Path('outputs/batch2/no_rag'))
right = latest_completed(Path('outputs/batch2/top1_rag'))
a = {x['id'] for x in read_jsonl(left / 'predictions.jsonl')}
b = {x['id'] for x in read_jsonl(right / 'predictions.jsonl')}
r = {x['sample_id'] for x in read_jsonl(Path('artifacts/ravenea/retrieval/pilot_seed42_top10.jsonl'))}
assert a == b and a <= r, 'Pilot ID invariant failed'
print('Same sample IDs:', len(a), '; retrieval coverage:', len(r))
for run in (left, right):
    print(run)
    print((run / 'metrics.json').read_text())
PY
```

#### GPU notes, discrepancies and troubleshooting

Use the RTX 3090 workflow validated for Batch 1 (user-reported). The retriever uses
FP32 and the 3B VLM uses BF16; they execute in separate processes, so retrieval
weights do not remain in VRAM during generation. Corpus embeddings occupy roughly
35 MB on disk; model activation VRAM depends on image/context size. Batch 2 GPU
memory usage has not been measured locally. No FAISS, vector database, distributed
inference, quantization or training framework is introduced.

The official downstream program uses vLLM; we retain the Batch 1 Transformers
adapter. Engines can produce different answers, so this is not an exact paper-result
reproduction. Image limits match official Qwen settings (784 to 1,003,520 pixels),
generation cap is 2,048, greedy temperature is zero, and total token budget is 4,096.
An over-budget prompt fails explicitly; there is no hidden evidence truncation policy.

- **Missing data/manifest:** run explicit preparation; inference never downloads data.
- **Missing checkpoint cache/revision:** run `python scripts/prepare_batch2_models.py`
  using the same config/HF_HOME as inference. A different revision may need its own snapshot.
- **Torchvision missing or incompatible:** use the Batch 1 repair instructions above.
- **Missing retrieval cache:** run retrieval and its validation command before Top-1 RAG.
- **Hash/config/pilot mismatch:** inspect inputs; use a new pilot path for a changed subset
  or explicit retrieval `--force` for an intentional embedding/cache change.
- **Missing/duplicate documents or images:** preparation fails with counts; fix inputs or
  explicitly reprepare the official archive. Do not silently skip examples.
- **CUDA unavailable/OOM:** check environment/driver and free GPU memory. No CPU or
  multi-GPU fallback occurs silently. Batch 2 GPU inference is still locally unverified.
- **Prompt exceeds token budget:** the run fails and records the error; do not silently
  change context/generation settings between paired runs. Any deliberate new settings
  need matching configs and documentation.
- **Malformed VLM answer:** its raw text is retained, parsed label is null and it counts
  as incorrect. Inspect predictions rather than rewriting outputs after evaluation.
- **Comparison rejected:** inspect both run metadata and confirm both are completed
  with identical samples, pinned model/revision and scientific settings.

#### Actual verification status

The official archive was downloaded and prepared locally with
`python scripts/prepare_ravenea.py --archive .cache/ravenea.zip`, then checked with
`python scripts/prepare_ravenea.py --validate-only`. The real adapter loaded the
persisted 50-question pilot and RGB PIL images. All **83 CPU tests pass**, including the original Batch 1 tests. CPU tests cover official-style
parsing/scoring, safe extraction, integrity hashes, deterministic IDs, Top-10 joins,
Top-1 context, paired runner wiring and transition counts using explicit test stubs.
Those stubs are not real model results. A direct audit against the pinned official
functions also matched 209 flattened samples, all 11,396 document contexts,
418 prompt strings and 3,135 parser/correctness checks. The persisted 50-question
pilot uses 41 unique images. These are semantic checks, not model predictions.

The local Batch 2 launcher stopped at `nvidia-smi unavailable`. Retrieval, real
No-RAG/Top-1 generation, measured accuracies/malformed counts, and real paired
comparison remain unverified. No real Batch 2 experiment output was fabricated.
Run the server commands above to complete the GPU pilot.

## Prompt and extension points

Every Batch 1 prediction passes `context=None` and uses:

```text
Answer the question based on the image.

Question: {question}

Give a concise answer.
```

The same VLM interface accepts optional context for later batches:

```text
Use the image and the provided evidence to answer the question.

Evidence:
{context}

Question:
{question}

Give a concise answer.
```

`MultimodalQADataset` normalizes each sample to `{id, image, question, answer}`;
`image` is a PIL image and the other values are strings. The RAVENEA adapter adds
options, official IDs and available country/category metadata. Full baseline
evaluation, unrestricted-corpus retrieval, oracle utility, learned reranking or
selection, fine-tuning, cIC, additional benchmarks and paper tables remain deferred.

## README policy for future batches

**Every coding task that implements new experiment functionality must update this
README in that same task**, including exact GPU-server commands, environment changes,
configuration, expected artifacts, and verification/blockers. Do not postpone these
instructions to a later documentation task.

## Local verification — 2026-10-07

Verified on Windows with Python 3.13.3 using the minimal CPU test environment:
`pytest -q` passed all 30 tests. Python compilation, Bash syntax, Git whitespace,
and ignore-rule checks passed. The smoke images total about 3 KB.

Executing the Bash launcher stopped at preflight with
`ERROR: nvidia-smi is unavailable; run on an NVIDIA GPU server.`
Real GPU inference and model-specific preprocessing remain unverified locally.
No model weights were downloaded and no persistent experiment run directory was
produced. The local `.venv` contains CPU test dependencies; install the full GPU
requirements on the server before inference. Test-stub artifacts existed only in
ignored pytest temporary folders and are not model results.
