# ICMR 2027

This project will study cultural multimodal question answering with retrieval-augmented
VLMs: does the highest-ranked retrieved document actually help the downstream VLM
answer correctly? **Batch 1 implements only a reproducible no-RAG baseline.**
It contains no retrieval, evidence selector, training, or paper result tables.

**Infrastructure-only synthetic smoke data. Never use these numbers in the paper.**
The three geometric images validate execution; their metrics have no scientific meaning.
No real cultural QA dataset has been selected or downloaded.

## Repository structure

```text
icmr-2027/
├── README.md
├── .gitignore
├── .gitattributes
├── requirements.txt
├── pytest.ini
├── configs/baseline/batch1_smoke.yaml
├── data/
│   ├── README.md
│   └── smoke/
│       ├── images/{smoke_001,smoke_002,smoke_003}.png
│       └── samples.jsonl
├── docs/README.md
├── outputs/.gitkeep
├── scripts/
│   ├── generate_smoke_data.py
│   ├── run_baseline.py
│   └── run_batch1_smoke.sh
├── src/icmr2027/
│   ├── __init__.py
│   ├── config.py
│   ├── datasets/{__init__.py,base.py}
│   ├── models/{__init__.py,vlm.py}
│   ├── evaluation/{__init__.py,metrics.py}
│   └── utils/{__init__.py,io.py,reproducibility.py,run_metadata.py}
└── tests/
    ├── test_config.py
    ├── test_datasets.py
    ├── test_metrics.py
    ├── test_io.py
    ├── test_vlm.py
    ├── test_run_metadata.py
    └── test_runner.py
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

Before installing requirements, select and execute the CUDA-enabled PyTorch pip
command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for your server's OS and supported CUDA build. The NVIDIA driver must support that
build; requirements intentionally do not choose a CUDA wheel index. Then:

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
python -m pip install pillow pyyaml pytest
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
keeps this smoke pipeline simple. Requirements use `transformers>=4.51,<5` to stay
on the stable 4.x API supporting the implemented Qwen2.5-VL adapter. No video
helpers or FlashAttention installation are needed.

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
`image` is a PIL image and the other values are strings. Batch 2 can add a real
cultural QA adapter and official scoring. Retrieval, BM25/dense retrieval, Top-K,
RAG experiments, oracle selection, utility scores/labels/reranking, selector training,
VLM fine-tuning, M4-RAG/RAVENEA reproduction, and paper result tables are deferred.

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
