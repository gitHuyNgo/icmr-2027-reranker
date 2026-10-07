# Inspected RAVENEA semantics

Official RAVENEA source commit inspected:
`b7b3c290b3ad2c9caf17f28fefebfd9b912170ec`.

Inspected files at that commit: `src/inference.py`, `src/downstream_cvqa.py`,
`src/metrics.py`, `src/helper.py`, `src/finetune/localclip.py`, and `pyproject.toml`.
See the [pinned official source](https://github.com/yfyuan01/RAVENEA/tree/b7b3c290b3ad2c9caf17f28fefebfd9b912170ec/src).

Inspected dataset: `jaagli/ravenea`, archive `ravenea.zip`, revision
`ff5a212a9bfa1515f82e0930b37b7d64e3e9ee2e`. Downloaded archive size:
783,426,902 bytes. The archive was actually downloaded and its released JSONL
files inspected before the adapter was written.

## Released schema and counts

`cvqa_downstream.jsonl` has 112 image rows, containing 209 aligned questions.
Each row has `file_name`, `country`, `task_type`, `questions`, `options`, `answers`,
`human_captions`, `category`, `enwiki_ids`, `culture_relevance`, and `generate_caption`.
Questions are strings; options are lists of four labelled strings (`A. text` through
`D. text`); answers are uppercase single letters A-D. No native row ID is present.
The adapter preserves the official helper's per-question ID:
`<original file_name>_<zero-based question index>`. It preserves the raw relative
filename, including `./ravenea/`, in this ID. Only disk path resolution changes.

There are 112 unique cVQA images, all present, and all belong to `metadata_test`.
The cVQA release contains seven countries; this is a task-specific subset of the
benchmark's broader country coverage, not fabricated missing metadata.

The corpus has 11,396 unique documents. Document fields are `date_modified`, `id`,
`text`, `title`, and `url`. IDs are strings such as `enwiki/256579`.
Metadata row counts: all 1,868; train 1,505; validation 74; test 161.
No missing cVQA images, duplicate corpus IDs, or missing annotated candidate IDs
were found. All integrity counts/hashes are also written into the ignored manifest.

## Retrieval scope and checkpoint

Official `inference.py` uses each query image alone; it supplies candidate document
texts from that row's `enwiki_ids`. It does not encode question text and does not
search unrestricted Wikipedia. Every released cVQA image has exactly ten candidate
IDs. This batch uses `candidate_scope: official_enwiki_ids`, preserving that scope.
It is candidate-pool ranking over the official corpus, not global corpus search.
Encoding all corpus documents once is an optimization, not a change of search scope.
Question/options enter only the VLM prompt. Ground-truth answers, culture relevance
and generated captions never enter CLIP feature encoding or the VLM prompt.
The candidate IDs define the official retrieval pool.

Retrieval text removes Markdown heading lines and surrounding whitespace. The
checkpoint's processor uses `use_fast=True`, 77-token max-length padding and
truncation. Projected image/text features are L2-normalized; raw scores are cosine
similarities multiplied by the learned `exp(logit_scale)`, matching CLIP image logits.
Ties retain the original candidate order. Cached ranks are 1-based (the official
TREC writer enumerates from zero), as required by our cache contract.

The checkpoint config names `LocalCLIPModel`, but has `model_type: clip` and no
`auto_map`. Official inference uses `AutoModel`, which selects standard CLIPModel.
The inspected LocalCLIP class adds training losses without changing feature
projections/normalization/logits or state-dict parameter names. We use the same
AutoModel inference path; no custom remote code or training code is copied.
Retriever revision: `890d85d3539a21fab2bb349d4874d5dfef5dd3ec`.

## Prompt, context and metric

Prompt meaning and option presentation follow `prepare_cvqa_input`: four original
labelled options, request an `Answer: $LETTER` final line, and insert the official
shared-culture document instruction only for RAG. We retain this instruction rather
than imposing a stricter one-letter-only format. Qwen's chat template inserts the
image and its default system message. Batch 1 open-ended prompting is unchanged.

Evidence follows `helper.load_documents`: remove heading lines, take the first
256 whitespace-separated words, retain the portion before the last `. ` separator,
and append a period. This can remove the final sentence even for a short article;
that behavior is preserved rather than silently repaired. Full raw corpus text can
be recovered by document ID. No summarizer, rewriting, utility truncation or
multi-document context is used.

Scoring follows `helper.is_match` and `metrics.evaluate_cvqa`: case-sensitive first
`Answer:` A-D match anywhere, otherwise an A-D label followed by `)` at the start,
otherwise a whitespace-stripped single uppercase letter. This is not a last-line
validator. `A. text` and lowercase outputs fail; `Answer: APPLE` is accepted as A,
reflecting the official permissive regex. Unparseable outputs become null and count
as incorrect; they remain in the denominator. Accuracy is a fraction in [0,1];
country accuracies and malformed counts are also saved. The released ground truths
are A-D, so the explicit parser preserves official correctness semantics.

## Implementation differences and verification limits

The official downstream code uses vLLM. We preserve the already validated Batch 1
Transformers Qwen adapter and run one image/question at a time. Different inference
engines can yield different answers; this is not a claim of identical paper results.
Batch 2 uses the official Qwen image bounds (784 to 1,003,520 pixels), 4,096 total
token budget, 2,048 generated-token cap and greedy decoding. Over-budget prompts
fail explicitly instead of introducing an undocumented truncation policy.
VLM revision: `66285546d2b821cf421d4f5eb2576359d3770cd3` in both conditions.

The 50-question seeded pilot, reusable normalized embeddings, JSONL cache/checksums,
explicit prerequisites, unique run directories and paired comparison are our
infrastructure additions. A question may share an image with another pilot question;
its retrieval ranking is reused while its official question ID remains distinct.
No official retrieval metric is computed because it is optional for this batch.

Local actual-data preparation and PIL loading were verified. CPU pipeline tests use
explicit synthetic fixtures and stubs, not real retriever/VLM predictions. This
workspace lacks NVIDIA GPU access. Full Batch 2 retrieval/generation/comparison and
pilot accuracies remain unverified until run on the GPU server.
Batch 1 GPU success is user-reported; this agent has not executed that remote run.

## Direct parity audit on the downloaded release

Independently executed the selected pinned official helper/prompt functions, with
UTF-8 file reading to reproduce Linux defaults on the Windows host. Verified exact
agreement for all 209 flattened IDs/questions/options/answers, all 11,396 prepared
document contexts, 418 No-RAG/RAG prompt strings, and 3,135 parser/correctness edge
checks. These checks concern functions and representations; they are not model
inference or pilot accuracy measurements. The persisted seed-42 pilot contains
50 questions using 41 unique images. All 83 offline CPU tests passed.
