# Batch 2.1 evaluation-validity report

This audit uses the actual GPU pilot outputs. It is a parser-only re-evaluation of unchanged raw predictions. These remain pilot results, not paper results. No hypothesis interpretation is made.

## Original runs

- no_rag: `/root/ICMR/icmr-2027/outputs/batch2/no_rag/20261007T103955703635Z_1fbc3d8a`
- top1_rag: `/root/ICMR/icmr-2027/outputs/batch2/top1_rag/20261007T104052504178Z_46a95bae`

## Every malformed case

All seven cases are `A. parser_failure`. Counts: parser failure 7 (No-RAG 1, RAG 6); verbose-but-identifiable 0; ambiguous 0; no-option/unrecoverable 0; prompt/context issues 0; other/manual-review 0. Classification uses deterministic formatting rules without ground truth.

Each ID below is the complete official sample ID; document previews and original options/questions are in the audit JSONL/CSV.

| Condition | Sample ID | Raw output | Category | Parsed after |
|---|---|---|---|---|
| no_rag | `./ravenea/images/cvqa_cap_5865939224274921551_0_Indonesia.jpg_2` | `B. 1` | parser_failure | B |
| top1_rag | `./ravenea/images/cvqa_cap_5865939244273074657_0_Indonesia.jpg_1` | `D. Bromo` | parser_failure | D |
| top1_rag | `./ravenea/images/cvqa_cap_5865926424276286375_0_Russia.jpg_0` | `D. Kalashnikov` | parser_failure | D |
| top1_rag | `./ravenea/images/cvqa_cap_5865939234273568028_0_Indonesia.jpg_1` | `D. Election ballot box` | parser_failure | D |
| top1_rag | `./ravenea/images/cvqa_cap_5865939244273074657_0_Indonesia.jpg_0` | `D. Jeep` | parser_failure | D |
| top1_rag | `./ravenea/images/cvqa_cap_5865780764275393867_0_Nigeria.jpg_1` | `D. University of Nigeria, Nsukka` | parser_failure | D |
| top1_rag | `./ravenea/images/cvqa_cap_5865926414278148074_0_Russia.jpg_0` | `D. Eternal Fire` | parser_failure | D |

The No-RAG answer becomes well-formed but remains incorrect. All six RAG labels become well-formed; correctness is evaluated only after parsing rules are fixed. No option-text or ground-truth matching occurs.

## Before and after

| Metric | Official before | Conservative after |
|---|---:|---:|
| no_rag_accuracy | 0.7 | 0.7 |
| top1_rag_accuracy | 0.66 | 0.78 |
| delta_accuracy | -0.04 | 0.08 |
| no_rag_num_malformed | 1 | 0 |
| top1_rag_num_malformed | 6 | 0 |
| no_rag_malformed_rate | 0.02 | 0 |
| top1_rag_malformed_rate | 0.12 | 0 |
| wrong_to_correct | 6 | 6 |
| correct_to_wrong | 8 | 2 |
| correct_to_correct | 27 | 33 |
| wrong_to_wrong | 9 | 9 |
| correction_rate | 0.4 | 0.4 |
| corruption_rate | 0.228571 | 0.0571429 |

## Parser and prompt decision

The official parser remains unchanged and is the default scorer. The separately named `conservative_v1` accepts one explicit boundary-aware A-D commitment in bare, bracketed, named-answer or leading labelled-option formats. It rejects conflicting labels, uncertainty, negation and letters inside words. Normalized scoring is semantically consistent with a single-option task but differs from the exact official regex; retain both protocols when reporting results. See `README.md` for the complete rule list and exact server commands.

No prompt, context policy, model/revision, generation setting or retrieval change was necessary. No inference or retrieval was rerun. Evidence placement follows the official template, whose formatting instruction appears before evidence. All 50 historical evidence lengths were verified; questions/options remain intact, document metadata is omitted, and token overflow fails without silent truncation. This audit identifies parser-format failures, without attributing the output-frequency imbalance to context length.

## Provenance and verification

- Same exact 50 IDs: yes; SHA-256 `5beca60bd8b92436093f82e08ad0a36e64fbe36ce483714be483d8e203b047fd`.
- All 100 original raw predictions unchanged: yes.
- Cache, Top-1 document IDs, ranks, scores and contexts unchanged: yes.
- Seven malformed cases after normalization: none remain.
- Local full regression suite: 146 passed, including all previous 83 tests.
- GPU-server parser-only audit/reparse/comparison: completed; full regression suite: 146 passed.
- Final checkpoint: passed output-format audit; source SHA-256 checks and all 100 raw-generation equality checks passed.
- Execution evidence: `outputs/batch2_1/verification.json`.

The locked input file hashes are in `outputs/batch2_1/audit/audit_meta.json`; exact before/after measurements are in `outputs/batch2_1/comparison/before_after.json`. All 50 old/new predictions and correctness values are in `outputs/batch2_1/comparison/per_sample_transitions.jsonl` and `.csv`. The malformed case files are `outputs/batch2_1/audit/malformed_cases.jsonl` and `.csv`. Context checks for all 50 RAG samples are in `outputs/batch2_1/audit/context_checks.jsonl` and `.csv`. Reparse metrics are under `outputs/batch2_1/reparse/`.

## Official-source recheck

Pinned commit: `b7b3c290b3ad2c9caf17f28fefebfd9b912170ec`. Downloaded source matched the prior inspected copies byte for byte:

- [helper.py](https://github.com/yfyuan01/RAVENEA/blob/b7b3c290b3ad2c9caf17f28fefebfd9b912170ec/src/helper.py): `5e01dbdf0ba2895b8d725803c7deb0fb071c3713127e4bb4e5630b243be9f5ae`.
- [downstream_cvqa.py](https://github.com/yfyuan01/RAVENEA/blob/b7b3c290b3ad2c9caf17f28fefebfd9b912170ec/src/downstream_cvqa.py): `5153eb3f54e5f135653b7806d531e8210345e29547c09b0f0db41989ff78e422`.
- [metrics.py](https://github.com/yfyuan01/RAVENEA/blob/b7b3c290b3ad2c9caf17f28fefebfd9b912170ec/src/metrics.py): `4f2c0442a498cb8744dcc6653a2193b0c7bcb85702e88140fa056f2110e616b5`.

The official format is `Answer: $LETTER`; matching is case-sensitive, first-match based, with a leading `A)` or stripped single-character fallback. All questions remain in the denominator and missing/unparsed predictions are wrong. The historical scores preserve those semantics.
