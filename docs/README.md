# Research notes

`ravenea_semantics.md` records the exact official commit and actual released data
inspected for Batch 2, prompt/parser/scoring behavior, retrieval scope, and deviations.
Batch 1 is the no-RAG synthetic infrastructure smoke test. Batch 2 adds real cVQA
No-RAG vs cached Top-1 candidate-pool RAG on one persisted 50-question pilot.

For every experiment batch, update the root README in the same coding task with
exact GPU-server commands and actual verification/blockers. Pilot numbers are not
paper results. Full baseline evaluation belongs to Batch 3; oracle utility, learned
selection, fine-tuning, cIC and other benchmarks remain deferred.
