# Data

`smoke/` contains three tracked synthetic geometric images/questions for Batch 1.
Infrastructure-only synthetic smoke data. Never use these numbers in the paper.
Regenerate them with `python scripts/generate_smoke_data.py`.

Batch 2 uses the real released RAVENEA cVQA task from `jaagli/ravenea`.
Run `python scripts/prepare_ravenea.py` explicitly to download/extract into
`data/ravenea/`. Run `python scripts/prepare_ravenea.py --validate-only` to inspect
counts and schemas. No dataset download occurs during VLM inference.
The optional `--archive` accepts a previously downloaded official ZIP.

The dataset, its images, Wikipedia text, `ravenea_manifest.json`, and extraction
backups are ignored by Git. The manifest stores relative filenames, sizes, SHA256
hashes, counts, and the source revision when known; no machine-specific paths.
A complete existing dataset is validated and reused unless `--force` is supplied.
The force path preserves previous data under a timestamped sibling directory.

Released cVQA counts verified locally: 112 image rows, 209 questions, 112 unique
images, 11,396 Wikipedia documents; zero missing images/duplicate document IDs.
See `docs/ravenea_semantics.md` for exact schema, IDs, candidate scope, and evaluation.
Synthetic schema fixtures live only in `tests/fixtures/ravenea/` and are never used
by real benchmark configs. Batch 2 pilot results are not final paper results.
