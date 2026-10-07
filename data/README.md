# Data

`smoke/` contains exactly three generated geometric images and color questions.
Infrastructure-only synthetic smoke data. Never use these numbers in the paper.
These samples are not research data and their metrics have no scientific meaning.
No cultural benchmark has been selected, downloaded, or fabricated.

The manifest `smoke/samples.jsonl` stores `id`, relative `image` path, `question`,
`answer`, and `infrastructure_only: true`. The adapter returns a loaded RGB PIL image.
Regenerate the tracked tiny files with `python scripts/generate_smoke_data.py`.

Real dataset directories are ignored. Batch 2 should add a documented adapter
normalizing each sample to `{id: str, image: PIL.Image.Image, question: str, answer: str}`,
with dataset provenance, licensing, splits, and official evaluation details.
