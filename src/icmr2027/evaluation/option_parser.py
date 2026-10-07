"""Versioned conservative A-D parser; never consults question, options, or truth.

This normalization is supplemental to the unchanged official RAVENEA scorer.
"""
import re

OFFICIAL = "official"
CONSERVATIVE = "conservative_v1"

# Full-token boundaries prevent option extraction from APPLE, CAB, IDs, etc.
TOKEN = r"(?:\(([A-D])\)|\[([A-D])\]|([A-D]))(?!\w)"
CONCISE = re.compile(r"\s*(?:" + TOKEN + r"|(?:Answer\s*:\s*|(?:The\s+)?(?:correct\s+)?answer\s+is\s+|Option\s+)"
                     + TOKEN + r")\s*\.?\s*", re.I)
COMMIT = re.compile(r"\b(?:Answer\s*:\s*|(?:The\s+)?(?:correct\s+)?answer\s+is\s+|Option\s+)"
                    + TOKEN, re.I)
LEADING = re.compile(r"^\s*(?:([A-D])[.)]\s+|\(([A-D])\)\s+|\[([A-D])\]\s+)", re.I)
UNCERTAIN = re.compile(r"\b(?:maybe|perhaps|could|might|cannot|unable|uncertain)\b|\bnot\s+(?:sure|certain)\b", re.I)
NEGATED = re.compile(r"\b(?:not|isn(?:'|\u2019)t|is\s+not|never)\s+(?:option\s+)?[([]?[A-D](?!\w)", re.I)


def parse_conservative_answer(raw: str) -> str | None:
    """Accept exactly one explicitly committed label, including labelled option text.

    Repeated same labels are allowed; any other standalone uppercase A-D or lower
    b/c/d is a conflict. Lowercase article 'a' in prose is not itself an option.
    Standalone lowercase a still works as a complete answer or explicit commitment.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    if UNCERTAIN.search(raw) or NEGATED.search(raw):
        return None
    explicit = []
    concise = CONCISE.fullmatch(raw)
    if concise:
        explicit += [label.upper() for label in concise.groups() if label]
    for match in COMMIT.finditer(raw):
        explicit += [label.upper() for label in match.groups() if label]
    leading = LEADING.match(raw)
    if leading:
        explicit += [label.upper() for label in leading.groups() if label]
    labels = set(explicit)
    # Reject ambiguous options even when one looks like an Answer: declaration.
    labels.update(label.upper() for label in re.findall(r"(?<!\w)([A-Db-d])(?!\w)", raw))
    labels.update(label.upper() for label in re.findall(r"[([]([a-d])[)\]]", raw))
    if len(labels) != 1 or not explicit:
        return None
    return next(iter(labels))


def answer_parser(version: str):
    if version == CONSERVATIVE:
        return parse_conservative_answer
    if version == OFFICIAL:
        from icmr2027.evaluation.ravenea import parse_cvqa_answer
        return parse_cvqa_answer
    raise ValueError(f"Unknown answer parser version: {version}")
