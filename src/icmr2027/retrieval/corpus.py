"""Official document cleaning, with distinct retrieval and downstream representations."""
from pathlib import Path
import re

from icmr2027.utils.io import read_jsonl


def retrieval_text(text: str) -> str:
    return re.sub(r"(?m)^#+\s.*$", "", text).strip()


def evidence_text(text: str) -> str:
    # Match official helper.load_documents exactly: 256 words, last '. ' boundary.
    cleaned = retrieval_text(text)
    return " ".join(cleaned.split()[:256]).rsplit(". ", 1)[0] + "."


class WikipediaCorpus:
    def __init__(self, path: Path):
        self.documents = {}
        for document in read_jsonl(path):
            identifier = document.get("id")
            if not isinstance(identifier, str) or not isinstance(document.get("text"), str):
                raise ValueError("Wikipedia document requires string id and text")
            if identifier in self.documents:
                raise ValueError(f"Duplicate Wikipedia document ID: {identifier}")
            self.documents[identifier] = document
        if not self.documents:
            raise ValueError("Empty Wikipedia corpus")
        self.doc_ids = sorted(self.documents)

    def context(self, doc_id: str) -> str:
        return evidence_text(self.documents[doc_id]["text"])
