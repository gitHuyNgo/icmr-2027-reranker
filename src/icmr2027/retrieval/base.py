from typing import Protocol, TypedDict


class RetrievedDocument(TypedDict):
    rank: int
    doc_id: str
    score: float
    text: str


class Retriever(Protocol):
    def retrieve(self, sample: dict, top_k: int) -> list[RetrievedDocument]:
        ...
