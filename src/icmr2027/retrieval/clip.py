"""Image-only RAVENEA-CLIP logits, using the official standard inference path."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from icmr2027.datasets.ravenea import OFFICIAL_SOURCE_COMMIT, file_sha256
from icmr2027.retrieval.base import RetrievedDocument
from icmr2027.retrieval.cache import rank_candidates
from icmr2027.retrieval.corpus import WikipediaCorpus, retrieval_text
from icmr2027.utils.io import write_json
from icmr2027.utils.run_metadata import package_version


class RaveneaCLIPRetriever:
    def __init__(self, config, corpus: WikipediaCorpus):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.config = config
        self.corpus = corpus
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for RAVENEA-CLIP retrieval")
        options = {"revision": config.revision, "local_files_only": config.local_files_only}
        # The checkpoint says LocalCLIPModel, but has model_type=clip and no auto_map.
        # Official inference also uses AutoModel: this loads standard CLIPModel.
        # LocalCLIP's training loss changes no inference features or parameter names.
        self.model = AutoModel.from_pretrained(config.name, dtype=torch.float32,
                                              attn_implementation="sdpa", **options).to(self.device)
        self.processor = AutoProcessor.from_pretrained(config.name, use_fast=True, **options)
        if self.model.config.model_type != "clip":
            raise ValueError("Expected the inspected CLIP-compatible checkpoint")
        self.model.eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        self.embeddings = None
        self.doc_index = {doc: index for index, doc in enumerate(corpus.doc_ids)}

    def prepare_corpus(self, cache_root: Path, wiki_hash: str, force: bool = False) -> None:
        import torch

        embedding_path = cache_root / "corpus_embeddings.npy"
        index_path = cache_root / "corpus_index.json"
        identity = {"doc_ids": self.corpus.doc_ids, "wiki_sha256": wiki_hash,
                    "retriever_model_id": self.config.name, "revision": self.config.revision,
                    "dtype": "float32", "max_text_tokens": self.config.max_text_tokens,
                    "use_fast": True, "normalization": "l2", "text_policy": "strip_markdown_heading_lines",
                    "transformers_version": package_version("transformers"),
                    "torch_version": package_version("torch"), "device": self.config.device,
                    "official_source_commit": OFFICIAL_SOURCE_COMMIT}
        if embedding_path.exists() and index_path.exists() and not force:
            saved = json.loads(index_path.read_text(encoding="utf-8"))
            if saved.get("identity") != identity or saved.get("embedding_sha256") != file_sha256(embedding_path):
                raise ValueError("Corpus embedding cache mismatch; rerun retrieval with --force")
            embeddings = np.load(embedding_path, allow_pickle=False)
        else:
            chunks = []
            total = len(self.corpus.doc_ids)
            for start in range(0, total, self.config.batch_size):
                ids = self.corpus.doc_ids[start:start + self.config.batch_size]
                texts = [retrieval_text(self.corpus.documents[doc]["text"]) for doc in ids]
                inputs = self.processor(text=texts, padding="max_length", truncation=True,
                                        max_length=self.config.max_text_tokens, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    features = self.model.get_text_features(**inputs)
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                chunks.append(features.float().cpu().numpy())
                if start % (self.config.batch_size * 50) == 0:
                    print(f"Corpus embeddings: {min(start + len(ids), total)}/{total}", flush=True)
            embeddings = np.concatenate(chunks).astype(np.float32)
            temporary = embedding_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, embeddings, allow_pickle=False)
            temporary.replace(embedding_path)
            write_json(index_path, {"identity": identity, "embedding_sha256": file_sha256(embedding_path),
                                    "shape": list(embeddings.shape)})
        if embeddings.shape != (len(self.corpus.doc_ids), self.model.config.projection_dim) or not (
            np.isfinite(embeddings).all() and np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4)
        ):
            raise ValueError("Invalid corpus embeddings: shape, finite values or normalization")
        self.embeddings = embeddings

    def retrieve(self, sample: dict, top_k: int) -> list[RetrievedDocument]:
        import torch

        if self.embeddings is None:
            raise RuntimeError("Prepare corpus embeddings before retrieval")
        candidates = sample["candidate_doc_ids"]
        if len(candidates) < top_k:
            raise ValueError("Official candidate pool is smaller than requested top_k")
        selected = self.embeddings[[self.doc_index[doc] for doc in candidates]]
        inputs = self.processor(images=sample["image"], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            image = self.model.get_image_features(**inputs)
            image = image / image.norm(p=2, dim=-1, keepdim=True)
            # Preserve raw logits_per_image semantics, including learned exp(logit_scale).
            logits = (image @ torch.from_numpy(selected).to(self.device).T) * self.model.logit_scale.exp()
        ranked = rank_candidates(candidates, logits.squeeze(0).float().cpu().tolist())[:top_k]
        return [{**candidate, "text": retrieval_text(self.corpus.documents[candidate["doc_id"]]["text"])}
                for candidate in ranked]
