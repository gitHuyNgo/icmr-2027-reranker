"""Small validated YAML config; relative paths are anchored at the repo root."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import math

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_root: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_path: str
    split: str
    limit: int | None = None
    sample_seed: int = 42
    pilot_path: str | None = None
    manifest_path: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 64
    temperature: float = 0.0
    revision: str | None = None
    local_files_only: bool = False
    prompt_format: str = "baseline"
    min_pixels: int | None = None
    max_pixels: int | None = None
    max_input_tokens: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    metric: str = "exact_match"
    strip_punctuation: bool = True


@dataclass(frozen=True)
class RagConfig:
    mode: str = "none"
    retrieval_path: str | None = None


@dataclass(frozen=True)
class RetrieverConfig:
    name: str
    revision: str
    device: str
    dtype: str
    batch_size: int
    top_k: int
    candidate_scope: str
    query_modality: str
    max_text_tokens: int
    local_files_only: bool = True


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    rag: RagConfig = RagConfig()
    retrieval: RetrieverConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nonempty(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    expected = {"experiment", "dataset", "model", "evaluation"}
    if not expected <= set(raw) or set(raw) - expected - {"rag", "retrieval"}:
        raise ValueError(f"Configuration requires exactly these sections: {sorted(expected)}")
    try:
        config = Config(
            ExperimentConfig(**raw["experiment"]),
            DatasetConfig(**raw["dataset"]),
            ModelConfig(**raw["model"]),
            EvaluationConfig(**raw["evaluation"]),
            RagConfig(**(raw.get("rag") or {})),
            RetrieverConfig(**raw["retrieval"]) if raw.get("retrieval") is not None else None,
        )
    except TypeError as exc:
        raise ValueError(f"Invalid configuration fields: {exc}") from exc
    for section in (config.experiment, config.dataset):
        for key, value in asdict(section).items():
            if key in {"name", "output_root", "data_path", "split"}:
                _nonempty(value, key)
    _nonempty(config.model.name, "model.name")
    if type(config.experiment.seed) is not int or not 0 <= config.experiment.seed < 2**32:
        raise ValueError("experiment.seed must be an integer in [0, 2**32)")
    if config.dataset.limit is not None and (
        type(config.dataset.limit) is not int or config.dataset.limit <= 0
    ):
        raise ValueError("dataset.limit must be a positive integer or null")
    if type(config.model.max_new_tokens) is not int or config.model.max_new_tokens <= 0:
        raise ValueError("model.max_new_tokens must be a positive integer")
    if type(config.model.temperature) not in {int, float} or not (
        math.isfinite(config.model.temperature) and config.model.temperature >= 0
    ):
        raise ValueError("model.temperature must be a finite nonnegative number")
    if config.model.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("model.dtype must be float32, float16, or bfloat16")
    _nonempty(config.model.device, "model.device")
    if config.model.revision is not None:
        _nonempty(config.model.revision, "model.revision")
    if type(config.model.local_files_only) is not bool:
        raise ValueError("model.local_files_only must be a boolean")
    if config.evaluation.metric not in {"exact_match", "ravenea_cvqa_accuracy"}:
        raise ValueError("Unknown evaluation metric")
    if config.rag.mode not in {"none", "top1"}:
        raise ValueError("rag.mode must be none or top1")
    if config.rag.mode == "top1":
        _nonempty(config.rag.retrieval_path, "rag.retrieval_path")
    if config.model.prompt_format not in {"baseline", "ravenea_cvqa"}:
        raise ValueError("Unknown model.prompt_format")
    for field in ("min_pixels", "max_pixels", "max_input_tokens"):
        value = getattr(config.model, field)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"model.{field} must be a positive integer or null")
    if config.model.min_pixels and config.model.max_pixels and config.model.min_pixels > config.model.max_pixels:
        raise ValueError("model.min_pixels exceeds max_pixels")
    if config.dataset.name == "smoke" and (
        config.rag.mode != "none" or config.evaluation.metric != "exact_match"
        or config.model.prompt_format != "baseline"
    ):
        raise ValueError("Batch 1 smoke requires baseline prompting, exact_match and no RAG")
    if config.dataset.name == "ravenea_cvqa":
        if config.dataset.split != "test" or config.dataset.limit is None:
            raise ValueError("RAVENEA requires the released test cVQA file and an explicit pilot limit")
        if type(config.dataset.sample_seed) is not int or not 0 <= config.dataset.sample_seed < 2**32:
            raise ValueError("dataset.sample_seed must be an integer in [0, 2**32)")
        for field in ("pilot_path", "manifest_path"):
            _nonempty(getattr(config.dataset, field), f"dataset.{field}")
        if config.evaluation.metric != "ravenea_cvqa_accuracy" or config.model.prompt_format != "ravenea_cvqa":
            raise ValueError("RAVENEA requires official-style cVQA prompting and evaluation")
        if not config.model.revision or not config.model.local_files_only:
            raise ValueError("Batch 2 requires a pinned VLM revision and local_files_only: true; prepare models first")
    if config.rag.mode == "top1" and config.retrieval is None:
        raise ValueError("Top-1 mode requires explicit retrieval settings")
    if config.retrieval is not None:
        ret = config.retrieval
        for field in ("name", "revision", "device"):
            _nonempty(getattr(ret, field), f"retrieval.{field}")
        if ret.dtype != "float32" or ret.candidate_scope != "official_enwiki_ids" or ret.query_modality != "image":
            raise ValueError("Batch 2 retrieval matches official float32 image-only enwiki_ids candidate ranking")
        if ret.top_k != 10 or ret.max_text_tokens != 77 or type(ret.batch_size) is not int or ret.batch_size <= 0:
            raise ValueError("Batch 2 retrieval requires top_k=10, max_text_tokens=77 and positive batch_size")
        if type(ret.local_files_only) is not bool or not ret.local_files_only:
            raise ValueError("Batch 2 retriever requires local_files_only: true; prepare models first")
    if type(config.evaluation.strip_punctuation) is not bool:
        raise ValueError("evaluation.strip_punctuation must be a boolean")
    return config


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()
