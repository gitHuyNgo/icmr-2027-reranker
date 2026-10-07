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


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 64
    temperature: float = 0.0
    revision: str | None = None
    local_files_only: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    metric: str = "exact_match"
    strip_punctuation: bool = True


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig

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
    if set(raw) != expected:
        raise ValueError(f"Configuration requires exactly these sections: {sorted(expected)}")
    try:
        config = Config(
            ExperimentConfig(**raw["experiment"]),
            DatasetConfig(**raw["dataset"]),
            ModelConfig(**raw["model"]),
            EvaluationConfig(**raw["evaluation"]),
        )
    except TypeError as exc:
        raise ValueError(f"Invalid configuration fields: {exc}") from exc
    for section in (config.experiment, config.dataset):
        for key, value in asdict(section).items():
            if key not in {"seed", "limit"}:
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
    if config.evaluation.metric != "exact_match":
        raise ValueError("Batch 1 supports only evaluation.metric: exact_match")
    if type(config.evaluation.strip_punctuation) is not bool:
        raise ValueError("evaluation.strip_punctuation must be a boolean")
    return config


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()
