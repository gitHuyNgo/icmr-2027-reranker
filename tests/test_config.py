from pathlib import Path

import pytest
import yaml

from icmr2027.config import load_config, resolve_path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_loads():
    config = load_config(ROOT / "configs/baseline/batch1_smoke.yaml")
    assert config.dataset.name == "smoke"
    assert config.dataset.limit == 3
    assert config.model.name == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert config.model.temperature == 0
    assert config.to_dict()["experiment"]["seed"] == 42


def test_relative_paths_ignore_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("data/smoke", ROOT) == ROOT / "data/smoke"
    assert resolve_path(tmp_path, ROOT) == tmp_path


@pytest.mark.parametrize("section,field,value", [
    ("model", "name", None), ("model", "max_new_tokens", 0),
    ("model", "temperature", -1), ("model", "temperature", float("nan")),
    ("model", "dtype", "int8"), ("model", "local_files_only", "false"),
    ("dataset", "limit", -1), ("dataset", "limit", True),
    ("experiment", "seed", -1), ("evaluation", "metric", "unimplemented"),
])
def test_invalid_config_rejected(tmp_path, section, field, value):
    raw = load_config(ROOT / "configs/baseline/batch1_smoke.yaml").to_dict()
    raw[section][field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_empty_yaml_rejected(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
