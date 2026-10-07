"""Offline wiring tests only. Stubs are not VLM inference or research results."""
import json
from pathlib import Path

import pytest
import yaml

from icmr2027.config import load_config
from icmr2027.utils.io import read_jsonl
from scripts import run_baseline as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_run(tmp_path, monkeypatch):
    raw = load_config(ROOT / "configs/baseline/batch1_smoke.yaml").to_dict()
    raw["experiment"]["output_root"] = str(tmp_path / "outputs")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(runner, "set_seed", lambda seed: None)
    monkeypatch.setattr(runner, "collect_run_metadata", lambda *args: {"timestamp": "test-only"})
    return config, tmp_path / "outputs"


def test_runner_saves_complete_artifacts_with_test_stub(isolated_run, monkeypatch):
    config, outputs = isolated_run

    class TestOnlyVLM:
        def __init__(self, config):
            self.resolved_revision = "test-stub"

        def generate(self, image, question, context=None):
            assert image.mode == "RGB"
            assert context is None
            return "red"

    monkeypatch.setattr(runner, "HuggingFaceVLM", TestOnlyVLM)
    run = runner.run_baseline(config, ROOT)
    assert set(path.name for path in run.iterdir()) == {
        "config.yaml", "run_meta.json", "predictions.jsonl", "metrics.json"
    }
    predictions = read_jsonl(run / "predictions.jsonl")
    assert len(predictions) == 3
    assert all(record["infrastructure_only"] for record in predictions)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["status"] == "completed"
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["num_predictions"] == 3
    assert meta["resolved_model_revision"] == "test-stub"
    assert runner.run_baseline(config, ROOT) != run


def test_model_failure_is_saved_without_success_metrics(isolated_run, monkeypatch):
    config, outputs = isolated_run

    def fail(config):
        raise RuntimeError("test-only model loading failure")

    monkeypatch.setattr(runner, "HuggingFaceVLM", fail)
    with pytest.raises(RuntimeError, match="test-only"):
        runner.run_baseline(config, ROOT)
    run = next(outputs.iterdir())
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["status"] == "failed"
    assert "model loading failure" in meta["error"]
    assert read_jsonl(run / "predictions.jsonl") == []
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["status"] == "failed"
    assert metrics["accuracy"] is None


def test_partial_predictions_survive_inference_failure(isolated_run, monkeypatch):
    config, outputs = isolated_run

    class FailsOnSecondSample:
        def __init__(self, config):
            self.resolved_revision = None
            self.calls = 0

        def generate(self, image, question, context=None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("test-only interrupted inference")
            return "red"

    monkeypatch.setattr(runner, "HuggingFaceVLM", FailsOnSecondSample)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run_baseline(config, ROOT)
    run = next(outputs.iterdir())
    assert len(read_jsonl(run / "predictions.jsonl")) == 1
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics == {"status": "failed", "num_samples": 1, "accuracy": None,
                       "infrastructure_only": True}
