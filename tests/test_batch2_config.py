from pathlib import Path

import pytest
import yaml

from icmr2027.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_batch2_configs_share_all_vlm_and_pilot_settings():
    left = load_config(ROOT / "configs/baseline/batch2_ravenea_no_rag.yaml")
    right = load_config(ROOT / "configs/baseline/batch2_ravenea_top1_rag.yaml")
    assert left.model == right.model
    assert left.dataset == right.dataset
    assert left.evaluation == right.evaluation
    assert left.experiment.seed == right.experiment.seed == 42
    assert left.dataset.limit == 50
    assert left.model.local_files_only and right.retrieval.local_files_only
    assert right.retrieval.top_k == 10
    assert right.retrieval.query_modality == "image"
    assert right.retrieval.candidate_scope == "official_enwiki_ids"


@pytest.mark.parametrize("section,field,value", [
    ("dataset", "limit", None), ("model", "revision", None), ("model", "local_files_only", False),
    ("model", "prompt_format", "baseline"), ("evaluation", "metric", "exact_match"),
    ("retrieval", "query_modality", "image_question"), ("retrieval", "top_k", 1),
    ("rag", "mode", "oracle"),
])
def test_scientifically_incompatible_batch2_config_fails(tmp_path, section, field, value):
    raw = load_config(ROOT / "configs/baseline/batch2_ravenea_top1_rag.yaml").to_dict()
    raw[section][field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
