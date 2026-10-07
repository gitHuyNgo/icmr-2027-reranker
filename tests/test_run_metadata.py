from unittest.mock import patch

from icmr2027.utils.run_metadata import collect_run_metadata


def test_metadata_without_git_or_gpu(tmp_path):
    # No subprocess/network or torch requirement in this unit test.
    with patch("icmr2027.utils.run_metadata._git", return_value=None):
        meta = collect_run_metadata(tmp_path, 42, "test-only-checkpoint", "cpu")
    assert meta["git_commit"] is None
    assert meta["git_dirty"] is None
    assert meta["seed"] == 42
    assert meta["model_name"] == "test-only-checkpoint"
    assert meta["python_version"]
    assert meta["timestamp"].endswith("+00:00")
    assert {"torch_version", "transformers_version", "cuda_available", "gpu_name"} <= meta.keys()
