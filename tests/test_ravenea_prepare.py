from pathlib import Path
import stat
import zipfile

import pytest

from icmr2027.datasets.prepare import prepare_dataset, safe_extract

FIXTURE = Path(__file__).parent / "fixtures/ravenea"


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "C:/escape.txt", "ravenea\\..\\escape.txt"])
def test_unsafe_zip_entries_are_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "staging")


def test_zip_symlinks_are_rejected(tmp_path):
    archive = tmp_path / "symlink.zip"
    entry = zipfile.ZipInfo("ravenea/link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(entry, "../../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "staging")


def test_prepare_locates_nested_root_and_reuses_complete_data(tmp_path):
    archive = tmp_path / "official-shaped-fixture.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in FIXTURE.rglob("*"):
            if path.is_file():
                zipped.write(path, "nested/ravenea/" + path.relative_to(FIXTURE).as_posix())
    data, manifest, cache = tmp_path / "data/ravenea", tmp_path / "data/manifest.json", tmp_path / "cache"
    result = prepare_dataset(data, manifest, cache, archive=archive, revision="test-only")
    assert result["integrity"]["num_cvqa_examples"] == 3
    # Invalid archive is never touched once all required data exists.
    reused = prepare_dataset(data, manifest, cache, archive=tmp_path / "does-not-exist.zip")
    assert reused["integrity"]["num_cvqa_examples"] == 3
    prepare_dataset(data, manifest, cache, force=True, archive=archive, revision="test-only")
    assert len(list(data.parent.glob("ravenea.previous_*"))) == 1
