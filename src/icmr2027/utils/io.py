"""UTF-8 artifacts and collision-safe run directories."""
from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_yaml(path: str | Path, value: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def create_run_directory(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = root / f"{stamp}_{uuid4().hex[:8]}"
        try:
            run.mkdir()  # Atomic allocation: never reuse or overwrite a run.
        except FileExistsError:
            continue
        return run
