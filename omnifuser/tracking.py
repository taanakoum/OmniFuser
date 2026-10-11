from __future__ import annotations

import csv
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _numeric_leaves(
    value: Any, prefix: str = ""
) -> Iterator[tuple[str, float]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _numeric_leaves(child, f"{prefix}.{index}")
    elif isinstance(value, bool):
        yield prefix, float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        yield prefix, float(value)


class ExperimentTracker:
    """Append-only, dependency-free experiment logs for local and hosted runs."""

    def __init__(
        self,
        output_dir: str | Path,
        metadata: dict[str, Any],
    ) -> None:
        self.directory = Path(output_dir) / "tracking"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "run.json"
        if self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            if not isinstance(manifest, dict) or "run_id" not in manifest:
                raise ValueError(f"invalid experiment tracking manifest: {self.manifest_path}")
            self.run_id = str(manifest["run_id"])
            manifest["updated_at_utc"] = _utc_now()
            manifest["metadata"] = metadata
        else:
            self.run_id = uuid.uuid4().hex
            manifest = {
                "run_id": self.run_id,
                "created_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "metadata": metadata,
            }
        with self.manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, allow_nan=False)

        self.events_path = self.directory / "metrics.jsonl"
        self.csv_path = self.directory / "metrics.csv"

    def log(
        self,
        *,
        split: str,
        metrics: dict[str, Any],
        fold: str | None = None,
        epoch: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp_utc": _utc_now(),
            "run_id": self.run_id,
            "fold": fold,
            "epoch": epoch,
            "split": split,
            "context": context or {},
            "metrics": metrics,
        }
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")

        rows = [
            {
                "timestamp_utc": event["timestamp_utc"],
                "run_id": self.run_id,
                "fold": fold or "",
                "epoch": "" if epoch is None else epoch,
                "split": split,
                "metric": name,
                "value": value,
            }
            for name, value in _numeric_leaves(metrics)
        ]
        if rows:
            write_header = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=(
                        "timestamp_utc",
                        "run_id",
                        "fold",
                        "epoch",
                        "split",
                        "metric",
                        "value",
                    ),
                )
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
