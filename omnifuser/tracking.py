from __future__ import annotations

import csv
import json
import math
import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CSV_FIELDS: tuple[str, ...] = (
    "timestamp_utc",
    "run_id",
    "fold",
    "epoch",
    "split",
    "evaluation_mode",
    "metric",
    "value",
)
# Hardware/library details may change when a run resumes on another machine.
RUNTIME_METADATA_KEY = "runtime"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _comparable(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in metadata.items() if key != RUNTIME_METADATA_KEY
    }


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
    elif isinstance(value, bool) or isinstance(value, (int, float)) and math.isfinite(float(value)):
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
            recorded = manifest.get("metadata")
            if not isinstance(recorded, dict) or _comparable(recorded) != _comparable(
                metadata
            ):
                raise ValueError(
                    "experiment output directory belongs to a different configuration; "
                    f"use a new output directory or resume with matching settings: {self.directory.parent}"
                )
            self.run_id = str(manifest["run_id"])
            manifest["updated_at_utc"] = _utc_now()
            runtime = metadata.get(RUNTIME_METADATA_KEY)
            history = manifest.setdefault("runtime_history", [])
            if runtime != (history[-1]["runtime"] if history else recorded.get(RUNTIME_METADATA_KEY)):
                history.append({"resumed_at_utc": _utc_now(), "runtime": runtime})
        else:
            self.run_id = uuid.uuid4().hex
            manifest = {
                "run_id": self.run_id,
                "created_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "metadata": metadata,
            }
        # Replace atomically, so a job killed mid-write keeps the previous manifest.
        temporary_manifest = self.manifest_path.with_suffix(".json.tmp")
        with temporary_manifest.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, allow_nan=False)
        os.replace(temporary_manifest, self.manifest_path)

        self.events_path = self.directory / "metrics.jsonl"
        self.csv_path = self.directory / "metrics.csv"
        self._migrate_csv_context_column()

    def _migrate_csv_context_column(self) -> None:
        if not self.csv_path.is_file():
            return
        fieldnames = CSV_FIELDS
        with self.csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fields = reader.fieldnames
            if existing_fields == list(fieldnames):
                return
            legacy_fields = [
                name for name in fieldnames if name != "evaluation_mode"
            ]
            if existing_fields != legacy_fields:
                raise ValueError(
                    f"unsupported experiment metrics CSV schema: {self.csv_path}"
                )
            rows = list(reader)
        temporary_path = self.csv_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "evaluation_mode": ""})
        os.replace(temporary_path, self.csv_path)

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
                "evaluation_mode": (context or {}).get("evaluation_mode", ""),
                "metric": name,
                "value": value,
            }
            for name, value in _numeric_leaves(metrics)
        ]
        if rows:
            write_header = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
