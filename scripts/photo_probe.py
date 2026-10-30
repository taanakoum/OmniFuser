"""How much wear-class information frozen ImageNet features keep at each photo setting.

A linear probe, without training OmniFuser: every tool photo goes through the frozen,
pretrained EfficientNet-B0 at a given input size and crop. A multinomial logistic
regression on the pooled features then predicts the photo's wear class, leave-one-tool-out
(trained on 9 tools, tested on the 10th). Settings that keep more of the thin wear band
at the cutting edge should score higher. Average and max pooling of the last feature map
are both probed, because a thin band covers few cells of the map.

It also locates the cutting edge in every photo (the row of the strongest vertical
brightness change), to check that a fixed crop contains it.

Example:
    python scripts/photo_probe.py --data-dir Nonastreda_Data
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from omnifuser.data import CLASS_TO_INDEX, IMAGE_MEAN, IMAGE_STD, Crop

EDGE_CROP: Crop = (0.30, 0.0, 0.85, 1.0)
SETTINGS: dict[str, tuple[int, int, Crop | None]] = {
    "full-128x384": (128, 384, None),  # the main runs
    "full-256x768": (256, 768, None),
    "full-512x1536": (512, 1536, None),
    "crop-128x640": (128, 640, EDGE_CROP),
    "crop-256x1280": (256, 1280, EDGE_CROP),
}
PENALTIES = (1e-3, 1e-2, 1e-1)
PRIMARY_PENALTY = 1e-2  # fixed before looking at the results; the others show robustness


def _pixels(image: Image.Image, height: int, width: int, crop: Crop | None) -> torch.Tensor:
    if crop is not None:
        top, left, bottom, right = crop
        image = image.crop((round(left * image.width), round(top * image.height),
                            round(right * image.width), round(bottom * image.height)))
    array = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(((array - IMAGE_MEAN) / IMAGE_STD).transpose(2, 0, 1).copy())


def _edge_row(image: Image.Image) -> float:
    """Row (fraction of the height) with the largest change of mean brightness."""
    gray = np.asarray(image.convert("L").resize((400, 200)), dtype=np.float32)
    profile = gray[:, 100:300].mean(axis=1)
    step = np.abs(np.diff(np.convolve(profile, np.ones(5) / 5, mode="same")))
    return float((np.argmax(step[5:-5]) + 5.5) / len(profile))


def extract(data_dir: Path, cache_dir: Path, threads: int) -> dict[str, Any]:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    torch.set_num_threads(threads)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {name: cache_dir / f"features-{name}.npz" for name in SETTINGS}
    with (data_dir / "labels.csv").open(encoding="utf-8-sig", newline="") as file:
        labels = {row["id"]: CLASS_TO_INDEX[row["image_label"].strip().lower()] for row in csv.DictReader(file)}
    ids = sorted(labels, key=lambda name: (int(name[1:].split("R")[0]), name))
    edges_path = cache_dir / "edge-rows.json"
    if all(path.is_file() for path in cached.values()) and edges_path.is_file():
        return {"ids": ids, "labels": labels, "edges": json.loads(edges_path.read_text(encoding="utf-8"))}
    features = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features.eval()
    pooled: dict[str, dict[str, list[np.ndarray]]] = {name: {"avg": [], "max": []} for name in SETTINGS}
    edges: dict[str, float] = {}
    pending: dict[str, list[torch.Tensor]] = {name: [] for name in SETTINGS}

    def flush(name: str) -> None:
        if not pending[name]:
            return
        with torch.no_grad():
            maps = features(torch.stack(pending[name]))
        pooled[name]["avg"].append(maps.mean(dim=(2, 3)).numpy())
        pooled[name]["max"].append(maps.amax(dim=(2, 3)).numpy())
        pending[name].clear()

    for position, record_id in enumerate(ids, start=1):
        with Image.open(data_dir / "tool" / f"{record_id}.jpg") as source:
            image = source.convert("RGB")
        edges[record_id] = _edge_row(image)
        for name, (height, width, crop) in SETTINGS.items():
            pending[name].append(_pixels(image, height, width, crop))
            if len(pending[name]) == 8:
                flush(name)
        if position % 64 == 0:
            print(f"  {position}/{len(ids)} photos", flush=True)
    for name in SETTINGS:
        flush(name)
        np.savez_compressed(cached[name], ids=np.asarray(ids),
                            avg=np.concatenate(pooled[name]["avg"]), max=np.concatenate(pooled[name]["max"]))
    edges_path.write_text(json.dumps(edges, indent=1), encoding="utf-8")
    return {"ids": ids, "labels": labels, "edges": edges}


def _logistic_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, penalty: float) -> np.ndarray:
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    x = torch.from_numpy((train_x - mean) / std).float()
    y = torch.from_numpy(train_y).long()
    weight = torch.zeros(x.shape[1], 3, requires_grad=True)
    bias = torch.zeros(3, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(x @ weight + bias, y) + 0.5 * penalty * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        logits = torch.from_numpy((test_x - mean) / std).float() @ weight + bias
    return logits.argmax(dim=1).numpy()


def probe(cache_dir: Path, ids: list[str], labels: dict[str, int]) -> dict[str, Any]:
    tools = np.asarray([name.split("R")[0] for name in ids])
    y = np.asarray([labels[name] for name in ids])
    results: dict[str, Any] = {}
    for name in SETTINGS:
        with np.load(cache_dir / f"features-{name}.npz") as arrays:
            assert list(arrays["ids"]) == ids
            pools = {"avg": arrays["avg"], "max": arrays["max"]}
        for pool, x in pools.items():
            for penalty in PENALTIES:
                per_tool: dict[str, float] = {}
                predicted = np.empty_like(y)
                for tool in sorted(set(tools), key=lambda t: int(t[1:])):
                    test = tools == tool
                    predicted[test] = _logistic_probe(x[~test], y[~test], x[test], penalty)
                    per_tool[tool] = float((predicted[test] == y[test]).mean())
                recalls = [float((predicted[y == k] == k).mean()) for k in range(3)]
                results[f"{name}/{pool}/{penalty:g}"] = {
                    "setting": name, "pool": pool, "penalty": penalty,
                    "accuracy_tool_mean": float(np.mean(list(per_tool.values()))),
                    "accuracy_pooled": float((predicted == y).mean()),
                    "recall_by_class": recalls,
                    "per_tool": per_tool,
                }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results/photo-probe"))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    data = extract(args.data_dir, args.output_dir, args.threads)
    results = probe(args.output_dir, data["ids"], data["labels"])
    edges = np.asarray(list(data["edges"].values()))
    summary = {
        "edge_row_fraction": {q: float(np.quantile(edges, float(q))) for q in ("0.01", "0.05", "0.5", "0.95", "0.99")},
        "edge_rows_inside_crop": float(((edges > EDGE_CROP[0]) & (edges < EDGE_CROP[2])).mean()),
        "crop": EDGE_CROP,
        "primary_penalty": PRIMARY_PENALTY,
        "probes": results,
    }
    (args.output_dir / "photo-probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("edge row (fraction of height):", summary["edge_row_fraction"],
          f"inside crop: {summary['edge_rows_inside_crop']:.3f}")
    for key, row in results.items():
        marker = " *" if row["penalty"] == PRIMARY_PENALTY else ""
        print(f"{key:32s} tool-mean {row['accuracy_tool_mean']:.3f} pooled {row['accuracy_pooled']:.3f} "
              f"recall {np.round(row['recall_by_class'], 2).tolist()}{marker}")


if __name__ == "__main__":
    main()
