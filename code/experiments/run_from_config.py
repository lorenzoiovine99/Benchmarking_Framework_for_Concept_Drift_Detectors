from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# IMPORTANT: import learner first (starts JVM with correct classpath)
from slidingheatmap_capymoa.sliding_heatmap import SlidingHeatmapClassifier

from capymoa.stream import stream_from_file
from capymoa.drift.detectors import ADWIN

from benchmarking.metrics.drift_metrics import compute_drift_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (has code/ and datasets/)


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (PROJECT_ROOT / pp).resolve()

def _normalize_csv_labels(
    src_csv: Path,
    dst_csv: Path,
    class_index: int,
) -> Path:
    """
    Create a normalized copy of src_csv where the class column is mapped to 0..K-1.
    Keeps header and all feature values unchanged.
    """
    dst_csv.parent.mkdir(parents=True, exist_ok=True)

    # If already exists, reuse (deterministic for fixed source file)
    if dst_csv.exists():
        return dst_csv

    # 1) First pass: collect unique labels (as strings to be safe)
    labels = set()
    with open(src_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        idx = class_index if class_index >= 0 else (len(header) + class_index)
        for row in r:
            if not row:
                continue
            labels.add(row[idx])

    # stable mapping
    sorted_labels = sorted(labels, key=lambda x: float(x) if x.replace(".","",1).isdigit() else x)
    mapping = {lab: str(i) for i, lab in enumerate(sorted_labels)}

    # 2) Second pass: write normalized csv
    with open(src_csv, "r", encoding="utf-8", newline="") as fin, open(dst_csv, "w", encoding="utf-8", newline="") as fout:
        r = csv.reader(fin)
        w = csv.writer(fout)
        header = next(r)
        w.writerow(header)

        idx = class_index if class_index >= 0 else (len(header) + class_index)
        for row in r:
            if not row:
                continue
            row[idx] = mapping[row[idx]]
            w.writerow(row)

    return dst_csv


def expand_grid(params_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(params_grid.keys())
    values = [params_grid[k] for k in keys]
    return [dict(zip(keys, prod)) for prod in itertools.product(*values)]


def load_metadata(metadata_path: Path) -> Tuple[List[int], int]:
    """
    Returns (drift_points, max_instances).
    Supports:
      - YAML: fields: drifts, max_instances
      - JSON: fields: n_samples + schedule with AbruptDrift positions
    """
    if metadata_path.suffix.lower() in [".yaml", ".yml"]:
        obj = load_yaml(metadata_path)
        drifts = list(map(int, obj["drifts"]))
        max_instances = int(obj["max_instances"])
        return drifts, max_instances

    if metadata_path.suffix.lower() == ".json":
        obj = load_json(metadata_path)
        max_instances = int(obj["n_samples"])
        drifts: List[int] = []
        for step in obj.get("schedule", []):
            if step.get("type") == "AbruptDrift" and "position" in step:
                drifts.append(int(step["position"]))
        return drifts, max_instances

    raise ValueError(f"Unsupported metadata format: {metadata_path}")


def make_stream(dataset_cfg: Dict[str, Any]):
    data_path = resolve_path(dataset_cfg["path"])
    dataset_name = dataset_cfg.get("name", data_path.stem)
    class_index = int(dataset_cfg.get("class_index", -1))
    target_type = dataset_cfg.get("target_type", "categorical")

    # --- normalize labels for real CSVs if requested ---
    if dataset_cfg.get("normalize_labels", False) and target_type == "categorical":
        norm_dir = resolve_path("tmp/normalized_datasets")
        norm_path = norm_dir / f"{dataset_name}.normalized.csv"
        data_path = _normalize_csv_labels(data_path, norm_path, class_index)

    return stream_from_file(
        str(data_path),
        dataset_name=dataset_name,
        class_index=class_index,
        target_type=target_type,
    )


def label_error_01(instance, prediction) -> int:
    # detector input: 1 if wrong, 0 if correct
    if prediction is None:
        return 1
    return int(instance.y_index != prediction)


def append_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_one(cfg: Dict[str, Any], learner_params: Dict[str, Any]) -> Dict[str, Any]:
    run_cfg = cfg["run"]
    dataset_cfg = cfg["dataset"]
    detector_cfg = cfg["detector"]

    metadata_path = resolve_path(dataset_cfg["metadata_path"])
    drift_points, max_instances_meta = load_metadata(metadata_path)

    max_instances = int(run_cfg.get("max_instances_override", max_instances_meta))

    stream = make_stream(dataset_cfg)

    learner = SlidingHeatmapClassifier(
        schema=stream.get_schema(),
        random_seed=int(run_cfg.get("seed", 1)),
        n_bins=int(learner_params["n_bins"]),
        buffer_window=int(learner_params["buffer_window"]),
    )

    # Detector: for now only ADWIN, extend later with a factory
    if detector_cfg["name"].upper() != "ADWIN":
        raise ValueError(f"Unsupported detector: {detector_cfg['name']}")
    delta = float(detector_cfg.get("params", {}).get("delta", 0.002))
    detector = ADWIN(delta=delta)

    detections: List[int] = []
    correct = 0
    seen = 0

    # manual prequential loop to collect detections
    i = 0
    while stream.has_more_instances() and i < max_instances:
        inst = stream.next_instance()
        y_pred = learner.predict(inst)

        err01 = label_error_01(inst, y_pred)
        if err01 == 0:
            correct += 1

        detector.add_element(float(err01))
        if detector.detected_change():
            detections.append(i)

        learner.train(inst)
        i += 1
        seen += 1

    accuracy = (correct / seen) if seen > 0 else 0.0

    drift_m = compute_drift_metrics(
        drifts=drift_points,
        detections=detections,
        buffer_window=int(learner_params["buffer_window"]),
    )

    row = {
        "dataset": dataset_cfg.get("name", dataset_cfg["path"]),
        "detector": detector_cfg["name"],
        "n_bins": int(learner_params["n_bins"]),
        "buffer_window": int(learner_params["buffer_window"]),
        "seed": int(run_cfg.get("seed", 1)),
        "max_instances": max_instances,

        # learner / prediction
        "accuracy": accuracy,
        "random_fallbacks": int(getattr(learner, "random_count", 0)),

        # drift metrics
        "TDR": drift_m.tdr,
        "FAR": drift_m.far,
        "MTD_norm": drift_m.mtd_norm,
        "M1": drift_m.m1,
        "true_det": drift_m.true_detections,
        "false_alarm": drift_m.false_alarms,
        "repeated_alarm": drift_m.repeated_alarms,
        "detected_drifts": drift_m.detected_drifts,
        "total_drifts": drift_m.total_drifts,
        "n_detections": len(detections),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = resolve_path(args.config)
    cfg = load_yaml(cfg_path)

    grid = expand_grid(cfg["learner"]["params_grid"])

    out_grid = resolve_path(cfg["run"]["output_grid_csv"])
    out_summary = resolve_path(cfg["run"]["output_summary_csv"])

    rows: List[Dict[str, Any]] = []
    for i, learner_params in enumerate(grid, start=1):
        row = run_one(cfg, learner_params)
        rows.append(row)
        append_csv(out_grid, list(row.keys()), row)
        print(f"[{i}/{len(grid)}] {row}")

    # Summary over the grid (mean of metrics across configurations)
    summary = {
        "dataset": rows[0]["dataset"] if rows else "",
        "detector": rows[0]["detector"] if rows else "",
        "grid_size": len(rows),

        "accuracy_mean": mean([r["accuracy"] for r in rows]),
        "random_fallbacks_mean": mean([float(r["random_fallbacks"]) for r in rows]),

        "TDR_mean": mean([r["TDR"] for r in rows]),
        "FAR_mean": mean([r["FAR"] for r in rows]),
        "MTD_norm_mean": mean([r["MTD_norm"] for r in rows]),
        "M1_mean": mean([r["M1"] for r in rows]),
    }

    append_csv(out_summary, list(summary.keys()), summary)

    print("\n=== SUMMARY (mean over grid) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved grid to: {out_grid}")
    print(f"Saved summary to: {out_summary}")


if __name__ == "__main__":
    main()
