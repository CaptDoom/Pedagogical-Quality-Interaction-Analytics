"""Evaluate the rubric with leave-one-recording-out validation.

Usage: python calibrate.py samples --model small [--frames]

Media are labeled from a parent folder or filename containing Poor, Average, Good, or Excellent.
Every held-out score is predicted using thresholds fitted only on the other recordings.
Per-recording analyses are cached in .cache/; cache entries are invalidated when the analysis schema changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from assessor.scoring import assess, label_for, load_config

LABELS = ["Poor", "Average", "Good", "Excellent"]
EXT = {".mp4", ".mkv", ".mov", ".webm", ".m4a", ".mp3", ".wav"}
CACHE_VERSION = 5


def label_of(path: Path) -> str | None:
    for part in (path.parent.name, path.stem):
        if re.search(r"\bquality[\s_-]+issue\b", part, re.I):
            return "Average"
        for label in LABELS:
            if re.search(rf"\b{label}\b", part, re.I) or part.lower().startswith(label.lower()):
                return label
    return None


def _cache_path(path: Path, model: str, frames: bool) -> Path:
    identity = f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}:{model}:{frames}:{CACHE_VERSION}"
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return Path(".cache") / f"{key}.json"


def _threshold_candidates(scores: list[float], current: float) -> list[float]:
    unique = sorted(set(float(np.clip(value, 1, 5)) for value in scores))
    candidates = {1.0, 5.0, float(np.clip(current, 1, 5))}
    candidates.update((left + right) / 2 for left, right in zip(unique, unique[1:]))
    return sorted(candidates)


def fit_thresholds(scores: list[float], labels: list[str], current: dict) -> dict:
    """Search score cutoffs while keeping boundaries unidentifiable from training labels unchanged."""
    result = dict(current)
    label_ranks = [LABELS.index(label) for label in labels]
    keys = ("Poor", "Average", "Good")
    candidates = []
    for boundary, key in enumerate(keys, start=1):
        if boundary - 1 in label_ranks and boundary in label_ranks:
            candidates.append(_threshold_candidates(scores, float(current[key])))
        else:
            candidates.append([float(current[key])])

    best_key = None
    best_thresholds = None
    for poor in candidates[0]:
        for average in candidates[1]:
            if average < poor:
                continue
            for good in candidates[2]:
                if good < average:
                    continue
                trial = {"Poor": poor, "Average": average, "Good": good}
                predicted = [LABELS.index(label_for(score, trial)) for score in scores]
                errors = [abs(actual - guess) for actual, guess in zip(label_ranks, predicted)]
                objective = (
                    sum(errors) / max(1, len(errors)),
                    -sum(error == 0 for error in errors),
                    sum(abs(trial[key] - float(current[key])) for key in keys),
                )
                if best_key is None or objective < best_key:
                    best_key, best_thresholds = objective, trial
    return best_thresholds or result


def fit_dimension_weights(rows: list[dict], base_weights: dict) -> dict:
    """Down-weight observed dimensions whose median score is not monotone across at least three levels."""
    fitted = dict(base_weights)
    for name, weight in base_weights.items():
        observations = [
            (LABELS.index(row["true"]), row["dimensions"].get(name))
            for row in rows if row["true"] in LABELS and row["dimensions"].get(name) is not None
        ]
        by_level = {
            rank: float(np.median([value for item_rank, value in observations if item_rank == rank]))
            for rank in sorted({rank for rank, _ in observations})
        }
        if len(by_level) < 3:
            continue
        changes = np.diff(list(by_level.values()))
        if np.allclose(changes, 0.0):
            fitted[name] = 0.0
        elif not (np.all(changes >= 0) or np.all(changes <= 0)):
            fitted[name] = float(weight) * 0.5
    return fitted


def _weighted_score(row: dict, weights: dict) -> float:
    available = [(value, weights.get(name, 0.0)) for name, value in row["dimensions"].items()]
    weighted = [(score, weight) for score, weight in available if weight > 0]
    total_weight = sum(weight for _, weight in weighted)
    return sum(score * weight for score, weight in weighted) / total_weight if total_weight else float(row["score"])


def _monotonicity(frame: pd.DataFrame, labels: pd.Series) -> list[tuple[str, str, float]]:
    rows = []
    ranks = labels.map({label: index for index, label in enumerate(LABELS)}).to_numpy(dtype=float)
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(ranks)
        if int(valid.sum()) < 3 or len(np.unique(ranks[valid])) < 3:
            continue
        correlation = float(pd.Series(values[valid]).corr(pd.Series(ranks[valid]), method="spearman"))
        medians = pd.DataFrame({"value": values[valid], "rank": ranks[valid]}).groupby("rank")["value"].median().to_numpy()
        diffs = np.diff(medians)
        steady = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
        rows.append((column, "monotone" if steady else "down-weight", correlation))
    return sorted(rows, key=lambda row: (row[1] == "monotone", abs(row[2])))


def _loo_predictions(
    rows: list[dict], current_thresholds: dict, base_weights: dict
) -> tuple[list[str], list[float], list[dict], list[dict]]:
    predictions: list[str] = []
    test_scores: list[float] = []
    fold_thresholds: list[dict] = []
    fold_weights: list[dict] = []
    for index, row in enumerate(rows):
        train = [item for position, item in enumerate(rows) if position != index and item["true"] in LABELS]
        weights = fit_dimension_weights(train, base_weights)
        train_scores = [_weighted_score(item, weights) for item in train]
        thresholds = fit_thresholds(
            train_scores,
            [item["true"] for item in train],
            current_thresholds,
        )
        fold_thresholds.append(thresholds)
        fold_weights.append(weights)
        score = _weighted_score(row, weights)
        test_scores.append(score)
        predictions.append(label_for(score, thresholds))
    return predictions, test_scores, fold_thresholds, fold_weights


def _print_accuracy(frame: pd.DataFrame) -> None:
    labeled = frame[frame["true"].isin(LABELS)]
    if labeled.empty:
        print("\nNo labels found; validation metrics are unavailable.")
        return
    true_ranks = labeled["true"].map({label: index for index, label in enumerate(LABELS)}).to_numpy()
    pred_ranks = labeled["pred"].map({label: index for index, label in enumerate(LABELS)}).to_numpy()
    exact = float(np.mean(true_ranks == pred_ranks))
    within_one = float(np.mean(np.abs(true_ranks - pred_ranks) <= 1))
    print(f"\nLeave-one-out exact accuracy: {exact:.1%} ({len(labeled)} recordings)")
    print(f"Leave-one-out within-one-level accuracy: {within_one:.1%}")

    matrix = pd.crosstab(labeled["true"], labeled["pred"]).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(matrix.to_string())

    concordant = comparable = 0.0
    for i in range(len(labeled)):
        for j in range(i + 1, len(labeled)):
            label_delta = true_ranks[i] - true_ranks[j]
            if label_delta == 0:
                continue
            comparable += 1
            score_delta = float(labeled.iloc[i]["score"]) - float(labeled.iloc[j]["score"])
            concordant += 1.0 if label_delta * score_delta > 0 else 0.5 if score_delta == 0 else 0.0
    if comparable:
        print(f"Pairwise ranking concordance: {concordant / comparable:.1%} "
              f"({int(comparable)} differently-labeled pairs)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--model", default="small")
    parser.add_argument("--frames", action="store_true")
    parser.add_argument("--write-config", action="store_true",
                        help="write the full-dataset fitted thresholds and monotone dimension weights to config.json")
    args = parser.parse_args()
    cache = Path(".cache")
    cache.mkdir(exist_ok=True)
    media = [path for path in sorted(Path(args.folder).rglob("*")) if path.suffix.lower() in EXT]
    files = [path for path in media if label_of(path) in LABELS]
    unlabeled = [path for path in media if label_of(path) not in LABELS]
    if not files:
        raise SystemExit(f"No supported media recordings found under {args.folder!r}.")
    if unlabeled:
        print(f"Skipping {len(unlabeled)} recording(s) without a recognized rubric label:")
        for path in unlabeled:
            print(f"  {path}")

    from assessor.pipeline import analyze

    rows = []
    for path in files:
        cache_file = _cache_path(path, args.model, args.frames)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            print(f"Analysing {path.name}")
            cached = analyze(str(path), args.model, args.frames, "none")
            cached.pop("transcript", None)
            cache_file.write_text(json.dumps(cached, default=float), encoding="utf-8")

        features = dict(cached["features"])
        features["_evidence"] = cached.get("evidence", {})
        report = assess(features, None)
        dimensions = {item["name"]: item["score"] for item in report["dimensions"]}
        row = {
            "file": path.name,
            "true": label_of(path),
            "score": report["score"],
            "features": report["features"],
            "dimensions": dimensions,
        }
        rows.append(row)

    labeled = rows
    cfg = load_config()
    predictions, loo_scores, fold_thresholds, fold_weights = _loo_predictions(
        labeled, cfg["thresholds"], cfg["weights"]
    )
    frame = pd.DataFrame([
        {"file": row["file"], "true": row["true"], "raw_score": row["score"],
         "score": loo_score, "pred": pred}
        for row, loo_score, pred in zip(labeled, loo_scores, predictions)
    ])
    export = frame[["file", "true", "score", "pred"]].copy()
    for index, row in enumerate(labeled):
        for name, value in row["features"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                export.loc[index, name] = value
        for name, value in row["dimensions"].items():
            export.loc[index, f"dimension_{name}"] = value
    export.to_csv("calibration.csv", index=False)

    print("\nLeave-one-out predictions")
    print(frame.to_string(index=False))
    _print_accuracy(frame)
    full_weights = fit_dimension_weights(labeled, cfg["weights"])
    full_scores = [_weighted_score(row, full_weights) for row in labeled]
    full_fit = fit_thresholds(full_scores, frame["true"].tolist(), cfg["thresholds"])
    print("\nThresholds fitted on all labeled recordings (not an independent validation):")
    print(json.dumps(full_fit, indent=2))
    print("\nDimension weights fitted from training-label monotonicity:")
    print(json.dumps(full_weights, indent=2))

    feature_frame = pd.DataFrame([row["features"] for row in labeled])
    print("\nSignal monotonicity across observed levels (non-monotone signals should be down-weighted):")
    monotonicity = _monotonicity(feature_frame, frame["true"])
    for feature, status, correlation in monotonicity:
        print(f"  {feature}: {status}, Spearman r={correlation:.2f}")
    present = set(frame["true"])
    missing = [label for label in LABELS if label not in present]
    if missing:
        print(f"\nUntested levels: {', '.join(missing)}. Their adjacent cutoffs cannot be validated.")
    print("\nLOO fold cutoffs:")
    for filename, thresholds, weights in zip(frame["file"], fold_thresholds, fold_weights):
        adjusted = {name: weight for name, weight in weights.items() if weight != cfg["weights"].get(name)}
        print(f"  {filename}: {json.dumps(thresholds, sort_keys=True)}; adjusted weights={json.dumps(adjusted)}")

    if args.write_config:
        config_path = Path(__file__).resolve().parent / "config.json"
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        saved["thresholds"] = full_fit
        saved["weights"] = full_weights
        config_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        print(f"\nUpdated {config_path} with full-dataset fitted thresholds and weights.")


if __name__ == "__main__":
    main()
