"""Calibrate thresholds on the labelled sample recordings.

Usage:  python calibrate.py /path/to/samples [--model small] [--holdout N]

--holdout N keeps N recordings (round-robin across labels) out of the tuning tables: means, Spearman
correlations and the printed tuning accuracy use the rest, and exact / within-one-level accuracy is
reported separately for the held-out recordings as an unseen test.

Folder layout (either works):
    samples/Excellent/*.mp4  samples/Good/*.mp4  samples/Average/*.mp4  samples/Poor/*.mp4
    samples/Excellent_class1.mp4 ...   (label as a prefix/word in the file name)

Prints a feature-by-label table (which signals actually separate the levels), the model's prediction for each
recording, and writes calibration.csv. Edit config.json (thresholds / weights) from what you learn.
Per-recording results are cached in .cache/ so reruns are instant.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from assessor.pipeline import analyze

LABELS = ["Poor", "Average", "Good", "Excellent"]
EXT = {".mp4", ".mkv", ".mov", ".webm", ".m4a", ".mp3", ".wav"}


def label_of(p: Path):
    for part in [p.parent.name, p.stem]:
        for l in LABELS:
            if re.search(rf"\b{l}\b", part, re.I) or part.lower().startswith(l.lower()):
                return l
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--model", default="small")
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--holdout", type=int, default=0, metavar="N",
                    help="Hold N recordings back as an unseen test: tuning tables and correlations use only the "
                         "remaining files, and exact / within-one-level accuracy is printed separately for the held-out set.")
    a = ap.parse_args()
    cache = Path(".cache")
    cache.mkdir(exist_ok=True)
    files = [p for p in sorted(Path(a.folder).rglob("*")) if p.suffix.lower() in EXT]
    # pick the held-out set up front, round-robin across labels so every level appears in the unseen test
    hold: set[Path] = set()
    if a.holdout > 0 and files:
        groups: dict = {}
        for p in files:
            groups.setdefault(label_of(p), []).append(p)
        while len(hold) < min(a.holdout, len(files)):
            progressed = False
            for lab in LABELS + [None]:
                nxt = [p for p in groups.get(lab, []) if p not in hold]
                if nxt and len(hold) < a.holdout:
                    hold.add(nxt[0])
                    progressed = True
            if not progressed:
                break
    rows = []
    for p in files:
        true = label_of(p)
        cp = cache / (p.stem + f".{a.model}.json")
        if cp.exists():
            cached = json.loads(cp.read_text())
        else:
            print("analysing", p.name)
            cached = analyze(str(p), a.model, a.frames, "none")
            cached.pop("transcript", None)
            cached.setdefault("evidence", {})
            cp.write_text(json.dumps(cached, default=float))
        if "features" in cached:
            # Re-score from cached features with the CURRENT config.json, so editing thresholds/weights
            # takes effect immediately without re-transcribing the recording.
            from assessor.scoring import assess

            f = dict(cached["features"])
            f["_evidence"] = cached.get("evidence", {})
            rep = assess(f, None)
            for k in ("language", "diarization", "notes"):
                if k in cached:
                    rep[k] = cached[k]
        else:
            rep = cached
        row = {"file": p.name, "true": true, "pred": rep["label"], "score": rep["score"],
               "set": "holdout" if p in hold else "tune"}
        row.update({k: v for k, v in rep["features"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("calibration.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPredictions\n", df[["file", "true", "pred", "score", "set"]].to_string(index=False))
    tune = df[df["set"] == "tune"] if len(hold) else df
    if tune["true"].notna().any():
        order = [l for l in LABELS if l in set(tune["true"])]
        print("\nMean of each signal by true label (look for monotone trends = useful; flat or inverted = misleading):" +
              ("  [tuning set only]" if len(hold) else ""))
        g = tune.groupby("true").mean(numeric_only=True).reindex(order).T
        print(g.round(2).to_string())
        print("\nSpearman correlation of each signal with quality level:")
        rank = tune["true"].map({l: i for i, l in enumerate(LABELS)})
        print(tune.drop(columns=["file", "true", "pred", "set"]).corrwith(rank, method="spearman").sort_values().round(2).to_string())

    def acc(part):
        part = part[part["true"].notna()]
        if not len(part):
            return None
        ok = (part["true"] == part["pred"]).mean()
        off1 = (part["true"].map(LABELS.index) - part["pred"].map(LABELS.index)).abs().le(1).mean()
        return ok, off1, len(part)

    res = acc(tune)
    if res:
        print(f"\nExact match {res[0]:.0%}, within one level {res[1]:.0%}" +
              (f"  [tuning set: {res[2]} recordings]" if len(hold) else ""))
    if len(hold):
        held = df[df["set"] == "holdout"]
        print(f"\nHeld-out (unseen) set, {len(held)} recordings:" if len(held) else "")
        if len(held):
            print(held[["file", "true", "pred", "score"]].to_string(index=False))
            hres = acc(held)
            if hres:
                print(f"Held-out exact match {hres[0]:.0%}, within one level {hres[1]:.0%}")


if __name__ == "__main__":
    main()
