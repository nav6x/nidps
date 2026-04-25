
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss

import datasets
from features import SKEWED, log_skewed

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
N_BINS = 10

def reliability(y_true: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> dict:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    bins = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if not count:
            continue
        conf = float(p[mask].mean())
        obs = float(y_true[mask].mean())
        bins.append({"count": count, "mean_pred": conf, "obs_freq": obs})
        gap = abs(conf - obs)
        ece += count / len(p) * gap
        mce = max(mce, gap)

    return {
        "bins": bins,
        "ece": ece,
        "mce": mce,
        "brier": float(brier_score_loss(y_true, p)),
    }

def run(key: str) -> dict:
    split = datasets.load(key)
    model = joblib.load(RESULTS / f"model-{key}-random-forest.joblib")
    Xte = log_skewed(split.X_test, SKEWED)
    p = model.predict_proba(Xte)[:, 1]
    rel = reliability(split.y_test.to_numpy(), p)
    rel["dataset"] = split.name
    print(f"{split.name:11s}  Brier={rel['brier']:.4f}  "
          f"ECE={rel['ece']:.4f}  MCE={rel['mce']:.4f}")
    return rel

def main() -> None:
    out = {key: run(key) for key in ("nsl-kdd", "unsw-nb15")}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {RESULTS / 'calibration.json'}")

if __name__ == "__main__":
    main()
