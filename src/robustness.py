
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import datasets
from features import SKEWED, log_skewed
from train import fit_and_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

def models(seed: int) -> dict:
    return {
        "logistic-regression": LogisticRegression(max_iter=2000, random_state=seed),
        "decision-tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=seed),
        "random-forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
    }

def one_seed(key: str, seed: int) -> dict:
    split = datasets.load(key)
    Xtr = log_skewed(split.X_train, SKEWED)
    Xte = log_skewed(split.X_test, SKEWED)

    X_all = pd.concat([Xtr, Xte], ignore_index=True)
    y_all = pd.concat([split.y_train, split.y_test], ignore_index=True)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X_all, y_all, test_size=0.25, stratify=y_all, random_state=seed)

    out = {}
    for name, clf in models(seed).items():
        _, y_pred_r, _, _, _ = fit_and_score(
            name, clf, Xa_tr, ya_tr, Xa_te, split.categorical)
        rnd = float((y_pred_r == ya_te.to_numpy()).mean())

        _, y_pred_o, _, _, _ = fit_and_score(
            name, models(seed)[name], Xtr, split.y_train, Xte, split.categorical)
        off = float((y_pred_o == split.y_test.to_numpy()).mean())

        out[name] = {"random": rnd, "official": off, "gap": rnd - off}
        print(f"  seed {seed}  {name:22s} random={rnd:.4f} "
              f"official={off:.4f} gap={rnd-off:+.4f}")
    return out

def summarise(per_seed: list[dict], model: str) -> dict:
    rnd = [s[model]["random"] for s in per_seed]
    off = [s[model]["official"] for s in per_seed]
    gap = [s[model]["gap"] for s in per_seed]
    sd = statistics.stdev if len(gap) > 1 else (lambda _x: 0.0)
    return {
        "random_mean": statistics.mean(rnd), "random_std": sd(rnd),
        "official_mean": statistics.mean(off), "official_std": sd(off),
        "gap_mean": statistics.mean(gap), "gap_std": sd(gap),
        "gap_min": min(gap), "gap_max": max(gap), "n_seeds": len(gap),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nsl-kdd")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))

    print(f"=== {args.dataset}: optimism gap across seeds {seeds} ===")
    per_seed = [one_seed(args.dataset, s) for s in seeds]

    model_names = list(per_seed[0].keys())
    summary = {m: summarise(per_seed, m) for m in model_names}

    print(f"\n--- gap summary ({len(seeds)} seeds) ---")
    for m in model_names:
        s = summary[m]
        print(f"{m:22s} gap {s['gap_mean']:+.4f} ± {s['gap_std']:.4f}  "
              f"(range {s['gap_min']:+.4f}..{s['gap_max']:+.4f})")

    RESULTS.mkdir(exist_ok=True)
    payload = {"dataset": datasets.load(args.dataset).name, "seeds": seeds,
               "per_seed": per_seed, "summary": summary}
    with open(RESULTS / f"robustness-{args.dataset}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {RESULTS / f'robustness-{args.dataset}.json'}")

if __name__ == "__main__":
    main()
