
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.ensemble import IsolationForest

import datasets
import evaluate as ev
from features import SKEWED, build_preprocessor, log_skewed

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 42
BUDGET = 0.01

def _dense(M):
    return M.toarray() if issparse(M) else np.asarray(M)

def _anomaly_score(iforest, X) -> np.ndarray:
    return -iforest.decision_function(X)

def run(key: str) -> dict:
    split = datasets.load(key)
    novel = split.novel_labels
    Xtr = log_skewed(split.X_train, SKEWED)
    Xte = log_skewed(split.X_test, SKEWED)
    yte = split.y_test

    prep = build_preprocessor(Xtr, split.categorical).fit(Xtr)
    Xtr_t = _dense(prep.transform(Xtr))
    Xte_t = _dense(prep.transform(Xte))

    benign = split.y_train.to_numpy() == 0
    iforest = IsolationForest(
        n_estimators=300, max_samples="auto", contamination="auto",
        random_state=SEED, n_jobs=-1,
    ).fit(Xtr_t[benign])
    if_score = _anomaly_score(iforest, Xte_t)

    rf_pipe = joblib.load(RESULTS / f"model-{key}-random-forest.joblib")
    rf_score = rf_pipe.predict_proba(Xte)[:, 1]

    reports: dict[str, ev.Report] = {}

    rf_thr = ev.threshold_for_fpr(yte, rf_score, BUDGET)
    reports["random-forest"] = ev.evaluate(
        yte, (rf_score >= rf_thr).astype(int), rf_score,
        model="random-forest", dataset=split.name,
        protocol=f"novelty@fpr={BUDGET}", labels=split.test_labels, novel=novel)

    if_thr = ev.threshold_for_fpr(yte, if_score, BUDGET)
    reports["isolation-forest"] = ev.evaluate(
        yte, (if_score >= if_thr).astype(int), if_score,
        model="isolation-forest", dataset=split.name,
        protocol=f"novelty@fpr={BUDGET}", labels=split.test_labels, novel=novel)

    rf_half = ev.threshold_for_fpr(yte, rf_score, BUDGET / 2)
    if_half = ev.threshold_for_fpr(yte, if_score, BUDGET / 2)
    hybrid_pred = ((rf_score >= rf_half) | (if_score >= if_half)).astype(int)
    union_score = np.maximum(_rankscale(rf_score), _rankscale(if_score))
    reports["hybrid-rf-or-if"] = ev.evaluate(
        yte, hybrid_pred, union_score,
        model="hybrid-rf-or-if", dataset=split.name,
        protocol=f"novelty@fpr={BUDGET}", labels=split.test_labels, novel=novel)

    print(f"\n=== {split.name}: {len(novel)} novel families ===")
    header = f"  {'detector':18s} {'recall':>8s} {'novel':>8s} {'known':>8s} {'FPR':>8s} {'fa/10k':>8s}"
    print(header)
    for name, r in reports.items():
        nov = "  N/A " if r.novel_recall is None else f"{r.novel_recall:.3f}"
        kno = "  N/A " if r.known_recall is None else f"{r.known_recall:.3f}"
        print(f"  {name:18s} {r.recall:8.3f} {nov:>8s} {kno:>8s} "
              f"{r.fpr:8.4f} {r.false_alerts_per_10k:8.1f}")

    return {name: r.to_dict() for name, r in reports.items()}

def _rankscale(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1) if len(x) > 1 else x

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["nsl-kdd", "unsw-nb15"])
    args = ap.parse_args()

    out = {key: run(key) for key in args.dataset}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "novelty.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {RESULTS / 'novelty.json'}")

if __name__ == "__main__":
    main()
