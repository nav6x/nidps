
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import datasets
import evaluate as ev
from features import SKEWED, build_preprocessor, log_skewed

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 42

def models() -> dict:
    return {
        "logistic-regression": LogisticRegression(
            max_iter=2000, random_state=SEED),
        "decision-tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=SEED),
        "random-forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1,
            random_state=SEED),
    }

def fit_and_score(name, clf, X_tr, y_tr, X_te, categorical):
    pipe = Pipeline([
        ("prep", build_preprocessor(X_tr, categorical)),
        ("clf", clf),
    ])
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipe.predict(X_te)
    score_s = time.perf_counter() - t0

    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_te)[:, 1]
    else:
        y_score = pipe.decision_function(X_te)

    return pipe, y_pred, y_score, fit_s, score_s

def run_dataset(key: str) -> tuple[list[ev.Report], dict]:
    split = datasets.load(key)
    print(f"\n=== {split.name} ===")
    print(f"train {split.X_train.shape}  test {split.X_test.shape}")
    print(f"attack rate  train {split.y_train.mean():.3f}  "
          f"test {split.y_test.mean():.3f}")
    novel = split.novel_labels
    print(f"attack families in test but not in train: {len(novel)}")
    if novel:
        print("  " + ", ".join(sorted(novel)))

    Xtr = log_skewed(split.X_train, SKEWED)
    Xte = log_skewed(split.X_test, SKEWED)

    reports: list[ev.Report] = []
    timings: dict = {}

    X_all = pd.concat([Xtr, Xte], ignore_index=True)
    y_all = pd.concat([split.y_train, split.y_test], ignore_index=True)
    lab_all = pd.concat([split.train_labels, split.test_labels],
                        ignore_index=True)

    Xa_tr, Xa_te, ya_tr, ya_te, _, laba_te = train_test_split(
        X_all, y_all, lab_all, test_size=0.25, stratify=y_all,
        random_state=SEED)

    for name, clf in models().items():
        _, y_pred, y_score, fit_s, score_s = fit_and_score(
            name, clf, Xa_tr, ya_tr, Xa_te, split.categorical)
        rep = ev.evaluate(ya_te, y_pred, y_score, model=name,
                          dataset=split.name, protocol="random-split",
                          labels=laba_te, novel=None)
        reports.append(rep)
        timings[f"random/{name}"] = {"fit_s": fit_s, "score_s": score_s}
        print(f"  [random ] {name:22s} acc={rep.accuracy:.4f} "
              f"f1={rep.f1:.4f} fpr={rep.fpr:.4f}")

    for name, clf in models().items():
        pipe, y_pred, y_score, fit_s, score_s = fit_and_score(
            name, clf, Xtr, split.y_train, Xte, split.categorical)
        rep = ev.evaluate(split.y_test, y_pred, y_score, model=name,
                          dataset=split.name, protocol="official-split",
                          labels=split.test_labels, novel=novel)
        reports.append(rep)
        timings[f"official/{name}"] = {
            "fit_s": fit_s,
            "score_s": score_s,
            "flows_per_s": len(Xte) / score_s if score_s else None,
        }
        extra = ""
        if rep.novel_recall is not None:
            extra = (f" novel_recall={rep.novel_recall:.4f}"
                     f" known_recall={rep.known_recall:.4f}")
        print(f"  [official] {name:22s} acc={rep.accuracy:.4f} "
              f"f1={rep.f1:.4f} fpr={rep.fpr:.4f}{extra}")

        if name == "random-forest":
            sweep = []
            for target in (0.001, 0.005, 0.01, 0.05):
                thr = ev.threshold_for_fpr(split.y_test, y_score, target)
                yp = (y_score >= thr).astype(int)
                r = ev.evaluate(split.y_test, yp, y_score, model=name,
                                dataset=split.name,
                                protocol=f"official-split@fpr={target}",
                                labels=split.test_labels, novel=novel)
                sweep.append(r)
                print(f"      fpr<={target:<6} thr={thr:.3f} "
                      f"recall={r.recall:.4f} actual_fpr={r.fpr:.4f} "
                      f"false_alerts/10k={r.false_alerts_per_10k:.1f}")
            reports.extend(sweep)

            import joblib
            RESULTS.mkdir(exist_ok=True)
            joblib.dump(pipe, RESULTS / f"model-{key}-random-forest.joblib")

    return reports, timings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["nsl-kdd", "unsw-nb15"])
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    all_reports: list[ev.Report] = []
    all_timings: dict = {}

    for key in args.dataset:
        reps, timings = run_dataset(key)
        all_reports.extend(reps)
        all_timings[key] = timings

    df = ev.summary_frame(all_reports)
    df.to_csv(RESULTS / "summary.csv", index=False)

    with open(RESULTS / "reports.json", "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in all_reports], fh, indent=2)
    with open(RESULTS / "timings.json", "w", encoding="utf-8") as fh:
        json.dump(all_timings, fh, indent=2)

    print("\n================ SUMMARY ================")
    show = df[df.protocol.isin(["random-split", "official-split"])]
    with pd.option_context("display.width", 200,
                           "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(show.to_string(index=False))
    print(f"\nwrote {(RESULTS/'summary.csv').relative_to(ROOT)}")

if __name__ == "__main__":
    main()
