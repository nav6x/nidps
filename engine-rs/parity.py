from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import datasets
import joblib
import ips
from features import SKEWED, log_skewed

DATA_FILE = {
    "nsl-kdd": ROOT / "data" / "raw" / "KDDTest+.txt",
    "unsw-nb15": ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv",
}


def python_decisions(dataset: str):
    split = datasets.load(dataset)
    model = joblib.load(ROOT / "results" / f"model-{dataset}-random-forest.joblib")
    Xte = log_skewed(split.X_test, SKEWED)
    X_raw = split.X_test
    y = split.y_test.to_numpy()

    policy = ips.Policy(target_fpr=0.01, block_threshold=0.98)
    engine = ips.PreventionEngine(
        model, policy, ips.default_signatures(set(X_raw.columns)))
    thr = engine.calibrate(y, Xte)
    scores = model.predict_proba(Xte)[:, 1]
    decisions = [d.action for d in engine.process(Xte, X_raw)]
    return decisions, scores, thr, y


def rust_decisions(dataset: str) -> list[str]:
    binary = ENGINE / "target" / "release" / (
        "nidps-engine.exe" if sys.platform == "win32" else "nidps-engine")
    if not binary.exists():
        raise SystemExit(f"build the engine first: cargo build --release "
                         f"(missing {binary})")
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    import os
    os.close(fd)
    out = Path(tmp)
    subprocess.run(
        [str(binary),
         "--model", str(ENGINE / "artifacts" / f"model-{dataset}.json"),
         "--data", str(DATA_FILE[dataset]),
         "--dump-decisions", str(out)],
        check=True, stdout=subprocess.DEVNULL)
    actions = out.read_text().split("\n")
    out.unlink(missing_ok=True)
    return actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nsl-kdd", choices=sorted(DATA_FILE))
    dataset = ap.parse_args().dataset
    print(f"dataset            {dataset}")

    py, scores, thr, y = python_decisions(dataset)
    rs = rust_decisions(dataset)

    if len(py) != len(rs):
        raise SystemExit(f"length mismatch: python {len(py)} vs rust {len(rs)}")

    n = len(py)
    mism = [i for i in range(n) if py[i] != rs[i]]
    agree = n - len(mism)

    print(f"flows compared     {n:,}")
    print(f"identical          {agree:,}  ({100*agree/n:.4f}%)")
    print(f"mismatched         {len(mism)}")
    print(f"alert threshold    {thr:.17g}")

    if mism:
        print("\n--- mismatched flows ---")
        for i in mism[:20]:
            dist = scores[i] - thr
            print(f"  flow {i:>6}  python={py[i]:<5} rust={rs[i]:<5} "
                  f"label={'attack' if y[i] else 'benign'}  "
                  f"score={scores[i]:.17g}  score-thr={dist:+.2e}")
        near = all(abs(scores[i] - thr) < 1e-9 for i in mism)
        print(f"\nall mismatches within 1e-9 of the threshold: {near}")

    boundary_only = all(abs(scores[i] - thr) < 1e-9 for i in mism)
    ok = (len(mism) == 0) or boundary_only
    print(f"\nPARITY: {'PASS' if ok else 'FAIL'}"
          + ("" if len(mism) == 0 else
             f" ({len(mism)} boundary flip{'s' if len(mism) != 1 else ''} at the FPR threshold)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
