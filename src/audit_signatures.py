
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import datasets
from ips import ALL_SIGNATURES

RESULTS = Path(__file__).resolve().parent.parent / "results"

def audit(key: str) -> list[dict]:
    split = datasets.load(key)
    X = split.X_test
    y = split.y_test.to_numpy()
    labels = split.test_labels.to_numpy()
    records = X.to_dict("records")
    base_rate = float(y.mean())

    print(f"\n=== {split.name} ===")
    print(f"{len(records):,} flows, base rate {base_rate:.1%} attack")
    print(f"{'sid':>5}  {'rule':46s} {'hits':>7} {'precision':>10} "
          f"{'coverage':>9}  verdict")

    rows = []
    for sig in ALL_SIGNATURES:
        cols = set(X.columns)
        if not sig.requires <= cols:
            continue
        hit = np.array([bool(_safe(sig, r)) for r in records])
        n = int(hit.sum())

        if n == 0:
            verdict = "no hits (untested here)"
            precision = None
            coverage = 0.0
        else:
            precision = float(y[hit].mean())
            coverage = float(hit[y == 1].sum() / max((y == 1).sum(), 1))
            if precision >= 0.95:
                verdict = "KEEP"
            elif precision > base_rate:
                verdict = "weak (beats base rate only)"
            else:
                verdict = "DROP (below base rate)"

        top = ""
        if n:
            vals, counts = np.unique(labels[hit], return_counts=True)
            order = np.argsort(-counts)[:3]
            top = ", ".join(f"{vals[i]}×{counts[i]}" for i in order)

        print(f"{sig.sid:>5}  {sig.name[:44]:46s} {n:>7,} "
              f"{'N/A' if precision is None else f'{precision:>10.4f}'} "
              f"{coverage:>9.4f}  {verdict}")
        if top:
            print(f"{'':>5}  {'':46s} mostly: {top}")

        rows.append({
            "dataset": split.name, "sid": sig.sid, "name": sig.name,
            "enabled": sig.enabled, "hits": n, "precision": precision,
            "attack_coverage": coverage, "base_rate": base_rate,
            "verdict": verdict,
        })
    return rows

def _safe(sig, row):
    try:
        return sig.test(row)
    except (TypeError, KeyError):
        return False

def main():
    rows = []
    for key in ("nsl-kdd", "unsw-nb15"):
        try:
            rows.extend(audit(key))
        except FileNotFoundError:
            print(f"skip {key}: raw data missing")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "signature-audit.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {RESULTS / 'signature-audit.json'}")

if __name__ == "__main__":
    main()
