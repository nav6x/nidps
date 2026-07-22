from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import datasets
import joblib
from features import SKEWED, log_skewed

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

def build_numeric_spec(prep) -> list[dict]:
    num_pipe = prep.named_transformers_["num"]
    imp = num_pipe.named_steps["impute"]
    sc = num_pipe.named_steps["scale"]
    cols = prep.transformers_[0][2]

    spec = []
    for i, col in enumerate(cols):
        spec.append({
            "source": col,
            "log1p": col in SKEWED,
            "median": float(imp.statistics_[i]),
            "mean": float(sc.mean_[i]),
            "scale": float(sc.scale_[i]),
        })
    return spec

def build_categorical_spec(prep) -> list[dict]:
    cat_pipe = prep.named_transformers_["cat"]
    enc = cat_pipe.named_steps["encode"]
    cat_cols = prep.transformers_[1][2]
    names = list(prep.get_feature_names_out())
    numeric_width = len(prep.transformers_[0][2])

    def out_index(colname: str) -> int:
        return names.index(colname)

    spec = []
    for f, feat in enumerate(cat_cols):
        infreq = enc.infrequent_categories_[f]
        infreq_set = set(infreq) if infreq is not None else set()
        value_to_index: dict[str, int] = {}

        for cat in enc.categories_[f]:
            if cat in infreq_set:
                idx = out_index(f"cat__{feat}_infrequent_sklearn")
            else:
                idx = out_index(f"cat__{feat}_{cat}")
            value_to_index[str(cat)] = idx

        spec.append({"source": feat, "value_to_index": value_to_index})
        _ = numeric_width
    return spec

def build_trees(rf) -> list[dict]:
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        proba1 = []
        for k in range(t.node_count):
            counts = t.value[k][0]
            total = counts.sum()
            proba1.append(float(counts[1] / total) if total else 0.0)
        trees.append({
            "feature": [int(x) for x in t.feature],
            "threshold": [float(x) for x in t.threshold],
            "left": [int(x) for x in t.children_left],
            "right": [int(x) for x in t.children_right],
            "proba1": proba1,
        })
    return trees

def reconstruct(X_raw, numeric_spec, categorical_spec, n_features) -> np.ndarray:
    n = len(X_raw)
    M = np.zeros((n, n_features), dtype=np.float64)
    records = X_raw.to_dict("records")

    for i, row in enumerate(records):
        for j, s in enumerate(numeric_spec):
            v = row.get(s["source"])
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = None
            if v is not None and s["log1p"]:
                v = np.log1p(max(float(v), 0.0))
            if v is None:
                v = s["median"]
            M[i, j] = (float(v) - s["mean"]) / s["scale"]

        for s in categorical_spec:
            val = row.get(s["source"])
            idx = s["value_to_index"].get(str(val))
            if idx is not None:
                M[i, idx] = 1.0
    return M

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nsl-kdd")
    args = ap.parse_args()

    results = ROOT / "results"
    model_path = results / f"model-{args.dataset}-random-forest.joblib"
    if not model_path.exists():
        raise SystemExit(f"no model at {model_path}; run src/train.py first")

    pipe = joblib.load(model_path)
    prep = pipe.named_steps["prep"]
    rf = pipe.named_steps["clf"]

    numeric_spec = build_numeric_spec(prep)
    categorical_spec = build_categorical_spec(prep)
    n_features = len(prep.get_feature_names_out())

    split = datasets.load(args.dataset)
    X_raw = split.X_test
    X_log = log_skewed(X_raw, SKEWED)
    ref = prep.transform(X_log)
    ref = ref.toarray() if hasattr(ref, "toarray") else np.asarray(ref)
    mine = reconstruct(X_raw, numeric_spec, categorical_spec, n_features)

    max_diff = float(np.abs(ref - mine).max())
    print(f"spec vs sklearn.transform: max abs diff = {max_diff:.2e} "
          f"over {ref.shape[0]:,} rows x {ref.shape[1]} features")
    if max_diff > 1e-9:
        raise SystemExit(f"EXPORT INVALID: feature reconstruction differs by "
                         f"{max_diff:.2e} (> 1e-9). Rust parity is impossible "
                         f"until this is fixed.")
    print("export validated: reconstruction is faithful to 1e-9")

    payload = {
        "dataset": split.name,
        "n_features": n_features,
        "raw_columns": list(X_raw.columns),
        "label_column": "label",
        "numeric": numeric_spec,
        "categorical": categorical_spec,
        "n_trees": len(rf.estimators_),
        "trees": build_trees(rf),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / f"model-{args.dataset}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out.relative_to(ROOT)}  ({size_mb:.1f} MB, "
          f"{payload['n_trees']} trees)")

if __name__ == "__main__":
    main()
