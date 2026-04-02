from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import datasets
import evaluate as ev
from features import build_preprocessor, log_skewed
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
has_data = (DATA / "KDDTrain+.txt").exists()
needs_data = pytest.mark.skipif(not has_data, reason="raw data not downloaded")


def test_preprocessor_is_fitted_on_train_only():
    train = pd.DataFrame({
        "proto": ["tcp", "udp", "tcp", "udp"] * 5,
        "x": np.arange(20, dtype=float),
    })
    test = pd.DataFrame({"proto": ["icmp"] * 4, "x": np.arange(4, dtype=float)})

    pre = build_preprocessor(train, ["proto"])
    pre.fit(train)
    out_train = pre.transform(train)
    out_test = pre.transform(test)

    assert out_train.shape[1] == out_test.shape[1], "feature width must be stable"
    dense = out_test.toarray() if hasattr(out_test, "toarray") else out_test
    assert dense[:, -2:].sum() == 0, "unknown category must encode as all-zero"


def test_scaler_does_not_see_test_statistics():
    train = pd.DataFrame({"proto": ["tcp"] * 10, "x": np.zeros(10)})
    train.loc[:4, "x"] = 1.0
    test = pd.DataFrame({"proto": ["tcp"] * 2, "x": [1000.0, 1000.0]})

    pre = build_preprocessor(train, ["proto"])
    pre.fit(train)
    dense = pre.transform(test)
    dense = dense.toarray() if hasattr(dense, "toarray") else dense
    assert abs(dense[0, 0]) > 100, (
        "a huge test value must stay huge; if it were scaled to ~0 the scaler "
        "had been fitted on test data"
    )


def test_log_skewed_handles_zero_and_negative():
    df = pd.DataFrame({"src_bytes": [0, -5, 10_000_000]})
    out = log_skewed(df, ["src_bytes"])
    assert out["src_bytes"].iloc[0] == 0.0
    assert out["src_bytes"].iloc[1] == 0.0
    assert np.isfinite(out["src_bytes"]).all()


def test_threshold_for_fpr_respects_budget():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(10_000), np.ones(1_000)])
    score = np.concatenate([rng.uniform(0, 1, 10_000),
                            rng.uniform(0.5, 1.5, 1_000)])

    for target in (0.001, 0.01, 0.05):
        thr = ev.threshold_for_fpr(y, score, target)
        realised = (score[y == 0] >= thr).mean()
        assert realised <= target * 1.5 + 1e-6, (
            f"FPR budget blown: asked {target}, got {realised}")


def test_threshold_is_rank_based_and_survives_miscalibration():
    rng = np.random.default_rng(1)
    y = np.concatenate([np.zeros(4_000), np.ones(1_000)])
    score = np.concatenate([rng.uniform(0, 1, 4_000),
                            rng.uniform(0.3, 1.3, 1_000)])

    warped = score ** 3
    thr = ev.threshold_for_fpr(y, score, 0.05)
    thr_w = ev.threshold_for_fpr(y, warped, 0.05)

    flagged = score >= thr
    flagged_warped = warped >= thr_w
    assert np.array_equal(flagged, flagged_warped), (
        "threshold selection changed under a monotone score distortion; it is "
        "supposed to be rank-based and therefore calibration-invariant")


def test_optimism_gap_is_positive_across_seeds_if_measured():
    import json

    path = (Path(__file__).resolve().parent.parent
            / "results" / "robustness-nsl-kdd.json")
    if not path.exists():
        pytest.skip("robustness run not generated")

    summary = json.loads(path.read_text())["summary"]
    for model, s in summary.items():
        assert s["gap_min"] > 0.05, (
            f"{model}: worst-seed optimism gap {s['gap_min']:.4f} is not clearly "
            f"positive; the headline claim would not survive this run")


def test_evaluate_counts_and_rates():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 0])
    rep = ev.evaluate(y, pred, np.array([.1, .9, .8, .2]),
                      model="m", dataset="d", protocol="p")
    assert (rep.tn, rep.fp, rep.fn, rep.tp) == (1, 1, 1, 1)
    assert rep.fpr == 0.5 and rep.fnr == 0.5
    assert rep.alerts_per_10k == 5000.0


def test_novel_recall_only_counts_novel_families():
    y = np.array([1, 1, 1, 0])
    pred = np.array([1, 0, 1, 0])
    labels = pd.Series(["neptune", "worm", "worm", "normal"])
    rep = ev.evaluate(y, pred, np.array([.9, .1, .9, .1]),
                      model="m", dataset="d", protocol="p",
                      labels=labels, novel={"worm"})
    assert rep.known_recall == 1.0
    assert rep.novel_recall == 0.5
    assert rep.per_family["worm"]["novel"] is True
    assert rep.per_family["neptune"]["novel"] is False


@needs_data
def test_nsl_kdd_shape_and_novelty():
    s = datasets.load("nsl-kdd")
    assert s.X_train.shape == (125_973, 41)
    assert s.X_test.shape == (22_544, 41)
    assert "difficulty" not in s.X_train.columns, "difficulty is target leakage"
    assert "label" not in s.X_train.columns
    assert len(s.novel_labels) == 17, "NSL-KDD test has 17 unseen families"
    assert set(s.y_train.unique()) == {0, 1}


@needs_data
def test_official_split_is_harder_than_random_split():
    s = datasets.load("nsl-kdd")
    Xtr, Xte = log_skewed(s.X_train, ["src_bytes"]), log_skewed(s.X_test, ["src_bytes"])

    clf = Pipeline([
        ("prep", build_preprocessor(Xtr, s.categorical)),
        ("clf", DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
                                       random_state=0)),
    ])
    clf.fit(Xtr, s.y_train)
    official = clf.score(Xte, s.y_test)

    pooled_X = pd.concat([Xtr, Xte], ignore_index=True)
    pooled_y = pd.concat([s.y_train, s.y_test], ignore_index=True)
    from sklearn.model_selection import train_test_split
    a_tr, a_te, b_tr, b_te = train_test_split(
        pooled_X, pooled_y, test_size=0.25, stratify=pooled_y, random_state=0)
    clf2 = Pipeline([
        ("prep", build_preprocessor(a_tr, s.categorical)),
        ("clf", DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
                                       random_state=0)),
    ])
    clf2.fit(a_tr, b_tr)
    random = clf2.score(a_te, b_te)

    assert random > official + 0.10, (
        f"expected a large optimism gap; got random={random:.4f} "
        f"official={official:.4f}")


def test_signatures_are_evaluated_on_raw_not_transformed_features():
    import ips

    raw = {"flag": "S0", "serror_rate": 1.0, "count": 511}
    transformed = dict(raw, count=float(np.log1p(511)))

    syn = next(s for s in ips.ALL_SIGNATURES if s.sid == 1000)
    assert syn.test(raw) is True
    assert syn.test(transformed) is False, (
        "rule fired on compressed counters; signatures must see raw units")


def test_disabled_signatures_stay_out_of_the_active_set():
    import ips

    cols = {"flag", "serror_rate", "count", "diff_srv_rate",
            "dst_host_srv_count", "root_shell", "logged_in",
            "num_failed_logins"}
    active = {s.sid for s in ips.default_signatures(cols)}

    assert 1000 in active and 1001 in active
    assert 1003 not in active, "rule 1003 measured below base rate; keep it off"
    for sig in ips.ALL_SIGNATURES:
        if not sig.enabled:
            assert "DISABLED" in sig.note, (
                f"sid {sig.sid} is off but does not say why")


def test_engine_scores_model_frame_and_rules_raw_frame():
    import ips

    class StubModel:
        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    X_model = pd.DataFrame({"flag": ["S0"], "serror_rate": [1.0],
                            "count": [np.log1p(511)]})
    X_raw = pd.DataFrame({"flag": ["S0"], "serror_rate": [1.0],
                          "count": [511]})

    engine = ips.PreventionEngine(
        StubModel(), ips.Policy(),
        ips.default_signatures(set(X_raw.columns)))
    engine.alert_threshold = 0.5

    decisions = engine.process(X_model, X_raw)
    assert decisions[0].action == ips.ALERT
    assert "SYN flood" in decisions[0].reason


@needs_data
def test_novelty_detector_beats_supervised_on_novel_families():
    import novelty

    res = novelty.run("nsl-kdd")
    rf_novel = res["random-forest"]["novel_recall"]
    if_novel = res["isolation-forest"]["novel_recall"]

    assert res["random-forest"]["fpr"] <= 0.02
    assert res["isolation-forest"]["fpr"] <= 0.02

    assert if_novel > rf_novel, (
        f"novelty detector ({if_novel:.3f}) should beat supervised "
        f"({rf_novel:.3f}) on novel families")
    assert if_novel > 2 * rf_novel, "the advantage should be large, not marginal"


@needs_data
def test_novelty_detector_never_sees_an_attack():
    import novelty

    split = datasets.load("nsl-kdd")
    benign = split.y_train.to_numpy() == 0
    assert benign.sum() < len(benign), "expected some attacks in training"
    assert set(split.y_train.to_numpy()[benign]) == {0}
