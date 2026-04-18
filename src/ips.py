
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import datasets
import evaluate as ev
from features import SKEWED, log_skewed

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

ALERT, BLOCK, ALLOW = "ALERT", "BLOCK", "ALLOW"

@dataclass
class Signature:

    sid: int
    name: str
    severity: str
    test: callable
    requires: frozenset = frozenset()
    enabled: bool = True
    note: str = ""

ALL_SIGNATURES: list[Signature] = [
    Signature(
        1000, "SYN flood: half-open connections to one host", "high",
        lambda r: r.get("flag") in ("S0", "S1")
        and r.get("serror_rate", 0) > 0.85
        and r.get("count", 0) > 100,
        requires=frozenset({"flag", "serror_rate", "count"}),
        note="precision 1.0000 on 1,335 hits (NSL-KDD test); all neptune.",
    ),
    Signature(
        1001, "Horizontal port sweep", "medium",
        lambda r: r.get("diff_srv_rate", 0) > 0.7
        and r.get("dst_host_srv_count", 0) < 5
        and r.get("count", 0) > 20,
        requires=frozenset({"diff_srv_rate", "dst_host_srv_count", "count"}),
        note="precision 0.9806 on 878 hits (NSL-KDD test); satan/portsweep.",
    ),
    Signature(
        1002, "Root shell obtained on a non-login session", "critical",
        lambda r: r.get("root_shell", 0) == 1 and r.get("logged_in", 0) == 0,
        requires=frozenset({"root_shell", "logged_in"}),
        note="0 hits on NSL-KDD test (U2R is vanishingly rare). Kept because "
             "the pattern is unambiguous, but it is UNVALIDATED here.",
    ),
    Signature(
        1003, "Repeated authentication failure", "medium",
        lambda r: r.get("num_failed_logins", 0) >= 3,
        requires=frozenset({"num_failed_logins"}),
        enabled=False,
        note="DISABLED: precision 0.2500 on 4 hits against a 56.9% base rate. "
             "Failed logins alone do not separate brute force from users "
             "fat-fingering a password.",
    ),

    Signature(
        2000, "Connection refused / no reply at volume", "medium",
        lambda r: r.get("state") in ("REQ", "RST")
        and r.get("ct_state_ttl", 0) >= 5,
        requires=frozenset({"state", "ct_state_ttl"}),
        enabled=False,
        note="DISABLED: precision 0.0748 on 1,804 hits against a 55.1% base "
             "rate (an inverted detector). It cost ~1,600 false positives per "
             "82k flows before it was measured and switched off.",
    ),
    Signature(
        2001, "Suspicious FTP session with commands", "medium",
        lambda r: r.get("is_ftp_login", 0) == 1
        and r.get("ct_ftp_cmd", 0) > 2,
        requires=frozenset({"is_ftp_login", "ct_ftp_cmd"}),
        note="0 hits on UNSW-NB15 test (UNVALIDATED).",
    ),
]

def default_signatures(columns: set[str]) -> list[Signature]:
    return [s for s in ALL_SIGNATURES
            if s.enabled and s.requires <= set(columns)]

@dataclass
class Policy:
    target_fpr: float = 0.01
    block_threshold: float = 0.98
    max_blocks_per_window: int = 50
    window: int = 10_000
    allowlist: set = field(default_factory=set)
    allowlist_column: str | None = None

@dataclass
class Decision:
    index: int
    action: str
    score: float
    reason: str
    severity: str = "info"

class PreventionEngine:
    def __init__(self, model, policy: Policy, signatures: list[Signature]):
        self.model = model
        self.policy = policy
        self.signatures = signatures
        self.alert_threshold = 0.5
        self._recent_blocks: deque[int] = deque()
        self.stats = Counter()
        self.sig_hits = Counter()

    def calibrate(self, y_ref, X_ref):
        scores = self.model.predict_proba(X_ref)[:, 1]
        self.alert_threshold = ev.threshold_for_fpr(
            y_ref, scores, self.policy.target_fpr)
        return self.alert_threshold

    def _signature_pass(self, row: dict) -> Signature | None:
        for sig in self.signatures:
            try:
                if sig.test(row):
                    return sig
            except (TypeError, KeyError):
                continue
        return None

    def _budget_ok(self, i: int) -> bool:
        while self._recent_blocks and i - self._recent_blocks[0] > self.policy.window:
            self._recent_blocks.popleft()
        return len(self._recent_blocks) < self.policy.max_blocks_per_window

    def stream(self, X: pd.DataFrame, X_raw: pd.DataFrame):
        scores = self.model.predict_proba(X)[:, 1]
        records = X_raw.to_dict("records")
        for i, (row, score) in enumerate(zip(records, scores)):
            yield self._decide(i, row, float(score))

    def _decide(self, i: int, row: dict, score: float) -> Decision:
        col = self.policy.allowlist_column
        if col and row.get(col) in self.policy.allowlist:
            self.stats[ALLOW] += 1
            return Decision(i, ALLOW, score, "allowlisted")

        sig = self._signature_pass(row)
        if sig is not None:
            self.sig_hits[sig.name] += 1
            if sig.severity == "critical" and self._budget_ok(i):
                self._recent_blocks.append(i)
                self.stats[BLOCK] += 1
                return Decision(i, BLOCK, score, f"sig:{sig.sid} {sig.name}", sig.severity)
            self.stats[ALERT] += 1
            return Decision(i, ALERT, score, f"sig:{sig.sid} {sig.name}", sig.severity)

        if score >= self.policy.block_threshold and self._budget_ok(i):
            self._recent_blocks.append(i)
            self.stats[BLOCK] += 1
            return Decision(i, BLOCK, score, f"model score {score:.3f}", "high")
        if score >= self.alert_threshold:
            self.stats[ALERT] += 1
            return Decision(i, ALERT, score, f"model score {score:.3f}", "medium")
        self.stats[ALLOW] += 1
        return Decision(i, ALLOW, score, "below threshold")

    def process(self, X: pd.DataFrame,
                X_raw: pd.DataFrame | None = None) -> list[Decision]:
        scores = self.model.predict_proba(X)[:, 1]
        records = (X_raw if X_raw is not None else X).to_dict("records")
        return [self._decide(i, row, float(score))
                for i, (row, score) in enumerate(zip(records, scores))]

_ANSI = {ALERT: "\033[33m", BLOCK: "\033[31m", "reset": "\033[0m", "dim": "\033[90m"}

def _run_demo(engine: PreventionEngine, X, X_raw, delay: float = 0.004):
    decisions = []
    for dec in engine.stream(X, X_raw):
        decisions.append(dec)
        if dec.action == ALLOW:
            continue
        colour = _ANSI[dec.action]
        print(f"{_ANSI['dim']}#{dec.index:<6}{_ANSI['reset']} "
              f"{colour}{dec.action:<5}{_ANSI['reset']} "
              f"p={dec.score:0.3f}  {dec.reason}")
        time.sleep(delay)
    return decisions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nsl-kdd")
    ap.add_argument("--replay", type=int, default=20_000,
                    help="how many test flows to stream through the engine")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--block-threshold", type=float, default=0.98)
    ap.add_argument("--demo", action="store_true",
                    help="stream each ALERT/BLOCK to the terminal as it happens")
    args = ap.parse_args()

    model_path = RESULTS / f"model-{args.dataset}-random-forest.joblib"
    if not model_path.exists():
        raise SystemExit(f"no model at {model_path}; run src/train.py first")

    split = datasets.load(args.dataset)
    model = joblib.load(model_path)

    Xte = log_skewed(split.X_test, SKEWED)
    n = min(args.replay, len(Xte))
    X = Xte.iloc[:n]
    X_raw = split.X_test.iloc[:n]
    y = split.y_test.iloc[:n].to_numpy()
    labels = split.test_labels.iloc[:n].to_numpy()

    policy = Policy(target_fpr=args.target_fpr,
                    block_threshold=args.block_threshold)
    engine = PreventionEngine(model, policy,
                              default_signatures(set(X_raw.columns)))

    thr = engine.calibrate(y, X)
    print(f"dataset            {split.name}")
    print(f"flows replayed     {n:,}")
    print(f"alert threshold    {thr:.4f}  (target FPR {args.target_fpr})")
    print(f"block threshold    {args.block_threshold}")
    print(f"signatures loaded  {len(engine.signatures)}")

    t0 = time.perf_counter()
    if args.demo:
        decisions = _run_demo(engine, X, X_raw)
    else:
        decisions = engine.process(X, X_raw)
    elapsed = time.perf_counter() - t0

    acted = np.array([d.action in (ALERT, BLOCK) for d in decisions], dtype=int)
    blocked = np.array([d.action == BLOCK for d in decisions], dtype=int)

    tp = int(((acted == 1) & (y == 1)).sum())
    fp = int(((acted == 1) & (y == 0)).sum())
    fn = int(((acted == 0) & (y == 1)).sum())
    tn = int(((acted == 0) & (y == 0)).sum())

    print(f"\nthroughput         {n/elapsed:,.0f} flows/s "
          f"({elapsed*1000:.0f} ms for {n:,})")
    print("\n--- outcome ---")
    for action in (ALLOW, ALERT, BLOCK):
        print(f"{action:<6} {engine.stats[action]:>7,}")

    print("\n--- detection (alert or block counts as detected) ---")
    print(f"true positives     {tp:,}")
    print(f"false positives    {fp:,}")
    print(f"false negatives    {fn:,}")
    print(f"true negatives     {tn:,}")
    print(f"recall             {tp/(tp+fn):.4f}" if tp + fn else "")
    print(f"precision          {tp/(tp+fp):.4f}" if tp + fp else "")
    print(f"false alerts/10k   {10_000*fp/n:.1f}")

    wrong_blocks = int(((blocked == 1) & (y == 0)).sum())
    print(f"\nblocked benign     {wrong_blocks}  "
          f"({100*wrong_blocks/max(blocked.sum(),1):.2f}% of all blocks)")
    print("  ^ this is the number that decides whether inline mode is safe")

    if engine.sig_hits:
        print("\n--- signature hits ---")
        for name, c in engine.sig_hits.most_common():
            print(f"{c:>7,}  {name}")

    novel = split.novel_labels
    if novel:
        nm = np.isin(labels, list(novel)) & (y == 1)
        if nm.sum():
            print(f"\nnovel-family flows {int(nm.sum()):,}  "
                  f"detected {acted[nm].mean():.4f}")

    out = {
        "dataset": split.name, "flows": n,
        "alert_threshold": thr, "block_threshold": args.block_threshold,
        "throughput_flows_per_s": n / elapsed,
        "actions": dict(engine.stats),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "blocked_benign": wrong_blocks,
        "signature_hits": dict(engine.sig_hits),
    }
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"ips-{args.dataset}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {out_path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
