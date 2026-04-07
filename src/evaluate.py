
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

@dataclass
class Report:
    model: str
    dataset: str
    protocol: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    fpr: float
    fnr: float
    tn: int
    fp: int
    fn: int
    tp: int
    alerts_per_10k: float
    false_alerts_per_10k: float
    per_family: dict = field(default_factory=dict)
    novel_recall: float | None = None
    known_recall: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

def evaluate(
    y_true,
    y_pred,
    y_score,
    *,
    model: str,
    dataset: str,
    protocol: str,
    labels: pd.Series | None = None,
    novel: set[str] | None = None,
) -> Report:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)

    rep = Report(
        model=model,
        dataset=dataset,
        protocol=protocol,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_score) if y_score is not None else float("nan"),
        pr_auc=average_precision_score(y_true, y_score) if y_score is not None else float("nan"),
        fpr=fp / (fp + tn) if (fp + tn) else 0.0,
        fnr=fn / (fn + tp) if (fn + tp) else 0.0,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        alerts_per_10k=10_000 * (tp + fp) / n,
        false_alerts_per_10k=10_000 * fp / n,
    )

    if labels is not None:
        labels = pd.Series(np.asarray(labels))
        detected = pd.Series(y_pred == 1)
        attack = pd.Series(y_true == 1)

        for lab in sorted(labels[attack].unique()):
            mask = (labels == lab) & attack
            if mask.sum():
                rep.per_family[str(lab)] = {
                    "n": int(mask.sum()),
                    "recall": float(detected[mask].mean()),
                    "novel": bool(novel and lab in novel),
                }

        if novel:
            nm = attack & labels.isin(novel)
            km = attack & ~labels.isin(novel)
            if nm.sum():
                rep.novel_recall = float(detected[nm].mean())
            if km.sum():
                rep.known_recall = float(detected[km].mean())

    return rep

def threshold_for_fpr(y_true, y_score, target_fpr: float) -> float:
    benign = np.asarray(y_score)[np.asarray(y_true) == 0]
    if len(benign) == 0:
        return 0.5
    return float(np.quantile(benign, 1.0 - target_fpr))

def summary_frame(reports: list[Report]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append({
            "dataset": r.dataset,
            "protocol": r.protocol,
            "model": r.model,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "roc_auc": r.roc_auc,
            "fpr": r.fpr,
            "novel_recall": r.novel_recall,
            "known_recall": r.known_recall,
            "false_alerts_per_10k": r.false_alerts_per_10k,
        })
    return pd.DataFrame(rows)
