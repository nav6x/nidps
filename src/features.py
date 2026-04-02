
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X, categorical: list[str]) -> ColumnTransformer:
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

def log_skewed(X, columns):
    X = X.copy()
    for c in columns:
        if c in X.columns:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X

SKEWED = [
    "src_bytes", "dst_bytes", "duration", "count", "srv_count",
    "sbytes", "dbytes", "dur", "sload", "dload", "spkts", "dpkts",
    "response_body_len", "sjit", "djit", "rate",
]
