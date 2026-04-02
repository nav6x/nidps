
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSL_CATEGORICAL = ["protocol_type", "service", "flag"]

NSL_FAMILY = {
    "neptune": "dos", "smurf": "dos", "back": "dos", "teardrop": "dos",
    "pod": "dos", "land": "dos", "apache2": "dos", "processtable": "dos",
    "mailbomb": "dos", "udpstorm": "dos",
    "satan": "probe", "ipsweep": "probe", "portsweep": "probe", "nmap": "probe",
    "mscan": "probe", "saint": "probe",
    "warezclient": "r2l", "guess_passwd": "r2l", "warezmaster": "r2l",
    "imap": "r2l", "ftp_write": "r2l", "multihop": "r2l", "phf": "r2l",
    "spy": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l", "httptunnel": "r2l",
    "named": "r2l", "sendmail": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "worm": "r2l",
    "buffer_overflow": "u2r", "rootkit": "u2r", "loadmodule": "u2r",
    "perl": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}

@dataclass
class Split:

    name: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_labels: pd.Series
    test_labels: pd.Series
    categorical: list[str]

    @property
    def novel_labels(self) -> set[str]:
        tr = set(self.train_labels.unique())
        te = set(self.test_labels.unique())
        return {lab for lab in te - tr if lab != "normal"}

def load_nsl_kdd() -> Split:
    train = pd.read_csv(RAW / "KDDTrain+.txt", header=None, names=NSL_COLUMNS)
    test = pd.read_csv(RAW / "KDDTest+.txt", header=None, names=NSL_COLUMNS)

    for df in (train, test):
        df.drop(columns=["difficulty"], inplace=True)

    y_train = (train["label"] != "normal").astype(int)
    y_test = (test["label"] != "normal").astype(int)

    return Split(
        name="NSL-KDD",
        X_train=train.drop(columns=["label"]),
        y_train=y_train,
        X_test=test.drop(columns=["label"]),
        y_test=y_test,
        train_labels=train["label"],
        test_labels=test["label"],
        categorical=list(NSL_CATEGORICAL),
    )

UNSW_CATEGORICAL = ["proto", "service", "state"]

def load_unsw_nb15() -> Split:
    train = pd.read_csv(RAW / "UNSW_NB15_training-set.csv")
    test = pd.read_csv(RAW / "UNSW_NB15_testing-set.csv")

    drop = ["id", "label", "attack_cat"]

    y_train = train["label"].astype(int)
    y_test = test["label"].astype(int)

    train_labels = train["attack_cat"].fillna("Normal")
    test_labels = test["attack_cat"].fillna("Normal")

    return Split(
        name="UNSW-NB15",
        X_train=train.drop(columns=drop),
        y_train=y_train,
        X_test=test.drop(columns=drop),
        y_test=y_test,
        train_labels=train_labels,
        test_labels=test_labels,
        categorical=list(UNSW_CATEGORICAL),
    )

LOADERS = {"nsl-kdd": load_nsl_kdd, "unsw-nb15": load_unsw_nb15}

def load(name: str) -> Split:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name]()
