
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = RESULTS / "figures"

BG_COLOR = "#181A1F"
CARD_COLOR = "#21262D"
GRID_COLOR = "#2D333B"
TEXT_COLOR = "#E6EDF3"
TEXT_MUTED = "#8B949E"
BORDER_COLOR = "#30363D"

RANDOM = "#F85149"
OFFICIAL = "#58A6FF"
KNOWN = "#3FB950"
NOVEL = "#F85149"
KEEP = "#58A6FF"
DROP = "#F85149"
UNVAL = "#8B949E"

MODELS = ["logistic-regression", "decision-tree", "random-forest"]
MODEL_LABEL = {
    "logistic-regression": "Logistic\nregression",
    "decision-tree": "Decision\ntree",
    "random-forest": "Random\nforest",
}

def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": BORDER_COLOR,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG_COLOR,
        "savefig.edgecolor": "none",
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
    })

def _load() -> list[dict]:
    return json.loads((RESULTS / "reports.json").read_text())

def _pick(reports, dataset, protocol, model):
    for r in reports:
        if r["dataset"] == dataset and r["protocol"] == protocol and r["model"] == model:
            return r
    raise KeyError(f"{dataset}/{protocol}/{model} not found")

def _spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def fig_optimism_gap(reports) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, dataset in zip(axes, ["NSL-KDD", "UNSW-NB15"]):
        rnd = [_pick(reports, dataset, "random-split", m)["accuracy"] for m in MODELS]
        off = [_pick(reports, dataset, "official-split", m)["accuracy"] for m in MODELS]
        x = range(len(MODELS))
        w = 0.36
        b1 = ax.bar([i - w / 2 for i in x], rnd, w, label="Random split (what papers report)", color=RANDOM)
        b2 = ax.bar([i + w / 2 for i in x], off, w, label="Official split (what the authors intended)", color=OFFICIAL)
        for i, (a, b) in enumerate(zip(rnd, off)):
            ax.annotate(f"−{a - b:.2f}", (i, max(a, b) + 0.015),
                        ha="center", fontsize=9.5, color=TEXT_COLOR, fontweight="bold")
        ax.set_title(dataset, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
        ax.set_ylim(0.6, 1.06)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        _spines(ax)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=10)
    plt.subplots_adjust(top=0.82)
    fig.suptitle("The optimism of a random split", fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.5, 0.90, "Same code, same models, same data: only the train/test protocol changes.",
             ha="center", fontsize=9.5, color=TEXT_MUTED)
    fig.savefig(FIGDIR / "optimism_gap.png")
    plt.close(fig)

def fig_novel_vs_known(reports) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    known = [_pick(reports, "NSL-KDD", "official-split", m)["known_recall"] for m in MODELS]
    novel = [_pick(reports, "NSL-KDD", "official-split", m)["novel_recall"] for m in MODELS]
    x = range(len(MODELS))
    w = 0.36
    ax.bar([i - w / 2 for i in x], known, w, label="Known families (seen in training)", color=KNOWN)
    ax.bar([i + w / 2 for i in x], novel, w, label="Novel families (never seen)", color=NOVEL)
    for i, (k, n) in enumerate(zip(known, novel)):
        ax.annotate(f"{k:.2f}", (i - w / 2, k + 0.012), ha="center", fontsize=9, color=TEXT_COLOR)
        ax.annotate(f"{n:.2f}", (i + w / 2, n + 0.012), ha="center", fontsize=9, color=TEXT_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Recall")
    ax.set_title("Best model on known attacks, worst on novel ones",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    _spines(ax)
    fig.savefig(FIGDIR / "novel_vs_known.png")
    plt.close(fig)

def fig_per_family(reports) -> None:
    r = _pick(reports, "NSL-KDD", "official-split", "random-forest")
    fams = r["per_family"]
    items = [(name, d["recall"], d["novel"]) for name, d in fams.items() if d["n"] >= 20]
    items.sort(key=lambda t: t[1])
    names = [t[0] for t in items]
    recalls = [t[1] for t in items]
    colors = [NOVEL if t[2] else KNOWN for t in items]

    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.32 * len(items))))
    ax.barh(range(len(items)), recalls, color=colors, height=0.7)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Recall")
    ax.set_title("Per-family recall: random forest, NSL-KDD official split",
                 fontsize=12.5, fontweight="bold", pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=KNOWN),
               plt.Rectangle((0, 0), 1, 1, color=NOVEL)]
    ax.legend(handles, ["Seen in training", "Novel (never seen)"],
              frameon=False, fontsize=9.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    _spines(ax)
    fig.savefig(FIGDIR / "per_family_recall.png")
    plt.close(fig)

def fig_signature_audit() -> None:
    audit = json.loads((RESULTS / "signature-audit.json").read_text())
    rows = [a for a in audit if a["hits"] > 0]
    rows.sort(key=lambda a: a["precision"])
    labels = [f"[{a['sid']}] {a['name'].split(chr(8212))[0].strip()[:28]}" for a in rows]
    prec = [a["precision"] for a in rows]
    colors = [KEEP if a["verdict"] == "KEEP" else DROP for a in rows]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(rows))
    ax.barh(y, prec, color=colors, height=0.62)
    for br in sorted({round(a["base_rate"], 3) for a in rows}):
        ax.axvline(br, color=TEXT_COLOR, linestyle="--", linewidth=1.1)
        ax.annotate(f"base rate {br:.0%}", (br, len(rows) - 0.4),
                    fontsize=8.5, color=TEXT_COLOR, rotation=90, va="top", ha="right")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Precision")
    ax.set_title("Every signature is measured: two fired below the base rate",
                 fontsize=12.5, fontweight="bold", pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=KEEP),
               plt.Rectangle((0, 0), 1, 1, color=DROP)]
    ax.legend(handles, ["Kept", "Disabled"], frameon=False, fontsize=9.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    _spines(ax)
    fig.savefig(FIGDIR / "signature_audit.png")
    plt.close(fig)

def fig_operating_points(reports) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for dataset, color in [("UNSW-NB15", OFFICIAL), ("NSL-KDD", RANDOM)]:
        pts = sorted(
            (r for r in reports
             if r["dataset"] == dataset and r["model"] == "random-forest"
             and "fpr=" in r["protocol"]),
            key=lambda r: r["fpr"],
        )
        xs = [r["fpr"] for r in pts]
        ys = [r["recall"] for r in pts]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6, label=dataset)
        for r in pts:
            budget = r["protocol"].split("fpr=")[1]
            ax.annotate(f"{r['recall']:.0%}", (r["fpr"], r["recall"]),
                        textcoords="offset points", xytext=(6, -10),
                        fontsize=8.5, color=color)
    ax.set_xlim(0, 0.055)
    ax.set_ylim(0.3, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("False-positive rate (budget swept 0.1% → 5%)")
    ax.set_ylabel("Recall")
    ax.set_title("Threshold tuning rescues a shifted balance, not unseen attacks",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    _spines(ax)
    fig.savefig(FIGDIR / "operating_points.png")
    plt.close(fig)

def fig_novelty() -> None:
    path = RESULTS / "novelty.json"
    if not path.exists():
        return
    nov = json.loads(path.read_text())
    order = ["random-forest", "isolation-forest", "hybrid-rf-or-if"]
    label = {"random-forest": "Random\nforest\n(supervised)",
             "isolation-forest": "Isolation\nforest\n(benign-only)",
             "hybrid-rf-or-if": "Hybrid\nRF ∪ IF"}
    colors = [OFFICIAL, KNOWN, "#b07aa1"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    nsl = nov["nsl-kdd"]
    vals = [nsl[m]["novel_recall"] for m in order]
    axes[0].bar(range(3), vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        axes[0].annotate(f"{v:.0%}", (i, v + 0.016), ha="center",
                         fontsize=11, fontweight="bold", color=TEXT_COLOR)
    axes[0].set_title("NSL-KDD: recall on 17 novel families",
                      fontsize=12, fontweight="bold", pad=12)
    axes[0].set_ylim(0, 0.65)

    unsw = nov["unsw-nb15"]
    vals2 = [unsw[m]["recall"] for m in order]
    axes[1].bar(range(3), vals2, color=colors, width=0.6)
    for i, v in enumerate(vals2):
        axes[1].annotate(f"{v:.0%}", (i, v + 0.02), ha="center",
                         fontsize=11, fontweight="bold", color=TEXT_COLOR)
    axes[1].set_title("UNSW-NB15 (no novel families): overall recall",
                      fontsize=12, fontweight="bold", pad=12)
    axes[1].set_ylim(0, 1.05)

    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels([label[m] for m in order], fontsize=9.5)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        _spines(ax)
    axes[0].set_ylabel("Recall")
    plt.subplots_adjust(top=0.82)
    fig.suptitle("Anomaly detection catches novel attacks: and only novel attacks",
                 fontsize=13.5, fontweight="bold", y=0.97)
    fig.text(0.5, 0.90,
             "Same 1% false-positive budget throughout. The isolation forest never saw an attack in training.",
             ha="center", fontsize=9.5, color=TEXT_MUTED)
    fig.savefig(FIGDIR / "novelty_comparison.png")
    plt.close(fig)

def fig_calibration() -> None:
    path = RESULTS / "calibration.json"
    if not path.exists():
        return
    cal = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    ax.plot([0, 1], [0, 1], "--", color=TEXT_MUTED, linewidth=1.3,
            label="Perfectly calibrated")
    for key, color, note in [("nsl-kdd", RANDOM, "under-confident"),
                             ("unsw-nb15", OFFICIAL, "over-confident")]:
        c = cal[key]
        xs = [b["mean_pred"] for b in c["bins"]]
        ys = [b["obs_freq"] for b in c["bins"]]
        ax.plot(xs, ys, "-o", color=color, linewidth=2.2, markersize=6.5,
                label=f"{c['dataset']}: {note} (ECE {c['ece']:.2f})")
    ax.fill_between([0, 1], [0, 1], [1, 1], color=RANDOM, alpha=0.04)
    ax.fill_between([0, 1], [0, 0], [0, 1], color=OFFICIAL, alpha=0.04)
    ax.text(0.48, 0.75, "scores too low\n(attacks look benign)", fontsize=9.5,
            color=RANDOM, ha="center", va="center", fontweight="bold")
    ax.text(0.78, 0.14, "scores too high\n(benign looks hostile)", fontsize=9.5,
            color=OFFICIAL, ha="center", va="center", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Mean predicted probability of attack", fontsize=11, labelpad=8)
    ax.set_ylabel("Observed fraction that were attacks", fontsize=11, labelpad=8)
    ax.set_title("The scores are rankings, not probabilities",
                 fontsize=13.5, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR,
              framealpha=0.95, fontsize=9.0, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=2)
    _spines(ax)
    fig.savefig(FIGDIR / "calibration.png", bbox_inches="tight")
    plt.close(fig)

def fig_robustness() -> None:
    path = RESULTS / "robustness-nsl-kdd.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    summ = data["summary"]
    models = ["logistic-regression", "decision-tree", "random-forest"]
    means = [summ[m]["gap_mean"] for m in models]
    stds = [summ[m]["gap_std"] for m in models]
    mins = [summ[m]["gap_min"] for m in models]
    n = data["summary"][models[0]]["n_seeds"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = range(len(models))
    ax.bar(x, means, 0.55, yerr=stds, color=RANDOM, capsize=6,
           error_kw={"elinewidth": 1.6, "ecolor": TEXT_COLOR}, label="mean gap ± 1σ")
    for i, (mn, mmin) in enumerate(zip(means, mins)):
        ax.plot([i - 0.27, i + 0.27], [mmin, mmin], color=TEXT_COLOR, linewidth=1.4,
                linestyle=":")
        ax.annotate(f"{mn:.3f}", (i, mn + stds[i] + 0.015), ha="center",
                    fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.axhline(0, color=TEXT_MUTED, linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models])
    ax.set_ylabel("Optimism gap (random − official accuracy)")
    ax.set_ylim(0, 0.30)
    ax.set_title(f"The gap survives {n} random seeds",
                 fontsize=13, fontweight="bold", pad=12)
    ax.annotate("dotted line: worst of the seeds", (0.97, 0.94),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=8.5, color=TEXT_MUTED)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _spines(ax)
    fig.savefig(FIGDIR / "robustness.png")
    plt.close(fig)

def main() -> None:
    _style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    reports = _load()
    fig_optimism_gap(reports)
    fig_novel_vs_known(reports)
    fig_per_family(reports)
    fig_signature_audit()
    fig_operating_points(reports)
    fig_novelty()
    fig_calibration()
    fig_robustness()
    made = sorted(p.name for p in FIGDIR.glob("*.png"))
    print(f"Wrote {len(made)} figures to {FIGDIR}:")
    for name in made:
        print(f"  {name}")

if __name__ == "__main__":
    main()
