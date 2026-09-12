from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import datasets
import ips
from features import SKEWED, log_skewed
import joblib

BG = (13, 17, 23)
PANEL = (33, 38, 45)
TEXT = (230, 237, 243)
DIM = (139, 148, 158)
CYAN = (88, 166, 255)
GREEN = (63, 185, 80)
AMBER = (210, 153, 34)
RED = (248, 81, 73)
BAR_BG = (45, 51, 59)

W, H = 920, 560
PAD = 22
LINE = 22
LOG_ROWS = 12
FRAMES = 72


def _font(size: int, bold: bool = False):
    for path in (r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    import matplotlib.font_manager as fm
    return ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans Mono")), size)


F = _font(15)
FB = _font(15, bold=True)
FBIG = _font(17, bold=True)
FSMALL = _font(13)
CHW = F.getbbox("M")[2]


def run_engine(dataset: str, n: int):
    model_path = ROOT / "results" / f"model-{dataset}-random-forest.joblib"
    split = datasets.load(dataset)
    model = joblib.load(model_path)
    Xte = log_skewed(split.X_test, SKEWED)
    n = min(n, len(Xte))
    X, X_raw = Xte.iloc[:n], split.X_test.iloc[:n]
    y = split.y_test.iloc[:n].to_numpy()

    policy = ips.Policy(target_fpr=0.01, block_threshold=0.98)
    engine = ips.PreventionEngine(model, policy,
                                  ips.default_signatures(set(X_raw.columns)))
    thr = engine.calibrate(y, X)
    decisions = list(engine.stream(X, X_raw))
    return split, decisions, y, thr, n


def _shorten(reason: str, width: int) -> str:
    reason = reason.replace("sig:", "")
    return reason if len(reason) <= width else reason[: width - 1] + "…"


def build_frames(dataset: str, n: int):
    split, decisions, y, thr, n = run_engine(dataset, n)

    act = np.array([d.action for d in decisions])
    is_alert = act == ips.ALERT
    is_block = act == ips.BLOCK
    acted = is_alert | is_block
    cum_allow = np.cumsum(act == ips.ALLOW)
    cum_alert = np.cumsum(is_alert)
    cum_block = np.cumsum(is_block)
    cum_tp = np.cumsum(acted & (y == 1))
    cum_fn = np.cumsum(~acted & (y == 1))
    cum_fp = np.cumsum(acted & (y == 0))

    notable = [(i, decisions[i]) for i in range(n) if acted[i]]
    throughput = 11_300

    frames = []
    cutoffs = [max(1, int(round(n * ((k + 1) / FRAMES) ** 2.2))) for k in range(FRAMES)]
    cutoffs[-1] = n
    for cut in cutoffs:
        j = cut - 1
        elapsed = cut / throughput
        clock = f"12:00:{int(elapsed):02d}.{int((elapsed % 1) * 1000):03d}"
        recent = [nd for nd in notable if nd[0] < cut][-LOG_ROWS:]
        tp, fn, fp = int(cum_tp[j]), int(cum_fn[j]), int(cum_fp[j])
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        frames.append({
            "clock": clock,
            "processed": cut,
            "n": n,
            "allow": int(cum_allow[j]),
            "alert": int(cum_alert[j]),
            "block": int(cum_block[j]),
            "recall": recall,
            "fa10k": 10_000 * fp / cut if cut else 0.0,
            "log": recent,
            "labels": split.test_labels.iloc[:n].to_numpy(),
        })
    return frames, split, thr, n


def draw_frame(fr, split, thr, n) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x = PAD
    y = PAD

    d.rounded_rectangle([PAD - 8, y - 8, W - PAD + 8, y + 26], 8, fill=PANEL)
    for k, c in enumerate((RED, AMBER, GREEN)):
        d.ellipse([x + k * 20, y + 4, x + k * 20 + 12, y + 16], fill=c)
    d.text((x + 78, y + 2), f"nidps · inline prevention engine ({split.name})",
           font=FB, fill=TEXT)
    d.text((W - PAD - 128, y + 3), f"clock {fr['clock']}", font=FSMALL, fill=DIM)
    y += 44

    d.text((x, y), "$ ", font=F, fill=GREEN)
    d.text((x + 2 * CHW, y), f"python src/ips.py --dataset {split.name.lower()} "
           f"--replay {n} --target-fpr 0.01 --demo", font=F, fill=TEXT)
    y += LINE
    d.text((x, y), f"alert threshold {thr:.4f}   block threshold 0.980   "
           f"signatures 2   policy: max 50 blocks / 10k window", font=FSMALL, fill=DIM)
    y += LINE + 6
    d.line([x, y, W - PAD, y], fill=BAR_BG, width=1)
    y += 12

    reason_w = 40
    for idx, dec in fr["log"]:
        if dec.action == ips.BLOCK:
            col, tag = RED, "BLOCK"
        else:
            col, tag = AMBER, "ALERT"
        t = idx / 11_300
        ev_clock = f"12:00:{int(t):02d}.{int((t % 1) * 100):02d}"
        line_y = y
        d.text((x, line_y), ev_clock, font=FSMALL, fill=DIM)
        d.text((x + 12 * CHW, line_y), tag, font=FB, fill=col)
        d.text((x + 18 * CHW, line_y), f"#{idx:<6}", font=F, fill=DIM)
        d.text((x + 26 * CHW, line_y), _shorten(dec.reason, reason_w), font=F, fill=TEXT)
        d.text((W - PAD - 10 * CHW, line_y), f"p={dec.score:0.3f}", font=F, fill=DIM)
        y += LINE
    y = PAD + 44 + 2 * LINE + 18 + LOG_ROWS * LINE + 10

    d.line([x, y, W - PAD, y], fill=BAR_BG, width=1)
    y += 14
    frac = fr["processed"] / fr["n"]
    bar_w = W - 2 * PAD
    d.rounded_rectangle([x, y, x + bar_w, y + 14], 6, fill=BAR_BG)
    d.rounded_rectangle([x, y, x + int(bar_w * frac), y + 14], 6, fill=CYAN)
    d.text((x, y + 22), f"{fr['processed']:,} / {fr['n']:,} flows   "
           f"{int(frac*100)}%   ~11,300 flows/s", font=FSMALL, fill=DIM)
    y += 48

    def tally(cx, label, val, col):
        d.text((cx, y), label, font=FSMALL, fill=DIM)
        d.text((cx, y + 16), f"{val:,}", font=FBIG, fill=col)

    tally(x, "ALLOW", fr["allow"], GREEN)
    tally(x + 180, "ALERT", fr["alert"], AMBER)
    tally(x + 360, "BLOCK", fr["block"], RED)
    d.text((x + 540, y), "RECALL", font=FSMALL, fill=DIM)
    d.text((x + 540, y + 16), f"{fr['recall']*100:.0f}%", font=FBIG, fill=CYAN)
    d.text((x + 700, y), "FALSE/10k", font=FSMALL, fill=DIM)
    d.text((x + 700, y + 16), f"{fr['fa10k']:.0f}", font=FBIG, fill=TEXT)

    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nsl-kdd")
    ap.add_argument("--flows", type=int, default=6000)
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "ips_demo.gif"))
    args = ap.parse_args()

    frames, split, thr, n = build_frames(args.dataset, args.flows)
    images = [draw_frame(fr, split, thr, n) for fr in frames]

    durations = [110] * len(images)
    durations[-1] = 2600
    durations[0] = 900
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({len(images)} frames, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
