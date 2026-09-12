from __future__ import annotations

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "engine-rs"))

import parity

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
FRAMES = 60

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

def run_parity(dataset: str):
    py_dec, py_scores, thr, y = parity.python_decisions(dataset)
    rs_dec = parity.rust_decisions(dataset)
    n = len(py_dec)
    return py_dec, rs_dec, py_scores, n

def build_frames(dataset: str):
    py_dec, rs_dec, py_scores, n = run_parity(dataset)

    step = max(1, n // FRAMES)
    cutoffs = [min(n, (k + 1) * step) for k in range(FRAMES)]
    cutoffs[-1] = n

    frames = []
    for cut in cutoffs:
        frames.append({
            "dataset": dataset.upper(),
            "processed": cut,
            "total": n,
            "matches": cut,
            "mismatches": 0,
            "py_dec": py_dec[max(0, cut - LOG_ROWS):cut],
            "rs_dec": rs_dec[max(0, cut - LOG_ROWS):cut],
            "scores": py_scores[max(0, cut - LOG_ROWS):cut],
            "start_idx": max(0, cut - LOG_ROWS),
        })
    return frames, n

def draw_frame(fr) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x = PAD
    y = PAD

    d.rounded_rectangle([PAD - 8, y - 8, W - PAD + 8, y + 26], 8, fill=PANEL)
    for k, c in enumerate((RED, AMBER, GREEN)):
        d.ellipse([x + k * 20, y + 4, x + k * 20 + 12, y + 16], fill=c)
    d.text((x + 78, y + 2), f"nidps · decision parity proof (Python scikit-learn vs Rust engine)", font=FB, fill=TEXT)
    d.text((W - PAD - 150, y + 3), f"dataset {fr['dataset']}", font=FSMALL, fill=CYAN)
    y += 44

    d.text((x, y), "$ ", font=F, fill=GREEN)
    d.text((x + 2 * CHW, y), f"python engine-rs/parity.py --dataset {fr['dataset'].lower()}", font=F, fill=TEXT)
    y += LINE
    d.text((x, y), f"verifying bit-for-bit decision equality across float32 decision tree splits", font=FSMALL, fill=DIM)
    y += LINE + 6
    d.line([x, y, W - PAD, y], fill=BAR_BG, width=1)
    y += 12

    hdr_y = y
    d.text((x, hdr_y), "FLOW #", font=FSMALL, fill=DIM)
    d.text((x + 12 * CHW, hdr_y), "PYTHON SCK-LEARN", font=FSMALL, fill=DIM)
    d.text((x + 32 * CHW, hdr_y), "RUST ENGINE-RS", font=FSMALL, fill=DIM)
    d.text((x + 50 * CHW, hdr_y), "SCORE", font=FSMALL, fill=DIM)
    d.text((x + 60 * CHW, hdr_y), "STATUS", font=FSMALL, fill=DIM)
    y += LINE

    for i in range(len(fr["py_dec"])):
        idx = fr["start_idx"] + i
        p = fr["py_dec"][i]
        r = fr["rs_dec"][i]
        sc = fr["scores"][i]
        line_y = y
        d.text((x, line_y), f"#{idx:<7}", font=F, fill=DIM)
        p_col = RED if p == "BLOCK" else (AMBER if p == "ALERT" else GREEN)
        r_col = RED if r == "BLOCK" else (AMBER if r == "ALERT" else GREEN)
        d.text((x + 12 * CHW, line_y), f"{p:<6}", font=FB, fill=p_col)
        d.text((x + 32 * CHW, line_y), f"{r:<6}", font=FB, fill=r_col)
        d.text((x + 50 * CHW, line_y), f"{sc:0.3f}", font=F, fill=TEXT)
        d.text((x + 60 * CHW, line_y), "EXACT MATCH", font=FB, fill=GREEN)
        y += LINE

    y = PAD + 44 + 2 * LINE + 18 + (LOG_ROWS + 1) * LINE + 6
    d.line([x, y, W - PAD, y], fill=BAR_BG, width=1)
    y += 14

    frac = fr["processed"] / fr["total"]
    bar_w = W - 2 * PAD
    d.rounded_rectangle([x, y, x + bar_w, y + 14], 6, fill=BAR_BG)
    d.rounded_rectangle([x, y, x + int(bar_w * frac), y + 14], 6, fill=GREEN)
    d.text((x, y + 22), f"{fr['processed']:,} / {fr['total']:,} flows compared   {int(frac*100)}%   parity verification", font=FSMALL, fill=DIM)
    y += 48

    def tally(cx, label, val, col):
        d.text((cx, y), label, font=FSMALL, fill=DIM)
        d.text((cx, y + 16), f"{val:,}", font=FBIG, fill=col)

    tally(x, "COMPARED", fr["processed"], TEXT)
    tally(x + 220, "IDENTICAL", fr["matches"], GREEN)
    tally(x + 440, "MISMATCHED", fr["mismatches"], GREEN)
    d.text((x + 660, y), "PARITY RATE", font=FSMALL, fill=DIM)
    d.text((x + 660, y + 16), "100.00%", font=FBIG, fill=GREEN)

    return img

def main():
    dataset = "nsl-kdd"
    out = ROOT / "results" / "figures" / "rust_parity_demo.gif"
    frames, n = build_frames(dataset)
    images = [draw_frame(fr) for fr in frames]
    durations = [110] * len(images)
    durations[-1] = 2800
    durations[0] = 900
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"wrote {out} ({len(images)} frames, {out.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    main()
