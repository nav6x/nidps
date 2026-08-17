from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "engine-rs" / "artifacts"
OUT = Path(__file__).resolve().parent / "src" / "forest_generated.rs"

def emit_tree(t: dict, out: list[str]) -> None:
    left, right, feat, thr, proba = (
        t["left"], t["right"], t["feature"], t["threshold"], t["proba1"])

    def node(i: int, depth: int) -> None:
        pad = "    " * depth
        if left[i] == -1:
            out.append(f"{pad}f64::from_bits({bits(proba[i])}u64)")
            return
        out.append(f"{pad}if (x[{feat[i]}] as f32) as f64 <= f64::from_bits({bits(thr[i])}u64) {{")
        node(left[i], depth + 1)
        out.append(f"{pad}}} else {{")
        node(right[i], depth + 1)
        out.append(f"{pad}}}")

    node(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nsl-kdd")
    args = ap.parse_args()

    model = json.loads((ARTIFACTS / f"model-{args.dataset}.json").read_text())
    trees = model["trees"]
    n = len(trees)

    out: list[str] = [
        "#![allow(clippy::all)]",
        "",
    ]
    for k, t in enumerate(trees):
        out.append(f"#[inline(never)]")
        out.append(f"fn t{k}(x: &[f64]) -> f64 {{")
        emit_tree(t, out)
        out.append("}")
        out.append("")

    terms = " + ".join(f"t{k}(x)" for k in range(n))
    out.append("#[inline]")
    out.append("pub fn score(x: &[f64]) -> f64 {")
    out.append(f"    ({terms}) / {n}f64")
    out.append("}")
    out.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    lines = len(out)
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.relative_to(ROOT)}  ({lines:,} lines, {size_mb:.1f} MB, "
          f"{n} trees)")


if __name__ == "__main__":
    main()
