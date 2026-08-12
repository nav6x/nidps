# `engine-native`: Compiling the Forest to Native Code

The interpreted engine in [`../engine-rs`](../engine-rs) walks node arrays at
runtime. Another approach for throughput is treelite-style compilation: emit
the forest as straight-line code (one Rust function per tree, with nested
`if/else` conditions over split tests and constants baked in) and let the
compiler generate branch code with zero node-array loads at runtime.
[`codegen.py`](codegen.py) implements this approach.

Decisions match the interpreter across all 22,544 test flows, but throughput is
roughly 2x slower. This directory documents the benchmark and profiling data
behind this negative result.

## Result

Evaluated on the NSL-KDD test set (22,544 flows, 200-tree forest, 8-core CPU)
using the same `score_all` harness as the interpreted engine within the same
binary.

| Scoring | Interpreted (`engine-rs`) | Native (compiled) | Ratio |
|---|---:|---:|---:|
| single-thread | ~53,000 flows/s | ~33,000 flows/s | **0.63x** |
| all-core (8) | ~300,000 flows/s | ~222,000 flows/s | **0.74x** |

- **Decisions: 100% identical** (22,544/22,544) to the interpreter, which in
  turn matches the Python reference detector.
- **Scores: within ~0.013, not bit-identical.** Thresholds and leaf values are
  emitted as exact IEEE-754 bit patterns (`f64::from_bits(..)`). A small number of
  flows score slightly differently: once a threshold is a compile-time constant,
  the compiler optimises the `(x[f] as f32) as f64 <= threshold` comparison,
  subtly shifting float32 boundary behaviour compared to a runtime threshold.
  These shifts never cross the decision boundary.
- **Compile cost:** ~90 s to compile the generated crate alone; ~6 min from a
  clean build, generating 330k lines / 26 MB of Rust source.

## Profiling Analysis

The interpreted engine uses cache-blocked, tree-major scoring, which keeps a
compact inner loop hot in L1 instruction cache while streaming feature data
through L2 cache (detailed in `engine-rs`).

The compiled version reverses this trade-off. Scoring each flow requires
executing 200 large functions sequentially. Machine code for 200 deep decision
trees far exceeds L1 instruction cache capacity, causing heavy instruction
cache thrashing. Embedding split thresholds into `.text` removes data cache
loads but introduces a large, branch-heavy code footprint.

For a 200-tree forest evaluated per flow, a cache-blocked interpreter outperforms
naive branch generation. Specialized native compilers like Treelite rely on
vectorized all-nodes-at-once evaluation (QuickScorer algorithms) or compact
leaf representations rather than direct nested `if/else` codegen.

## Reproduction

Prerequisites: exported NSL-KDD model (`make engine` from repository root,
generating `../engine-rs/artifacts/model-nsl-kdd.json`) and raw dataset files.

```bash
make native
```

Or step-by-step:

```bash
python engine-native/codegen.py --dataset nsl-kdd
cd engine-native
cargo build --release
./target/release/nidps-engine-native \
    --model ../engine-rs/artifacts/model-nsl-kdd.json \
    --data ../data/raw/KDDTest+.txt
```

The test harness shares feature parsing, preprocessing, and classification
logic with `engine-rs`, isolating throughput differences strictly to the tree
traversal implementation.

## Conclusion

`engine-rs` remains the production deployment engine. Native code generation is
retained as a benchmark reference. Further native optimization would require
vectorized bitmask evaluation rather than straight-line branch code.
