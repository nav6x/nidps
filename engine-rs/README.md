# `engine-rs`: Rust Prevention Engine with Verified Parity

The Python engine in [`../src/ips.py`](../src/ips.py) is the reference
implementation: it evaluates each network flow to produce an ALLOW, ALERT, or
BLOCK decision. `engine-rs` is a standalone Rust reimplementation of that
decision path: preprocessing, random forest scoring, signatures, and blocking
policy.

It answers one central question:

> Can a compiled, dependency-free engine make **exactly** the same decisions as
> the scikit-learn detector, without a Python runtime?

**Parity is verified across every flow in both test sets: 22,544 NSL-KDD and
82,332 UNSW-NB15.** The primary objective is operational deployment: delivering
an engine that makes identical decisions to the reference detector in a compact,
standalone binary.

```
flow -> preprocess (exported spec) -> forest score -> signatures -> policy -> ALLOW / ALERT / BLOCK
```

## Decision Parity

Model training remains in Python. `export_model.py` serializes the fitted
pipeline (scaler, imputer, log1p parameters, one-hot column mapping, and every
tree in the ensemble) to JSON. Before writing to disk, it validates the export
against scikit-learn by reconstructing the full feature matrix and verifying
that it matches `pipeline.transform` across all 22,544 test rows.

`parity.py` executes both the Python engine and the compiled Rust binary over the
same test sets and compares every individual decision:

```
$ python engine-rs/parity.py --dataset nsl-kdd
flows compared     22,544
identical          22,544  (100.0000%)
mismatched         0
PARITY: PASS

$ python engine-rs/parity.py --dataset unsw-nb15
flows compared     82,332
identical          82,332  (100.0000%)
mismatched         0
PARITY: PASS
```

### Numerical Precision and Float32 Boundary Casts

Achieving 100% parity required matching scikit-learn's internal floating point
semantics. scikit-learn decision trees cast input features to `float32` prior to
evaluating split conditions (the tree stores split thresholds in `float64`, but
evaluates them against `float32` features).

One NSL-KDD test flow has a feature value between its `float32` and `float64`
representations, resting precisely on a split threshold. A standard `float64`
comparison branches differently in a single tree, altering the ensemble score by
~0.005 and changing the final classification. `src/model.rs` replicates this
cast by rounding features to `f32` during node evaluation, matching scikit-learn
output across all flows.

## Operational Rationale

Scoring benchmarks indicate that `engine-rs` matches scikit-learn's Cython
implementation in single-threaded throughput and provides 1.9x to 2.3x
throughput across multi-core workloads. The primary advantages are operational:

- **Zero Python dependencies:** A single ~2 MB static binary replaces the full
  Python runtime environment (Python, NumPy, SciPy, scikit-learn, joblib).
- **Deterministic execution:** Eliminates garbage collection pauses and Python
  interpreter startup latency.
- **Memory safety:** The execution path is strongly typed, with explicit bounds
  validation during tree traversal.

## Benchmark

Scoring throughput for 200-tree forests measured on an 8-core CPU. scikit-learn
runs `predict_proba` on precomputed feature matrices with `n_jobs` configured
for all cores and for single-thread execution.

| Dataset (flows x features) | Threads | Rust engine | scikit-learn | Ratio |
|---|---|---:|---:|---:|
| NSL-KDD (22,544 x 115) | single | ~49,000 flows/s | ~60,800 flows/s | 0.8x |
| NSL-KDD | all-core (8) | ~300,000 flows/s | ~159,000 flows/s | 1.9x |
| UNSW-NB15 (82,332 x 192) | single | ~41,000 flows/s | ~37,000 flows/s | 1.1x |
| UNSW-NB15 | all-core (8) | ~210,000 flows/s | ~91,000 flows/s | 2.3x |

Key observations:

- **Single-threaded performance is balanced:** scikit-learn's optimized Cython
  inner loop leads on NSL-KDD, while `engine-rs` leads on UNSW-NB15.
- **Multi-core scaling favors the Rust engine (1.9x to 2.3x):** `engine-rs` scales
  by ~6x across 8 cores, whereas scikit-learn plateaus at ~2.5x due to thread
  synchronization overhead.
- **Cache architecture:** Scoring uses a tree-major, cache-blocked layout (flows
  are scored in blocks that fit in L2 cache across each tree). Scoring within
  each block is deterministic, producing output bit-identical to sequential runs.

## Build and Run

Prerequisites: trained Python model artifact (`../results/model-nsl-kdd-random-forest.joblib`
via `make train`) and raw datasets (`make data`).

```bash
# from the repository root
make engine      # export models to JSON and build release binary
make parity      # run full decision parity verification

# direct dataset execution
python engine-rs/export_model.py --dataset unsw-nb15
cd engine-rs
cargo build --release
./target/release/nidps-engine \
    --model artifacts/model-unsw-nb15.json \
    --data ../data/raw/UNSW_NB15_testing-set.csv --bench
```

The binary automatically selects the appropriate parser based on the model's
`dataset` field: headerless for NSL-KDD, headered CSV for UNSW-NB15.

Unit tests in `tests/units.rs` validate quantile calculation, `f32` tree
traversal, and policy logic without requiring dataset files.

## Project Structure

```
export_model.py   Export and validate model/pipeline spec to JSON
parity.py         Execute Python and Rust pipelines; verify decision parity
src/model.rs      Spec deserialization, preprocessing, f32 tree evaluation
src/engine.rs     Signature rules, thresholding, policy resolution
src/main.rs       CLI interface, scoring harness, benchmarking
src/lib.rs        Library interface and dataset CSV parsers
tests/units.rs    Unit tests for quantiles, tree evaluation, and decisions
```

## Scope and Dataset Support

Both NSL-KDD and UNSW-NB15 are supported end-to-end. The serialization schema is
dataset-agnostic. Signature rules inspect specific feature names, preventing
cross-dataset false triggers. Adding support for a new dataset requires a CSV
reader and optional signature definitions; the preprocessing and tree evaluation
engine remains unchanged.
