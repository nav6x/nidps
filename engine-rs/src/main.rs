
use nidps_engine::data;
use nidps_engine::engine::{decide, threshold_for_fpr, Action, Policy};
use nidps_engine::model::Model;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    data: PathBuf,
    replay: Option<usize>,
    target_fpr: f64,
    block_threshold: f64,
    dump: Option<PathBuf>,
    bench: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::from("artifacts/model-nsl-kdd.json"),
        data: PathBuf::from("../data/raw/KDDTest+.txt"),
        replay: None,
        target_fpr: 0.01,
        block_threshold: 0.98,
        dump: None,
        bench: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => a.model = it.next().expect("--model needs a path").into(),
            "--data" => a.data = it.next().expect("--data needs a path").into(),
            "--replay" => a.replay = Some(it.next().unwrap().parse().expect("int")),
            "--target-fpr" => a.target_fpr = it.next().unwrap().parse().expect("float"),
            "--block-threshold" => {
                a.block_threshold = it.next().unwrap().parse().expect("float")
            }
            "--dump-decisions" => a.dump = Some(it.next().expect("path").into()),
            "--bench" => a.bench = true,
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }
    a
}

fn main() {
    let args = parse_args();

    let model = Model::load(&args.model).unwrap_or_else(|e| {
        eprintln!("{e}\n(run `python engine-rs/export_model.py` first)");
        std::process::exit(1);
    });

    let (mut rows, mut y) = data::read(&args.data, &model).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if let Some(n) = args.replay {
        rows.truncate(n);
        y.truncate(n);
    }
    let n = rows.len();

    let plan = model.plan();
    let t0 = Instant::now();
    let features: Vec<Vec<f64>> = rows.par_iter().map(|r| plan.feature_vector(r)).collect();
    let prep_secs = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let scores: Vec<f64> = model.score_all(&features);
    let score_secs = t0.elapsed().as_secs_f64();

    let policy = Policy { target_fpr: args.target_fpr, block_threshold: args.block_threshold, ..Default::default() };
    let alert_threshold = threshold_for_fpr(&scores, &y, policy.target_fpr);
    let decisions = decide(&model, &rows, &scores, alert_threshold, &policy);

    report(&model, n, &y, &decisions, alert_threshold, &policy, prep_secs, score_secs);

    if let Some(path) = &args.dump {
        let body: String = decisions.iter().map(|d| d.as_str()).collect::<Vec<_>>().join("\n");
        std::fs::write(path, body).expect("write decisions");
        println!("\nwrote {} decisions to {}", n, path.display());
    }

    if args.bench {
        benchmark(&model, &features);
    }
}

#[allow(clippy::too_many_arguments)]
fn report(
    model: &Model,
    n: usize,
    y: &[u8],
    decisions: &[Action],
    alert_threshold: f64,
    policy: &Policy,
    prep_secs: f64,
    score_secs: f64,
) {
    let (mut tp, mut fp, mut fn_, mut tn) = (0usize, 0usize, 0usize, 0usize);
    let (mut allow, mut alert, mut block, mut blocked_benign) = (0usize, 0usize, 0usize, 0usize);
    for (d, &label) in decisions.iter().zip(y) {
        match d {
            Action::Allow => allow += 1,
            Action::Alert => alert += 1,
            Action::Block => {
                block += 1;
                if label == 0 {
                    blocked_benign += 1;
                }
            }
        }
        match (d.acted(), label == 1) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }

    println!("dataset            {}", model.dataset);
    println!("flows replayed     {n}");
    println!("alert threshold    {alert_threshold:.4}  (target FPR {})", policy.target_fpr);
    println!("block threshold    {}", policy.block_threshold);
    println!("\npreprocess         {:>10.0} flows/s  ({:.0} ms)", n as f64 / prep_secs, prep_secs * 1000.0);
    println!("score (hot path)   {:>10.0} flows/s  ({:.0} ms)", n as f64 / score_secs, score_secs * 1000.0);
    println!("end-to-end         {:>10.0} flows/s", n as f64 / (prep_secs + score_secs));

    println!("\n--- outcome ---");
    println!("ALLOW   {allow:>7}");
    println!("ALERT   {alert:>7}");
    println!("BLOCK   {block:>7}");

    println!("\n--- detection (alert or block counts as detected) ---");
    println!("true positives     {tp}");
    println!("false positives    {fp}");
    println!("false negatives    {fn_}");
    println!("true negatives     {tn}");
    if tp + fn_ > 0 {
        println!("recall             {:.4}", tp as f64 / (tp + fn_) as f64);
    }
    if tp + fp > 0 {
        println!("precision          {:.4}", tp as f64 / (tp + fp) as f64);
    }
    println!("false alerts/10k   {:.1}", 10_000.0 * fp as f64 / n as f64);
    println!("\nblocked benign     {blocked_benign}  ({:.2}% of all blocks)", 100.0 * blocked_benign as f64 / block.max(1) as f64);
}

fn benchmark(model: &Model, features: &[Vec<f64>]) {
    let reps = 20;
    let n = features.len();

    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = model.score_all_seq(features);
    }
    let single = t0.elapsed().as_secs_f64() / reps as f64;

    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = model.score_all(features);
    }
    let multi = t0.elapsed().as_secs_f64() / reps as f64;

    println!("\n--- scoring benchmark (mean of {reps}, {n} flows, precomputed features) ---");
    println!("single-thread      {:>12.0} flows/s", n as f64 / single);
    println!("all-core           {:>12.0} flows/s  ({:.1}x)", n as f64 / multi, single / multi);
}
