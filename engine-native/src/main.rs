use nidps_engine::data;
use nidps_engine::engine::{decide, threshold_for_fpr, Policy};
use nidps_engine::model::Model;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

#[path = "forest_generated.rs"]
mod forest;

fn main() {
    let mut model_path = PathBuf::from("../engine-rs/artifacts/model-nsl-kdd.json");
    let mut data_path = PathBuf::from("../data/raw/KDDTest+.txt");
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model_path = it.next().unwrap().into(),
            "--data" => data_path = it.next().unwrap().into(),
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let model = Model::load(&model_path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let (rows, y) = data::read(&data_path, &model).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let n = rows.len();
    let plan = model.plan();
    let features: Vec<Vec<f64>> = rows.par_iter().map(|r| plan.feature_vector(r)).collect();

    let interp = model.score_all(&features);
    let generated: Vec<f64> = features.iter().map(|x| forest::score(x)).collect();
    let max_diff = interp
        .iter()
        .zip(&generated)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    let policy = Policy::default();
    let thr_i = threshold_for_fpr(&interp, &y, policy.target_fpr);
    let thr_g = threshold_for_fpr(&generated, &y, policy.target_fpr);
    let dec_i = decide(&model, &rows, &interp, thr_i, &policy);
    let dec_g = decide(&model, &rows, &generated, thr_g, &policy);
    let mism = dec_i.iter().zip(&dec_g).filter(|(a, b)| a != b).count();

    println!("dataset            {}", model.dataset);
    println!("flows              {n}");
    println!("max score diff     {max_diff:.2e}  (generated vs interpreted)");
    println!(
        "decisions identical {}  ({}/{}){}",
        if mism == 0 { "YES" } else { "NO" },
        n - mism,
        n,
        if mism == 0 { "" } else { " <-- MISMATCH" }
    );

    let reps = 20;

    let t = Instant::now();
    for _ in 0..reps {
        let _ = model.score_all_seq(&features);
    }
    let interp_1 = n as f64 / (t.elapsed().as_secs_f64() / reps as f64);

    let t = Instant::now();
    for _ in 0..reps {
        let _: Vec<f64> = features.iter().map(|x| forest::score(x)).collect();
    }
    let gen_1 = n as f64 / (t.elapsed().as_secs_f64() / reps as f64);

    let t = Instant::now();
    for _ in 0..reps {
        let _ = model.score_all(&features);
    }
    let interp_n = n as f64 / (t.elapsed().as_secs_f64() / reps as f64);

    let t = Instant::now();
    for _ in 0..reps {
        let _: Vec<f64> = features.par_iter().map(|x| forest::score(x)).collect();
    }
    let gen_n = n as f64 / (t.elapsed().as_secs_f64() / reps as f64);

    println!("\n--- scoring throughput (mean of {reps}) ---");
    println!("{:22}{:>14}{:>14}", "", "interpreted", "native");
    println!("{:22}{:>12.0}/s{:>12.0}/s   ({:.2}x)", "single-thread", interp_1, gen_1, gen_1 / interp_1);
    println!("{:22}{:>12.0}/s{:>12.0}/s   ({:.2}x)", "all-core", interp_n, gen_n, gen_n / interp_n);
}
