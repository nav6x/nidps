use nidps_engine::engine::{decide, quantile_linear, threshold_for_fpr, Action, Policy};
use nidps_engine::model::Model;

#[test]
fn quantile_matches_numpy_linear() {
    let a = [0.0, 1.0, 2.0, 3.0, 4.0];
    assert!((quantile_linear(&a, 0.5) - 2.0).abs() < 1e-12);
    assert!((quantile_linear(&a, 0.99) - 3.96).abs() < 1e-12);
    assert_eq!(quantile_linear(&a, 0.0), 0.0);
    assert_eq!(quantile_linear(&a, 1.0), 4.0);
    assert_eq!(quantile_linear(&[], 0.99), 0.5);
}

#[test]
fn threshold_respects_budget() {
    let scores: Vec<f64> = (0..10).map(|i| i as f64 / 10.0).collect();
    let y = vec![0u8; 10];
    let thr = threshold_for_fpr(&scores, &y, 0.1);
    let realised = scores.iter().filter(|&&s| s >= thr).count() as f64 / 10.0;
    assert!(realised <= 0.1 + 1e-9, "budget blown: {realised}");
}

fn stub_model() -> Model {
    let json = r#"{
        "dataset": "stub",
        "n_features": 1,
        "raw_columns": ["flag","serror_rate","count","diff_srv_rate",
                        "dst_host_srv_count","root_shell","logged_in"],
        "label_column": "label",
        "numeric": [],
        "categorical": [],
        "n_trees": 1,
        "trees": [{"feature":[-2],"threshold":[-2.0],"left":[-1],"right":[-1],"proba1":[0.0]}]
    }"#;
    Model::from_json(json).unwrap()
}

fn row(fields: &[&str]) -> Vec<String> {
    fields.iter().map(|s| s.to_string()).collect()
}

#[test]
fn tree_eval_picks_the_right_leaf() {
    let json = r#"{
        "dataset":"t","n_features":1,"raw_columns":[],"label_column":"l",
        "numeric":[],"categorical":[],"n_trees":1,
        "trees":[{"feature":[0,-2,-2],"threshold":[0.5,-2.0,-2.0],
                  "left":[1,-1,-1],"right":[2,-1,-1],"proba1":[0.0,0.1,0.9]}]
    }"#;
    let m: Model = Model::from_json(json).unwrap();
    assert!((m.score(&[0.2]) - 0.1).abs() < 1e-12);
    assert!((m.score(&[0.8]) - 0.9).abs() < 1e-12);
}

#[test]
fn signatures_and_policy_decide_like_python() {
    let m = stub_model();
    let rows = vec![
        row(&["S0", "1.0", "200", "0", "0", "0", "0"]),
        row(&["SF", "0", "0", "0", "0", "1", "0"]),
        row(&["SF", "0", "0", "0", "0", "0", "1"]),
    ];
    let scores = vec![0.0, 0.0, 0.0];
    let decisions = decide(&m, &rows, &scores, 0.5, &Policy::default());
    assert_eq!(decisions[0], Action::Alert, "SYN flood is high-severity: alert");
    assert_eq!(decisions[1], Action::Block, "critical signature blocks");
    assert_eq!(decisions[2], Action::Allow, "benign flow below threshold");
}
