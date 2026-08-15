
use crate::model::{Model, RawRow};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Allow,
    Alert,
    Block,
}

impl Action {
    pub fn as_str(&self) -> &'static str {
        match self {
            Action::Allow => "ALLOW",
            Action::Alert => "ALERT",
            Action::Block => "BLOCK",
        }
    }
    pub fn acted(&self) -> bool {
        !matches!(self, Action::Allow)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Severity {
    Critical,
    Other,
}

pub struct Policy {
    pub target_fpr: f64,
    pub block_threshold: f64,
    pub max_blocks_per_window: usize,
    pub window: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            target_fpr: 0.01,
            block_threshold: 0.98,
            max_blocks_per_window: 50,
            window: 10_000,
        }
    }
}

fn signature_severity(row: &RawRow, col: &HashMap<&str, usize>) -> Option<Severity> {
    let f = |name: &str| -> f64 {
        col.get(name)
            .and_then(|&i| row.get(i))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    let s = |name: &str| -> &str {
        col.get(name).and_then(|&i| row.get(i)).map(|s| s.as_str()).unwrap_or("")
    };

    let flag = s("flag");
    if (flag == "S0" || flag == "S1") && f("serror_rate") > 0.85 && f("count") > 100.0 {
        return Some(Severity::Other);
    }
    if f("diff_srv_rate") > 0.7 && f("dst_host_srv_count") < 5.0 && f("count") > 20.0 {
        return Some(Severity::Other);
    }
    if f("root_shell") == 1.0 && f("logged_in") == 0.0 {
        return Some(Severity::Critical);
    }
    if f("is_ftp_login") == 1.0 && f("ct_ftp_cmd") > 2.0 {
        return Some(Severity::Other);
    }
    None
}

pub fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => return 0.5,
        1 => return sorted[0],
        _ => {}
    }
    let n = sorted.len();
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

pub fn threshold_for_fpr(scores: &[f64], y: &[u8], target_fpr: f64) -> f64 {
    let mut benign: Vec<f64> = scores
        .iter()
        .zip(y)
        .filter(|(_, &label)| label == 0)
        .map(|(&s, _)| s)
        .collect();
    benign.sort_by(|a, b| a.partial_cmp(b).unwrap());
    quantile_linear(&benign, 1.0 - target_fpr)
}

struct BlockBudget {
    recent: VecDeque<usize>,
    window: usize,
    max: usize,
}

impl BlockBudget {
    fn new(window: usize, max: usize) -> Self {
        BlockBudget { recent: VecDeque::new(), window, max }
    }
    fn ok(&mut self, i: usize) -> bool {
        while let Some(&front) = self.recent.front() {
            if i - front > self.window {
                self.recent.pop_front();
            } else {
                break;
            }
        }
        self.recent.len() < self.max
    }
    fn record(&mut self, i: usize) {
        self.recent.push_back(i);
    }
}

pub fn decide(
    model: &Model,
    rows: &[RawRow],
    scores: &[f64],
    alert_threshold: f64,
    policy: &Policy,
) -> Vec<Action> {
    let col = model.column_index();
    let mut budget = BlockBudget::new(policy.window, policy.max_blocks_per_window);
    let mut out = Vec::with_capacity(rows.len());

    for (i, row) in rows.iter().enumerate() {
        let action = if let Some(sev) = signature_severity(row, &col) {
            if sev == Severity::Critical && budget.ok(i) {
                budget.record(i);
                Action::Block
            } else {
                Action::Alert
            }
        } else if scores[i] >= policy.block_threshold && budget.ok(i) {
            budget.record(i);
            Action::Block
        } else if scores[i] >= alert_threshold {
            Action::Alert
        } else {
            Action::Allow
        };
        out.push(action);
    }
    out
}
