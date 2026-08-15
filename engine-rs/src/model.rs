
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

const CHUNK: usize = 256;

#[derive(Deserialize)]
pub struct NumericSpec {
    pub source: String,
    pub log1p: bool,
    pub median: f64,
    pub mean: f64,
    pub scale: f64,
}

#[derive(Deserialize)]
pub struct CategoricalSpec {
    pub source: String,
    pub value_to_index: HashMap<String, usize>,
}

#[derive(Deserialize)]
struct RawTree {
    feature: Vec<i64>,
    threshold: Vec<f64>,
    left: Vec<i64>,
    right: Vec<i64>,
    proba1: Vec<f64>,
}

#[derive(Deserialize)]
struct ModelFile {
    dataset: String,
    n_features: usize,
    raw_columns: Vec<String>,
    label_column: String,
    numeric: Vec<NumericSpec>,
    categorical: Vec<CategoricalSpec>,
    n_trees: usize,
    trees: Vec<RawTree>,
}

#[derive(Clone, Copy)]
struct Node {
    threshold: f64,
    feature: i32,
    left: i32,
    right: i32,
}

pub struct Tree {
    nodes: Vec<Node>,
    proba1: Vec<f64>,
}

impl Tree {
    #[inline]
    pub fn eval(&self, x: &[f64]) -> f64 {
        let mut i = 0usize;
        loop {
            let nd = unsafe { self.nodes.get_unchecked(i) };
            if nd.left < 0 {
                return unsafe { *self.proba1.get_unchecked(i) };
            }
            let v = (x[nd.feature as usize] as f32) as f64;
            i = if v <= nd.threshold { nd.left as usize } else { nd.right as usize };
        }
    }
}

pub struct Model {
    pub dataset: String,
    pub n_features: usize,
    pub raw_columns: Vec<String>,
    pub label_column: String,
    pub numeric: Vec<NumericSpec>,
    pub categorical: Vec<CategoricalSpec>,
    pub n_trees: usize,
    pub trees: Vec<Tree>,
}

pub type RawRow = Vec<String>;

impl From<ModelFile> for Model {
    fn from(f: ModelFile) -> Self {
        let trees = f
            .trees
            .into_iter()
            .map(|t| {
                let nodes = (0..t.feature.len())
                    .map(|k| Node {
                        threshold: t.threshold[k],
                        feature: t.feature[k] as i32,
                        left: t.left[k] as i32,
                        right: t.right[k] as i32,
                    })
                    .collect();
                Tree { nodes, proba1: t.proba1 }
            })
            .collect();
        Model {
            dataset: f.dataset,
            n_features: f.n_features,
            raw_columns: f.raw_columns,
            label_column: f.label_column,
            numeric: f.numeric,
            categorical: f.categorical,
            n_trees: f.n_trees,
            trees,
        }
    }
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
        let file: ModelFile =
            serde_json::from_slice(&bytes).map_err(|e| format!("parse {path:?}: {e}"))?;
        Ok(file.into())
    }

    pub fn from_json(s: &str) -> Result<Self, String> {
        let file: ModelFile = serde_json::from_str(s).map_err(|e| e.to_string())?;
        Ok(file.into())
    }

    pub fn column_index(&self) -> HashMap<&str, usize> {
        self.raw_columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect()
    }

    pub fn plan(&self) -> FeaturePlan {
        let col = self.column_index();
        let numeric = self
            .numeric
            .iter()
            .map(|s| NumPlan {
                idx: col[s.source.as_str()],
                log1p: s.log1p,
                median: s.median,
                mean: s.mean,
                scale: s.scale,
            })
            .collect();
        let categorical = self
            .categorical
            .iter()
            .map(|s| CatPlan { idx: col[s.source.as_str()], map: s.value_to_index.clone() })
            .collect();
        FeaturePlan { n_features: self.n_features, numeric, categorical }
    }

    #[inline]
    pub fn score(&self, x: &[f64]) -> f64 {
        let mut acc = 0.0f64;
        for t in &self.trees {
            acc += t.eval(x);
        }
        acc / self.trees.len() as f64
    }

    #[inline]
    fn score_block(&self, feats: &[Vec<f64>], out: &mut [f64]) {
        for t in &self.trees {
            for (a, x) in out.iter_mut().zip(feats.iter()) {
                *a += t.eval(x);
            }
        }
        let k = self.trees.len() as f64;
        for v in out.iter_mut() {
            *v /= k;
        }
    }

    pub fn score_all_seq(&self, features: &[Vec<f64>]) -> Vec<f64> {
        let mut acc = vec![0.0f64; features.len()];
        for (feats, out) in features.chunks(CHUNK).zip(acc.chunks_mut(CHUNK)) {
            self.score_block(feats, out);
        }
        acc
    }

    pub fn score_all(&self, features: &[Vec<f64>]) -> Vec<f64> {
        let mut acc = vec![0.0f64; features.len()];
        features
            .par_chunks(CHUNK)
            .zip(acc.par_chunks_mut(CHUNK))
            .for_each(|(feats, out)| self.score_block(feats, out));
        acc
    }
}

pub struct NumPlan {
    pub idx: usize,
    pub log1p: bool,
    pub median: f64,
    pub mean: f64,
    pub scale: f64,
}

pub struct CatPlan {
    pub idx: usize,
    pub map: HashMap<String, usize>,
}

pub struct FeaturePlan {
    pub n_features: usize,
    pub numeric: Vec<NumPlan>,
    pub categorical: Vec<CatPlan>,
}

impl FeaturePlan {
    #[inline]
    pub fn feature_vector(&self, row: &RawRow) -> Vec<f64> {
        let mut x = vec![0.0f64; self.n_features];
        for (j, s) in self.numeric.iter().enumerate() {
            let mut v: Option<f64> = match row.get(s.idx) {
                Some(t) if !t.is_empty() => t.parse::<f64>().ok(),
                _ => None,
            };
            if let (Some(val), true) = (v, s.log1p) {
                v = Some(val.max(0.0).ln_1p());
            }
            let val = v.unwrap_or(s.median);
            x[j] = (val - s.mean) / s.scale;
        }
        for s in &self.categorical {
            if let Some(val) = row.get(s.idx) {
                if let Some(&idx) = s.map.get(val) {
                    x[idx] = 1.0;
                }
            }
        }
        x
    }
}
