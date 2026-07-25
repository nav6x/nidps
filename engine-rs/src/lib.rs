
pub mod engine;
pub mod model;

pub mod data {

    use crate::model::{Model, RawRow};
    use std::collections::HashMap;
    use std::path::Path;

    pub fn read(path: &Path, model: &Model) -> Result<(Vec<RawRow>, Vec<u8>), String> {
        if model.dataset.eq_ignore_ascii_case("UNSW-NB15") {
            read_unsw(path, model)
        } else {
            read_nsl(path, model)
        }
    }

    pub fn read_nsl(path: &Path, model: &Model) -> Result<(Vec<RawRow>, Vec<u8>), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
        let label_idx = model.raw_columns.len();

        let mut rows = Vec::new();
        let mut y = Vec::new();
        for line in text.lines() {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let fields: Vec<String> = line.split(',').map(|s| s.to_string()).collect();
            if fields.len() <= label_idx {
                return Err(format!("row has {} fields, expected > {label_idx}", fields.len()));
            }
            y.push(if fields[label_idx] != "normal" { 1 } else { 0 });
            rows.push(fields);
        }
        Ok((rows, y))
    }

    pub fn read_unsw(path: &Path, model: &Model) -> Result<(Vec<RawRow>, Vec<u8>), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
        let mut lines = text.lines();
        let header = lines.next().ok_or("empty file")?;
        let header = header.trim_start_matches('\u{feff}').trim_end_matches('\r');
        let pos: HashMap<&str, usize> =
            header.split(',').enumerate().map(|(i, c)| (c, i)).collect();

        let resolve = |name: &str| -> Result<usize, String> {
            pos.get(name).copied().ok_or_else(|| format!("column {name:?} not in header"))
        };
        let col_idx: Vec<usize> = model
            .raw_columns
            .iter()
            .map(|c| resolve(c))
            .collect::<Result<_, _>>()?;
        let label_idx = resolve(&model.label_column)?;
        let max_idx = col_idx.iter().copied().max().unwrap_or(0).max(label_idx);

        let mut rows = Vec::new();
        let mut y = Vec::new();
        for line in lines {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() <= max_idx {
                return Err(format!("row has {} fields, expected > {max_idx}", fields.len()));
            }
            let row: RawRow = col_idx.iter().map(|&i| fields[i].to_string()).collect();
            let attack = fields[label_idx].trim().parse::<i64>().unwrap_or(0) != 0;
            y.push(if attack { 1 } else { 0 });
            rows.push(row);
        }
        Ok((rows, y))
    }
}
