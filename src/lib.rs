mod matrix_ops;
mod qhat;
mod util;

use ndarray::prelude::*;
use matrix_ops::calc_diff_matrix;
use qhat::{get_qhat_values, qhat_values};
use rand::prelude::SliceRandom;
use util::{argmax, get_windows, maximum};

const DEFAULT_PVALUE: f64 = 0.01;
const DEFAULT_PERMUTATIONS: usize = 100;

pub struct EDivisive {
    pvalue: f64,
    n_permutations: usize,
}

#[derive(PartialEq, Copy, Clone, Debug)]
struct ChangePoint {
    index: usize,
    qhat: f64,
}

fn get_best_change_point(
    diff_matrix: &ArrayView2<f64>,
    known_change_points: &Vec<ChangePoint>,
) -> ChangePoint {
    let series_len = diff_matrix.nrows();
    let mut change_points: Vec<ChangePoint> = vec![];

    let boundaries: Vec<usize> = get_windows(&cp_indexes(&known_change_points), series_len);
    for bounds in boundaries.windows(2) {
        let a = bounds[0];
        let b = bounds[1];

        let qhats = qhat_values(&diff_matrix.slice(s!(a..b, a..b)));
        let max_idx = argmax(&qhats);
        change_points.push(ChangePoint{index: max_idx + a, qhat: qhats[max_idx]});
    }

    let max_index = argmax(&change_points.iter().map(|cp| cp.qhat).collect());

    change_points[max_index]
}

fn cp_indexes(change_points: &Vec<ChangePoint>) -> Vec<usize> {
    change_points.iter().map(|cp| cp.index).collect()
}

impl EDivisive {
    pub fn default() -> EDivisive {
        EDivisive {
            pvalue: DEFAULT_PVALUE,
            n_permutations: DEFAULT_PERMUTATIONS,
        }
    }

    pub fn new(pvalue: f64, n_permutations: usize) -> EDivisive {
        EDivisive {
            pvalue,
            n_permutations,
        }
    }

    pub fn get_change_points(&self, series: &Vec<f64>) -> Vec<usize> {
        let diff_matrix = calc_diff_matrix(series);
        let mut change_points: Vec<ChangePoint> = vec![];

        let mut best_candidate = get_best_change_point(&diff_matrix.view(), &change_points);
        let mut windows = get_windows(&cp_indexes(&change_points), series.len());
        while self.is_significant(&best_candidate, series, &windows) {
            if change_points.contains(&best_candidate) {
                break;
            }
            change_points.push(best_candidate);
            windows = get_windows(&cp_indexes(&change_points), series.len());
            best_candidate = get_best_change_point(&diff_matrix.view(), &change_points);
        }

        cp_indexes(&change_points)
    }

    fn is_significant(
        &self,
        candidate: &ChangePoint,
        series: &Vec<f64>,
        windows: &Vec<usize>,
    ) -> bool {
        if candidate.qhat < 1e-9 {
            return false;
        }
        let permutes_with_higher = (0..self.n_permutations)
            .map(|_| permutation_test(series, windows))
            .filter(|v| v > &candidate.qhat)
            .count();
        let probability = permutes_with_higher as f64 / (self.n_permutations + 1) as f64;

        probability <= self.pvalue
    }
}

fn permutation_test(series: &Vec<f64>, windows: &Vec<usize>) -> f64 {
    let mut rng = rand::thread_rng();
    let mut permuted_qhat_values: Vec<f64> = vec![];

    for bounds in windows.windows(2) {
        let a = bounds[0];
        let b = bounds[1];

        let mut window: Vec<f64> = vec![0.; b - a];
        window.copy_from_slice(&series[a..b]);
        window.shuffle(&mut rng);

        let q_list = get_qhat_values(&window);
        let (_, max_qhat) = maximum(&q_list);
        permuted_qhat_values.push(max_qhat);
    }

    let (_, max_value) = maximum(&permuted_qhat_values);
    max_value
}
