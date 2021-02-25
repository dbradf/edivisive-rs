mod matrix_ops;
mod qhat;
mod util;

use matrix_ops::calc_diff_matrix;
use ndarray::prelude::*;
use qhat::{get_qhat_values, qhat_values};
use rand::prelude::SliceRandom;
use util::{argmax, get_windows, maximum};

fn get_best_change_point(
    diff_matrix: &ArrayView2<f64>,
    known_change_points: &Vec<usize>,
) -> (usize, f64) {
    let series_len = diff_matrix.nrows();
    let mut change_points: Vec<(usize, f64)> = vec![];

    let boundaries: Vec<usize> = get_windows(known_change_points, series_len);
    for bounds in boundaries.windows(2) {
        let a = bounds[0];
        let b = bounds[1];

        let qhats = qhat_values(&diff_matrix.slice(s!(a..b, a..b)));
        let max_idx = argmax(&qhats);
        change_points.push((max_idx + a, qhats[max_idx]));
    }

    let max_cp = argmax(&change_points.iter().map(|(_, val)| *val).collect());

    change_points[max_cp]
}

pub fn get_change_points(series: &Vec<f64>) -> Vec<usize> {
    let pvalue = 0.05;
    let permutations = 10;
    let diff_matrix = calc_diff_matrix(series);
    let mut change_points: Vec<usize> = vec![];

    let mut best_candidate = get_best_change_point(&diff_matrix.view(), &change_points);
    let mut windows = get_windows(&change_points, series.len());
    while is_significant(best_candidate.1, series, permutations, pvalue, &windows) {
        if change_points.contains(&best_candidate.0) {
            break;
        }
        change_points.push(best_candidate.0);
        windows = get_windows(&change_points, series.len());
        best_candidate = get_best_change_point(&diff_matrix.view(), &change_points);
    }

    change_points
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

fn is_significant(
    candidate: f64,
    series: &Vec<f64>,
    permutations: usize,
    pvalue: f64,
    windows: &Vec<usize>,
) -> bool {
    let permutes_with_higher = (0..permutations)
        .map(|_| permutation_test(series, windows))
        .filter(|v| v > &candidate)
        .count();
    let probability = permutes_with_higher as f64 / (permutations + 1) as f64;

    probability <= pvalue
}
