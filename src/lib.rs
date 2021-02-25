use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::{Array, Array2};
use rand::prelude::SliceRandom;

fn diff_row(series: &Vec<f64>, value: f64) -> Vec<f64> {
    series.iter().map(|s| (*s - value).abs()).collect()
}

#[test]
fn test_diff_row() {
    let list = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let diff = diff_row(&list, 3.0);
    assert_eq!(diff, vec![2.0, 1.0, 0.0, 1.0, 2.0])
}

fn calc_diff_matrix(series: &Vec<f64>) -> Array2<f64> {
    let series_len = series.len();
    let diff_vectors: Vec<f64> = series.iter().flat_map(|i| diff_row(series, *i)).collect();

    Array::from_shape_vec((series_len, series_len), diff_vectors).unwrap()
}

#[test]
fn test_calc_diff_matrix() {
    let list = vec![1.0, 2.0, 3.0];
    let diff = calc_diff_matrix(&list);

    assert_eq!(
        diff,
        arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    );
}

fn sum_square(
    matrix: &ArrayView2<f64>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> f64 {
    matrix
        .slice(s![row_start..row_end, col_start..col_end])
        .sum()
}

#[test]
fn test_sum_square() {
    let matrix = arr2(&[
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
    ]);

    assert_eq!(sum_square(&matrix.view(), 0, 1, 0, 1), 1.0);
    assert_eq!(sum_square(&matrix.view(), 0, 2, 0, 2), 8.0);
    assert_eq!(sum_square(&matrix.view(), 1, 3, 1, 3), 16.0);
}

fn calc_q(cross_term: f64, x_term: f64, y_term: f64, x_len: usize, y_len: usize) -> f64 {
    let x_len = x_len as f64;
    let y_len = y_len as f64;

    let cross_term_reg = if x_len < 1.0 || y_len < 1.0 {
        0.0
    } else {
        cross_term * (2.0 / (x_len * y_len))
    };

    let x_term_reg = if x_len < 2.0 {
        0.0
    } else {
        x_term * (2.0 / (x_len * (x_len - 1.0)))
    };

    let y_term_reg = if y_len < 2.0 {
        0.0
    } else {
        y_term * (2.0 / (y_len * (y_len - 1.0)))
    };

    let factor = (x_len * y_len as f64) / (x_len + y_len as f64);
    factor * (cross_term_reg - x_term_reg - y_term_reg)
}

fn qhat_values(diff_matrix: &ArrayView2<f64>) -> Vec<f64> {
    // We will partition our signal into:
    // X = {Xi; 0 <= i < tau}
    // Y = {Yj; tau <= j < len(signal) }
    // and look for argmax(tau)Q(tau)
    let series_len = diff_matrix.nrows();

    // sum |Xi - Yj| for i < tau <= j
    let mut cross_term = 0.0;
    // sum |Xi - Xj| for i < j < tau
    let mut x_term = 0.0;
    // sum |Yi - Yj| for tau <= i < j
    let mut y_term = 0.0;

    for row in 0..series_len {
        y_term += sum_square(diff_matrix, row, row + 1, row, series_len);
    }

    (0..series_len)
        .map(|tau| {
            let q = calc_q(cross_term, x_term, y_term, tau, series_len - tau);

            let column_delta = sum_square(diff_matrix, 0, tau, tau, tau + 1);
            let row_delta = sum_square(diff_matrix, tau, tau + 1, tau, series_len);

            cross_term = cross_term - column_delta + row_delta;
            x_term += column_delta;
            y_term -= row_delta;

            q
        })
        .collect()
}

#[test]
fn test_series() {
    let series = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];
    let q_values = get_qhat_values(&series);

    assert_eq!(q_values, vec!(0.0, 0.0));
}

fn get_qhat_values(series: &Vec<f64>) -> Vec<f64> {
    let diff_matrix = calc_diff_matrix(series);
    qhat_values(&diff_matrix.view())
}

fn maximum(list: &Vec<f64>) -> (usize, f64) {
    list.iter()
        .enumerate()
        .fold((0, 0.0), |(idx_max, val_max), (idx, val)| {
            if &val_max > val {
                (idx_max, val_max)
            } else {
                (idx, *val)
            }
        })
}

fn argmax(list: &Vec<f64>) -> usize {
    let (max_idx, _) = maximum(list);
    max_idx
}

#[test]
fn test_argmax() {
    let list = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(argmax(&list), 3);

    let list = vec![4.0, 3.0, 2.0, 1.0];
    assert_eq!(argmax(&list), 0);

    let list = vec![1.0, 1.0, 8.0, 1.0];
    assert_eq!(argmax(&list), 2);
}

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

fn get_windows(change_points: &Vec<usize>, series_len: usize) -> Vec<usize> {
    let mut boundaries: Vec<usize> = vec![0];
    boundaries.extend(change_points.iter().sorted());
    if boundaries.last().unwrap() != &series_len {
        boundaries.push(series_len);
    }

    boundaries
}

#[test]
fn test_get_windows() {
    let change_points: Vec<usize> = vec![];
    assert_eq!(get_windows(&change_points, 1), vec!(0, 1));

    let change_points: Vec<usize> = vec![3, 6, 9];
    assert_eq!(get_windows(&change_points, 12), vec!(0, 3, 6, 9, 12));

    let change_points: Vec<usize> = vec![7, 2, 9];
    assert_eq!(get_windows(&change_points, 15), vec!(0, 2, 7, 9, 15));
}

pub fn get_change_points(series: &Vec<f64>) -> Vec<usize> {
    let pvalue = 0.05;
    let permutations = 100;
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
