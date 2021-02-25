use ndarray::prelude::*;
use ndarray::{Array, Array2};

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
    let diff_matrix = calc_diff_matrix(&series);
    let q_values = qhat_values(&diff_matrix.view());

    assert_eq!(q_values, vec!(0.0, 0.0));
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

fn get_best_change_point(diff_matrix: &ArrayView2<f64>) -> usize {
    let mut change_points: Vec<(usize, f64)> = vec![];

    let boundaries: Vec<usize> = vec![0, diff_matrix.nrows()];
    for bounds in boundaries.windows(2) {
        let a = bounds[0];
        let b = bounds[1];

        let qhats = qhat_values(&diff_matrix.slice(s!(a..b, a..b)));
        let max_idx = argmax(&qhats);
        change_points.push((max_idx + a, qhats[max_idx]));
    }

    let max_cp = argmax(&change_points.iter().map(|(_, val)| *val).collect());

    change_points[max_cp].0
}

pub fn get_change_points(series: &Vec<f64>) -> Vec<usize> {
    let diff_matrix = calc_diff_matrix(series);
    let mut changes_points: Vec<usize> = vec![];

    let best_candidate = get_best_change_point(&diff_matrix.view());
    changes_points.push(best_candidate);

    changes_points
}
