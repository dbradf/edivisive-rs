use ndarray::ArrayView2;

use crate::matrix_ops::{calc_diff_matrix, sum_square};

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

pub fn qhat_values(diff_matrix: &ArrayView2<f64>) -> Vec<f64> {
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

pub fn get_qhat_values(series: &Vec<f64>) -> Vec<f64> {
    let diff_matrix = calc_diff_matrix(series);
    qhat_values(&diff_matrix.view())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series() {
        let series = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let q_values = get_qhat_values(&series);

        assert_eq!(q_values, vec!(0.0, 0.0));
    }
}
