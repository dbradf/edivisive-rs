use ndarray::prelude::*;
use ndarray::{Array, Array2};

fn diff_row(series: &Vec<f64>, value: f64) -> Vec<f64> {
    series.iter().map(|s| (*s - value).abs()).collect()
}

pub fn calc_diff_matrix(series: &Vec<f64>) -> Array2<f64> {
    let series_len = series.len();
    let diff_vectors: Vec<f64> = series.iter().flat_map(|i| diff_row(series, *i)).collect();

    Array::from_shape_vec((series_len, series_len), diff_vectors).unwrap()
}

pub fn sum_square(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_row() {
        let list = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let diff = diff_row(&list, 3.0);
        assert_eq!(diff, vec![2.0, 1.0, 0.0, 1.0, 2.0])
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
}
