use itertools::Itertools;

pub fn maximum(list: &Vec<f64>) -> (usize, f64) {
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

pub fn argmax(list: &Vec<f64>) -> usize {
    let (max_idx, _) = maximum(list);
    max_idx
}

pub fn get_windows(change_points: &Vec<usize>, series_len: usize) -> Vec<usize> {
    let mut boundaries: Vec<usize> = vec![0];
    boundaries.extend(change_points.iter().sorted());
    if boundaries.last().unwrap() != &series_len {
        boundaries.push(series_len);
    }

    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let list = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(argmax(&list), 3);

        let list = vec![4.0, 3.0, 2.0, 1.0];
        assert_eq!(argmax(&list), 0);

        let list = vec![1.0, 1.0, 8.0, 1.0];
        assert_eq!(argmax(&list), 2);
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
}
