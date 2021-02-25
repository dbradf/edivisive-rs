use edivisive::get_change_points;

fn main() {
    // let a: Vec<f64> = (0..1000).map(|d| d as f64).collect();
    // let diff_matrix = calc_diff_matrix(&a);
    // let q = qhat_values(&diff_matrix);
    // println!("{:?}", q);

    let series: Vec<f64> = vec![0., 0., 0., 0., 0., 1., 1., 1., 1., 1.];
    let change_points = get_change_points(&series);

    println!("Series: {:?}", series);
    println!("Change Points: {:?}", change_points);
}
