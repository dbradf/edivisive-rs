use rstest::rstest;
use std::{fs, path::Path};
use edivisive::EDivisive;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct SampleSeries {
    series: Vec<f64>,
    expected: usize,
}

impl SampleSeries {
    fn from_file(series_name: &str) -> SampleSeries {
        let destination = format!("./tests/data/{}.json", series_name);
        let path = Path::new(&destination);
        let contents = fs::read_to_string(path).expect("Count not read file");
        serde_json::from_str(&contents).expect("Count not parse JSON")
    }
}

#[rstest(sample_series,  
    case("small"),
    case("short"),
    case("medium"),
)]
fn test_short_series(sample_series: &str) {
    let e_divisive = EDivisive::default();
    let sample_data = SampleSeries::from_file(sample_series);

    let change_points = e_divisive.get_change_points(&sample_data.series);
    
    assert_eq!(change_points.len(), sample_data.expected);
}
