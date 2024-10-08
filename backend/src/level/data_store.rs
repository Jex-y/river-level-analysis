use chrono::{DateTime, Utc};
use ndarray::Array2;
use std::ops::{Div, Sub};

type FrequencySeconds = u32;

#[derive(Debug, serde::Deserialize)]
pub struct Feature {
    pub value: f32,

    #[serde(alias = "dateTime")]
    pub datetime: DateTime<Utc>,
}

#[derive(Debug)]
pub struct FeatureColumn {
    pub features: Vec<Feature>,
    pub name: String,
    pub frequency: FrequencySeconds,
}

impl FeatureColumn {
    pub fn new(name: String, features: Vec<Feature>, frequency: FrequencySeconds) -> Self {
        #[cfg(debug_assertions)]
        {
            if features.is_empty() {
                panic!("FeatureColumn must have at least one feature");
            }

            if features.is_sorted_by(|a: &Feature, b: &Feature| a.datetime < b.datetime) {
                panic!("Features must be sorted by datetime");
            }
        }

        return Self {
            name,
            features,
            frequency,
        };
    }

    pub fn oldest(&self) -> &Feature {
        self.features
            .first()
            .expect("FeatureColumn must have at least one feature")
    }

    pub fn newest(&self) -> &Feature {
        self.features
            .last()
            .expect("FeatureColumn must have at least one feature")
    }

    fn is_continuous(&self) -> bool {
        // Check if the features are continuous in time.
        let expected_features = (self.oldest().datetime - self.newest().datetime)
            .num_seconds()
            .div(self.frequency as i64)
            + 1;

        return self.features.len() as i64 == expected_features;
    }

    fn num_non_nan(&self) -> usize {
        self.features.len()
    }

    fn num_features(&self) -> usize {
        self.newest()
            .datetime
            .sub(self.oldest().datetime)
            .num_seconds()
            .div(self.frequency as i64) as usize
            + 1
    }

    pub fn to_array(columns: &[Self], timesteps: usize) -> (Array2<f32>, DateTime<Utc>) {
        let most_recent_datetime = columns
            .iter()
            .map(|col| col.newest().datetime)
            .max()
            .unwrap();

        // let array = Array2::uninit((timesteps, columns.len()));

        let array = Array2::from_shape_fn((timesteps, columns.len()), |(i, j)| {
            let col = &columns[j];
            let index = (most_recent_datetime - col.oldest().datetime)
                .num_seconds()
                .div(col.frequency as i64)
                .sub(i as i64)
                .div(col.frequency as i64) as usize;

            if index >= col.num_non_nan() {
                return f32::NAN;
            }

            return col.features[index].value;
        });

        return (array, most_recent_datetime);
    }
}

// impl Into<Array1<f32>> for FeatureColumn {
//     fn into(self) -> Array1<f32> {
//         if self.is_continuous() {
//             Array1::from_iter(self.features.iter().map(|f| f.value))
//         } else {
//             let mut array = Array1::from_elem(self.num_features(), f32::NAN);

//             let discontinuous_features: Vec<(DateTime<Utc>, f32)> = self
//                 .features
//                 .windows(2)
//                 .filter_map(|window| {
//                     let (a, b) = (window[0], window[1]);

//                     if b.datetime - a.datetime ==
// Duration::seconds(self.frequency as i64) {                         return
// None;                     }

//                     return Some((a.datetime, a.value));
//                 })

//     }
// }
