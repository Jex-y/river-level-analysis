use chrono::{DateTime, Duration, Utc};
use ndarray::{s, Array, Array1, Array2};
use std::ops::{Div, Sub};
use thiserror::Error;

type FrequencySeconds = u32;

#[derive(Debug, serde::Deserialize)]
pub struct Feature {
    value: f32,

    #[serde(alias = "dateTime")]
    datetime: DateTime<Utc>,
}

pub struct FeatureColumn {
    features: Vec<Feature>,
    name: String,
    frequency: FrequencySeconds,
}

impl FeatureColumn {
    pub fn new(name: String, features: Vec<Feature>, frequency: FrequencySeconds) -> Self {
        #[cfg(debug_assertions)]
        {
            if features.is_empty() {
                panic!("FeatureColumn must have at least one feature");
            }

            if features.is_sorted_by(|a: &Feature, b: &Feature| a.datetime > b.datetime) {
                panic!("Features must be sorted by datetime");
            }
        }

        return Self {
            name,
            features,
            frequency,
        };
    }

    fn oldest(&self) -> &Feature {
        self.features
            .first()
            .expect("FeatureColumn must have at least one feature")
    }

    fn newest(&self) -> &Feature {
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
        self.oldest()
            .datetime
            .sub(self.newest().datetime)
            .num_seconds()
            .div(self.frequency as i64) as usize
            + 1
    }
}

impl Into<Array1<f32>> for FeatureColumn {
    fn into(self) -> Array1<f32> {
        if self.is_continuous() {
            Array1::from_iter(self.features.iter().map(|f| f.value))
        } else {
            let mut array = Array1::from_elem(self.num_features(), f32::NAN);

            for feature in self.features.iter() {
                let index = (feature.datetime - self.oldest().datetime)
                    .num_seconds()
                    .div(self.frequency as i64) as usize;
                array[index] = feature.value;
            }

            array
        }
    }
}

pub struct TimeSeriesBuffer {
    buffer: Array2<f32>,
    feature_names: Vec<String>,

    /// The timestamp of the oldest feature of any reading.
    buffer_start_datetime: Option<DateTime<Utc>>,

    /// The index in the buffer of the oldest feature.
    buffer_start_index: usize,

    /// The index in the buffer of the most recent feature.
    buffer_end_index: usize,

    /// The timestamp of the most recent feature of the reading that is furthest
    /// behind in time.
    least_recent_feature: Option<DateTime<Utc>>,

    buffer_size: usize,
    frequency: FrequencySeconds,

    initialised: bool,
}

#[derive(Debug, Error)]
pub enum TimeSeriesBufferError {
    #[error("FeatureColumn {0} has a different frequency to the buffer")]
    MismatchedFrequency(String),

    #[error("Buffer is not initialised")]
    BufferNotInitialised,

    #[error("Column {0} is not in the buffer")]
    ColumnNotInBuffer(String),

    #[error("Index {0} is out of bounds")]
    IndexOutOfBounds(usize),

    #[error("FeatureColumn {0} is not continuous in time")]
    NonContinuousFeatureColumn(String),

    #[error("Buffer already initialised")]
    BufferAlreadyInitialised,
}

type Result<T> = std::result::Result<T, TimeSeriesBufferError>;

impl TimeSeriesBuffer {
    fn empty(column_names: Vec<String>, buffer_size: usize, frequency: FrequencySeconds) -> Self {
        return Self {
            buffer: Array2::from_elem((buffer_size, column_names.len()), f32::NAN),
            feature_names: column_names,
            buffer_start_datetime: None,
            buffer_start_index: 0,
            buffer_end_index: 0,
            least_recent_feature: None,
            buffer_size,
            frequency,
            initialised: false,
        };
    }

    fn datetime_to_buffer_index(&self, datetime: DateTime<Utc>) -> Result<usize> {
        Ok((datetime - self.buffer_start_datetime()?)
            .num_seconds()
            .div(self.frequency as i64) as usize)
    }

    fn datetime_to_array_index(&self, datetime: DateTime<Utc>) -> Result<usize> {
        Ok((self.datetime_to_buffer_index(datetime)? + self.buffer_start_index) % self.buffer_size)
    }

    fn buffer_index_to_datetime(&self, index: usize) -> Result<DateTime<Utc>> {
        Ok(self.buffer_start_datetime()? + Duration::seconds(index as i64 * self.frequency as i64))
    }

    fn array_index_to_datetime(&self, index: usize) -> Result<DateTime<Utc>> {
        Ok(self.buffer_index_to_datetime(
            (index + self.buffer_size - self.buffer_start_index) % self.buffer_size,
        )?)
    }

    /// The timestamp of the oldest feature row in the buffer
    fn buffer_start_datetime(&self) -> Result<DateTime<Utc>> {
        self.buffer_start_datetime
            .ok_or(TimeSeriesBufferError::BufferNotInitialised)
    }

    fn least_recent_feature(&self) -> Result<DateTime<Utc>> {
        self.least_recent_feature
            .ok_or(TimeSeriesBufferError::BufferNotInitialised)
    }

    /// The timestamp of the most recent feature row in the buffer
    fn buffer_end_datetime(&self) -> Result<DateTime<Utc>> {
        Ok(self.buffer_index_to_datetime(self.buffer_end_index)?)
    }

    fn init(&mut self, buffer_start_datetime: DateTime<Utc>, initial_size: usize) -> Result<()> {
        if self.initialised {
            return Err(TimeSeriesBufferError::BufferAlreadyInitialised);
        }

        self.buffer_start_datetime = Some(buffer_start_datetime);
        self.buffer_start_index = 0;
        self.buffer_end_index = initial_size - 1;
        self.initialised = true;

        return Ok(());
    }

    fn clear_space(&mut self, n: usize) -> Result<()> {
        // Clear space for n new rows in the buffer.

        if n > self.buffer_size {
            return Err(TimeSeriesBufferError::IndexOutOfBounds(n));
        }

        if n == 0 {
            return Ok(());
        }

        if n == self.buffer_size {
            self.buffer.fill(f32::NAN);
            return Ok(());
        }

        // Clear the n oldest rows in the buffer.
        let i = self.buffer_start_index;
        let j = (self.buffer_start_index + n) % self.buffer_size;
        self.buffer_start_index = j;

        self.buffer_start_datetime = Some(self.buffer_index_to_datetime(j)?);

        if i < j {
            self.buffer.slice_mut(s![i..j, ..]).fill(f32::NAN);
            return Ok(());
        }

        self.buffer.slice_mut(s![i.., ..]).fill(f32::NAN);
        self.buffer.slice_mut(s![..j, ..]).fill(f32::NAN);
        return Ok(());
    }

    fn update_column(&mut self, col: FeatureColumn) -> Result<()> {
        if col.frequency != self.frequency {
            return Err(TimeSeriesBufferError::MismatchedFrequency(col.name.clone()));
        }

        let column_index = self
            .feature_names
            .iter()
            .position(|name| name == &col.name)
            .ok_or(TimeSeriesBufferError::ColumnNotInBuffer(col.name.clone()))?;

        let col_oldest = col.oldest().datetime;
        let col_newest = col.newest().datetime;

        if !self.initialised {
            self.init(col_oldest, col.num_features())?;
        } else if col_newest > self.buffer_end_datetime()? {
            self.clear_space(self.datetime_to_buffer_index(col_newest)? - self.buffer_end_index)?;
        }

        let data_start_index = self.datetime_to_array_index(col.oldest().datetime)?;
        let data_end_index = self.datetime_to_array_index(col.newest().datetime)?;

        // If data_end_index is before data_start_index, the data wraps around the
        // buffer.

        let col_array: Array1<f32> = col.into();

        if data_start_index <= data_end_index {
            self.buffer
                .slice_mut(s![data_start_index..=data_end_index, column_index])
                .assign(&col_array);
            return Ok(());
        }

        self.buffer
            .slice_mut(s![data_start_index.., column_index])
            .assign(&col_array.slice(s![..col_array.len() - (data_start_index - data_end_index)]));

        return Ok(());
    }

    /// Get the most recent n features for each column in the buffer.
    fn tail(&self, n: usize) -> Result<(Array2<f32>, DateTime<Utc>)> {
        // Get data from self.end_index - n to self.end_index

        if n > self.buffer_size {
            panic!("n must be less than or equal to buffer_size");
        }

        let i = ((self.buffer_end_index as i64 - n as i64) % self.buffer_size as i64 + 1) as usize;
        let j = self.buffer_end_index % self.buffer_size + 1;
        println!("buffer: {:?}, i: {}, j: {}", self.buffer, i, j);

        if i < j {
            return Ok((
                self.buffer.slice(s![i..j, ..]).to_owned(),
                self.buffer_end_datetime()?,
            ));
        }

        // Data wraps around
        let mut data = Array2::uninit((n, self.buffer.shape()[1]));

        self.buffer
            .slice(s![i.., ..])
            .assign_to(&mut data.slice_mut(s![0.., ..]));

        self.buffer
            .slice(s![..j, ..])
            .assign_to(&mut data.slice_mut(s![n - (j - i).., ..]));

        // All elements have now been initialised
        return Ok((unsafe { data.assume_init() }, self.buffer_end_datetime()?));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_datetime_to_index() {
        const N: usize = 10;

        let mut buffer = TimeSeriesBuffer::empty(vec!["Column a".to_string()], N, 1);

        let start_datetime: DateTime<Utc> = "1970-01-01T00:00:00Z".parse().unwrap();
        let end_datetime: DateTime<Utc> = "1970-01-01T00:00:09Z".parse().unwrap();
        let data: Vec<Feature> = (0..N)
            .map(|i| Feature {
                datetime: start_datetime + Duration::seconds(i as i64),
                value: i as f32,
            })
            .collect();

        let col = FeatureColumn::new("Column a".to_string(), data, 1);

        buffer.update_column(col).unwrap();

        assert_eq!(buffer.datetime_to_buffer_index(start_datetime).unwrap(), 0);
        assert_eq!(buffer.datetime_to_buffer_index(end_datetime).unwrap(), 9);

        assert_eq!(buffer.buffer_index_to_datetime(0).unwrap(), start_datetime);
        assert_eq!(buffer.buffer_index_to_datetime(9).unwrap(), end_datetime);
    }

    #[test]
    fn test_single_column() {
        const N: usize = 10;

        let mut buffer = TimeSeriesBuffer::empty(vec!["Column a".to_string()], N, 1);

        let start_datetime: DateTime<Utc> = "1970-01-01T00:00:00Z".parse().unwrap();
        let end_datetime: DateTime<Utc> = "1970-01-01T00:00:09Z".parse().unwrap();

        let data: Vec<Feature> = (0..N)
            .map(|i| Feature {
                datetime: start_datetime + Duration::seconds(i as i64),
                value: i as f32,
            })
            .collect();

        let col = FeatureColumn::new("Column a".to_string(), data, 1);

        buffer.update_column(col).unwrap();

        let (tail, datetime) = buffer.tail(5).unwrap();

        assert_eq!(tail.shape(), &[5, 1]);

        for i in 0..5 {
            assert_eq!(tail[[i, 0]], (i + 5) as f32);
        }

        assert_eq!(datetime, end_datetime);
    }

    #[test]
    fn test_single_column_overwrite() {
        let mut buffer = TimeSeriesBuffer::empty(vec!["Column a".to_string()], 10, 1);

        let start_datetime_1: DateTime<Utc> = "1970-01-01T00:00:00Z".parse().unwrap();

        let data_1: Vec<Feature> = (0..10)
            .map(|i| Feature {
                datetime: start_datetime_1 + Duration::seconds(i as i64),
                value: i as f32,
            })
            .collect();

        let col_1 = FeatureColumn::new("Column a".to_string(), data_1, 1);

        buffer.update_column(col_1).unwrap();

        let start_datetime_2: DateTime<Utc> = "1970-01-01T00:00:07Z".parse().unwrap();
        let end_datetime_2: DateTime<Utc> = "1970-01-01T00:00:12Z".parse().unwrap();

        let data_2: Vec<Feature> = (0..5)
            .map(|i| Feature {
                datetime: start_datetime_2 + Duration::seconds(i as i64),
                value: (-i - 1) as f32,
            })
            .collect();

        let col_2 = FeatureColumn::new("Column a".to_string(), data_2, 1);

        buffer.update_column(col_2).unwrap();

        let (tail, datetime) = buffer.tail(10).unwrap();

        assert_eq!(tail.shape(), &[10, 1]);

        assert_eq!(
            tail,
            array![
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
                [-1.0],
                [-2.0],
                [-3.0],
                [-4.0],
                [-5.0],
            ]
        );

        assert_eq!(datetime, end_datetime_2);
    }
}
