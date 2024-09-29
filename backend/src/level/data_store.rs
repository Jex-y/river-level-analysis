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

#[derive(Debug)]
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
        self.newest()
            .datetime
            .sub(self.oldest().datetime)
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
    head_datetime: Option<DateTime<Utc>>,

    /// The index in the buffer of the oldest feature.
    head_index: usize,

    /// The index in the buffer of the most recent feature.
    tail_index: usize,

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
            head_datetime: None,
            head_index: 0,
            tail_index: 0,
            buffer_size,
            frequency,
            initialised: false,
        };
    }

    fn datetime_to_index(&self, datetime: DateTime<Utc>) -> Result<usize> {
        Ok((datetime - self.get_head_datetime()?)
            .num_seconds()
            .div(self.frequency as i64) as usize)
    }

    fn datetime_to_offset(&self, datetime: DateTime<Utc>) -> Result<usize> {
        Ok((self.datetime_to_index(datetime)? + self.head_index) % self.buffer_size)
    }

    fn index_to_datetime(&self, index: usize) -> Result<DateTime<Utc>> {
        let offest = (index + self.buffer_size - self.head_index) % self.buffer_size;

        Ok(self.get_head_datetime()? + Duration::seconds(offest as i64 * self.frequency as i64))
    }

    fn buffer_offset_to_datetime(&self, index: usize) -> Result<DateTime<Utc>> {
        Ok(self
            .index_to_datetime((index + self.buffer_size - self.head_index) % self.buffer_size)?)
    }

    /// The timestamp of the oldest feature row in the buffer
    fn get_head_datetime(&self) -> Result<DateTime<Utc>> {
        self.head_datetime
            .ok_or(TimeSeriesBufferError::BufferNotInitialised)
    }

    /// The timestamp of the most recent feature row in the buffer
    fn get_tail_datetime(&self) -> Result<DateTime<Utc>> {
        Ok(self.index_to_datetime(self.tail_index)?)
    }

    fn init(&mut self, buffer_start_datetime: DateTime<Utc>, initial_size: usize) -> Result<()> {
        if self.initialised {
            return Err(TimeSeriesBufferError::BufferAlreadyInitialised);
        }

        self.head_datetime = Some(buffer_start_datetime);
        self.head_index = 0;
        self.tail_index = initial_size - 1;
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
            self.head_index = 0;
            self.tail_index = self.buffer_size - 1;
            self.head_datetime = Some(
                self.get_head_datetime()?
                    + Duration::seconds(self.buffer_size as i64 * self.frequency as i64),
            );

            return Ok(());
        }

        // If there is space between the tail and the head of the buffer, clear
        // that space.

        let remaining_space = self.buffer_size - self.tail_index - 1;
        let use_from_remaining = std::cmp::min(remaining_space, n);
        let use_from_start = n - use_from_remaining;

        if use_from_remaining > 0 {
            self.buffer
                .slice_mut(s![
                    self.tail_index + 1..self.tail_index + 1 + use_from_remaining,
                    ..
                ])
                .fill(f32::NAN);
        }

        // Clear the use_from_start oldest rows in the buffer.
        let prev_head_index = self.head_index;
        self.tail_index = (self.tail_index + use_from_start) % self.buffer_size;
        self.head_index = (self.head_index + use_from_start) % self.buffer_size;

        self.head_datetime = Some(
            self.get_head_datetime()?
                + Duration::seconds(use_from_start as i64 * self.frequency as i64),
        );

        if prev_head_index < self.head_index {
            self.buffer
                .slice_mut(s![prev_head_index..self.head_index, ..])
                .fill(f32::NAN);
            return Ok(());
        }

        self.buffer
            .slice_mut(s![prev_head_index.., ..])
            .fill(f32::NAN);
        self.buffer
            .slice_mut(s![..self.head_index, ..])
            .fill(f32::NAN);
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
        } else if col_newest > self.get_tail_datetime()? {
            self.clear_space(self.datetime_to_index(col_newest)? - self.tail_index)?;
        }

        #[cfg(debug_assertions)]
        {
            assert_eq!(col_newest, self.get_tail_datetime()?);
        }

        let data_start_offset = self.datetime_to_offset(col.oldest().datetime)?;
        let data_end_offset = self.datetime_to_offset(col.newest().datetime)?;

        let col_array: Array1<f32> = col.into();

        // If data_end_index is before data_start_index, the data wraps around the
        // buffer.
        if data_start_offset <= data_end_offset {
            self.buffer
                .slice_mut(s![data_start_offset..=data_end_offset, column_index])
                .assign(&col_array);
            return Ok(());
        }

        // Data wraps around
        self.buffer
            .slice_mut(s![data_start_offset.., column_index])
            .assign(&col_array.slice(s![..(self.buffer_size - data_start_offset)]));

        self.buffer
            .slice_mut(s![..=data_end_offset, column_index])
            .assign(&col_array.slice(s![col_array.len() - (data_end_offset + 1)..]));

        return Ok(());
    }

    /// Get the most recent n features for each column in the buffer.
    fn tail(&self, n: usize) -> Result<(Array2<f32>, DateTime<Utc>)> {
        // Get data from self.end_index - n to self.end_index

        if n > self.buffer_size {
            panic!("n must be less than or equal to buffer_size");
        }

        let i = (self.tail_index + self.buffer_size - n + 1) % self.buffer_size;
        let j = (self.tail_index) % self.buffer_size;

        if i < j {
            return Ok((
                self.buffer.slice(s![i..=j, ..]).to_owned(),
                self.get_tail_datetime()?,
            ));
        }

        // Data wraps around
        let mut data = Array2::uninit((n, self.buffer.shape()[1]));

        self.buffer
            .slice(s![i.., ..])
            .assign_to(&mut data.slice_mut(s![..(self.buffer_size - i), ..]));

        self.buffer
            .slice(s![..=j, ..])
            .assign_to(&mut data.slice_mut(s![self.buffer_size - i.., ..]));

        // All elements have now been initialised
        return Ok((unsafe { data.assume_init() }, self.get_tail_datetime()?));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn generate_column(
        column_name: &str,
        start_datetime: DateTime<Utc>,
        frequency: FrequencySeconds,
        data: Vec<f32>,
    ) -> FeatureColumn {
        let features: Vec<Feature> = data
            .iter()
            .enumerate()
            .map(|(i, value)| Feature {
                datetime: start_datetime + Duration::seconds(i as i64 * frequency as i64),
                value: *value,
            })
            .collect();

        return FeatureColumn::new(column_name.to_string(), features, frequency);
    }

    #[test]
    fn test_datetime_to_index() {
        const N: usize = 10;

        let mut buffer = TimeSeriesBuffer::empty(vec!["Column a".to_string()], N, 1);

        let start_datetime: DateTime<Utc> = "1970-01-01T00:00:00Z".parse().unwrap();
        let end_datetime: DateTime<Utc> = "1970-01-01T00:00:09Z".parse().unwrap();

        let col = generate_column(
            "Column a",
            start_datetime,
            1,
            (0..N).map(|i| i as f32).collect(),
        );

        assert_eq!(col.newest().datetime, end_datetime);

        buffer.update_column(col).unwrap();

        assert_eq!(buffer.datetime_to_index(start_datetime).unwrap(), 0);
        assert_eq!(buffer.datetime_to_index(end_datetime).unwrap(), 9);

        assert_eq!(buffer.index_to_datetime(0).unwrap(), start_datetime);
        assert_eq!(buffer.index_to_datetime(9).unwrap(), end_datetime);
    }

    #[test]
    fn test_single_column() {
        let mut buffer = TimeSeriesBuffer::empty(vec!["Column a".to_string()], 10, 1);

        let col = generate_column(
            "Column a",
            "1970-01-01T00:00:00Z".parse().unwrap(),
            1,
            (0..10).map(|i| i as f32).collect(),
        );

        let col_newest = col.newest().datetime;

        buffer.update_column(col).unwrap();

        let (tail, datetime) = buffer.tail(5).unwrap();

        assert_eq!(tail.shape(), &[5, 1]);

        for i in 0..5 {
            assert_eq!(tail[[i, 0]], (i + 5) as f32);
        }

        assert_eq!(datetime, col_newest);
    }

    #[test]
    fn test_single_column_overwrite() {
        let mut buffer = TimeSeriesBuffer::empty(vec!["Column a".to_string()], 10, 1);

        let col_1 = generate_column(
            "Column a",
            "1970-01-01T00:00:00Z".parse().unwrap(),
            1,
            (0..10).map(|i| i as f32).collect(),
        );
        buffer.update_column(col_1).unwrap();

        let col_2 = generate_column(
            "Column a",
            "1970-01-01T00:00:07Z".parse().unwrap(),
            1,
            (0..5).map(|i| (-i - 1) as f32).collect(),
        );

        let col_2_newest = col_2.newest().datetime;

        buffer.update_column(col_2).unwrap();

        let (tail, datetime) = buffer.tail(10).unwrap();

        assert_eq!(
            tail,
            array![
                [2.0],
                [3.0],
                [4.0],
                [5.0],
                [6.0],
                [-1.0],
                [-2.0],
                [-3.0],
                [-4.0],
                [-5.0],
            ]
        );

        assert_eq!(datetime, col_2_newest);
    }

    #[test]
    fn test_two_columns() {
        let mut buffer =
            TimeSeriesBuffer::empty(vec!["Column a".to_string(), "Column b".to_string()], 10, 1);

        let col_a = generate_column(
            "Column a",
            "1970-01-01T00:00:00Z".parse().unwrap(),
            1,
            (0..10).map(|i| i as f32).collect(),
        );

        let col_b = generate_column(
            "Column b",
            "1970-01-01T00:00:00Z".parse().unwrap(),
            1,
            (10..20).map(|i| i as f32).collect(),
        );

        let col_newest = col_a.newest().datetime;

        buffer.update_column(col_a).unwrap();

        buffer.update_column(col_b).unwrap();

        let (tail, datetime) = buffer.tail(5).unwrap();

        assert_eq!(tail.shape(), &[5, 2]);
        assert_eq!(
            tail,
            array![
                [5.0, 15.0],
                [6.0, 16.0],
                [7.0, 17.0],
                [8.0, 18.0],
                [9.0, 19.0],
            ]
        );

        assert_eq!(datetime, col_newest);
    }

    #[test]
    fn test_unaligned_columns() {
        let mut buffer =
            TimeSeriesBuffer::empty(vec!["Column a".to_string(), "Column b".to_string()], 10, 1);

        let col_a = generate_column(
            "Column a",
            "1970-01-01T00:00:00Z".parse().unwrap(),
            1,
            (0..10).map(|i| i as f32).collect(),
        );

        let col_b = generate_column(
            "Column b",
            "1970-01-01T00:00:00Z".parse().unwrap(),
            1,
            (10..18).map(|i| i as f32).collect(),
        );

        let col_newest = col_a.newest().datetime;

        buffer.update_column(col_a).unwrap();

        let (tail, datetime) = buffer.tail(5).unwrap();

        assert_eq!(tail.shape(), &[5, 2]);

        assert_eq!(
            tail,
            array![
                [5.0, f32::NAN],
                [6.0, f32::NAN],
                [7.0, f32::NAN],
                [8.0, f32::NAN],
                [9.0, f32::NAN],
            ]
        );

        assert_eq!(datetime, col_newest);

        buffer.update_column(col_b).unwrap();

        let (tail, datetime) = buffer.tail(5).unwrap();

        assert_eq!(tail.shape(), &[5, 2]);

        assert_eq!(
            tail,
            array![
                [5.0, 15.0],
                [6.0, 16.0],
                [7.0, 17.0],
                [8.0, f32::NAN],
                [9.0, f32::NAN],
            ]
        );

        assert_eq!(datetime, col_newest);
    }
}
