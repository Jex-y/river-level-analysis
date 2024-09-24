use chrono::{DateTime, Utc};
use ndarray::{s, Array2};
use thiserror::Error;

pub struct TimeSeriesBuffer {
    data: Array2<f32>,
    feature_names: Vec<String>,

    /// The timestamp of the most recent feature of any reading.
    most_recent_feature: Option<DateTime<Utc>>,

    /// The timestamp of the most recent feature of the reading that is furthest
    /// behind in time.
    least_recent_feature: Option<DateTime<Utc>>,

    /// The timestamp of the oldest feature of any reading.
    oldest_feature: Option<DateTime<Utc>>,

    initialised: bool,
    buffer_size: usize,
    frequency: u32,

    /// The index in the buffer of the oldest feature.
    boundary: usize,
}

pub struct Feature {
    value: f32,
    datetime: DateTime<Utc>,
}

pub struct FeatureColumn {
    features: Vec<Feature>,
    name: String,
    frequency: u32,
}

impl FeatureColumn {
    fn new(name: String, features: Vec<Feature>, frequency: u32) -> Self {
        #[cfg(debug_assertions)]
        {
            if features.is_empty() {
                panic!("FeatureColumn must have at least one feature");
            }

            if features.is_sorted_by_key(|f| f.datetime) {
                panic!("Features must be sorted by datetime");
            }
        }

        return Self {
            name,
            features,
            frequency,
        };
    }

    fn most_recent(&self) -> &Feature {
        self.features
            .last()
            .expect("FeatureColumn must have at least one feature")
    }
}

#[derive(Debug, Error)]
enum TimeSeriesBufferError {
    #[error("FeatureColumn {0} has a different frequency to the buffer")]
    MismatchedFrequency(String),

    #[error("Buffer is not initialised")]
    BufferNotInitialised,

    #[error("Column {0} is not in the buffer")]
    ColumnNotInBuffer(String),

    #[error("Index {0} is out of bounds")]
    IndexOutOfBounds(usize),
}

type Result<T> = std::result::Result<T, TimeSeriesBufferError>;

impl TimeSeriesBuffer {
    /// TimeSeriesBuffer is a fixed size buffer that stores time series data.
    /// It has an implicit time axis, where the most recent data is at the end
    /// of the buffer, and the least recent data is at the start. The buffer
    /// is updated with new data, and old data is dropped when the buffer is
    /// full.

    fn new(column_names: Vec<String>, buffer_size: usize, frequency: u32) -> Self {
        return Self {
            data: Array2::from_elem((buffer_size, column_names.len()), f32::NAN),
            feature_names: column_names,
            most_recent_feature: None,
            least_recent_feature: None,
            oldest_feature: None,
            initialised: false,
            buffer_size,
            frequency,
            boundary: 0,
        };
    }

    fn datetime_to_index(&self, datetime: DateTime<Utc>) -> Result<usize> {
        // Should return buffer_size -1 for the most recent feature
        let time_diff = datetime
            - self
                .oldest_feature
                .ok_or(TimeSeriesBufferError::BufferNotInitialised)?;
        let seconds = time_diff.num_seconds();
        return Ok((seconds / self.frequency as i64) as usize);
    }

    fn index_to_datetime(&self, index: usize) -> Result<DateTime<Utc>> {
        let seconds = index as i64 * self.frequency as i64;
        return Ok(self
            .oldest_feature
            .ok_or(TimeSeriesBufferError::BufferNotInitialised)?
            + chrono::Duration::seconds(seconds));
    }

    fn clear_space(&mut self, n: usize) -> Result<()> {
        if n > self.buffer_size {
            return Err(TimeSeriesBufferError::IndexOutOfBounds(n));
        }

        if n == self.buffer_size || n == 0 {
            self.data.fill(f32::NAN);
            return Ok(());
        }

        let i = self.boundary;
        let j = (self.boundary + n) % self.buffer_size;
        self.boundary = j;

        if i < j {
            self.data.slice_mut(s![i..j, ..]).fill(f32::NAN);
            return Ok(());
        }

        self.data.slice_mut(s![i.., ..]).fill(f32::NAN);
        self.data.slice_mut(s![..j, ..]).fill(f32::NAN);
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

        // Find out if we need to clear space in the buffer

        let most_recent_new_data = col.most_recent().datetime;
        let most_recent_buffer_data = self
            .most_recent_feature
            .ok_or(TimeSeriesBufferError::BufferNotInitialised)?;

        let mut offset = (self.datetime_to_index(most_recent_new_data)? as isize)
            - ((self.buffer_size - 1) as isize);

        if offset > 0 {
            self.clear_space(offset as usize)?;
            offset = 0;
            self.most_recent_feature = Some(most_recent_new_data);
        }

        // Write new data to buffer at the correct index

        todo!();
    }

    // fn update(&mut self, new_data: Vec<FeatureColumn>) {
    //     // Update buffer with new data

    //     let (oldest_new_data, newest_new_data) = new_data
    //         .iter()
    //         .map(|col| col.most_recent().datetime)
    //         .minmax()
    //         .into_option()
    //         .expect("No data in new_data");
    // }

    /// Get the most recent n features for each column in the buffer.
    fn tail(&self, n: usize) -> (Array2<f32>, DateTime<Utc>) {
        // Get data from self.end_index - n to self.end_index

        let returned_datetime = self.most_recent_feature.expect("Buffer not initialised");

        if n > self.buffer_size {
            panic!("n must be less than or equal to buffer_size");
        }

        let i = (self.end_index - n) % self.buffer_size;
        let j = self.end_index % self.buffer_size;

        if i < j {
            return (self.data.slice(s![i..j, ..]).to_owned(), returned_datetime);
        }

        // Data wraps around
        let mut data = Array2::uninit((n, self.data.shape()[1]));
        self.data
            .slice(s![i.., ..])
            .assign_to(&mut data.slice_mut(s![0.., ..]));
        self.data
            .slice(s![..j, ..])
            .assign_to(&mut data.slice_mut(s![n - (j - i).., ..]));

        // All elements have now been initialised
        return (unsafe { data.assume_init() }, returned_datetime);
    }
}
