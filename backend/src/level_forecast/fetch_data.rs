use chrono::{DateTime, Utc};
use futures::{stream, StreamExt, TryStreamExt};
use ndarray::Array2;
use polars::prelude::*;
use reqwest::{header::ACCEPT, Client, Url};
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub enum Parameter {
    Level,
    Rainfall,
}

impl std::fmt::Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Parameter::Level => write!(f, "level"),
            Parameter::Rainfall => write!(f, "rainfall"),
        }
    }
}

#[derive(Debug, Error)]
pub enum FetchDataError {
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Deserialisation error: {0}")]
    DeserialisationError(#[from] polars::error::PolarsError),

    #[error("Failed to join all tasks: {0}")]
    JoinError(#[from] tokio::task::JoinError),
}

#[derive(Debug, Clone)]
pub struct ColSpec {
    pub station_id: String,
    pub parameter: Parameter,
}

async fn get_station_readings(
    http_client: &Client,
    col_spec: ColSpec,
    last_n: u32,
) -> Result<DataFrame, FetchDataError> {
    let units = match col_spec.parameter {
        Parameter::Level => "level-stage-i-15_min-m",
        Parameter::Rainfall => "rainfall-tipping_bucket_raingauge-t-15_min-mm",
    };

    const BASE_URL: &str = "https://environment.data.gov.uk/flood-monitoring/";

    let url = BASE_URL
        .parse::<Url>()
        .unwrap()
        .join(&format!(
            "id/measures/{station_notation}-{units}/readings.csv",
            station_notation = col_spec.station_id,
            units = units,
        ))
        .unwrap();

    let response = http_client
        .get(url)
        .header(ACCEPT, "text/csv")
        .query(&[("_sorted", "true"), ("_limit", last_n.to_string().as_ref())])
        .send()
        .await?
        .error_for_status()?;

    let response_bytes = response.bytes().await?;

    Ok(CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(
            ["dateTime", "value"].iter().map(|&s| s.into()).collect(),
        ))
        .into_reader_with_file_handle(std::io::Cursor::new(response_bytes))
        .finish()?
        .lazy()
        .select([
            col("dateTime")
                .str()
                .to_datetime(
                    Some(TimeUnit::Milliseconds),
                    Some("UTC".into()),
                    StrptimeOptions::default(),
                    lit("raise"),
                )
                .alias("datetime"),
            col("value").cast(DataType::Float32),
        ])
        .sort(["datetime"], Default::default())
        .collect()?
        .upsample::<[String; 0]>([], "datetime", Duration::parse("15m"))?)
}

async fn get_readings(
    http_client: &Client,
    col_specs: &[ColSpec],
    last_n: u32,
    max_concurrent_requests: usize,
) -> Result<DataFrame, FetchDataError> {
    let task_results: Vec<Result<_, FetchDataError>> = stream::iter(col_specs.iter().cloned())
        .map(|col_spec: ColSpec| {
            let http_client = http_client.clone();
            let col_name = format!("{}_{}", col_spec.station_id, col_spec.parameter);
            let col_spec: ColSpec = col_spec.clone();
            tokio::task::spawn(async move {
                Ok(get_station_readings(&http_client, col_spec, last_n)
                    .await?
                    .rename("value", col_name.into())?
                    .to_owned())
            })
        })
        .buffered(max_concurrent_requests)
        .try_collect()
        .await?;

    Ok(task_results
        .into_iter()
        .collect::<Result<Vec<DataFrame>, FetchDataError>>()?
        .into_iter()
        .map(|df: DataFrame| df.lazy())
        .reduce(|df1: LazyFrame, df2: LazyFrame| {
            df1.join(
                df2,
                [col("datetime")],
                [col("datetime")],
                JoinArgs::new(JoinType::Left),
            )
        })
        .unwrap()
        .collect()?)
}

#[tracing::instrument]
pub async fn fetch_data(
    http_client: &Client,
    col_specs: &Vec<ColSpec>,
    required_timesteps: usize,
    max_concurrent_requests: Option<usize>,
) -> Result<(Array2<f32>, DateTime<Utc>), FetchDataError> {
    let readings = get_readings(
        http_client,
        col_specs,
        required_timesteps as u32,
        max_concurrent_requests.unwrap_or(col_specs.len()),
    )
    .await?
    .fill_null(FillNullStrategy::Backward(None))?
    .tail(Some(required_timesteps));

    let most_recent_data: i64 = readings
        .column("datetime")
        .unwrap()
        .datetime()
        .unwrap()
        .max()
        .unwrap();

    let most_recent_data = DateTime::from_timestamp_millis(most_recent_data).unwrap();

    let data = readings
        .drop("datetime")?
        .to_ndarray::<Float32Type>(IndexOrder::C)?;

    Ok((data, most_recent_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fetch_data() {
        let http_client = Client::new();
        let col_specs = vec![
            ColSpec {
                station_id: "025878".to_string(),
                parameter: Parameter::Rainfall,
            },
            ColSpec {
                station_id: "0240120".to_string(),
                parameter: Parameter::Level,
            },
        ];

        let required_timesteps = 10;

        let max_concurrent_requests = None;

        let (data, _most_recent_data) = fetch_data(
            &http_client,
            &col_specs,
            required_timesteps,
            max_concurrent_requests,
        )
        .await
        .unwrap();

        assert_eq!(data.shape(), &[10, 2]);
    }
}
