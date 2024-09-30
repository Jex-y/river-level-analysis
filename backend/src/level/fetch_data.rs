use super::{
    errors::GetReadingsError,
    models::{ColSpec, Parameter, StationQuery},
};
use chrono::{DateTime, Duration, Utc};
use futures::{stream, StreamExt, TryStreamExt};
use polars::prelude::*;
use reqwest::{header, Client, Url};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

pub async fn get_station_readings(
    http_client: &Client,
    col_spec: ColSpec,
    query: StationQuery,
) -> Result<DataFrame, GetReadingsError> {
    debug!(
        "Fetching readings for {:?} with query {:?}",
        col_spec, query
    );

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
        .get(url.clone())
        .header(header::ACCEPT, "text/csv")
        .query(&query)
        .send()
        .await?
        .error_for_status()?;

    let response_bytes = response.bytes().await?;

    if response_bytes.is_empty() {
        let datetime_col = Series::new_empty(
            "datetime".into(),
            &DataType::Datetime(TimeUnit::Milliseconds, None),
        );
        let value_col = Series::new_empty("value".into(), &DataType::Float32);

        return Ok(DataFrame::new(vec![datetime_col, value_col])?);
    }

    // TODO: Improve CSV parsing performance
    // This seems to block the thread!
    // Could also keep the dataframes around and only add new data when needed

    let result = CsvReadOptions::default()
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
        .upsample::<[String; 0]>([], "datetime", polars::time::Duration::parse("15m"))?;

    debug!("Fetched {} readings for {:?}", result.height(), col_spec);

    Ok(result)
}

fn merge_dataframes(current_data: Option<DataFrame>, new_data: Vec<DataFrame>) -> DataFrame {
    current_data
        .map(|df| vec![df])
        .unwrap_or_default()
        .into_iter()
        .chain(new_data.into_iter())
        .into_iter()
        .map(|df| df.lazy())
        .reduce(|acc, df| {
            acc.join(
                df,
                [col("datetime")],
                [col("datetime")],
                JoinArgs::new(JoinType::Full).with_coalesce(JoinCoalesce::CoalesceColumns),
            )
        })
        .map(|df: LazyFrame| {
            // After joining, updated columns with have a suffic of _right
            // Merge these columns into the original columns
            let mut df = df;
            let schema = df.collect_schema().expect("Failed to collect schema");

            let right_columns = schema
                .iter()
                .filter(|field| field.0.ends_with("_right"))
                .map(|field| field.0.clone())
                .collect::<Vec<_>>();

            for right in right_columns.iter() {
                let left = right.replace("_right", "");

                df = df.with_column(col(left.clone()).fill_null(col(right.clone())))
            }

            df.drop(right_columns)
        })
        .map(|df| df.collect().expect("Failed to merged dataframes"))
        .expect("No dataframes to merge")
}

fn get_column_most_recent(df: &DataFrame, col_name: &str) -> Result<DateTime<Utc>, anyhow::Error> {
    df.select(["datetime", col_name])?
        .drop_nulls::<String>(None)?
        .column("datetime")?
        .datetime()?
        .max()
        .and_then(DateTime::from_timestamp_millis)
        .ok_or_else(|| anyhow::anyhow!("Failed to get most recent datetime"))
}

async fn collect_data_from_scratch(
    http_client: &reqwest::Client,
    col_specs: &[ColSpec],
    required_timesteps: usize,
    max_concurrent_requests: Option<usize>,
) -> Result<DataFrame, GetReadingsError> {
    let new_data = fetch_col_queries_concurrent(
        http_client,
        &col_specs
            .iter()
            .map(|col_spec| (col_spec.clone(), StationQuery::last_n(required_timesteps)))
            .collect::<Vec<_>>(),
        max_concurrent_requests,
    )
    .await?;

    Ok(merge_dataframes(None, new_data).tail(Some(required_timesteps)))
}

async fn fetch_col_queries_concurrent(
    http_client: &reqwest::Client,
    queries: &[(ColSpec, StationQuery)],
    max_concurrent_requests: Option<usize>,
) -> Result<Vec<DataFrame>, GetReadingsError> {
    let futures = queries
        .iter()
        .cloned()
        .map(|(col_spec, query): (ColSpec, StationQuery)| {
            let http_client = http_client.clone();
            let col_spec = col_spec.clone();
            let col_name = col_spec.to_col_name();
            let query = query.clone();

            tokio::task::spawn(async move {
                Ok::<_, GetReadingsError>(
                    get_station_readings(&http_client, col_spec, query)
                        .await?
                        .rename("value", col_name.into())?
                        .to_owned(),
                )
            })
        });

    let results = stream::iter(futures)
        .buffered(max_concurrent_requests.unwrap_or(queries.len()))
        .try_collect::<Vec<_>>()
        .await?
        .into_iter()
        .collect::<Result<Vec<DataFrame>, GetReadingsError>>()?;

    Ok(results)
}

async fn update_dataframe(
    http_client: &reqwest::Client,
    col_specs: &[ColSpec],
    current_data: &DataFrame,
    required_timesteps: usize,
    max_concurrent_requests: Option<usize>,
) -> Result<DataFrame, GetReadingsError> {
    let now = Utc::now();

    let col_queries: Vec<(ColSpec, StationQuery)> = col_specs
        .iter()
        .filter_map(|col_spec| {
            let column_most_recent =
                get_column_most_recent(current_data, &col_spec.to_col_name()).ok()?;

            if now - column_most_recent < Duration::minutes(15) {
                debug!("Column {} is up to date", col_spec.to_col_name());

                return None;
            }

            debug!(
                "Column {} is out of date, fetching new data",
                col_spec.to_col_name()
            );

            Some((col_spec.clone(), StationQuery::since(column_most_recent)))
        })
        .collect();

    let new_data = fetch_col_queries_concurrent(http_client, &col_queries, max_concurrent_requests);

    Ok(
        merge_dataframes(Some(current_data.clone()), new_data.await?)
            .tail(Some(required_timesteps)),
    )
}

pub async fn get_latest_data(
    http_client: &reqwest::Client,
    col_specs: &[ColSpec],
    current_data: Arc<Mutex<Option<DataFrame>>>,
    required_timesteps: usize,
    max_concurrent_requests: Option<usize>,
) -> Result<DataFrame, GetReadingsError> {
    debug!("Waiting for lock on current data");

    let mut current_data = current_data.lock().await;

    let updated_df = match current_data.take() {
        Some(df) => {
            debug!("Updating existing dataframe");

            update_dataframe(
                http_client,
                col_specs,
                &df,
                required_timesteps,
                max_concurrent_requests,
            )
            .await?
        }
        None => {
            debug!("No existing dataframe, fetching from scratch");

            collect_data_from_scratch(
                http_client,
                col_specs,
                required_timesteps,
                max_concurrent_requests,
            )
            .await?
        }
    };

    *current_data = Some(updated_df.clone());

    Ok(updated_df)
}
