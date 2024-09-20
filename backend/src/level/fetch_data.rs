use super::{
    errors::GetReadingsError,
    models::{ColSpec, Parameter},
};
use futures::{stream, StreamExt, TryStreamExt};
use polars::prelude::*;
use reqwest::{header, Client, Url};

pub async fn get_station_readings(
    http_client: &Client,
    col_spec: ColSpec,
    last_n: usize,
) -> Result<DataFrame, GetReadingsError> {
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
        .header(header::ACCEPT, "text/csv")
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

pub async fn get_many_readings(
    http_client: &Client,
    col_specs: &[ColSpec],
    last_n: usize,
    max_concurrent_requests: usize,
) -> Result<DataFrame, GetReadingsError> {
    let task_results: Vec<Result<_, GetReadingsError>> = stream::iter(col_specs.iter().cloned())
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
        .collect::<Result<Vec<DataFrame>, GetReadingsError>>()?
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
