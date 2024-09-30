use super::{
    data_store::{Feature, FeatureColumn},
    errors::GetReadingsError,
    models::{ColSpec, Parameter},
};
use futures::{stream, StreamExt, TryStreamExt};
use reqwest::{header, Client, Url};
use tokio::task::JoinError;

pub async fn get_station_readings(
    http_client: &Client,
    col_spec: ColSpec,
    last_n: usize,
) -> Result<FeatureColumn, GetReadingsError> {
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
        .query(&[("_sorted", "true"), ("_limit", last_n.to_string().as_ref())])
        .send()
        .await?
        .error_for_status()?;

    let response_bytes = response.bytes().await?;
    let mut reader = csv::Reader::from_reader(response_bytes.as_ref());
    let records: Vec<Feature> = reader
        .deserialize::<Feature>()
        .collect::<Result<Vec<Feature>, csv::Error>>()?;

    return Ok(FeatureColumn::new(
        col_spec.station_id.clone(),
        records,
        900,
    ));
}

pub async fn get_many_readings(
    http_client: &Client,
    col_specs: &[ColSpec],
    last_n: usize,
    max_concurrent_requests: usize,
) -> Result<Vec<FeatureColumn>, GetReadingsError> {
    let futures = col_specs.iter().cloned().map(|col_spec: ColSpec| {
        let http_client = http_client.clone();
        let col_spec = col_spec.clone();
        tokio::task::spawn(
            async move { get_station_readings(&http_client, col_spec, last_n).await },
        )
    });

    let results = stream::iter(futures)
        .buffered(max_concurrent_requests)
        .try_collect::<Vec<_>>()
        .await?
        .into_iter()
        .collect::<Result<Vec<FeatureColumn>, GetReadingsError>>()?;

    Ok(results)
}
