use super::{
    errors::{GetReadingsError, LevelApiError, ModelExecutionError},
    fetch_data::get_many_readings,
    models::{ColSpec, ForecastRecord, Parameter, ServiceState},
};
use axum::{
    extract::State,
    http::{header, HeaderMap},
    response::IntoResponse,
    Json,
};
use chrono::{DateTime, Datelike, Utc};
use ndarray::{s, Array2};
use ort::Session;
use polars::prelude::*;
use reqwest::Client;
use std::sync::Arc;

async fn collect_data(
    http_client: &Client,
    col_specs: &Vec<ColSpec>,
    required_timesteps: usize,
    max_concurrent_requests: Option<usize>,
) -> Result<(Array2<f32>, DateTime<Utc>), GetReadingsError> {
    let readings = get_many_readings(
        http_client,
        col_specs,
        required_timesteps,
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

async fn run_model(
    model: Arc<Session>,
    context_data: Array2<f32>,
    most_recent_context_data: DateTime<Utc>,
    thresholds: Vec<f32>,
) -> Result<Vec<ForecastRecord>, ModelExecutionError> {
    // Model requires day of year and year to be passed as input
    let day_of_year = most_recent_context_data.ordinal0() as i64;
    let year = most_recent_context_data.year() as i64;

    let time_input = ndarray::array![day_of_year, year];

    // Model expects a batch size dimension of 1

    let time_input = time_input.insert_axis(ndarray::Axis(0));
    let context_data = context_data.insert_axis(ndarray::Axis(0));

    let outputs = model.run(ort::inputs![time_input, context_data]?)?;
    // Model has 3 outputs, mean, std, p lower than threshold

    let mean = outputs[0].try_extract_tensor::<f32>()?;
    let std = outputs[1].try_extract_tensor::<f32>()?;
    let p_lower = outputs[2].try_extract_tensor::<f32>()?;

    let num_prediction_timesteps = mean.shape()[1];

    let mean = mean.into_shape(num_prediction_timesteps)?;
    let std = std.into_shape(num_prediction_timesteps)?;
    let p_lower = p_lower.into_shape((num_prediction_timesteps, thresholds.len()))?;

    Ok((0..num_prediction_timesteps)
        .map(|i| {
            ForecastRecord::new(
                most_recent_context_data + chrono::Duration::minutes(((i + 1) * 15) as i64),
                mean[i],
                std[i],
                thresholds
                    .iter()
                    .zip(p_lower.slice(s![i, ..]).iter())
                    .map(|x| x.into())
                    .collect(),
            )
        })
        .collect())
}

pub async fn get_forecast(
    State(state): State<ServiceState>,
) -> Result<impl IntoResponse, LevelApiError> {
    let (data, most_recent) = collect_data(
        &state.http_client,
        &state.config.model_input_columns,
        state.config.required_timesteps,
        state.config.max_concurrent_requests,
    )
    .await?;

    let forecast = run_model(
        state.forecast_model,
        data,
        most_recent,
        state.config.thresholds.clone(),
    )
    .await?;

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CACHE_CONTROL,
        "public, max-age=300".parse().unwrap(),
    );

    Ok((headers, Json(forecast)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_collect_data() {
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

        let (data, _most_recent_data) = collect_data(
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
