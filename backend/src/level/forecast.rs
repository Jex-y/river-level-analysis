use super::{
    config::{InferenceConfig, LevelServiceConfig},
    errors::{GetReadingsError, LevelApiError, ModelExecutionError},
    fetch_data::get_many_readings,
    models::{ColSpec, ForecastRecord, ServiceState},
};
use crate::level::data_store::FeatureColumn;
use axum::{extract::State, response::IntoResponse, Json};
use chrono::{DateTime, Datelike, Utc};
use google_cloud_storage::{
    client::{Client as GcsClient, ClientConfig},
    http::objects::{download::Range, get::GetObjectRequest},
};
use ndarray::{s, Array2};
use ort::Session;
use reqwest::Client;
use std::sync::Arc;
use tracing::debug;

async fn download_bytes_from_bucket(
    client: &GcsClient,
    bucket: &str,
    object: &str,
) -> Result<Vec<u8>, anyhow::Error> {
    let download_config = GetObjectRequest {
        bucket: bucket.to_string(),
        object: object.to_string(),
        ..Default::default()
    };
    let download_range = Range::default();
    let bytes = client
        .download_object(&download_config, &download_range)
        .await?;
    Ok(bytes)
}

pub async fn load_model(
    config: &LevelServiceConfig,
) -> Result<(Session, InferenceConfig), anyhow::Error> {
    let client_config = ClientConfig::default().with_auth().await?;
    let client = GcsClient::new(client_config);

    debug!(
        "Downloading model and config from bucket {}",
        config.model_bucket
    );

    let (model_bytes, config_bytes) = tokio::try_join!(
        download_bytes_from_bucket(&client, &config.model_bucket, &config.bucket_model_path),
        download_bytes_from_bucket(&client, &config.model_bucket, &config.bucket_config_path)
    )?;

    let inference_config: InferenceConfig = serde_json::from_slice(&config_bytes)?;
    debug!("Initialising onnx runtime session");

    let model = ort::Session::builder()?
        .with_intra_threads(config.model_inference_threads)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .commit_from_memory(&model_bytes)?;
    Ok((model, inference_config))
}

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
    .await?;

    Ok(FeatureColumn::to_array(&readings, required_timesteps))
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
        &state.model_config.input_columns,
        state.model_config.prev_timesteps,
        state.config.max_concurrent_requests,
    )
    .await?;

    let forecast = run_model(
        state.forecast_model,
        data,
        most_recent,
        state.model_config.thresholds.clone(),
    )
    .await?;

    Ok(Json(forecast))
}
