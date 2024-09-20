use crate::level_forecast::fetch_data::{fetch_data, ColSpec};
use axum::{extract::State, http::StatusCode, routing::get, Json, Router};
use chrono::{DateTime, Datelike, Utc};
use ndarray::{s, Array2};
use ort::Session;
use reqwest::Client;
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ForecastConfig {
    /// Number of threads to use for model inference. Defaults to the number of logical CPUs.
    pub model_inference_threads: Option<usize>,

    /// Maximum number of concurrent requests to make to the data source. Defaults to no limit.
    pub max_concurrent_requests: Option<usize>,

    /// Path to the ONNX model file.
    pub model_onnx_path: String,

    /// Number of timesteps required for the model to make a prediction.
    pub required_timesteps: usize,

    /// Columns required by the model for input.
    pub model_input_columns: Vec<ColSpec>,

    /// Thresholds for the model to use.
    pub thresholds: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ForecastState {
    pub forecast_model: Arc<Session>,
    pub http_client: Client,
    pub config: ForecastConfig,
}

#[derive(Debug, Clone, Serialize)]
struct ThrehsoldProbability {
    value: f32,
    probability_gt: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ForecastRecord {
    timestamp: DateTime<Utc>,
    mean: f32,
    std: f32,
    thresholds: Vec<ThrehsoldProbability>,
}

async fn run_model(
    model: Arc<Session>,
    context_data: Array2<f32>,
    most_recent_context_data: DateTime<Utc>,
    thresholds: Vec<f32>,
) -> Result<Vec<ForecastRecord>, anyhow::Error> {
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
        .map(|i| ForecastRecord {
            timestamp: most_recent_context_data + chrono::Duration::minutes(((i + 1) * 15) as i64),
            mean: mean[i],
            std: std[i],
            thresholds: thresholds
                .iter()
                .zip(p_lower.slice(s![i, ..]).iter())
                .map(|(value, probability_gt)| ThrehsoldProbability {
                    value: *value,
                    probability_gt: *probability_gt,
                })
                .collect(),
        })
        .collect())
}

async fn get_forecast(
    State(state): State<ForecastState>,
) -> Result<Json<Vec<ForecastRecord>>, StatusCode> {
    let (data, most_recent) = fetch_data(
        &state.http_client,
        &state.config.model_input_columns,
        state.config.required_timesteps,
        state.config.max_concurrent_requests,
    )
    .await
    .map_err(|e| {
        tracing::error!("Failed to fetch data: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let forecast = run_model(
        state.forecast_model,
        data,
        most_recent,
        state.config.thresholds.clone(),
    )
    .await
    .map_err(|e| {
        tracing::error!("Failed to run model: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(forecast))
}

pub fn create_forecast_routes(config: ForecastConfig) -> Result<Router<()>, ort::Error> {
    let http_client = reqwest::Client::new();
    let model = ort::Session::builder()?
        .with_intra_threads(config.model_inference_threads.unwrap_or(num_cpus::get()))?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_parallel_execution(true)?
        .commit_from_file(&config.model_onnx_path)?;

    Ok(Router::new()
        .route("/", get(get_forecast))
        .with_state(ForecastState {
            forecast_model: model.into(),
            http_client,
            config,
        }))
}
