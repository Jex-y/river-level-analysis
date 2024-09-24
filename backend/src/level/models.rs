use super::config::LevelServiceConfig;
use axum::extract::FromRef;
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::{fmt::Display, str::FromStr, sync::Arc};

#[derive(Debug, Clone, FromRef)]
pub struct ServiceState {
    pub forecast_model: Arc<ort::Session>,
    pub http_client: reqwest::Client,
    pub config: LevelServiceConfig,
}

impl TryFrom<LevelServiceConfig> for ServiceState {
    type Error = anyhow::Error;

    fn try_from(config: LevelServiceConfig) -> anyhow::Result<Self> {
        let http_client = reqwest::Client::new();

        let cloud_storage_client = cloud_storage::Client::new();
        // let model_bucket = cloud_storage_client.bucket().read(&config.model_bucket)?;
        // model_bucket.

        let model = ort::Session::builder()?
            .with_intra_threads(config.model_inference_threads.unwrap_or(num_cpus::get()))?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .commit_from_file(&config.model_onnx_path)?;

        Ok(Self {
            forecast_model: Arc::new(model),
            http_client,
            config,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ThrehsoldProbability {
    value: f32,
    probability_gt: f32,
}

impl From<(&f32, &f32)> for ThrehsoldProbability {
    fn from((value, probability_gt): (&f32, &f32)) -> Self {
        Self {
            value: *value,
            probability_gt: *probability_gt,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ForecastRecord {
    timestamp: DateTime<Utc>,
    mean: f32,
    std: f32,
    thresholds: Vec<ThrehsoldProbability>,
}

impl ForecastRecord {
    pub fn new(
        timestamp: DateTime<Utc>,
        mean: f32,
        std: f32,
        thresholds: Vec<ThrehsoldProbability>,
    ) -> Self {
        Self {
            timestamp,
            mean,
            std,
            thresholds,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ObservationRecord {
    timestamp: DateTime<Utc>,
    value: f32,
}

impl ObservationRecord {
    pub fn new(timestamp: DateTime<Utc>, value: f32) -> Self {
        Self { timestamp, value }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Parameter {
    Level,
    Rainfall,
}

impl Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Parameter::Level => write!(f, "level"),
            Parameter::Rainfall => write!(f, "rainfall"),
        }
    }
}

impl FromStr for Parameter {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "level" => Ok(Self::Level),
            "rainfall" => Ok(Self::Rainfall),
            _ => Err(anyhow::anyhow!("Invalid parameter: {}", s)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColSpec {
    pub station_id: String,
    pub parameter: Parameter,
}
