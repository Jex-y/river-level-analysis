use super::config::{InferenceConfig, LevelServiceConfig};
use axum::extract::FromRef;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{fmt::Display, str::FromStr, sync::Arc};

#[derive(Debug, Clone, FromRef)]
pub struct ServiceState {
    pub forecast_model: Arc<ort::Session>,
    pub model_config: InferenceConfig,
    pub http_client: reqwest::Client,
    pub config: LevelServiceConfig,
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

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
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

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ColSpec {
    pub station_id: String,
    pub parameter: Parameter,
}
