use super::config::{InferenceConfig, LevelServiceConfig};
use axum::extract::FromRef;
use chrono::{DateTime, Utc};
use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, sync::Arc};
use tokio::sync::Mutex;

#[derive(Debug, Clone, FromRef)]
pub struct ServiceState {
    pub forecast_model: Arc<ort::Session>,
    pub http_client: reqwest::Client,
    pub config: LevelServiceConfig,
    pub inference_config: InferenceConfig,
    pub data: Arc<Mutex<Option<DataFrame>>>,
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

#[derive(Debug, Clone, Deserialize)]
pub struct ColSpec {
    pub station_id: String,
    pub parameter: Parameter,
}

impl ColSpec {
    pub fn to_col_name(&self) -> String {
        format!("{}-{}", self.station_id, self.parameter)
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct StationQuery {
    since: Option<DateTime<Utc>>,

    #[serde(rename = "_limit")]
    limit: Option<usize>,

    #[serde(rename = "_sorted")]
    sort: bool,
}

impl StationQuery {
    pub fn since(since: DateTime<Utc>) -> Self {
        Self {
            since: Some(since),
            limit: None,
            sort: true,
        }
    }

    pub fn last_n(n: usize) -> Self {
        Self {
            since: None,
            limit: Some(n),
            sort: true,
        }
    }
}
