use super::models::{ColSpec, Parameter};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct LevelServiceConfig {
    /// Number of threads to use for model inference. Defaults to the number of
    /// logical CPUs.
    pub model_inference_threads: Option<usize>,

    /// Maximum number of concurrent requests to make to the data source.
    /// Defaults to no limit.
    pub max_concurrent_requests: Option<usize>,

    pub model_bucket: String,

    /// Path to the ONNX model file
    pub bucket_model_path: String,

    /// Path to the bucket config file
    pub bucket_config_path: String,

    /// Target station ID
    pub target_station_id: String,

    /// Target parameter
    pub target_parameter: Parameter,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct InferenceConfig {
    // prediction_length: usize,
    pub prev_timesteps: usize,
    pub input_columns: Vec<ColSpec>,
    pub quantiles: Vec<f32>,
    pub thresholds: Vec<f32>,
}
