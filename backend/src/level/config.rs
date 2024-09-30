use super::models::{ColSpec, Parameter};
use serde::Deserialize;

#[derive(serde::Deserialize, Debug, Clone)]

pub struct InferenceConfig {
    // prediction_length: usize,
    pub prev_timesteps: usize,
    pub input_columns: Vec<ColSpec>,
    pub thresholds: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LevelServiceConfig {
    /// Number of threads to use for model inference. Defaults to the number of
    /// logical CPUs.
    pub model_inference_threads: usize,

    /// Maximum number of concurrent requests to make to the data source.
    /// Defaults to no limit.
    pub max_concurrent_requests: Option<usize>,

    /// Path to the ONNX model file.
    pub model_bucket: String,

    pub model_path: String,

    pub config_path: String,

    /// Target station ID
    pub target_station_id: String,

    /// Target parameter
    pub target_parameter: Parameter,

    /// Cache ttl. Defaults to 300 seconds.
    pub cache_ttl: Option<u64>,
}

impl Default for LevelServiceConfig {
    fn default() -> Self {
        Self {
            model_inference_threads: num_cpus::get(),
            max_concurrent_requests: None,
            model_bucket: "durham-river-level-models".to_string(),
            model_path: "dev/model.onnx".to_string(),
            config_path: "dev/config.json".to_string(),
            target_station_id: "025878".to_string(),
            target_parameter: Parameter::Level,
            cache_ttl: Some(300),
        }
    }
}
