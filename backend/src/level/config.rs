use super::models::{ColSpec, Parameter};

#[derive(Debug, Clone)]
pub struct LevelServiceConfig {
    /// Number of threads to use for model inference. Defaults to the number of
    /// logical CPUs.
    pub model_inference_threads: Option<usize>,

    /// Maximum number of concurrent requests to make to the data source.
    /// Defaults to no limit.
    pub max_concurrent_requests: Option<usize>,

    /// Path to the ONNX model file.
    pub model_onnx_path: String,

    /// Number of timesteps required for the model to make a prediction.
    pub required_timesteps: usize,

    /// Columns required by the model for input.
    pub model_input_columns: Vec<ColSpec>,

    /// Thresholds for the model to use.
    pub thresholds: Vec<f32>,

    /// Target station ID
    pub target_station_id: String,

    /// Target parameter
    pub target_parameter: Parameter,
}
