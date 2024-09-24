use super::models::{ColSpec, Parameter};

#[derive(serde::Deserialize)]
struct ConfigFileInputColumn {
    station_id: String,
    parameter: String,
    label: String,
}

#[derive(serde::Deserialize)]
pub struct LevelServiceConfigFile {
    prediction_length: usize,
    prev_timesteps: usize,
    input_columns: Vec<ConfigFileInputColumn>,
    thresholds: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LevelServiceConfig {
    /// Number of threads to use for model inference. Defaults to the number of
    /// logical CPUs.
    pub model_inference_threads: Option<usize>,

    /// Maximum number of concurrent requests to make to the data source.
    /// Defaults to no limit.
    pub max_concurrent_requests: Option<usize>,

    /// Path to the ONNX model file.s
    pub model_bucket: String,

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

    /// Cache ttl. Defaults to 300 seconds.
    pub cache_ttl: Option<u64>,
}

impl TryFrom<LevelServiceConfigFile> for LevelServiceConfig {
    type Error = anyhow::Error;

    fn try_from(config: LevelServiceConfigFile) -> Result<Self, Self::Error> {
        let mut model_input_columns = Vec::new();
        for col in config.input_columns {
            model_input_columns.push(ColSpec {
                station_id: col.station_id,
                parameter: col.parameter.parse()?,
            });
        }

        Ok(Self {
            model_inference_threads: None,
            max_concurrent_requests: None,
            model_onnx_path: "model.onnx".to_string(),
            required_timesteps: config.prev_timesteps,
            model_input_columns,
            thresholds: config.thresholds,
            target_station_id: "station_id".to_string(),
            target_parameter: Parameter::Level,
            cache_ttl: None,
        })
    }
}
