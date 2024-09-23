LevelServiceConfig {
    model_inference_threads: None,
    model_onnx_path: "./model/model.onnx".to_string(),
    max_concurrent_requests: None,
    required_timesteps: config.prev_timesteps,
    thresholds: config.thresholds,
    target_station_id: "0240120".to_string(),
    cache_ttl: None,
    target_parameter: Parameter::Level,
    model_input_columns: config
        .input_columns
        .iter()
        .map(|input_column| ColSpec {
            station_id: input_column.station_id.clone(),
            parameter: match input_column.parameter.as_str() {
                "level" => Parameter::Level,
                "rainfall" => Parameter::Rainfall,
                _ => panic!("Unknown parameter"),
            },
        })
        .collect(),
}
