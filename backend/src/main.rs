use axum::Router;

mod level_forecast;
use level_forecast::{create_forecast_routes, ColSpec, ForecastConfig, Parameter};
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;

#[derive(serde::Deserialize)]
struct ForecastConfigFileInputColumn {
    station_id: String,
    parameter: String,
    label: String,
}

#[derive(serde::Deserialize)]
struct ForecastConfigFile {
    prediction_length: usize,
    prev_timesteps: usize,
    input_columns: Vec<ForecastConfigFileInputColumn>,
    thresholds: Vec<f32>,
}

async fn index() -> &'static str {
    "Hello, World!"
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Read forecast config from file and deserialize it
    let config: ForecastConfigFile = serde_json::from_str(
        &std::fs::read_to_string("./model/inference_config.json")
            .expect("Inference config file not found."),
    )?;

    let app = Router::new()
        .route("/", axum::routing::get(index))
        .nest(
            "/level-forecast",
            create_forecast_routes(ForecastConfig {
                model_inference_threads: None,
                model_onnx_path: "./model/model.onnx".to_string(),
                max_concurrent_requests: None,
                required_timesteps: config.prev_timesteps,
                thresholds: config.thresholds,
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
            })?,
        )
        .layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    tracing::info!("Listening on {}", listener.local_addr()?);

    axum::serve(listener, app).await?;

    Ok(())
}
