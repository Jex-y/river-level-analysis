mod config;
mod data_store;
mod errors;
mod fetch_data;
mod forecast;
mod models;
mod observed;

use axum::{extract::State, http::header, middleware, response::Response, routing::get, Router};
pub use config::LevelServiceConfig;
use forecast::load_model;
pub use models::ColSpec;
use models::ServiceState;

pub async fn create_level_routes(config: LevelServiceConfig) -> anyhow::Result<axum::Router<()>> {
    async fn add_cache_control<B>(
        State(config): State<LevelServiceConfig>,
        mut response: Response<B>,
    ) -> Response<B> {
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            format!("public, max-age={}", config.cache_ttl.unwrap_or(300))
                .parse()
                .unwrap(),
        );
        response
    }

    let (forecast_model, model_config) = load_model(&config).await?;

    let state = ServiceState {
        forecast_model: forecast_model.into(),
        http_client: reqwest::Client::new(),
        config: config.clone(),
        model_config,
    };

    Ok(Router::new()
        .route("/forecast", get(forecast::get_forecast))
        .route("/observed", get(observed::get_observed))
        .layer(middleware::map_response_with_state(
            state.clone(),
            add_cache_control,
        ))
        .with_state(state))
}
