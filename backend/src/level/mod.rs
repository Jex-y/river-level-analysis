mod config;
mod errors;
mod fetch_data;
mod forecast;
mod models;
mod observed;

use axum::{extract::State, http::header, middleware, response::Response, routing::get, Router};
pub use config::LevelServiceConfig;
use models::ServiceState;
pub use models::{ColSpec, Parameter};

pub fn create_level_routes(config: LevelServiceConfig) -> anyhow::Result<axum::Router<()>> {
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

    let state = ServiceState::try_from(config)?;

    Ok(Router::new()
        .route("/forecast", get(forecast::get_forecast))
        .route("/observed", get(observed::get_observed))
        .layer(middleware::map_response_with_state(
            state.clone(),
            add_cache_control,
        ))
        .with_state(state))
}
