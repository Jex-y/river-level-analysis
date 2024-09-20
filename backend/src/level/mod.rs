mod config;
mod errors;
mod fetch_data;
mod forecast;
mod models;
mod observed;

use axum::{routing::get, Router};
pub use config::LevelServiceConfig;
use models::ServiceState;
pub use models::{ColSpec, Parameter};

pub fn create_level_routes(config: LevelServiceConfig) -> anyhow::Result<axum::Router<()>> {
    Ok(Router::new()
        .route("/forecast", get(forecast::get_forecast))
        .route("/observed", get(observed::get_observed))
        .with_state(ServiceState::try_from(config)?))
}
