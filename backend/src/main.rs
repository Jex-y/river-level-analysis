use axum::{http::HeaderValue, middleware, response::Response, Router};

mod config;
mod level;
mod spills;

use config::load_config;
use http::{header, Method};
use level::create_level_routes;
use spills::create_spill_routes;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();

    let config = load_config();

    let app = Router::new()
        .nest(
            "/api/level",
            create_level_routes(config.level_service).await?,
        )
        .nest(
            "/api/spills",
            create_spill_routes(config.spill_service).await?,
        )
        .layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()))
        .layer(
            CorsLayer::new()
                .allow_origin("http://localhost:3000".parse::<HeaderValue>().unwrap())
                .allow_methods([Method::GET]),
        )
        .layer(middleware::map_response(
            move |mut res: Response| async move {
                res.headers_mut().insert(
                    header::CACHE_CONTROL,
                    format!("public, max-age={}", config.http.cache_duration_sec)
                        .parse()
                        .unwrap(),
                );
                res
            },
        ));

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", config.http.host, config.http.port)).await?;

    tracing::info!("Listening on {}", listener.local_addr()?);

    axum::serve(listener, app).await?;

    Ok(())
}
