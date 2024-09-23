use axum::{http::HeaderValue, Router};

mod level;
use http::Method;
use level::{create_level_routes, LevelServiceConfigFile};
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

#[derive(serde::Deserialize)]
struct ConfigFile {
    #[serde(flatten)]
    level_service: LevelServiceConfigFile,

    /// Host to listen on.
    host: &'static str,

    /// Port to listen on.
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let config: ConfigFile = config::Config::builder()
        .add_source(config::File::with_name("./config/default.json"))
        .add_source(config::Environment::with_prefix("APP"))
        .build()
        .expect("Failed to build config")
        .try_deserialize()
        .expect("Failed to deserialize config");

    let app = Router::new()
        .nest(
            "/api/level",
            create_level_routes(config.level_service.try_into()?)?,
        )
        .layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()))
        .layer(
            CorsLayer::new()
                .allow_origin("http://localhost:3000".parse::<HeaderValue>().unwrap())
                .allow_methods([Method::GET]),
        );

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", config.host, config.port)).await?;

    tracing::info!("Listening on {}", listener.local_addr()?);

    axum::serve(listener, app).await?;

    Ok(())
}
