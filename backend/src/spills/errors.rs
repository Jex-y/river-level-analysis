use axum::response::IntoResponse;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SpillServiceError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] mongodb::error::Error),

    #[error("Deserialisation error: {0}")]
    DeserialisationError(#[from] mongodb::bson::de::Error),
}

impl IntoResponse for SpillServiceError {
    fn into_response(self) -> axum::http::Response<axum::body::Body> {
        axum::http::Response::builder()
            .status(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
            .body(axum::body::Body::from(format!(
                "Internal server error: {}",
                self
            )))
            .unwrap()
    }
}
