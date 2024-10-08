use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GetReadingsError {
    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),

    #[error("Deserialisation error: {0}")]
    Deserialisation(#[from] csv::Error),

    #[error("Failed to join all tasks: {0}")]
    Join(#[from] tokio::task::JoinError),
}

#[derive(Error, Debug)]
pub enum ModelExecutionError {
    #[error("Ort error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Shape error: {0}")]
    Ndarray(#[from] ndarray::ShapeError),
}

#[derive(Error, Debug)]
pub enum LevelApiError {
    #[error("Error fetching data from flooding API: {0}")]
    FetchData(#[from] GetReadingsError),

    #[error("Failed to execute model: {0}")]
    ModelExecution(#[from] ModelExecutionError),
}

impl IntoResponse for LevelApiError {
    fn into_response(self) -> Response {
        let status = match self {
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let body = format!("Internal server error: {}", self);

        (status, body).into_response()
    }
}
