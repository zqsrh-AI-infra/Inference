use thiserror::Error;
use axum::{
    response::{IntoResponse, Response},
    http::StatusCode,
    Json,
};
use serde_json::json;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Model is loading, please wait: {0}")]
    ModelLoading(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Capability not supported: {0}")]
    CapabilityNotSupported(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Inference error: {0}")]
    InferenceExecutionError(String),
}

impl IntoResponse for InferenceError {
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            InferenceError::ModelNotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            InferenceError::ModelNotLoaded(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            InferenceError::ModelLoading(_) => (StatusCode::ACCEPTED, self.to_string()),
            InferenceError::BackendError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            InferenceError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            InferenceError::CapabilityNotSupported(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            InferenceError::SerializationError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            InferenceError::IoError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            InferenceError::TokenizationError(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            InferenceError::InferenceExecutionError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = Json(json!({
            "error": error_message,
            "code": status.as_u16(),
        }));

        (status, body).into_response()
    }
}