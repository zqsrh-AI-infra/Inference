use axum::{
    extract::{Path, State, Json},
    response::IntoResponse,
};
use serde::Deserialize;
use std::sync::Arc;
use tokio::time::Instant;

use crate::models::ModelManager;
use crate::scheduler::Scheduler;
use crate::metrics::SharedMetrics;
use crate::inference::InferenceRequest;
use crate::config::ModelConfig;

use crate::inference::InferenceError;

#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
    pub scheduler: Arc<Scheduler>,
    pub metrics: SharedMetrics,
}

pub async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "inference-gateway",
    }))
}

pub async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let metric_families = state.metrics.registry().gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

pub async fn list_models_handler(State(state): State<AppState>) -> impl IntoResponse {
    let models = state.model_manager.list_models();
    Json(serde_json::json!({
        "object": "list",
        "data": models,
    }))
}

pub async fn get_model_handler(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, InferenceError> {
    match state.model_manager.get_model_info(&model_id) {
        Some(info) => Ok(Json(serde_json::json!({
            "object": "model",
            "data": info,
        }))),
        None => Err(InferenceError::ModelNotFound(model_id)),
    }
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub config: ModelConfig,
}

pub async fn load_model_handler(
    State(state): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> Result<Json<serde_json::Value>, InferenceError> {
    let start = Instant::now();
    match state.model_manager.load_model(request.config).await {
        Ok(info) => {
            state.metrics.record_inference(
                &info.id,
                "management",
                "success",
                start.elapsed().as_secs_f64(),
            );
            Ok(Json(serde_json::json!({
                "status": "loaded",
                "model": info,
            })))
        }
        Err(e) => Err(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct UnloadModelRequest {
    pub model_id: String,
}

pub async fn unload_model_handler(
    State(state): State<AppState>,
    Json(request): Json<UnloadModelRequest>,
) -> Result<Json<serde_json::Value>, InferenceError> {
    match state.model_manager.unload_model(&request.model_id).await {
        Ok(()) => Ok(Json(serde_json::json!({
            "status": "unloaded",
            "model_id": request.model_id,
        }))),
        Err(e) => Err(e),
    }
}

pub async fn inference_handler(
    State(state): State<AppState>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<crate::inference::InferenceResponse>, InferenceError> {
    let start = Instant::now();
    let model_id = request.model.clone();

    match state.scheduler.schedule_inference(request).await {
        Ok(response) => {
            state.metrics.record_inference(
                &model_id,
                "inference",
                "success",
                start.elapsed().as_secs_f64(),
            );
            state.metrics.increment_model_inference(&model_id, "inference");
            Ok(Json(response))
        }
        Err(e) => {
            state.metrics.record_inference(
                &model_id,
                "inference",
                "error",
                start.elapsed().as_secs_f64(),
            );
            Err(e)
        }
    }
}