use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;

use crate::models::ModelManager;
use crate::scheduler::Scheduler;
use crate::metrics::SharedMetrics;

use super::handlers::{
    AppState, health_handler, metrics_handler, list_models_handler, get_model_handler,
    load_model_handler, unload_model_handler, inference_handler,
};
use super::chat::chat_completions_handler;
use super::embeddings::embeddings_handler;
use super::rerank::rerank_handler;

pub fn create_router(
    model_manager: Arc<ModelManager>,
    scheduler: Arc<Scheduler>,
    metrics: SharedMetrics,
) -> Router {
    let app_state = AppState {
        model_manager,
        scheduler,
        metrics,
    };

    let api_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(list_models_handler))
        .route("/v1/models/:id", get(get_model_handler))
        .route("/v1/models/load", post(load_model_handler))
        .route("/v1/models/unload", post(unload_model_handler))
        .route("/v1/inference", post(inference_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/embeddings", post(embeddings_handler))
        .route("/v1/rerank", post(rerank_handler))
        .with_state(app_state.clone());

    Router::new()
        .nest("/api", api_routes)
        .with_state(app_state)
}