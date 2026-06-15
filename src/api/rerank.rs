use axum::{extract::State, Json};
use tokio::time::Instant;

use crate::api::handlers::AppState;
use crate::inference::{RerankRequest, RerankResponse, InferenceRequest, InferenceError};

pub async fn rerank_handler(
    State(state): State<AppState>,
    Json(request): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, InferenceError> {
    let start = Instant::now();

    let inference_request = InferenceRequest {
        model: request.model.clone(),
        input: serde_json::json!({
            "capability": "rerank",
            "query": request.query,
            "documents": request.documents,
            "top_n": request.top_n,
        }),
        parameters: Default::default(),
    };

    match state.scheduler.schedule_inference(inference_request).await {
        Ok(response) => {
            state.metrics.record_inference(
                &request.model,
                "rerank",
                "success",
                start.elapsed().as_secs_f64(),
            );

            let rerank_response = match response {
                crate::inference::InferenceResponse::Rerank(rerank) => rerank,
                _ => RerankResponse {
                    id: format!("rerank-{}", uuid::Uuid::new_v4()),
                    results: vec![],
                    model: request.model.clone(),
                    usage: crate::inference::Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                },
            };

            Ok(Json(rerank_response))
        }
        Err(e) => {
            state.metrics.record_inference(
                &request.model,
                "rerank",
                "error",
                start.elapsed().as_secs_f64(),
            );
            Err(e)
        }
    }
}