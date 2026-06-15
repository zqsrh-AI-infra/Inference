use axum::{extract::State, Json};
use tokio::time::Instant;

use crate::api::handlers::AppState;
use crate::inference::{EmbeddingRequest, EmbeddingResponse, InferenceRequest, InferenceError};

pub async fn embeddings_handler(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, InferenceError> {
    let start = Instant::now();

    let inference_request = InferenceRequest {
        model: request.model.clone(),
        input: serde_json::json!({
            "capability": "embedding",
            "input": request.input,
        }),
        parameters: Default::default(),
    };

    match state.scheduler.schedule_inference(inference_request).await {
        Ok(response) => {
            state.metrics.record_inference(
                &request.model,
                "embedding",
                "success",
                start.elapsed().as_secs_f64(),
            );

            let embedding_response = match response {
                crate::inference::InferenceResponse::Embedding(emb) => emb,
                _ => EmbeddingResponse {
                    object: "list".to_string(),
                    data: vec![],
                    model: request.model.clone(),
                    usage: crate::inference::Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                },
            };

            Ok(Json(embedding_response))
        }
        Err(e) => {
            state.metrics.record_inference(
                &request.model,
                "embedding",
                "error",
                start.elapsed().as_secs_f64(),
            );
            Err(e)
        }
    }
}