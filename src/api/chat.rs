use axum::{extract::State, Json};
use tokio::time::Instant;

use crate::api::handlers::AppState;
use crate::inference::{
    ChatCompletionRequest, ChatResponse, InferenceRequest, Usage, ChatChoice, ChatMessage,
};
use crate::inference::InferenceError;

pub async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatResponse>, InferenceError> {
    let start = Instant::now();

    let inference_request = InferenceRequest {
        model: request.model.clone(),
        input: serde_json::json!({
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }),
        parameters: Default::default(),
    };

    match state.scheduler.schedule_inference(inference_request).await {
        Ok(response) => {
            state.metrics.record_inference(
                &request.model,
                "chat",
                "success",
                start.elapsed().as_secs_f64(),
            );

            let chat_response = match response {
                crate::inference::InferenceResponse::Chat(chat) => chat,
                _ => ChatResponse {
                    id: format!("chat-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion".to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: request.model.clone(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: "Model response".to_string(),
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                },
            };

            Ok(Json(chat_response))
        }
        Err(e) => {
            state.metrics.record_inference(
                &request.model,
                "chat",
                "error",
                start.elapsed().as_secs_f64(),
            );
            Err(e)
        }
    }
}

pub async fn chat_completions_stream_handler(
    State(state): State<AppState>,
    Json(_request): Json<ChatCompletionRequest>,
) -> Result<impl axum::response::IntoResponse, InferenceError> {
    let start = Instant::now();

    state.metrics.record_inference(
        "unknown",
        "chat_stream",
        "success",
        start.elapsed().as_secs_f64(),
    );

    Err::<Json<ChatResponse>, InferenceError>(InferenceError::BackendError(
        "Streaming not yet implemented".to_string(),
    ))
}