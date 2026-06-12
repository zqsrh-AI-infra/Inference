use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::ServiceExt;
use inference_gateway::api::create_router;
use inference_gateway::models::ModelManager;
use inference_gateway::scheduler::Scheduler;
use inference_gateway::metrics::create_shared_metrics;
use std::sync::Arc;

#[cfg(test)]
mod health_tests {
    use super::*;

    async fn create_test_app() -> axum::Router {
        let model_manager = Arc::new(ModelManager::new());
        let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
        let metrics = create_shared_metrics();
        create_router(model_manager, scheduler, metrics)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "healthy");
        assert_eq!(json["service"], "inference-gateway");
    }
}

#[cfg(test)]
mod metrics_tests {
    use super::*;

    async fn create_test_app() -> axum::Router {
        let model_manager = Arc::new(ModelManager::new());
        let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
        let metrics = create_shared_metrics();
        create_router(model_manager, scheduler, metrics)
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}

#[cfg(test)]
mod models_api_tests {
    use super::*;

    async fn create_test_app() -> axum::Router {
        let model_manager = Arc::new(ModelManager::new());
        let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
        let metrics = create_shared_metrics();
        create_router(model_manager, scheduler, metrics)
    }

    #[tokio::test]
    async fn test_list_models_empty() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert!(json["data"].is_array());
        assert_eq!(json["data"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_model_not_found() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/models/nonexistent")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_unload_model_not_found() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/models/unload")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"model_id": "nonexistent"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}

#[cfg(test)]
mod inference_api_tests {
    use super::*;

    async fn create_test_app() -> axum::Router {
        let model_manager = Arc::new(ModelManager::new());
        let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
        let metrics = create_shared_metrics();
        create_router(model_manager, scheduler, metrics)
    }

    #[tokio::test]
    async fn test_inference_model_not_found() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/inference")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"model": "nonexistent", "input": {}}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}

#[cfg(test)]
mod chat_api_tests {
    use super::*;

    async fn create_test_app() -> axum::Router {
        let model_manager = Arc::new(ModelManager::new());
        let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
        let metrics = create_shared_metrics();
        create_router(model_manager, scheduler, metrics)
    }

    #[tokio::test]
    async fn test_chat_completions_invalid_request() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/chat/completions")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"invalid": "request"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn test_chat_completions_model_not_found() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/chat/completions")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{
                        "model": "nonexistent",
                        "messages": [{"role": "user", "content": "hi"}]
                    }"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}

#[cfg(test)]
mod embeddings_api_tests {
    use super::*;

    async fn create_test_app() -> axum::Router {
        let model_manager = Arc::new(ModelManager::new());
        let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
        let metrics = create_shared_metrics();
        create_router(model_manager, scheduler, metrics)
    }

    #[tokio::test]
    async fn test_embeddings_model_not_found() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/embeddings")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{
                        "model": "nonexistent",
                        "input": "Hello world"
                    }"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}

#[cfg(test)]
mod rerank_api_tests {
    use super::*;

    async fn create_test_app() -> axum::Router {
        let model_manager = Arc::new(ModelManager::new());
        let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
        let metrics = create_shared_metrics();
        create_router(model_manager, scheduler, metrics)
    }

    #[tokio::test]
    async fn test_rerank_model_not_found() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/rerank")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{
                        "model": "nonexistent",
                        "query": "What is Rust?",
                        "documents": ["Rust is a programming language"]
                    }"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}