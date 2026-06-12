use inference_gateway::backends::{BackendType, ModelBackend, ModelInfo};
use inference_gateway::inference::{Capability, InferenceError, InferenceRequest, InferenceResponse};
use std::sync::Arc;

pub struct TestBackend {
    model_info: ModelInfo,
    should_error: bool,
}

impl TestBackend {
    pub fn new(id: &str, capabilities: Vec<Capability>) -> Self {
        Self {
            model_info: ModelInfo {
                id: id.to_string(),
                name: id.to_string(),
                backend: BackendType::Onnx,
                capabilities,
                metadata: Default::default(),
            },
            should_error: false,
        }
    }

    pub fn with_error(mut self) -> Self {
        self.should_error = true;
        self
    }
}

#[async_trait::async_trait]
impl ModelBackend for TestBackend {
    async fn infer(
        &self,
        _request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        if self.should_error {
            Err(InferenceError::ModelNotFound("test".to_string()))
        } else {
            Ok(InferenceResponse::Raw(inference_gateway::inference::RawResponse {
                output: serde_json::json!({"test": "output"}),
                metadata: Default::default(),
            }))
        }
    }

    fn get_model_info(&self) -> ModelInfo {
        self.model_info.clone()
    }

    fn capabilities(&self) -> Vec<Capability> {
        self.model_info.capabilities.clone()
    }

    async fn warmup(&self) -> Result<(), InferenceError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_backend_infer_success() {
        let backend = TestBackend::new("test-model", vec![Capability::Chat]);

        let request = InferenceRequest {
            model: "test-model".to_string(),
            input: serde_json::json!({"messages": [{"role": "user", "content": "hello"}]}),
            parameters: Default::default(),
        };

        let result = backend.infer(request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_model_backend_infer_with_error() {
        let backend = TestBackend::new("test-model", vec![Capability::Chat])
            .with_error();

        let request = InferenceRequest {
            model: "test-model".to_string(),
            input: serde_json::json!({}),
            parameters: Default::default(),
        };

        let result = backend.infer(request).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_model_backend_get_info() {
        let backend = TestBackend::new("test-model", vec![Capability::Chat, Capability::Embedding]);

        let info = backend.get_model_info();
        assert_eq!(info.id, "test-model");
        assert_eq!(info.backend, BackendType::Onnx);
        assert!(info.capabilities.contains(&Capability::Chat));
        assert!(info.capabilities.contains(&Capability::Embedding));
    }

    #[test]
    fn test_model_backend_capabilities() {
        let capabilities = vec![Capability::Rerank, Capability::Classification];
        let backend = TestBackend::new("test-model", capabilities.clone());

        assert_eq!(backend.capabilities(), capabilities);
    }

    #[tokio::test]
    async fn test_model_backend_warmup() {
        let backend = TestBackend::new("test-model", vec![Capability::Chat]);
        let result = backend.warmup().await;
        assert!(result.is_ok());
    }
}