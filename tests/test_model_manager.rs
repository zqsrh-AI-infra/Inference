use inference_gateway::backends::{BackendType, ModelBackend, ModelInfo};
use inference_gateway::config::{DeviceConfig, DeviceType, ModelConfig};
use inference_gateway::inference::{Capability, InferenceError, InferenceRequest, InferenceResponse, ChatResponse, Usage};
use inference_gateway::models::ModelManager;
use std::path::PathBuf;
use std::sync::Arc;

pub struct MockBackend {
    model_info: ModelInfo,
}

impl MockBackend {
    pub fn new(id: &str, capabilities: Vec<Capability>) -> Self {
        Self {
            model_info: ModelInfo {
                id: id.to_string(),
                name: id.to_string(),
                backend: BackendType::Onnx,
                capabilities,
                metadata: Default::default(),
            },
        }
    }
}

#[async_trait::async_trait]
impl ModelBackend for MockBackend {
    async fn infer(
        &self,
        _request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        Ok(InferenceResponse::Chat(ChatResponse {
            id: "test-chat".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: self.model_info.id.clone(),
            choices: vec![],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        }))
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

fn create_test_model_config(id: &str) -> ModelConfig {
    ModelConfig {
        id: id.to_string(),
        name: id.to_string(),
        backend: BackendType::Onnx,
        path: PathBuf::from(format!("/tmp/test-model-{}.onnx", id)),
        capabilities: vec![Capability::Chat, Capability::Embedding],
        metadata: Default::default(),
        device: DeviceConfig {
            device_type: DeviceType::Auto,
            device_ids: vec![],
        },
        batch_size: None,
        max_sequence_length: None,
        inference_timeout_secs: 300,
        use_gpu: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_manager_list_empty() {
        let manager = ModelManager::new();
        let models = manager.list_models();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_model_manager_default_model_none() {
        let manager = ModelManager::new();
        let default = manager.get_default_model();
        assert!(default.is_none());
    }

    #[tokio::test]
    async fn test_model_manager_is_model_loaded_false() {
        let manager = ModelManager::new();
        assert!(!manager.is_model_loaded("nonexistent"));
    }

    #[tokio::test]
    async fn test_model_manager_get_model_info_none() {
        let manager = ModelManager::new();
        let info = manager.get_model_info("nonexistent");
        assert!(info.is_none());
    }

    #[tokio::test]
    async fn test_model_manager_get_model_none() {
        let manager = ModelManager::new();
        let model = manager.get_model("nonexistent");
        assert!(model.is_none());
    }

    #[tokio::test]
    async fn test_model_manager_unload_nonexistent() {
        let manager = ModelManager::new();
        let result = manager.unload_model("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_model_manager_set_default_model_not_loaded() {
        let manager = ModelManager::new();
        let result = manager.set_default_model("nonexistent");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_model_manager_get_models_by_capability() {
        let manager = ModelManager::new();
        let models = manager.get_models_by_capability(Capability::Chat);
        assert!(models.is_empty());
    }

    #[test]
    fn test_model_config_default_device() {
        let config = create_test_model_config("test");
        assert_eq!(config.device.device_type, DeviceType::Auto);
    }

    #[test]
    fn test_model_config_capabilities() {
        let config = create_test_model_config("test");
        assert!(config.capabilities.contains(&Capability::Chat));
        assert!(config.capabilities.contains(&Capability::Embedding));
    }

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::Onnx.to_string(), "onnx");
        assert_eq!(BackendType::Gguf.to_string(), "gguf");
        assert_eq!(BackendType::Candle.to_string(), "candle");
        assert_eq!(BackendType::TensorRT.to_string(), "tensorrt");
    }
}