use crate::backends::{BackendType, ModelBackend};
use crate::config::ModelConfig;
use crate::inference::{
    Capability, ChatResponse, EmbeddingResponse, InferenceError, InferenceRequest,
    InferenceResponse, RawResponse,
};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

pub struct GgufBackend {
    config: ModelConfig,
    model: Arc<RwLock<Option<()>>>,
}

impl GgufBackend {
    pub fn new(config: &ModelConfig) -> Result<Self, InferenceError> {
        Ok(Self {
            config: config.clone(),
            model: Arc::new(RwLock::new(None)),
        })
    }

    fn ensure_model(&self) -> Result<(), InferenceError> {
        let mut model_guard = self.model.write();
        if model_guard.is_some() {
            return Ok(());
        }

        info!("Creating GGUF model from: {:?}", self.config.path);

        if !self.config.path.exists() {
            return Err(InferenceError::BackendError(format!(
                "Model file not found: {:?}",
                self.config.path
            )));
        }

        *model_guard = Some(());
        Ok(())
    }

    fn infer_chat(&self, _input: serde_json::Value) -> Result<ChatResponse, InferenceError> {
        Ok(ChatResponse {
            id: format!("chat-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: self.config.id.clone(),
            choices: vec![],
            usage: crate::inference::Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }

    fn infer_embedding(&self, _input: serde_json::Value) -> Result<EmbeddingResponse, InferenceError> {
        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data: vec![],
            model: self.config.id.clone(),
            usage: crate::inference::Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }
}

#[async_trait]
impl ModelBackend for GgufBackend {
    async fn infer(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        self.ensure_model()?;
        let capability = self.determine_capability(&request.input)?;

        match capability {
            Capability::Chat => {
                let response = self.infer_chat(request.input)?;
                Ok(InferenceResponse::Chat(response))
            }
            Capability::Embedding => {
                let response = self.infer_embedding(request.input)?;
                Ok(InferenceResponse::Embedding(response))
            }
            _ => Ok(InferenceResponse::Raw(RawResponse {
                output: serde_json::json!({"status": "placeholder"}),
                metadata: HashMap::new(),
            })),
        }
    }

    fn get_model_info(&self) -> crate::backends::ModelInfo {
        crate::backends::ModelInfo {
            id: self.config.id.clone(),
            name: self.config.name.clone(),
            backend: BackendType::Gguf,
            capabilities: self.config.capabilities.clone(),
            metadata: self.config.metadata.clone(),
        }
    }

    fn capabilities(&self) -> Vec<Capability> {
        self.config.capabilities.clone()
    }

    async fn warmup(&self) -> Result<(), InferenceError> {
        info!("Warming up GGUF backend for model: {}", self.config.id);
        self.ensure_model()?;
        Ok(())
    }
}

impl GgufBackend {
    fn determine_capability(
        &self,
        input: &serde_json::Value,
    ) -> Result<Capability, InferenceError> {
        if let Some(cap) = input.get("capability").and_then(|v| v.as_str()) {
            cap.parse::<Capability>()
                .map_err(|_| InferenceError::CapabilityNotSupported(cap.to_string()))
        } else if self.config.capabilities.contains(&Capability::Chat) {
            Ok(Capability::Chat)
        } else if self.config.capabilities.contains(&Capability::Embedding) {
            Ok(Capability::Embedding)
        } else {
            Ok(self.config.capabilities.first().cloned().unwrap_or(Capability::Chat))
        }
    }
}