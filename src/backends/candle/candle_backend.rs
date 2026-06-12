use crate::backends::{BackendType, ModelBackend};
use crate::config::ModelConfig;
use crate::inference::{Capability, InferenceError, InferenceRequest, InferenceResponse};
use async_trait::async_trait;

pub struct CandleBackend {
    config: ModelConfig,
}

impl CandleBackend {
    pub fn new(config: &ModelConfig) -> Result<Self, InferenceError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

#[async_trait]
impl ModelBackend for CandleBackend {
    async fn infer(
        &self,
        _request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        Err(InferenceError::BackendError(
            "Candle backend not yet implemented".to_string(),
        ))
    }

    fn get_model_info(&self) -> crate::backends::ModelInfo {
        crate::backends::ModelInfo {
            id: self.config.id.clone(),
            name: self.config.name.clone(),
            backend: BackendType::Candle,
            capabilities: self.config.capabilities.clone(),
            metadata: self.config.metadata.clone(),
        }
    }

    fn capabilities(&self) -> Vec<Capability> {
        self.config.capabilities.clone()
    }

    async fn warmup(&self) -> Result<(), InferenceError> {
        Ok(())
    }
}