use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use crate::inference::{Capability, InferenceRequest, InferenceResponse};

pub mod onnx;
pub mod gguf;
pub mod candle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub backend: BackendType,
    pub capabilities: Vec<Capability>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BackendType {
    Onnx,
    Gguf,
    Candle,
    TensorRT,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Onnx => write!(f, "onnx"),
            BackendType::Gguf => write!(f, "gguf"),
            BackendType::Candle => write!(f, "candle"),
            BackendType::TensorRT => write!(f, "tensorrt"),
        }
    }
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    async fn infer(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, crate::inference::InferenceError>;

    fn get_model_info(&self) -> ModelInfo;

    fn capabilities(&self) -> Vec<Capability>;

    async fn warmup(&self) -> Result<(), crate::inference::InferenceError>;
}

pub type BackendResult<T> = Result<T, crate::inference::InferenceError>;