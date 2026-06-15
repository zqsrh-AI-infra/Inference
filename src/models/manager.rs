use crate::backends::{BackendType, ModelBackend, ModelInfo};
use crate::inference::{Capability, InferenceError, InferenceRequest, InferenceResponse};
use crate::config::{LoadPolicy, ModelConfig};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::collections::HashMap;
use tracing::info;

#[derive(Debug, Clone)]
pub struct ModelLoadingState {
    pub model_id: String,
    pub started_at: std::time::Instant,
}

pub struct ModelManager {
    backends: DashMap<String, Arc<dyn ModelBackend>>,
    model_configs: DashMap<String, ModelConfig>,
    loading_models: DashMap<String, ModelLoadingState>,
    default_model: RwLock<Option<String>>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            backends: DashMap::new(),
            model_configs: DashMap::new(),
            loading_models: DashMap::new(),
            default_model: RwLock::new(None),
        }
    }

    pub async fn load_model(&self, config: ModelConfig) -> Result<ModelInfo, InferenceError> {
        let model_id = config.id.clone();

        if self.backends.contains_key(&model_id) {
            return Err(InferenceError::BackendError(format!(
                "Model {} is already loaded",
                model_id
            )));
        }

        if self.loading_models.contains_key(&model_id) {
            return Err(InferenceError::ModelLoading(format!(
                "Model {} is currently being loaded, please wait",
                model_id
            )));
        }

        self.loading_models.insert(model_id.clone(), ModelLoadingState {
            model_id: model_id.clone(),
            started_at: std::time::Instant::now(),
        });

        info!("Loading model {} from {:?}", model_id, config.path);

        let result = async {
            let backend: Arc<dyn ModelBackend> = match config.backend {
                BackendType::Onnx => {
                    Arc::new(crate::backends::onnx::OnnxBackend::new(&config)?)
                }
                BackendType::Gguf => {
                    Arc::new(crate::backends::gguf::GgufBackend::new(&config)?)
                }
                BackendType::Candle => {
                    return Err(InferenceError::BackendError(
                        "Candle backend not yet implemented".to_string(),
                    ));
                }
                BackendType::TensorRT => {
                    return Err(InferenceError::BackendError(
                        "TensorRT backend not yet implemented".to_string(),
                    ));
                }
            };

            let is_lazy = config.load_policy == LoadPolicy::Lazy;

            if !is_lazy {
                backend.warmup().await?;
            } else {
                info!(
                    "Model {} loaded in lazy mode, warmup deferred to first request",
                    model_id
                );
            }

            let model_info = backend.get_model_info();
            self.backends.insert(model_id.clone(), backend);
            self.model_configs.insert(model_id.clone(), config);

            info!("Model {} loaded successfully", model_id);
            Ok(model_info)
        }
        .await;

        self.loading_models.remove(&model_id);

        result
    }

    pub async fn unload_model(&self, model_id: &str) -> Result<(), InferenceError> {
        if let Some(_backend) = self.backends.remove(model_id) {
            self.model_configs.remove(model_id);
            info!("Model {} unloaded successfully", model_id);
            Ok(())
        } else {
            Err(InferenceError::ModelNotFound(format!(
                "Model {} not found",
                model_id
            )))
        }
    }

    pub fn get_model(&self, model_id: &str) -> Option<Arc<dyn ModelBackend>> {
        self.backends.get(model_id).map(|r| r.value().clone())
    }

    pub fn get_model_info(&self, model_id: &str) -> Option<ModelInfo> {
        self.backends
            .get(model_id)
            .map(|r| r.value().get_model_info())
    }

    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.backends
            .iter()
            .map(|r| r.value().get_model_info())
            .collect()
    }

    pub async fn infer(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let model_id = request.model.clone();

        if self.is_model_loading(&model_id) {
            return Err(InferenceError::ModelLoading(format!(
                "Model {} is currently being loaded, please retry in a few moments",
                model_id
            )));
        }

        let backend = self
            .get_model(&model_id)
            .ok_or_else(|| InferenceError::ModelNotFound(model_id.clone()))?;

        backend.infer(request).await
    }

    pub fn set_default_model(&self, model_id: &str) -> Result<(), InferenceError> {
        if !self.backends.contains_key(model_id) {
            return Err(InferenceError::ModelNotFound(format!(
                "Model {} not found",
                model_id
            )));
        }
        *self.default_model.write() = Some(model_id.to_string());
        Ok(())
    }

    pub fn get_default_model(&self) -> Option<String> {
        self.default_model.read().clone()
    }

    pub fn is_model_loaded(&self, model_id: &str) -> bool {
        self.backends.contains_key(model_id)
    }

    pub fn is_model_loading(&self, model_id: &str) -> bool {
        self.loading_models.contains_key(model_id)
    }

    pub fn get_model_loading_time(&self, model_id: &str) -> Option<std::time::Duration> {
        self.loading_models
            .get(model_id)
            .map(|state| state.started_at.elapsed())
    }

    pub fn get_models_by_capability(&self, capability: Capability) -> Vec<ModelInfo> {
        self.backends
            .iter()
            .filter(|r| r.value().capabilities().contains(&capability))
            .map(|r| r.value().get_model_info())
            .collect()
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub info: ModelInfo,
    pub backend_type: BackendType,
    pub capabilities: Vec<Capability>,
    pub metadata: HashMap<String, String>,
}