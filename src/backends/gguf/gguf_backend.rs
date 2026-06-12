use crate::backends::{BackendType, ModelBackend};
use crate::config::ModelConfig;
use crate::inference::{
    Capability, ChatResponse, ChatMessage, ChatChoice, InferenceError, InferenceRequest,
    InferenceResponse,
};
use async_trait::async_trait;
use llama_rs::Engine;
use llama_rs::EngineConfig;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use tracing::info;

pub struct GgufBackend {
    config: ModelConfig,
    engine: Arc<RwLock<Option<Arc<Engine>>>>,
}

impl GgufBackend {
    pub fn new(config: &ModelConfig) -> Result<Self, InferenceError> {
        Ok(Self {
            config: config.clone(),
            engine: Arc::new(RwLock::new(None)),
        })
    }

    fn ensure_engine(&self) -> Result<Arc<Engine>, InferenceError> {
        let mut engine_guard = self.engine.write();
        if let Some(ref engine) = *engine_guard {
            return Ok(engine.clone());
        }

        info!("Loading GGUF model from: {:?}", self.config.path);

        if !self.config.path.exists() {
            return Err(InferenceError::BackendError(format!(
                "Model file not found: {:?}",
                self.config.path
            )));
        }

        let model_path = self.config.path.to_string_lossy().to_string();

        let engine_config = EngineConfig {
            model_path,
            tokenizer_path: None,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
            max_tokens: 128,
            seed: None,
            use_gpu: self.config.use_gpu,
            max_context_len: Some(1024),
            kv_cache_type: Default::default(),
        };

        let engine = Engine::load(engine_config).map_err(|e| {
            InferenceError::BackendError(format!("Failed to load GGUF model: {:?}", e))
        })?;

        let engine = Arc::new(engine);
        *engine_guard = Some(engine.clone());
        Ok(engine)
    }

    fn build_prompt_from_messages(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str(&format!("System: {}\n", msg.content));
                }
                "user" => {
                    prompt.push_str(&format!("User: {}\n", msg.content));
                }
                "assistant" => {
                    prompt.push_str(&format!("Assistant: {}\n", msg.content));
                }
                _ => {
                    prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
                }
            }
        }

        prompt.push_str("Assistant: ");
        prompt
    }

    async fn infer_chat(&self, input: serde_json::Value) -> Result<ChatResponse, InferenceError> {
        let engine = self.ensure_engine()?;

        let _temperature = input
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;

        let max_tokens = input
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;

        let timeout_secs = input
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.config.inference_timeout_secs);

        let messages: Vec<ChatMessage> = input
            .get("messages")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let prompt = self.build_prompt_from_messages(&messages);

        info!("Running inference with prompt: {}...", &prompt[..prompt.len().min(100)]);

        let engine_clone = engine.clone();
        let prompt_clone = prompt.clone();

        let spawn_result = timeout(
            Duration::from_secs(timeout_secs),
            tokio::task::spawn_blocking(move || {
                engine_clone.generate(&prompt_clone, max_tokens)
            })
        )
        .await;

        let generated: String = match spawn_result {
            Ok(Ok(Ok(s))) => s,
            Ok(Ok(Err(e))) => {
                return Err(InferenceError::BackendError(format!(
                    "Inference failed: {:?}", e
                )));
            }
            Ok(Err(e)) => {
                return Err(InferenceError::BackendError(format!(
                    "Join error: {:?}", e
                )));
            }
            Err(_) => {
                return Err(InferenceError::InferenceExecutionError(format!(
                    "Inference timeout after {} seconds", timeout_secs
                )));
            }
        };

        info!("Generated response: {}...", &generated[..generated.len().min(100)]);

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(ChatResponse {
            id: format!("chat-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created,
            model: self.config.id.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: generated,
                },
                finish_reason: Some("stop".to_string()),
            }],
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
        let capability = self.determine_capability(&request.input)?;

        match capability {
            Capability::Chat => {
                let response = self.infer_chat(request.input).await?;
                Ok(InferenceResponse::Chat(response))
            }
            _ => Err(InferenceError::CapabilityNotSupported(
                capability.to_string(),
            )),
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
        self.ensure_engine()?;
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