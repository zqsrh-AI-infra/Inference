use crate::backends::{BackendType, ModelBackend};
use crate::config::ModelConfig;
use crate::inference::{
    Capability, ChatResponse, ChatMessage, ChatChoice, EmbeddingData, EmbeddingResponse,
    InferenceError, InferenceRequest, InferenceResponse,
};
use async_trait::async_trait;
use llama_rs::Engine;
use llama_rs::EngineConfig;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::timeout;
use tracing::{info, warn};

pub struct GgufBackend {
    config: ModelConfig,
    engine: Arc<RwLock<Option<Arc<Engine>>>>,
    semaphore: Arc<Semaphore>,
}

impl GgufBackend {
    pub fn new(config: &ModelConfig) -> Result<Self, InferenceError> {
        let max_concurrent = config
            .max_concurrent_requests
            .unwrap_or(1)
            .max(1);
        Ok(Self {
            config: config.clone(),
            engine: Arc::new(RwLock::new(None)),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
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

    /// Build a chat prompt using the engine's detected chat template.
    ///
    /// The prompt is formatted with chat template markers (e.g. `<|user|>`, `<|assistant|>`)
    /// so that `Engine::generate()` detects them and skips its own `wrap_prompt()` call.
    /// This avoids double-wrapping which corrupts the prompt format.
    fn build_chat_prompt(
        &self,
        engine: &Engine,
        messages: &[ChatMessage],
    ) -> String {
        let template = engine.chat_template();

        let system_msg = messages.iter().find(|m| m.role == "system");
        let non_system: Vec<&ChatMessage> =
            messages.iter().filter(|m| m.role != "system").collect();

        if non_system.is_empty() {
            return String::new();
        }

        let mut prompt = String::new();
        let mut first_user_handled = false;

        for (i, msg) in non_system.iter().enumerate() {
            match msg.role.as_str() {
                "user" => {
                    if !first_user_handled {
                        first_user_handled = true;
                        if let Some(sys) = system_msg {
                            prompt.push_str(
                                &template.format_first_turn(&sys.content, &msg.content),
                            );
                        } else {
                            // wrap_prompt produces markers (e.g. <|user|>...<|assistant|>)
                            // that generate() will pass through unchanged
                            prompt.push_str(&template.wrap_prompt(&msg.content));
                        }
                    } else {
                        prompt.push_str(&template.format_continuation(&msg.content));
                    }
                }
                "assistant" => {
                    // Append assistant response as raw text (no template wrapping)
                    // Check if there's a following user message to determine
                    // whether to add format_continuation after this
                    let has_next_user = non_system
                        .get(i + 1)
                        .map(|m| m.role == "user")
                        .unwrap_or(false);
                    prompt.push_str(&msg.content);
                    if !has_next_user {
                        // If this is the last message and it's an assistant,
                        // still need to trigger generation. Add empty continuation.
                        prompt.push('\n');
                    }
                }
                _ => {} // Ignore unknown roles
            }
        }

        // Ensure prompt ends with assistant marker so the model generates a response
        if !prompt.contains("<|assistant|>")
            && !prompt.contains("<|im_start|>assistant")
            && !prompt.contains("[/INST]")
        {
            // The template already added the assistant prefix via wrap_prompt/format_*
        }

        prompt
    }

    async fn infer_chat(&self, input: serde_json::Value) -> Result<ChatResponse, InferenceError> {
        let engine = self.ensure_engine()?;

        // NOTE: Per-request temperature/top_k/top_p are NOT applied because
        // llama-rs Engine::generate() uses the sampler config set at engine load time.
        // See EngineConfig in ensure_engine() for the effective values.
        let _temperature = input
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;

        let max_tokens = input
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;

        let timeout_secs = input
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.config.inference_timeout_secs);

        let messages: Vec<ChatMessage> = if let Some(msgs) = input
            .get("messages")
            .and_then(|v| serde_json::from_value::<Vec<ChatMessage>>(v.clone()).ok())
        {
            if msgs.is_empty() {
                return Err(InferenceError::InvalidRequest(
                    "messages array is empty, at least one message is required".to_string(),
                ));
            }
            msgs
        } else if let Some(text) = input.as_str() {
            vec![ChatMessage {
                role: "user".to_string(),
                content: text.to_string(),
            }]
        } else if let Some(text) = input.get("input").and_then(|v| v.as_str()) {
            vec![ChatMessage {
                role: "user".to_string(),
                content: text.to_string(),
            }]
        } else {
            return Err(InferenceError::InvalidRequest(
                "No messages or string input found. Provide 'messages' array or a string 'input'.".to_string(),
            ));
        };

        let prompt = self.build_chat_prompt(&engine, &messages);

        info!(
            "Chat template: {:?}, prompt: {}...",
            engine.chat_template(),
            &prompt[..prompt.len().min(100)]
        );

        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| InferenceError::BackendError("Semaphore closed".to_string()))?;

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
                warn!(
                    "Inference timeout after {} seconds. Note: blocking task may still be running.",
                    timeout_secs
                );
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

    async fn infer_embedding(
        &self,
        input: serde_json::Value,
    ) -> Result<EmbeddingResponse, InferenceError> {
        let engine = self.ensure_engine()?;

        let texts: Vec<String> = if let Some(s) = input.get("input").and_then(|v| v.as_str()) {
            vec![s.to_string()]
        } else if let Some(arr) = input.get("input").and_then(|v| v.as_array()) {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            return Err(InferenceError::InvalidRequest(
                "Embedding input must be a string or array of strings".to_string(),
            ));
        };

        if texts.is_empty() {
            return Err(InferenceError::InvalidRequest(
                "Embedding input is empty".to_string(),
            ));
        }

        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| InferenceError::BackendError("Semaphore closed".to_string()))?;

        let engine_clone = engine.clone();
        let texts_clone = texts.clone();

        let embeddings: Vec<Vec<f32>> = tokio::task::spawn_blocking(move || {
            texts_clone
                .iter()
                .map(|text| {
                    engine_clone
                        .embed(text)
                        .map_err(|e| format!("Embedding failed: {:?}", e))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .await
        .map_err(|e| InferenceError::BackendError(format!("Join error: {:?}", e)))?
        .map_err(|e| InferenceError::BackendError(e))?;

        let data: Vec<EmbeddingData> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, embedding)| EmbeddingData {
                object: "embedding".to_string(),
                embedding,
                index: i,
            })
            .collect();

        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data,
            model: self.config.id.clone(),
            usage: crate::inference::Usage {
                prompt_tokens: texts.iter().map(|t| t.len()).sum(),
                completion_tokens: 0,
                total_tokens: texts.iter().map(|t| t.len()).sum(),
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
            Capability::Embedding => {
                let response = self.infer_embedding(request.input).await?;
                Ok(InferenceResponse::Embedding(response))
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
        if let Some(cap_str) = input.get("capability").and_then(|v| v.as_str()) {
            let cap = cap_str
                .parse::<Capability>()
                .map_err(|_| InferenceError::CapabilityNotSupported(cap_str.to_string()))?;
            if !self.config.capabilities.contains(&cap) {
                return Err(InferenceError::CapabilityNotSupported(format!(
                    "Capability '{}' is not supported by model '{}'",
                    cap_str, self.config.id
                )));
            }
            Ok(cap)
        } else if self.config.capabilities.contains(&Capability::Chat) {
            Ok(Capability::Chat)
        } else if self.config.capabilities.contains(&Capability::Embedding) {
            Ok(Capability::Embedding)
        } else {
            Ok(self.config.capabilities.first().cloned().unwrap_or(Capability::Chat))
        }
    }
}