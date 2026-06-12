use crate::models::ModelManager;
use crate::inference::{InferenceRequest, InferenceResponse, InferenceError};
use std::sync::Arc;
use tracing::info;

pub struct Scheduler {
    model_manager: Arc<ModelManager>,
    routing_strategy: RoutingStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum RoutingStrategy {
    RoundRobin,
    LeastLoaded,
    CapabilityBased,
}

impl Scheduler {
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self {
            model_manager,
            routing_strategy: RoutingStrategy::LeastLoaded,
        }
    }

    pub async fn schedule_inference(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let model_id = self.select_model(&request).await?;
        let model_request = InferenceRequest {
            model: model_id,
            ..request
        };

        self.model_manager.infer(model_request).await
    }

    async fn select_model(&self, request: &InferenceRequest) -> Result<String, InferenceError> {
        if self.model_manager.is_model_loaded(&request.model) {
            return Ok(request.model.clone());
        }

        let default_model = self.model_manager.get_default_model();
        if let Some(model_id) = default_model {
            info!("Using default model: {}", model_id);
            return Ok(model_id);
        }

        Err(InferenceError::ModelNotFound(
            "No suitable model found for request".to_string(),
        ))
    }

    pub fn set_routing_strategy(&mut self, strategy: RoutingStrategy) {
        self.routing_strategy = strategy;
    }
}

pub struct QueueManager {
    max_queue_size: usize,
    current_queue_size: std::sync::atomic::AtomicUsize,
}

impl QueueManager {
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            max_queue_size,
            current_queue_size: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn try_enqueue(&self) -> bool {
        let current = self.current_queue_size.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        current < self.max_queue_size
    }

    pub fn dequeue(&self) {
        self.current_queue_size.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}