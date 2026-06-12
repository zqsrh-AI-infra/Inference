use inference_gateway::inference::{InferenceRequest, InferenceError};
use inference_gateway::models::ModelManager;
use inference_gateway::scheduler::Scheduler;
use std::sync::Arc;

#[cfg(test)]
mod scheduler_tests {
    use super::*;

    fn create_test_scheduler() -> (Arc<Scheduler>, Arc<ModelManager>) {
        let model_manager = Arc::new(ModelManager::new());
        let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
        (scheduler, model_manager)
    }

    #[tokio::test]
    async fn test_scheduler_creation() {
        let (scheduler, _manager) = create_test_scheduler();
        assert!(Arc::ptr_eq(&scheduler, &scheduler));
    }

    #[tokio::test]
    async fn test_scheduler_schedule_inference_no_model() {
        let (scheduler, _manager) = create_test_scheduler();

        let request = InferenceRequest {
            model: "nonexistent".to_string(),
            input: serde_json::json!({}),
            parameters: Default::default(),
        };

        let result = scheduler.schedule_inference(request).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, InferenceError::ModelNotFound(_)));
    }
}