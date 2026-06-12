use prometheus::{
    HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry,
};
use std::sync::Arc;

pub struct Metrics {
    registry: Registry,
    inference_requests: IntCounterVec,
    inference_duration: HistogramVec,
    models_loaded: IntGaugeVec,
    models_inference_total: IntCounterVec,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let inference_requests = IntCounterVec::new(
            Opts::new("inference_requests_total", "Total number of inference requests"),
            &["model", "capability", "status"],
        )
        .expect("Failed to create inference_requests counter");

        let inference_duration = HistogramVec::new(
            HistogramOpts::new("inference_duration_seconds", "Inference duration in seconds")
                .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
            &["model", "capability"],
        )
        .expect("Failed to create inference_duration histogram");

        let models_loaded = IntGaugeVec::new(
            Opts::new("models_loaded", "Number of models currently loaded"),
            &["backend"],
        )
        .expect("Failed to create models_loaded gauge");

        let models_inference_total = IntCounterVec::new(
            Opts::new("models_inference_total", "Total inferences per model"),
            &["model_id", "backend"],
        )
        .expect("Failed to create models_inference_total counter");

        registry
            .register(Box::new(inference_requests.clone()))
            .expect("Failed to register inference_requests");
        registry
            .register(Box::new(inference_duration.clone()))
            .expect("Failed to register inference_duration");
        registry
            .register(Box::new(models_loaded.clone()))
            .expect("Failed to register models_loaded");
        registry
            .register(Box::new(models_inference_total.clone()))
            .expect("Failed to register models_inference_total");

        Self {
            registry,
            inference_requests,
            inference_duration,
            models_loaded,
            models_inference_total,
        }
    }

    pub fn record_inference(
        &self,
        model: &str,
        capability: &str,
        status: &str,
        duration_secs: f64,
    ) {
        self.inference_requests
            .with_label_values(&[model, capability, status])
            .inc();
        self.inference_duration
            .with_label_values(&[model, capability])
            .observe(duration_secs);
    }

    pub fn set_models_loaded(&self, backend: &str, count: i64) {
        self.models_loaded
            .with_label_values(&[backend])
            .set(count);
    }

    pub fn increment_model_inference(&self, model_id: &str, backend: &str) {
        self.models_inference_total
            .with_label_values(&[model_id, backend])
            .inc();
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Metrics {
    fn clone(&self) -> Self {
        Self::new()
    }
}

pub type SharedMetrics = Arc<Metrics>;

pub fn create_shared_metrics() -> SharedMetrics {
    Arc::new(Metrics::new())
}