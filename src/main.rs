mod api;
mod backends;
mod config;
mod inference;
mod metrics;
mod models;
mod scheduler;

use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::api::create_router;
use crate::config::AppConfig;
use crate::models::ModelManager;
use crate::scheduler::Scheduler;
use crate::metrics::create_shared_metrics;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = AppConfig::load_or_default(None);

    init_tracing(&config);

    tracing::info!("Starting Inference Gateway");
    tracing::info!("Server binding to {}:{}", config.server.host, config.server.port);

    let model_manager = Arc::new(ModelManager::new());
    let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
    let metrics = create_shared_metrics();

    let app = create_router(model_manager, scheduler, metrics);

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .expect("Invalid socket address");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Inference Gateway listening on {}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Inference Gateway shutting down");
    Ok(())
}

fn init_tracing(config: &AppConfig) {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&config.logging.level));

    let fmt_layer = tracing_subscriber::fmt::layer();

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();
}

async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received");
}