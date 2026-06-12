mod api;
mod backends;
mod config;
mod inference;
mod metrics;
mod models;
mod scheduler;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::api::create_router;
use crate::config::AppConfig;
use crate::models::ModelManager;
use crate::scheduler::Scheduler;
use crate::metrics::create_shared_metrics;

#[derive(Parser, Debug)]
#[command(name = "inference-gateway", about = "Rust unified model inference service")]
struct Args {
    #[arg(short, long, help = "Path to configuration file")]
    config: Option<PathBuf>,

    #[arg(short, long, help = "Model ID to load on startup (overrides config default)")]
    model: Option<String>,
}

fn get_config_path(args: &Args) -> Option<PathBuf> {
    args.config.clone()
        .or_else(|| std::env::var("INFERENCE_CONFIG").ok().map(PathBuf::from))
        .or_else(|| {
            let default_path = PathBuf::from("config/config.yaml");
            if default_path.exists() {
                Some(default_path)
            } else {
                None
            }
        })
}

fn get_default_model_id(args: &Args, config: &AppConfig) -> Option<String> {
    args.model.clone()
        .or_else(|| std::env::var("INFERENCE_DEFAULT_MODEL").ok())
        .or(config.default_model.clone())
        .or(config.models.first().map(|m| m.id.clone()))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config_path = get_config_path(&args);
    let config = AppConfig::load_or_default(config_path);

    init_tracing(&config);

    tracing::info!("Starting Inference Gateway");
    tracing::info!("Server binding to {}:{}", config.server.host, config.server.port);

    let model_manager = Arc::new(ModelManager::new());
    let scheduler = Arc::new(Scheduler::new(model_manager.clone()));
    let metrics = create_shared_metrics();

    if let Some(default_model_id) = get_default_model_id(&args, &config) {
        tracing::info!("Loading default model: {}", default_model_id);

        if let Some(model_config) = config.models.iter().find(|m| m.id == default_model_id) {
            match model_manager.load_model(model_config.clone()).await {
                Ok(info) => {
                    tracing::info!("Successfully loaded default model: {} from {:?}", info.id, model_config.path);
                    model_manager.set_default_model(&info.id)?;
                }
                Err(e) => {
                    tracing::error!("Failed to load default model {}: {}", default_model_id, e);
                }
            }
        } else {
            tracing::warn!("Default model '{}' not found in configuration", default_model_id);
        }
    } else if !config.models.is_empty() {
        tracing::info!("No default model specified, {} models available in config", config.models.len());
    }

    let app = create_router(model_manager, scheduler, metrics);

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .expect("Invalid socket address");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Inference Gateway listening on {}", addr);

    let server = axum::serve(listener, app);

    tokio::select! {
        result = server => {
            if let Err(e) = result {
                tracing::error!("Server error: {}", e);
            }
        }
        _ = shutdown_signal() => {
            tracing::info!("Shutdown signal received, exiting immediately");
        }
    }

    std::process::exit(0);
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