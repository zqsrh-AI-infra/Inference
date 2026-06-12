use crate::backends::BackendType;
use crate::inference::Capability;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub default_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "default_workers")]
    pub workers: usize,
    #[serde(default)]
    pub tls: Option<TlsConfig>,
}

fn default_workers() -> usize {
    num_cpus::get()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default)]
    pub format: LogFormat,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: LogFormat::default(),
        }
    }
}

fn default_log_level() -> String {
    "info".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Json,
    #[default]
    Pretty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    pub backend: BackendType,
    pub path: PathBuf,
    #[serde(default)]
    pub capabilities: Vec<Capability>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    #[serde(default)]
    pub device: DeviceConfig,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub max_sequence_length: Option<usize>,
    #[serde(default = "default_inference_timeout")]
    pub inference_timeout_secs: u64,
}

fn default_inference_timeout() -> u64 {
    300
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceConfig {
    #[serde(default = "default_device")]
    pub device_type: DeviceType,
    #[serde(default)]
    pub device_ids: Vec<usize>,
}

fn default_device() -> DeviceType {
    DeviceType::Auto
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
    TensorRT,
}

impl AppConfig {
    pub fn load(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn load_or_default(path: Option<PathBuf>) -> Self {
        if let Some(config_path) = path {
            if config_path.exists() {
                Self::load(&config_path).unwrap_or_else(|e| {
                    tracing::warn!("Failed to load config from {:?}: {}", config_path, e);
                    Self::default()
                })
            } else {
                Self::default()
            }
        } else {
            Self::default()
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                workers: default_workers(),
                tls: None,
            },
            models: Vec::new(),
            logging: LoggingConfig {
                level: default_log_level(),
                format: LogFormat::Pretty,
            },
            default_model: None,
        }
    }
}