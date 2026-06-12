use inference_gateway::config::{
    AppConfig, ServerConfig, ModelConfig, LogFormat,
};
use inference_gateway::backends::BackendType;
use inference_gateway::inference::Capability;
use std::path::PathBuf;

#[cfg(test)]
mod app_config_tests {
    use super::*;

    #[test]
    fn test_app_config_default() {
        let config = AppConfig::default();

        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert!(config.models.is_empty());
        assert_eq!(config.logging.level, "info");
    }

    #[test]
    fn test_app_config_serialization() {
        let config = AppConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 3000,
                workers: 4,
                tls: None,
            },
            models: vec![],
            logging: inference_gateway::config::LoggingConfig {
                level: "debug".to_string(),
                format: LogFormat::Json,
            },
            default_model: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("127.0.0.1"));
        assert!(json.contains("3000"));
        assert!(json.contains("debug"));
        assert!(json.contains("json"));
    }

    #[test]
    fn test_app_config_deserialization() {
        let json = r#"{
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 4
            },
            "models": [],
            "logging": {
                "level": "info",
                "format": "pretty"
            }
        }"#;

        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.workers, 4);
        assert_eq!(config.logging.level, "info");
    }
}

#[cfg(test)]
mod server_config_tests {
    use super::*;

    #[test]
    fn test_server_config_default_workers() {
        let config = ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 8080,
            workers: num_cpus::get(),
            tls: None,
        };

        assert!(config.workers > 0);
    }

    #[test]
    fn test_server_config_with_tls() {
        let config = ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 8443,
            workers: 4,
            tls: Some(inference_gateway::config::TlsConfig {
                cert_path: PathBuf::from("/certs/cert.pem"),
                key_path: PathBuf::from("/certs/key.pem"),
            }),
        };

        assert!(config.tls.is_some());
        let tls = config.tls.unwrap();
        assert_eq!(tls.cert_path, PathBuf::from("/certs/cert.pem"));
    }
}

#[cfg(test)]
mod model_config_tests {
    use super::*;

    fn create_test_model_config() -> ModelConfig {
        ModelConfig {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            backend: BackendType::Onnx,
            path: PathBuf::from("/models/test.onnx"),
            capabilities: vec![Capability::Chat, Capability::Embedding],
            metadata: Default::default(),
            device: inference_gateway::config::DeviceConfig {
                device_type: inference_gateway::config::DeviceType::Auto,
                device_ids: vec![],
            },
            batch_size: Some(32),
            max_sequence_length: Some(2048),
            inference_timeout_secs: 300,
        }
    }

    #[test]
    fn test_model_config_serialization() {
        let config = create_test_model_config();
        let json = serde_json::to_string(&config).unwrap();

        assert!(json.contains("test-model"));
        assert!(json.contains("onnx"));
        assert!(json.contains("chat"));
        assert!(json.contains("embedding"));
    }

    #[test]
    fn test_model_config_deserialization() {
        let json = r#"{
            "id": "qwen-chat",
            "name": "Qwen Chat",
            "backend": "Onnx",
            "path": "/models/qwen.onnx",
            "capabilities": ["chat", "embedding"],
            "device": {"device_type": "auto", "device_ids": [0, 1]}
        }"#;

        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.id, "qwen-chat");
        assert_eq!(config.backend, BackendType::Onnx);
        assert!(config.capabilities.contains(&Capability::Chat));
        assert!(config.capabilities.contains(&Capability::Embedding));
    }

    #[test]
    fn test_model_config_batch_size() {
        let mut config = create_test_model_config();
        config.batch_size = Some(64);
        assert_eq!(config.batch_size, Some(64));
    }

    #[test]
    fn test_model_config_max_sequence_length() {
        let mut config = create_test_model_config();
        config.max_sequence_length = Some(4096);
        assert_eq!(config.max_sequence_length, Some(4096));
    }
}

#[cfg(test)]
mod log_format_tests {
    use super::*;

    #[test]
    fn test_log_format_json() {
        let format = LogFormat::Json;
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"json\"");
    }

    #[test]
    fn test_log_format_pretty() {
        let format = LogFormat::Pretty;
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"pretty\"");
    }

    #[test]
    fn test_log_format_deserialization() {
        let json = r#""json""#;
        let format: LogFormat = serde_json::from_str(json).unwrap();
        assert_eq!(format, LogFormat::Json);

        let json = r#""pretty""#;
        let format: LogFormat = serde_json::from_str(json).unwrap();
        assert_eq!(format, LogFormat::Pretty);
    }
}

#[cfg(test)]
mod logging_config_tests {
    use inference_gateway::config::LoggingConfig;

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, "info");
    }
}