pub mod types;
pub mod error;

pub use types::*;
pub use error::InferenceError;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Chat,
    Embedding,
    Rerank,
    Classification,
    Detection,
    OcR,
    Asr,
    Tts,
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Capability::Chat => write!(f, "chat"),
            Capability::Embedding => write!(f, "embedding"),
            Capability::Rerank => write!(f, "rerank"),
            Capability::Classification => write!(f, "classification"),
            Capability::Detection => write!(f, "detection"),
            Capability::OcR => write!(f, "ocr"),
            Capability::Asr => write!(f, "asr"),
            Capability::Tts => write!(f, "tts"),
        }
    }
}

impl std::str::FromStr for Capability {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "chat" => Ok(Capability::Chat),
            "embedding" => Ok(Capability::Embedding),
            "rerank" => Ok(Capability::Rerank),
            "classification" => Ok(Capability::Classification),
            "detection" => Ok(Capability::Detection),
            "ocr" => Ok(Capability::OcR),
            "asr" => Ok(Capability::Asr),
            "tts" => Ok(Capability::Tts),
            _ => Err(format!("Unknown capability: {}", s)),
        }
    }
}