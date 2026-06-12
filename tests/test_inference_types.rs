use inference_gateway::inference::{
    Capability, ChatCompletionRequest, ChatMessage, ChatResponse, ChatChoice,
    EmbeddingRequest, EmbeddingResponse, InputType,
    RerankRequest, RerankResponse,
    InferenceRequest, InferenceResponse, Usage,
};

#[cfg(test)]
mod capability_tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_capability_from_str() {
        assert_eq!(Capability::from_str("chat").unwrap(), Capability::Chat);
        assert_eq!(Capability::from_str("embedding").unwrap(), Capability::Embedding);
        assert_eq!(Capability::from_str("rerank").unwrap(), Capability::Rerank);
        assert_eq!(Capability::from_str("classification").unwrap(), Capability::Classification);
        assert_eq!(Capability::from_str("detection").unwrap(), Capability::Detection);
        assert_eq!(Capability::from_str("ocr").unwrap(), Capability::OcR);
        assert_eq!(Capability::from_str("asr").unwrap(), Capability::Asr);
        assert_eq!(Capability::from_str("tts").unwrap(), Capability::Tts);
    }

    #[test]
    fn test_capability_from_str_case_insensitive() {
        assert_eq!(Capability::from_str("CHAT").unwrap(), Capability::Chat);
        assert_eq!(Capability::from_str("Chat").unwrap(), Capability::Chat);
        assert_eq!(Capability::from_str("EMBEDDING").unwrap(), Capability::Embedding);
    }

    #[test]
    fn test_capability_from_str_invalid() {
        assert!(Capability::from_str("invalid").is_err());
        assert!(Capability::from_str("").is_err());
    }

    #[test]
    fn test_capability_display() {
        assert_eq!(Capability::Chat.to_string(), "chat");
        assert_eq!(Capability::Embedding.to_string(), "embedding");
        assert_eq!(Capability::Rerank.to_string(), "rerank");
        assert_eq!(Capability::OcR.to_string(), "ocr");
    }
}

#[cfg(test)]
mod chat_completion_tests {
    use super::*;

    #[test]
    fn test_chat_message_serialization() {
        let message = ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_chat_message_deserialization() {
        let json = r#"{"role": "assistant", "content": "Hi there"}"#;
        let message: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.role, "assistant");
        assert_eq!(message.content, "Hi there");
    }

    #[test]
    fn test_chat_completion_request_defaults() {
        let json = r#"{"model": "gpt-4", "messages": []}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, "gpt-4");
        assert!(request.messages.is_empty());
        assert!(!request.stream);
        assert_eq!(request.temperature, 1.0);
        assert!(request.max_tokens.is_none());
    }

    #[test]
    fn test_chat_response_serialization() {
        let response = ChatResponse {
            id: "chat-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("chat-123"));
        assert!(json.contains("chat.completion"));
        assert!(json.contains("Hello!"));
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;

    #[test]
    fn test_input_type_string() {
        let input = InputType::String("Hello world".to_string());
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("Hello world"));
    }

    #[test]
    fn test_input_type_array_of_strings() {
        let input = InputType::ArrayOfStrings(vec!["hello".to_string(), "world".to_string()]);
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("world"));
    }

    #[test]
    fn test_embedding_request_deserialization() {
        let json = r#"{"model": "text-embedding-3-small", "input": "Hello world"}"#;
        let request: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, "text-embedding-3-small");
        match request.input {
            InputType::String(s) => assert_eq!(s, "Hello world"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_embedding_request_array_input() {
        let json = r#"{"model": "text-embedding-3-small", "input": ["hello", "world"]}"#;
        let request: EmbeddingRequest = serde_json::from_str(json).unwrap();
        match request.input {
            InputType::ArrayOfStrings(arr) => {
                assert_eq!(arr.len(), 2);
                assert_eq!(arr[0], "hello");
            }
            _ => panic!("Expected ArrayOfStrings variant"),
        }
    }

    #[test]
    fn test_embedding_response_serialization() {
        let response = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![
                inference_gateway::inference::EmbeddingData {
                    object: "embedding".to_string(),
                    embedding: vec![0.1, 0.2, 0.3],
                    index: 0,
                },
            ],
            model: "text-embedding-3-small".to_string(),
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 0,
                total_tokens: 5,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("list"));
        assert!(json.contains("embedding"));
        assert!(json.contains("0.1"));
    }
}

#[cfg(test)]
mod rerank_tests {
    use super::*;

    #[test]
    fn test_rerank_request_deserialization() {
        let json = r#"{
            "model": "bge-reranker",
            "query": "What is Rust?",
            "documents": ["Rust is a programming language", "Rust is a metal"]
        }"#;
        let request: RerankRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, "bge-reranker");
        assert_eq!(request.query, "What is Rust?");
        assert_eq!(request.documents.len(), 2);
        assert_eq!(request.top_n, Some(10));
    }

    #[test]
    fn test_rerank_request_with_top_n() {
        let json = r#"{
            "model": "bge-reranker",
            "query": "What is Rust?",
            "documents": ["Rust is a programming language"],
            "top_n": 5
        }"#;
        let request: RerankRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.top_n, Some(5));
    }

    #[test]
    fn test_rerank_response_serialization() {
        let response = RerankResponse {
            id: "rerank-123".to_string(),
            results: vec![
                inference_gateway::inference::RerankResult {
                    index: 0,
                    relevance_score: 0.95,
                    document: Some("Rust is a programming language".to_string()),
                },
            ],
            model: "bge-reranker".to_string(),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 0,
                total_tokens: 10,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("rerank-123"));
        assert!(json.contains("0.95"));
    }
}

#[cfg(test)]
mod inference_request_tests {
    use super::*;

    #[test]
    fn test_inference_request_deserialization() {
        let json = r#"{
            "model": "qwen-chat",
            "input": {"messages": [{"role": "user", "content": "hi"}]}
        }"#;
        let request: InferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, "qwen-chat");
        assert!(request.input.get("messages").is_some());
        assert!(request.parameters.is_empty());
    }

    #[test]
    fn test_inference_request_with_parameters() {
        let json = r#"{
            "model": "qwen-chat",
            "input": {},
            "parameters": {"temperature": 0.7, "max_tokens": 100}
        }"#;
        let request: InferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.parameters.get("temperature").unwrap().as_f64().unwrap(), 0.7);
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;

    #[test]
    fn test_usage_serialization() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };

        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("100"));
        assert!(json.contains("50"));
        assert!(json.contains("150"));
    }

    #[test]
    fn test_usage_deserialization() {
        let json = r#"{"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }
}