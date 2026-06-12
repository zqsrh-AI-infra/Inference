# Inference Gateway

[English](#english) | [中文](#中文)

---

## English

### Overview

Inference Gateway is a unified model inference service built with Rust, designed to load AI models in various formats and expose a unified HTTP API for orchestration systems like Dify, LangGraph, Coze, and Flowise.

### Features

- **Multi-Backend Support**: ONNX, GGUF (llama.cpp), Candle (reserved)
- **Unified API**: OpenAI-compatible Chat Completions, Embeddings, and Rerank endpoints
- **Dynamic Model Management**: Load/unload models at runtime
- **Backend Plugin Architecture**: Easy to extend with new model backends
- **Prometheus Metrics**: Built-in observability for inference requests
- **Async/Await**: Full async implementation with Tokio
- **Multiple Capabilities**: Chat, Embedding, Rerank, Classification, Detection, OCR, ASR, TTS

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (Axum)                        │
│  /v1/chat/completions  /v1/embeddings  /v1/rerank  /v1/models │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Scheduler                                 │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   ModelManager                               │
│         ┌──────────┬──────────┬──────────┐                  │
│         │ OnnxBack │ GgufBack │ Candle.. │                  │
└─────────────────────────────────────────────────────────────┘
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/metrics` | Prometheus metrics |
| GET | `/api/v1/models` | List all loaded models |
| GET | `/api/v1/models/{id}` | Get model info |
| POST | `/api/v1/models/load` | Load a model |
| POST | `/api/v1/models/unload` | Unload a model |
| POST | `/api/v1/inference` | Generic inference |
| POST | `/api/v1/chat/completions` | Chat completion (OpenAI compatible) |
| POST | `/api/v1/embeddings` | Get embeddings |
| POST | `/api/v1/rerank` | Rerank documents |

### Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 8

models:
  - id: "qwen-chat"
    name: "Qwen Chat"
    backend: onnx
    path: "/models/qwen.onnx"
    capabilities:
      - chat
      - embedding
```

### Quick Start

```bash
# Build
cargo build --release

# Run
cargo run

# Or with custom config
INFERENCE_CONFIG=/path/to/config.yaml cargo run
```

### Supported Models

| Model Format | Backend | Status |
|-------------|---------|--------|
| ONNX | onnxruntime-rs | ✅ Implemented |
| GGUF | llama.cpp-rs | ✅ Implemented |
| Safetensors | Candle | 🔜 Reserved |
| TorchScript | tch-rs | 🔜 Reserved |
| TensorRT | TensorRT Backend | 🔜 Reserved |

### Capabilities

```rust
pub enum Capability {
    Chat,           // Conversational AI
    Embedding,      // Text vectorization
    Rerank,         // Document ranking
    Classification, // Text classification
    Detection,      // Object detection
    OCR,            // Optical character recognition
    ASR,            // Automatic speech recognition
    TTS,            // Text-to-speech
}
```

---

## 中文

### 概述

Inference Gateway 是一个基于 Rust 构建的统一模型推理服务，旨在加载不同格式的 AI 模型，并对外暴露统一的 HTTP API，供 Dify、LangGraph、Coze、Flowise 等编排系统调用。

上层无需关心模型格式和推理实现，仅通过统一接口完成模型调用。

### 功能特性

- **多后端支持**: ONNX, GGUF (llama.cpp), Candle (预留)
- **统一 API**: 兼容 OpenAI 的 Chat Completions、Embeddings 和 Rerank 接口
- **动态模型管理**: 运行时加载/卸载模型
- **后端插件架构**: 易于扩展新的模型后端
- **Prometheus 监控**: 内置推理请求可观测性
- **异步处理**: 基于 Tokio 的全异步实现
- **多能力支持**: Chat, Embedding, Rerank, Classification, Detection, OCR, ASR, TTS

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      API 层 (Axum)                           │
│  /v1/chat/completions  /v1/embeddings  /v1/rerank  /v1/models │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      调度器                                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    模型管理器                                │
│         ┌──────────┬──────────┬──────────┐                  │
│         │ OnnxBack │ GgufBack │ Candle.. │                  │
└─────────────────────────────────────────────────────────────┘
```

### API 端点

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/metrics` | Prometheus 指标 |
| GET | `/api/v1/models` | 列出所有已加载模型 |
| GET | `/api/v1/models/{id}` | 获取模型信息 |
| POST | `/api/v1/models/load` | 加载模型 |
| POST | `/api/v1/models/unload` | 卸载模型 |
| POST | `/api/v1/inference` | 通用推理 |
| POST | `/api/v1/chat/completions` | 聊天补全 (兼容 OpenAI) |
| POST | `/api/v1/embeddings` | 获取向量嵌入 |
| POST | `/api/v1/rerank` | 文档重排序 |

### 配置示例

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 8

models:
  - id: "qwen-chat"
    name: "Qwen Chat"
    backend: onnx
    path: "/models/qwen.onnx"
    capabilities:
      - chat
      - embedding
```

### 快速开始

```bash
# 编译
cargo build --release

# 运行
cargo run

# 或指定配置文件
INFERENCE_CONFIG=/path/to/config.yaml cargo run
```

### 支持的模型

| 模型格式 | 实现库 | 状态 |
|---------|--------|------|
| ONNX | onnxruntime-rs | ✅ 已实现 |
| GGUF | llama.cpp-rs | ✅ 已实现 |
| Safetensors | Candle | 🔜 预留 |
| TorchScript | tch-rs | 🔜 预留 |
| TensorRT | TensorRT Backend | 🔜 预留 |

### 能力类型

```rust
pub enum Capability {
    Chat,           // 对话AI
    Embedding,      // 文本向量化
    Rerank,         // 文档排序
    Classification, // 文本分类
    Detection,      // 目标检测
    OCR,            // 文字识别
    ASR,            // 语音识别
    TTS,            // 语音合成
}
```

### 目录结构

```
src/
├── api/                    # HTTP API 层
│   ├── chat.rs            # 聊天补全 API
│   ├── embeddings.rs      # 向量嵌入 API
│   ├── handlers.rs        # 核心处理器
│   ├── rerank.rs          # 重排序 API
│   └── routes.rs          # 路由配置
├── backends/              # 后端插件架构
│   ├── mod.rs            # ModelBackend trait
│   ├── onnx/             # ONNX 后端
│   ├── gguf/             # GGUF 后端
│   └── candle/           # Candle 后端 (预留)
├── config/                # 配置管理
├── inference/             # 推理类型和错误处理
├── metrics/               # Prometheus 指标
├── models/                # 模型管理器
├── scheduler/             # 调度器
└── main.rs
```

### 技术栈

- **Runtime**: Tokio
- **Web Framework**: Axum
- **Logging**: tracing + tracing-subscriber
- **Metrics**: Prometheus
- **Serialization**: serde + serde_json

---

## License

MIT