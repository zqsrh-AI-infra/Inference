# Chat/通用推理超时问题 — 根因分析与修复方案

## 问题现象

```bash
curl -X POST http://localhost:8080/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-it",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'

# 返回
{
  "code": 500,
  "error": "Inference error: Inference timeout after 60 seconds"
}
```

日志显示 prompt 仅为 `Assistant: `（修复前），修复后 prompt 为 `User: Hello, who are you?\nAssistant: `，但推理依然超时。

---

## 根因分析

### 核心问题：Prompt 双层模板包装 (Double Template Wrapping)

**问题链路追踪：**

1. **用户请求** → `chat_completions_handler` 构造 `InferenceRequest { input: {"messages": [...]} }`
2. **GGUF 后端** → `infer_chat()` 调用 `build_prompt_from_messages()` 生成 prompt
3. **build_prompt_from_messages()** (旧代码) 输出：
   ```
   User: Hello, who are you?
   Assistant:
   ```
4. **Engine::generate(prompt, max_tokens)** 被调用

#### 关键发现：`llama-rs Engine::generate()` 的内部行为

```rust
// llama-rs 0.16.1 src/engine.rs:1025-1031
pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, EngineError> {
    let mut ctx = self.create_inference_context();
    let mut sampler = Sampler::new(self.sampler_config.clone(), self.config.vocab_size);

    // ⚠️ 关键：始终对 prompt 调用 chat_template.wrap_prompt()
    let formatted = self.chat_template.wrap_prompt(prompt);
    let mut tokens = self.tokenizer.encode(&formatted, self.add_bos)?;
    // ...
}
```

**`wrap_prompt()` 的逻辑**（`ChatTemplate::UserAssistant` 即 Gemma 模板）：

```rust
ChatTemplate::UserAssistant => {
    format!("<|user|>\n{}<|assistant|>\n", prompt)
}
```

**但它有一个跳过条件**：如果 prompt 已含 `<|user|>`、`<|im_start|>` 或 `[INST]`，则直接返回原文。

#### 双层包装的产生

| 步骤 | 内容 | 说明 |
|------|------|------|
| ① 我们构造 | `User: Hello, who are you?\nAssistant: ` | 不含任何模板标记 |
| ② `wrap_prompt()` 包装 | `<\|user\|>\nUser: Hello, who are you?\nAssistant: \n<\|assistant\|>\n` | 被误判为"原始文本"再次包装 |

**结果**：模型收到的是嵌套垃圾 prompt：

```
<|user|>
User: Hello, who are you?      ← "User:" 是人类可读文本，不是模板标记
Assistant:                      ← "Assistant:" 同上
<|assistant|>
```

Gemma 的 tokenizer 训练时从未见过这种将 `User: xxx Assistant:` 嵌入 `<|user|>...</|assistant|>` 的格式，导致：

- 模型无法识别对话起止点
- 无法命中 stop token (`<|end|>` 或 `<|user|>`)
- 生成长篇乱码直到 `max_tokens` 耗尽（默认 128）或超时

**为什么修复后（转为 user message）仍然超时？**

修复后 prompt 变为 `User: Hello, who are you?\nAssistant: `，本质没有变化——仍然是手写格式，`wrap_prompt()` 仍然会检测不到模板标记而二次包装。

---

### 次要问题：Per-request 采样参数无效

```rust
// 旧代码
let _temperature = input.get("temperature")...  // 读取了但从未使用
```

`Engine::generate()` 内部从 `self.sampler_config` 创建 Sampler，该配置在 `Engine::load()` 时固化，**不支持 per-request 覆盖**。

```rust
// EngineConfig 在 ensure_engine() 中硬编码：
temperature: 0.7,
top_k: 40,
top_p: 0.9,
repeat_penalty: 1.1,
max_tokens: 128,  // 修复后默认 64
max_context_len: Some(1024),
```

这意味着无论请求传什么 `temperature`/`top_p`，实际使用的一律是 load 时的值。这是 llama-rs 0.16.1 的 API 限制。

---

## 修复方案

### 修复 1：消除 Prompt 双层包装（核心）

**策略**：不再手写 prompt 格式，直接使用 `Engine` 检测到的 `ChatTemplate` 来格式化消息。

**`build_chat_prompt()` 新逻辑**：

1. 读取 `engine.chat_template()` 获取引擎自动检测的模板
2. 对于第一条 user 消息：
   - 有 system prompt → 调用 `template.format_first_turn(system, user)`
   - 无 system prompt → 调用 `template.wrap_prompt(user)`
3. 对于后续 assistant 消息 → 直接拼接原文
4. 对于后续 user 消息 → 调用 `template.format_continuation(user)`
5. 生成的 prompt 包含 `<|user|>` 等模板标记，`generate()` 的 `wrap_prompt()` 检测到后直接透传

**效果对比**：

| 场景 | 旧 prompt（传给 generate） | `wrap_prompt` 二次包装结果 |
|------|------------------------|-------------------------|
| `"Hello"` | `User: Hello\nAssistant: ` | `<\|user\|>\nUser: Hello\n...<\|assistant\|>\n` ❌ |
| `"Hello"` (修复后) | `<\|user\|>\nHello<\|assistant\|>\n` | 检测到 `<\|user\|>` → 透传 ✓ |

**Gemma 模型实际收到的 prompt**（修复后）：

```
<|user|>
Hello, who are you?
<|assistant|>
```

这正是 Gemma instruct 微调时使用的格式，模型能正确理解对话边界并正常停止。

### 修复 2：增加模板日志

```rust
info!(
    "Chat template: {:?}, prompt: {}...",
    engine.chat_template(),
    &prompt[..prompt.len().min(100)]
);
```

方便运维确认实际使用的模板。

### 修复 3：采样参数无效说明

在代码中显式注释说明 per-request `temperature`/`top_k`/`top_p` 在 llama-rs 0.16.1 下不生效，有效值在 `ensure_engine()` 的 `EngineConfig` 中。

---

## 修改文件

| 文件 | 修改 |
|------|------|
| `src/backends/gguf/gguf_backend.rs` | 删除 `build_prompt_from_messages()`，新增 `build_chat_prompt()` 使用 `engine.chat_template()` |

---

## 验证方法

1. **模型启动后**，查看日志中的 `Chat template` 输出，确认检测到的模板
2. **发送单条 chat 请求**，观察日志中 prompt 格式是否包含 `<|user|>...<|assistant|>` 标记
3. **确认无超时**：Gemma 4B on CPU 首次推理（含 `ensure_engine`）应在 ~30-120 秒内返回（视硬件而定）
4. **质量验证**：模型回复应语义连贯，非乱码

### 测试 curl

```bash
curl -X POST http://localhost:8080/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-it",
    "messages": [{"role": "user", "content": "What is 1+1?"}],
    "max_tokens": 32
  }'
```

---

## 未解决的限制

| 问题 | 状态 | 说明 |
|------|------|------|
| Per-request 采样参数无效 | ⚠️ 已知限制 | llama-rs 0.16.1 不支持 per-request override |
| 多轮对话 KV cache 不复用 | ⚠️ 设计限制 | `generate()` 每次创建新 context，历史 token 需重算 |
| Token 计数为 0 | ⚠️ 未修复 | `Usage` 字段未实现 tokenizer 级别的计数 |
| Gemma 4B CPU 推理慢 | ⚠️ 硬件限制 | 4B 参数在 CPU 上推理是预期行为 |
