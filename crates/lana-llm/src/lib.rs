//! Local LLM inference for Lana using Qwen3 GGUF via candle.
//!
//! Loads a quantised Qwen3 GGUF file (e.g. `Qwen3-1.7B-Q5_K_M.gguf`) onto
//! Metal and exposes a streaming chat API consumable by the orchestrator.

#![forbid(unsafe_code)]

mod engine;
mod error;
mod token_stream;

pub use engine::{EngineConfig, GenerationConfig, LlmEngine, Message, Role, TokenChunk};
pub use error::LlmError;
