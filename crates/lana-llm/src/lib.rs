//! Local LLM inference for Lana via candle.
//!
//! Loads a quantised LFM2 GGUF (Luth-LFM2, French-specialised) onto Metal
//! and exposes a streaming chat API consumable by the orchestrator.

#![forbid(unsafe_code)]

mod engine;
mod error;
mod token_stream;

pub use engine::{
    EngineConfig, GenerationConfig, LlmEngine, Message, Role, TokenChunk, resolve_model_assets,
};
pub use error::LlmError;
