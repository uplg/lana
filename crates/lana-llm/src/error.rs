//! Error types for the local LLM engine.

use thiserror::Error;

/// Errors returned by [`crate::LlmEngine`].
#[derive(Debug, Error)]
pub enum LlmError {
    /// Loading model weights, tokenizer or device initialisation failed.
    #[error("model load failed: {0}")]
    Load(String),

    /// A streaming inference request failed mid-flight.
    #[error("inference failed: {0}")]
    Inference(String),
}
