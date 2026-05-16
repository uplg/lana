//! Error types for the speech-to-text engine.

use thiserror::Error;

/// Errors returned by [`crate::SttEngine`].
#[derive(Debug, Error)]
pub enum SttError {
    /// Loading the Parakeet ONNX model failed.
    #[error("stt init failed: {0}")]
    Init(String),

    /// A transcription request failed.
    #[error("stt transcription failed: {0}")]
    Transcribe(String),
}
