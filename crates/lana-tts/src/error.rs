//! Error types for the text-to-speech engine.

use thiserror::Error;

/// Errors returned by [`crate::TtsEngine`].
#[derive(Debug, Error)]
pub enum TtsError {
    /// Loading the model or resolving the voice failed.
    #[error("tts init failed: {0}")]
    Init(String),

    /// A synthesis request failed.
    #[error("tts synthesis failed: {0}")]
    Synthesis(String),
}
