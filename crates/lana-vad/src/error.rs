//! Error types for voice activity detection.

use thiserror::Error;

/// Errors returned by [`crate::VadEngine`].
#[derive(Debug, Error)]
pub enum VadError {
    /// Initialisation of the `earshot` detector failed.
    #[error("vad init failed: {0}")]
    Init(String),

    /// Scoring an audio chunk failed.
    #[error("vad processing failed: {0}")]
    Process(String),
}
