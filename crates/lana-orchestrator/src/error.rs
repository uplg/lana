//! Error type for the orchestrator.

use thiserror::Error;

/// Fatal errors that stop the conversational loop. Per-turn failures
/// (a bad transcription, a TTS hiccup) are reported as
/// [`crate::OrchestratorEvent::Error`] and the loop continues.
#[derive(Debug, Error)]
pub enum OrchestratorError {
    /// The microphone stream closed unexpectedly.
    #[error("microphone stream ended")]
    MicClosed,

    /// Audio playback subsystem failed irrecoverably.
    #[error("audio output failed: {0}")]
    Audio(String),
}
