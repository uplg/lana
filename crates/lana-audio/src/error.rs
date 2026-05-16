//! Error types for audio capture and playback.

use thiserror::Error;

/// Errors returned by the audio subsystem.
#[derive(Debug, Error)]
pub enum AudioError {
    /// No default input/output device was available.
    #[error("audio device unavailable: {0}")]
    Device(String),

    /// The device's stream configuration is unsupported by this pipeline.
    #[error("unsupported stream config: {0}")]
    Config(String),

    /// Building or starting the `cpal` stream failed.
    #[error("stream error: {0}")]
    Stream(String),

    /// Decoding a synthesized WAV buffer failed.
    #[error("wav decode failed: {0}")]
    WavDecode(String),
}
