//! Voice activity detection and utterance segmentation for Lana.
//!
//! [`VadEngine`] wraps `FluidAudio`'s on-device VAD (its own instance, so it
//! can run concurrently with STT/TTS for barge-in). [`UtteranceSegmenter`]
//! is a pure state machine turning per-chunk voice flags into
//! start/end-of-turn [`VadEvent`]s.

#![forbid(unsafe_code)]

mod engine;
mod error;
mod segmenter;

pub use engine::{VadConfig, VadEngine};
pub use error::VadError;
pub use segmenter::{SegmenterConfig, UtteranceSegmenter, VadEvent};
