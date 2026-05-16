//! Text-to-speech for Lana using the native Rust port of Kyutai Pocket TTS.
//!
//! Runs on candle (Metal). Voices load from a Kyutai predefined-voice
//! embedding (e.g. the real French Estelle) or are cloned from a
//! reference WAV. Native per-Mimi-frame streaming.

#![forbid(unsafe_code)]

mod engine;
mod error;

pub use engine::{Speech, TtsConfig, TtsEngine, VoiceSource};
pub use error::TtsError;
