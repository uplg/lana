//! Text-to-speech for Lana using the native Rust port of Kyutai Pocket TTS.
//!
//! Runs on candle (Metal). Voices load directly from Kyutai's native
//! `.safetensors` voice prompts (e.g. Estelle) or are cloned from a
//! reference WAV. This replaces the `FluidAudio` `PocketTTS` path, whose
//! voice/cloning pipeline was broken.

#![forbid(unsafe_code)]

mod engine;
mod error;

pub use engine::{Speech, TtsConfig, TtsEngine, VoiceSource};
pub use error::TtsError;
