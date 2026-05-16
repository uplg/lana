//! Audio capture, resampling and playback plumbing for Lana.
//!
//! Capture: default input device via `cpal` (`CoreAudio` on macOS), mixed
//! to mono and decimated to 16 kHz, exposed as a stream of fixed 256 ms
//! chunks for VAD/STT. Playback (output of synthesized speech) is added
//! alongside it.

#![forbid(unsafe_code)]

mod capture;
mod error;
mod playback;
mod resample;

pub use capture::{CHUNK_SAMPLES, MicCapture, TARGET_RATE};
pub use error::AudioError;
pub use playback::AudioOutput;
