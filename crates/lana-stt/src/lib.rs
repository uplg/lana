//! Speech-to-text for Lana using NVIDIA Parakeet-TDT-0.6B-v3.
//!
//! Backed by `parakeet-rs` (ONNX Runtime, pure Rust — no Swift).
//! Multilingual (25 EU languages incl. French) with automatic language
//! detection.

#![forbid(unsafe_code)]

mod engine;
mod error;

pub use engine::{SttConfig, SttEngine, Transcript, resolve_model_dir};
pub use error::SttError;
