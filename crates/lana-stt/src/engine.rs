//! [`SttEngine`] — async facade over `parakeet-rs` (Parakeet-TDT-0.6B-v3,
//! ONNX Runtime, pure Rust — no Swift, no `CoreML`).
//!
//! The model ONNX files (`encoder-model.onnx`, `encoder-model.onnx.data`,
//! `decoder_joint-model.onnx`, `vocab.txt`) live in one directory; get them
//! from HF `istupakov/parakeet-tdt-0.6b-v3-onnx`. ONNX Runtime CPU is the
//! default execution provider (`CoreML` is unstable for this model and CPU
//! is fast enough on Apple Silicon).

use std::path::PathBuf;
use std::sync::Arc;

use parakeet_rs::{ParakeetTDT, Transcriber};
use parking_lot::Mutex;
use tokio::task::spawn_blocking;
use tracing::info;

use crate::error::SttError;

/// Static configuration of an [`SttEngine`].
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// Directory containing the Parakeet-TDT v3 ONNX files.
    pub model_dir: PathBuf,
}

/// A transcript.
#[derive(Debug, Clone)]
pub struct Transcript {
    /// Decoded text.
    pub text: String,
}

/// Streaming-capable speech-to-text engine backed by Parakeet-TDT v3.
pub struct SttEngine {
    model: Arc<Mutex<ParakeetTDT>>,
}

impl std::fmt::Debug for SttEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SttEngine").finish_non_exhaustive()
    }
}

impl SttEngine {
    /// Load the Parakeet-TDT v3 ONNX model from `config.model_dir`.
    ///
    /// Heavy work (ONNX session build) runs on a blocking thread.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::Init`] if the model directory is missing or the
    /// ONNX session cannot be built.
    pub async fn new(config: SttConfig) -> Result<Self, SttError> {
        let model = spawn_blocking(move || -> Result<ParakeetTDT, SttError> {
            info!(model_dir = %config.model_dir.display(), "loading parakeet-tdt v3");
            let model = ParakeetTDT::from_pretrained(&config.model_dir, None)
                .map_err(|e| SttError::Init(e.to_string()))?;
            info!("parakeet ready");
            Ok(model)
        })
        .await
        .map_err(|e| SttError::Init(format!("blocking join: {e}")))??;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }

    /// Transcribe in-memory PCM samples. `parakeet-rs` resamples internally
    /// from `sample_rate` to the model rate, so any rate is accepted (the
    /// orchestrator passes 16 kHz mono).
    ///
    /// # Errors
    ///
    /// Returns [`SttError::Transcribe`] if inference fails.
    pub async fn transcribe_samples(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Transcript, SttError> {
        let model = Arc::clone(&self.model);
        spawn_blocking(move || -> Result<Transcript, SttError> {
            let text = {
                let mut guard = model.lock();
                guard
                    .transcribe_samples(samples, sample_rate, channels, None)
                    .map_err(|e| SttError::Transcribe(e.to_string()))?
                    .text
            };
            Ok(Transcript { text })
        })
        .await
        .map_err(|e| SttError::Transcribe(format!("blocking join: {e}")))?
    }
}
