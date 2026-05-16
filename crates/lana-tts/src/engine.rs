//! [`TtsEngine`] — async facade over the native Rust Kyutai Pocket TTS
//! (`pocket-tts` crate, candle/Metal).
//!
//! The real Kyutai voices load from a predefined-voice embedding (e.g.
//! Estelle) and cloning runs the candle Mimi encoder.

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use parking_lot::Mutex;
use pocket_tts::{ModelState, TTSModel};
use tokio::sync::mpsc;
use tokio::task::spawn_blocking;
use tracing::info;

use crate::error::TtsError;

/// Where the speaking voice comes from.
#[derive(Debug, Clone)]
pub enum VoiceSource {
    /// A pre-computed Kyutai voice prompt: a `.safetensors` carrying an
    /// `audio_prompt` tensor (the native format of the files in
    /// `huggingface.co/kyutai/tts-voices`, e.g. Estelle).
    PromptFile(PathBuf),
    /// Clone the voice from a reference WAV (native candle Mimi encoding).
    CloneWav(PathBuf),
    /// A Kyutai predefined-voice embedding (a per-FlowLM-layer KV cache),
    /// e.g. the real Estelle. The spec is a local path or an
    /// `hf://repo/path@rev` URL. Loads token-free with the
    /// without-voice-cloning checkpoint (no Mimi encode, no speaker proj).
    KyutaiEmbedding(String),
}

/// Static configuration of a [`TtsEngine`].
#[derive(Debug, Clone)]
pub struct TtsConfig {
    /// Pocket TTS config variant. `french_24l` is the 24-layer French
    /// checkpoint (Estelle voice); `b6369a24` is the English base.
    pub variant: String,
    /// The voice to speak with.
    pub voice: VoiceSource,
    /// Sampling temperature (higher = more variation).
    pub temperature: f32,
    /// Flow-matching decode steps (more = better quality, slower).
    pub lsd_decode_steps: usize,
    /// End-of-sequence threshold (more negative = longer audio).
    pub eos_threshold: f32,
}

impl TtsConfig {
    /// Config with the default French variant (`french_24l`) and sane
    /// generation parameters, speaking with `voice`.
    #[must_use]
    pub fn new(voice: VoiceSource) -> Self {
        Self {
            variant: "french_24l".to_owned(),
            voice,
            temperature: 0.7,
            lsd_decode_steps: 1,
            eos_threshold: -2.0,
        }
    }
}

/// Synthesised speech: a complete WAV byte buffer.
#[derive(Debug, Clone)]
pub struct Speech {
    /// WAV-encoded audio bytes (model sample rate, mono, 16-bit).
    pub wav: Vec<u8>,
}

struct Inner {
    model: TTSModel,
    voice: ModelState,
}

/// Text-to-speech engine backed by native Kyutai Pocket TTS on Metal.
pub struct TtsEngine {
    inner: Arc<Mutex<Inner>>,
    sample_rate: u32,
}

impl std::fmt::Debug for TtsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsEngine")
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}

impl TtsEngine {
    /// Load the model on Metal and resolve the voice.
    ///
    /// Heavy work (weight download/load, voice encoding) runs on a blocking
    /// thread so the async runtime stays responsive.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::Init`] if the Metal device, model weights or
    /// voice cannot be loaded.
    pub async fn new(config: TtsConfig) -> Result<Self, TtsError> {
        let loaded = spawn_blocking(move || -> Result<(Inner, u32), TtsError> {
            let device =
                Device::new_metal(0).map_err(|e| TtsError::Init(format!("metal device: {e}")))?;

            info!(variant = %config.variant, "loading pocket-tts model");
            let model = TTSModel::load_with_params_device(
                &config.variant,
                config.temperature,
                config.lsd_decode_steps,
                config.eos_threshold,
                None,
                &device,
            )
            .map_err(|e| TtsError::Init(format!("model load: {e}")))?;

            let voice = match &config.voice {
                VoiceSource::PromptFile(path) => {
                    info!(prompt = %path.display(), "loading voice prompt");
                    model
                        .get_voice_state_from_prompt_file(path)
                        .map_err(|e| TtsError::Init(format!("voice prompt: {e}")))?
                }
                VoiceSource::CloneWav(path) => {
                    info!(reference = %path.display(), "cloning voice from wav");
                    model
                        .get_voice_state(path)
                        .map_err(|e| TtsError::Init(format!("voice clone: {e}")))?
                }
                VoiceSource::KyutaiEmbedding(spec) => {
                    info!(embedding = %spec, "loading Kyutai voice embedding");
                    let path = pocket_tts::weights::download_if_necessary(spec.as_str())
                        .map_err(|e| TtsError::Init(format!("voice embedding download: {e}")))?;
                    model
                        .get_voice_state_from_kyutai_embedding(&path)
                        .map_err(|e| TtsError::Init(format!("voice embedding: {e}")))?
                }
            };

            let sample_rate = u32::try_from(model.sample_rate).unwrap_or(24_000);
            info!(sample_rate, "pocket-tts ready");
            Ok((Inner { model, voice }, sample_rate))
        })
        .await
        .map_err(|e| TtsError::Init(format!("blocking join: {e}")))??;

        Ok(Self {
            inner: Arc::new(Mutex::new(loaded.0)),
            sample_rate: loaded.1,
        })
    }

    /// Model output sample rate (Hz).
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Synthesise `text` into a WAV buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::Synthesis`] if generation or WAV encoding fails.
    pub async fn synthesize(&self, text: &str) -> Result<Speech, TtsError> {
        let inner = Arc::clone(&self.inner);
        let text = text.to_owned();
        let sample_rate = self.sample_rate;
        spawn_blocking(move || -> Result<Speech, TtsError> {
            let audio = {
                let guard = inner.lock();
                guard
                    .model
                    .generate(&text, &guard.voice)
                    .map_err(|e| TtsError::Synthesis(format!("generate: {e}")))?
            };

            let mut cursor = std::io::Cursor::new(Vec::new());
            pocket_tts::audio::write_wav_to_writer(&mut cursor, &audio, sample_rate)
                .map_err(|e| TtsError::Synthesis(format!("wav encode: {e}")))?;
            Ok(Speech {
                wav: cursor.into_inner(),
            })
        })
        .await
        .map_err(|e| TtsError::Synthesis(format!("blocking join: {e}")))?
    }

    /// Stream synthesis: yields one mono f32 PCM frame (at
    /// [`Self::sample_rate`]) per Mimi step, as soon as it is produced, so
    /// playback can start after the first frame instead of the whole
    /// utterance. The returned channel closes when generation ends; an
    /// `Err` item is terminal.
    #[must_use]
    pub fn synthesize_stream(&self, text: &str) -> mpsc::Receiver<Result<Vec<f32>, TtsError>> {
        let (tx, rx) = mpsc::channel(64);
        let inner = Arc::clone(&self.inner);
        let text = text.to_owned();
        spawn_blocking(move || {
            let frames = {
                let guard = inner.lock();
                guard.model.generate_stream_owned(&text, &guard.voice)
            };
            for item in frames {
                let msg = item
                    .map_err(|e| TtsError::Synthesis(format!("generate: {e}")))
                    .and_then(|frame| frame_to_mono(&frame));
                let terminal = msg.is_err();
                if tx.blocking_send(msg).is_err() || terminal {
                    break;
                }
            }
        });
        rx
    }
}

/// Flatten a `[B, C, T]` audio frame tensor to mono f32 PCM (mean over
/// channels). The model is mono (`C = 1`) so this is usually identity.
fn frame_to_mono(frame: &Tensor) -> Result<Vec<f32>, TtsError> {
    let map = |e: candle_core::Error| TtsError::Synthesis(format!("frame decode: {e}"));
    frame
        .squeeze(0)
        .and_then(|f| f.mean(0))
        .and_then(|f| f.contiguous())
        .and_then(|f| f.to_vec1::<f32>())
        .map_err(map)
}
