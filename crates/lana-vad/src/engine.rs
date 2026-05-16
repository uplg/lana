//! [`VadEngine`] — pure-Rust voice activity detection via `earshot`.
//!
//! `earshot` is a tiny (~110 KiB) neural VAD: no ONNX, no Swift, no model
//! download. It scores fixed 256-sample (16 ms @ 16 kHz) frames. The
//! orchestrator feeds 4096-sample (256 ms) chunks, so each call slices the
//! chunk into 16 frames, scores them and reports the chunk voiced when a
//! sufficient fraction crosses the threshold (debounces single blips).

use earshot::Detector;
use parking_lot::Mutex;
use tracing::info;

use crate::error::VadError;

/// Frame size `earshot` requires (16 ms @ 16 kHz).
const FRAME: usize = 256;
/// Fraction of a chunk's frames that must be voiced for the chunk to count
/// as speech. ~25 % ≈ 64 ms of voice within a 256 ms chunk.
const VOICED_FRACTION: f32 = 0.25;

/// Static configuration of a [`VadEngine`].
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Per-frame score (0..1) above which a frame counts as voiced. Lower =
    /// more sensitive.
    pub threshold: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

/// Voice activity detector. Holds one `earshot` `Detector` (it carries
/// per-stream state) behind a mutex; scoring is microsecond-cheap so no
/// blocking offload is needed.
pub struct VadEngine {
    detector: Mutex<Detector>,
    threshold: f32,
}

impl std::fmt::Debug for VadEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VadEngine")
            .field("threshold", &self.threshold)
            .finish_non_exhaustive()
    }
}

impl VadEngine {
    /// Create the detector. Infallible in practice; kept `async`/`Result`
    /// for API symmetry with the other engines and future backends.
    ///
    /// # Errors
    ///
    /// Currently never returns an error.
    #[expect(
        clippy::unused_async,
        reason = "API symmetry with the STT/TTS engines; callers `.await` it"
    )]
    pub async fn new(config: VadConfig) -> Result<Self, VadError> {
        info!(threshold = config.threshold, "initialising earshot vad");
        Ok(Self {
            detector: Mutex::new(Detector::default()),
            threshold: config.threshold,
        })
    }

    /// Classify one chunk of 16 kHz mono `f32` samples: `true` if a
    /// sufficient fraction of its 256-sample frames score above the
    /// threshold. Any trailing partial frame (< 256 samples) is ignored.
    ///
    /// # Errors
    ///
    /// Currently never returns an error.
    #[expect(
        clippy::unused_async,
        reason = "API symmetry with the STT engine; the orchestrator `.await`s it"
    )]
    pub async fn voice_active(&self, samples: Vec<f32>) -> Result<bool, VadError> {
        let mut detector = self.detector.lock();
        let mut frames: u32 = 0;
        let mut voiced: u32 = 0;
        for frame in samples.chunks_exact(FRAME) {
            frames = frames.saturating_add(1);
            if detector.predict_f32(frame) >= self.threshold {
                voiced = voiced.saturating_add(1);
            }
        }
        drop(detector);
        if frames == 0 {
            return Ok(false);
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "frame counts are tiny (<= 16 per chunk); exact in f32"
        )]
        let ratio = voiced as f32 / frames as f32;
        Ok(ratio >= VOICED_FRACTION)
    }
}
