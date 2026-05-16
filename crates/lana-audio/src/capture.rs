//! Microphone capture: `cpal` input → mono → 16 kHz → fixed chunks.
//!
//! `cpal`'s stream is `!Send` and its data callback runs on a realtime
//! audio thread, so the stream lives entirely on a dedicated OS thread.
//! That thread owns the stream, drains raw samples from the callback over a
//! `crossbeam` channel, downmixes to mono, decimates 48→16 kHz and forwards
//! fixed 256 ms chunks to the async side over a Tokio channel.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::error::AudioError;
use crate::resample::Decimator3;

/// Output sample rate fed to VAD/STT.
pub const TARGET_RATE: u32 = 16_000;
/// Input rate the capture path is built for (exact 3:1 to `TARGET_RATE`).
const EXPECTED_INPUT_RATE: u32 = 48_000;
/// Samples per emitted chunk: 256 ms at 16 kHz.
///
/// Chosen for the VAD/utterance-segmentation granularity (start/end-of-turn
/// and the post-speech grace are tracked per chunk). Not a hard constraint;
/// it just happens to be a whole multiple of `earshot`'s 256-sample frame
/// (4096 = 16 × 256), so the VAD's `chunks_exact(256)` discards no audio.
pub const CHUNK_SAMPLES: usize = 4_096;
/// Bound on the chunk channel; back-pressure if the consumer stalls.
const CHUNK_CHANNEL_CAPACITY: usize = 32;

/// A running microphone capture. Dropping it stops the stream and joins the
/// audio thread.
#[derive(Debug)]
pub struct MicCapture {
    running: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl MicCapture {
    /// Open the default input device and start streaming 16 kHz mono chunks.
    ///
    /// Returns the handle plus the receiver of `Vec<f32>` chunks (each
    /// [`CHUNK_SAMPLES`] long, except possibly the last before shutdown).
    ///
    /// # Errors
    ///
    /// Returns [`AudioError`] if no input device exists, the device is not
    /// 48 kHz, or the stream cannot be built.
    pub fn start() -> Result<(Self, mpsc::Receiver<Vec<f32>>), AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| AudioError::Device("no default input device".to_owned()))?;
        let supported = device
            .default_input_config()
            .map_err(|e| AudioError::Config(e.to_string()))?;

        let sample_rate = supported.sample_rate();
        if sample_rate != EXPECTED_INPUT_RATE {
            return Err(AudioError::Config(format!(
                "input device runs at {sample_rate} Hz; this build expects \
                 {EXPECTED_INPUT_RATE} Hz (exact 3:1 to {TARGET_RATE})"
            )));
        }
        let channels = supported.channels() as usize;
        let sample_format = supported.sample_format();
        info!(
            rate = sample_rate,
            channels,
            ?sample_format,
            "opening input device"
        );

        let (chunk_tx, chunk_rx) = mpsc::channel::<Vec<f32>>(CHUNK_CHANNEL_CAPACITY);
        let running = Arc::new(AtomicBool::new(true));
        let thread_running = Arc::clone(&running);
        let stream_config: cpal::StreamConfig = supported.config();

        let thread = std::thread::Builder::new()
            .name("lana-mic".to_owned())
            .spawn(move || {
                run_capture_thread(
                    &device,
                    &stream_config,
                    sample_format,
                    channels,
                    sample_rate,
                    &thread_running,
                    &chunk_tx,
                );
            })
            .map_err(|e| AudioError::Stream(format!("spawn capture thread: {e}")))?;

        Ok((
            Self {
                running,
                thread: Some(thread),
            },
            chunk_rx,
        ))
    }
}

impl Drop for MicCapture {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn run_capture_thread(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    channels: usize,
    input_rate: u32,
    running: &AtomicBool,
    chunk_tx: &mpsc::Sender<Vec<f32>>,
) {
    // Raw interleaved samples from the realtime callback. Unbounded so the
    // audio thread never blocks; the worker keeps it drained.
    let (raw_tx, raw_rx) = crossbeam_channel::unbounded::<Vec<f32>>();

    let err_tx = raw_tx.clone();
    let stream = match build_stream(device, config, sample_format, &raw_tx) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "failed to build input stream");
            return;
        }
    };
    drop(err_tx);

    if let Err(e) = stream.play() {
        warn!(error = %e, "failed to start input stream");
        return;
    }

    let mut decimator = Decimator3::new(input_rate);
    let mut mono_scratch: Vec<f32> = Vec::with_capacity(8_192);
    let mut resampled: Vec<f32> = Vec::with_capacity(8_192);
    let mut pending: Vec<f32> = Vec::with_capacity(CHUNK_SAMPLES * 2);

    while running.load(Ordering::SeqCst) {
        let Ok(raw) = raw_rx.recv_timeout(Duration::from_millis(100)) else {
            continue;
        };

        downmix_into(&raw, channels, &mut mono_scratch);
        resampled.clear();
        decimator.process(&mono_scratch, &mut resampled);
        pending.extend_from_slice(&resampled);

        while pending.len() >= CHUNK_SAMPLES {
            let chunk: Vec<f32> = pending.drain(..CHUNK_SAMPLES).collect();
            if chunk_tx.blocking_send(chunk).is_err() {
                // Consumer gone; stop capturing.
                return;
            }
        }
    }
}

fn build_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    raw_tx: &crossbeam_channel::Sender<Vec<f32>>,
) -> Result<cpal::Stream, AudioError> {
    let err_fn = |e| warn!(error = %e, "input stream error");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let tx = raw_tx.clone();
            device.build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let _ = tx.send(data.to_vec());
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let tx = raw_tx.clone();
            device.build_input_stream(
                config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut v = Vec::with_capacity(data.len());
                    for &s in data {
                        v.push(f32::from(s) / f32::from(i16::MAX));
                    }
                    let _ = tx.send(v);
                },
                err_fn,
                None,
            )
        }
        other => {
            return Err(AudioError::Config(format!(
                "unsupported sample format {other:?} (expected F32 or I16)"
            )));
        }
    };

    stream.map_err(|e| AudioError::Stream(e.to_string()))
}

/// Average interleaved channels down to mono into `out` (cleared first).
fn downmix_into(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    out.clear();
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    #[expect(clippy::cast_precision_loss, reason = "channel count is tiny (1-8)")]
    let inv = 1.0_f32 / channels as f32;
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        out.push(sum * inv);
    }
}
