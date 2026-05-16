//! Speech playback: decode the TTS WAV and play it on the default output
//! device, cancellable mid-stream for barge-in.
//!
//! Same architecture as capture: `cpal`'s output stream is `!Send` and its
//! callback is realtime, so the stream lives on a dedicated thread. The
//! async side hands decoded+resampled f32 mono samples to a shared queue;
//! the realtime callback drains it (up-mixing to the device channel count).
//! Cancelling clears the queue so the current utterance stops immediately.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use tracing::{info, warn};

use crate::error::AudioError;

/// Sample rate `PocketTTS` emits.
const TTS_RATE: u32 = 24_000;

/// Shared playback queue plus a generation counter. Bumping the generation
/// (on cancel) invalidates whatever is still queued without tearing down
/// the stream.
#[derive(Debug, Default)]
struct Shared {
    queue: VecDeque<f32>,
    /// Incremented on cancel; the feeder checks it to abort early.
    generation: u64,
    /// Streaming-resampler continuity for [`AudioOutput::enqueue_pcm`].
    /// Per-frame independent resampling left a discontinuity at every
    /// frame seam (~80 ms) → audible crackle. We carry an integer phase
    /// accumulator and the previous frame's last sample so consecutive
    /// frames resample as one continuous stream.
    rs_phase: u32,
    rs_prev: f32,
    /// Source rate the carried state applies to; a change resets it.
    rs_src: u32,
    /// Jitter buffer: the realtime callback does not drain the queue until
    /// it first holds `prebuffer` samples, and re-arms on underrun. This
    /// stops the early-generation crackle (the callback would otherwise
    /// greedily consume the first frame before the LLM→sentence→TTS
    /// pipeline is primed, starving between the first short chunk and the
    /// next). Cost: ~`prebuffer / device_rate` s of leading silence.
    started: bool,
    /// Samples to accumulate before playback starts / resumes (set once
    /// from the device rate in [`AudioOutput::start`]).
    prebuffer: usize,
}

/// Live audio output. One device/stream reused across utterances.
#[derive(Debug)]
pub struct AudioOutput {
    shared: Arc<Mutex<Shared>>,
    device_rate: u32,
    device_channels: usize,
    running: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl AudioOutput {
    /// Open the default output device and start an (initially silent)
    /// stream.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError`] if no output device exists or the stream
    /// cannot be built.
    pub fn start() -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| AudioError::Device("no default output device".to_owned()))?;
        let supported = device
            .default_output_config()
            .map_err(|e| AudioError::Config(e.to_string()))?;
        let device_rate = supported.sample_rate();
        let device_channels = supported.channels() as usize;
        let sample_format = supported.sample_format();
        info!(
            rate = device_rate,
            channels = device_channels,
            ?sample_format,
            "opening output device"
        );

        let shared = Arc::new(Mutex::new(Shared::default()));
        // ~250 ms jitter buffer before playback starts/resumes.
        shared.lock().prebuffer = (device_rate as usize) / 4;
        let running = Arc::new(AtomicBool::new(true));
        let cfg: cpal::StreamConfig = supported.config();

        let thread_shared = Arc::clone(&shared);
        let thread_running = Arc::clone(&running);
        let thread = std::thread::Builder::new()
            .name("lana-out".to_owned())
            .spawn(move || {
                run_output_thread(
                    &device,
                    &cfg,
                    sample_format,
                    device_channels,
                    &thread_shared,
                    &thread_running,
                );
            })
            .map_err(|e| AudioError::Stream(format!("spawn output thread: {e}")))?;

        Ok(Self {
            shared,
            device_rate,
            device_channels,
            running,
            thread: Some(thread),
        })
    }

    /// Decode a WAV buffer and enqueue it for playback. Returns once the
    /// audio has been queued (not when it finishes playing); use
    /// [`AudioOutput::wait_drained`] to await completion.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::WavDecode`] if the buffer is not decodable.
    pub fn enqueue_wav(&self, wav: &[u8]) -> Result<(), AudioError> {
        let mono = decode_wav_mono(wav)?;
        let resampled = linear_resample(&mono, TTS_RATE, self.device_rate);
        {
            let mut guard = self.shared.lock();
            guard.queue.extend(resampled);
        }
        Ok(())
    }

    /// Enqueue a raw mono f32 PCM frame (samples in `[-1, 1]`) sampled at
    /// `src_rate`, resampling to the device rate. Used for native streaming
    /// TTS: each Mimi frame is played as soon as it is produced.
    ///
    /// The resampler is *stateful across frames*: it carries the fractional
    /// read position and the previous frame's last sample, so frame seams
    /// no longer introduce the periodic discontinuity that crackled.
    pub fn enqueue_pcm(&self, samples: &[f32], src_rate: u32) {
        if samples.is_empty() {
            return;
        }
        let device_rate = self.device_rate;
        let last_sample = samples.last().copied().unwrap_or(0.0); // non-empty
        let mut shared = self.shared.lock();

        if src_rate != shared.rs_src {
            // First use, or the source rate changed: (re)start continuity
            // from this frame's first sample (no leading click).
            shared.rs_src = src_rate;
            shared.rs_phase = 0;
            shared.rs_prev = samples[0];
        }

        if src_rate == device_rate {
            shared.queue.extend(samples.iter().copied());
            shared.rs_prev = last_sample;
            return;
        }

        // Continuous linear resample via an integer phase accumulator.
        // Between two consecutive input samples `prev` and `cur`, an output
        // lands at fractional position `phase / device_rate`; advancing the
        // phase by `src_rate` per output and consuming one input sample
        // every `device_rate` keeps the rational ratio exact. Carrying
        // `phase` and `prev` across calls makes frame seams continuous.
        let mut phase = shared.rs_phase;
        let mut prev = shared.rs_prev;
        let mut out = Vec::new();
        for &cur in samples {
            while phase < device_rate {
                // `phase` < `device_rate` ≤ 192 kHz: exact in f32.
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "phase < device_rate ≤ 192 kHz is exact in f32"
                )]
                let frac = phase as f32 / device_rate as f32;
                out.push((cur - prev).mul_add(frac, prev));
                phase = phase.saturating_add(src_rate);
            }
            phase = phase.saturating_sub(device_rate);
            prev = cur;
        }
        shared.rs_phase = phase;
        shared.rs_prev = last_sample;
        shared.queue.extend(out);
    }

    /// Stop the current utterance immediately (barge-in): drop everything
    /// queued and bump the generation.
    pub fn cancel(&self) {
        let mut guard = self.shared.lock();
        guard.queue.clear();
        guard.generation = guard.generation.wrapping_add(1);
        // Next utterance is unrelated audio: restart resampler continuity
        // and re-arm the jitter buffer.
        guard.rs_phase = 0;
        guard.rs_prev = 0.0;
        guard.rs_src = 0;
        guard.started = false;
    }

    /// Force the jitter buffer to release: play whatever is queued even if
    /// it never reached the prebuffer threshold. Call once a turn's audio
    /// has been fully enqueued so a reply shorter than the prebuffer (and
    /// the final tail) still plays out instead of stalling.
    pub fn flush_tail(&self) {
        self.shared.lock().started = true;
    }

    /// `true` while there is still audio queued to play.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        !self.shared.lock().queue.is_empty()
    }

    /// Block (cheaply, polling) until the queue is empty or playback is
    /// cancelled.
    pub fn wait_drained(&self) {
        while self.is_playing() && self.running.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// Device output rate, exposed for diagnostics.
    #[must_use]
    pub const fn device_rate(&self) -> u32 {
        self.device_rate
    }

    /// Device channel count, exposed for diagnostics.
    #[must_use]
    pub const fn device_channels(&self) -> usize {
        self.device_channels
    }
}

impl Drop for AudioOutput {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn run_output_thread(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    channels: usize,
    shared: &Arc<Mutex<Shared>>,
    running: &AtomicBool,
) {
    let stream = match build_output_stream(device, config, sample_format, channels, shared) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "failed to build output stream");
            return;
        }
    };
    if let Err(e) = stream.play() {
        warn!(error = %e, "failed to start output stream");
        return;
    }
    while running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    channels: usize,
    shared: &Arc<Mutex<Shared>>,
) -> Result<cpal::Stream, AudioError> {
    let err_fn = |e| warn!(error = %e, "output stream error");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let shared = Arc::clone(shared);
            device.build_output_stream(
                config,
                move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut guard = shared.lock();

                    // Jitter buffer: stay silent until the queue is primed.
                    if !guard.started {
                        if guard.queue.len() >= guard.prebuffer {
                            guard.started = true;
                        } else {
                            out.fill(0.0);
                            return;
                        }
                    }

                    for frame in out.chunks_mut(channels) {
                        if let Some(s) = guard.queue.pop_front() {
                            for slot in frame.iter_mut() {
                                *slot = s;
                            }
                        } else {
                            // Underrun: re-arm the jitter buffer so the next
                            // audio re-primes instead of crackling, and emit
                            // silence for the rest of this block.
                            guard.started = false;
                            for slot in frame.iter_mut() {
                                *slot = 0.0;
                            }
                        }
                    }
                    drop(guard);
                },
                err_fn,
                None,
            )
        }
        other => {
            return Err(AudioError::Config(format!(
                "unsupported output sample format {other:?} (expected F32)"
            )));
        }
    };

    stream.map_err(|e| AudioError::Stream(e.to_string()))
}

/// Decode a 16-bit (or float) WAV into mono f32 in `[-1, 1]`.
fn decode_wav_mono(wav: &[u8]) -> Result<Vec<f32>, AudioError> {
    let cursor = std::io::Cursor::new(wav);
    let mut reader =
        hound::WavReader::new(cursor).map_err(|e| AudioError::WavDecode(e.to_string()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0_f32 / f32::from(i16::MAX);
            reader
                .samples::<i16>()
                .map(|s| s.map(|v| f32::from(v) * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| AudioError::WavDecode(e.to_string()))?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| AudioError::WavDecode(e.to_string()))?,
    };

    if channels <= 1 {
        return Ok(interleaved);
    }
    #[expect(clippy::cast_precision_loss, reason = "channel count is tiny (1-8)")]
    let inv = 1.0_f32 / channels as f32;
    Ok(interleaved
        .chunks_exact(channels)
        .map(|f| f.iter().sum::<f32>() * inv)
        .collect())
}

/// Linear-interpolation resample of mono audio. Speech output tolerates
/// linear interpolation perceptually; the heavier windowed-sinc path is
/// reserved for the STT *input* where it affects recognition accuracy.
fn linear_resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == to_rate {
        return input.to_vec();
    }
    let from = u64::from(from_rate);
    let to = u64::from(to_rate);
    let in_len = u64::try_from(input.len()).unwrap_or(u64::MAX);
    let out_len_u64 = in_len.saturating_mul(to).checked_div(from).unwrap_or(0);
    let out_len = usize::try_from(out_len_u64).unwrap_or(usize::MAX);

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = u64::try_from(i).unwrap_or(u64::MAX).saturating_mul(from);
        let idx = usize::try_from(pos.checked_div(to).unwrap_or(0)).unwrap_or(usize::MAX);
        let rem = pos.checked_rem(to).unwrap_or(0);
        // The only int→float conversion a resampler cannot avoid. `rem < to`
        // and `to` is a device sample rate (<= 96 kHz < 2^24), so both
        // values are exactly representable in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "rem < to <= 96 kHz < 2^24; exact in f32"
        )]
        let frac = rem as f32 / to as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx.saturating_add(1)).copied().unwrap_or(a);
        out.push((b - a).mul_add(frac, a));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::linear_resample;

    #[test]
    fn same_rate_is_identity() {
        let s = vec![0.1, 0.2, 0.3];
        assert_eq!(linear_resample(&s, 24_000, 24_000), s);
    }

    #[test]
    fn doubling_rate_roughly_doubles_length() {
        let s = vec![0.0_f32; 1_000];
        let out = linear_resample(&s, 24_000, 48_000);
        assert!(out.len().abs_diff(2_000) < 4, "len {}", out.len());
    }
}
