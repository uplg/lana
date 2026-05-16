//! Audio → viseme analysis for Lana's lip-sync.
//!
//! Short-time analysis of a finished mono PCM utterance: per-frame loudness
//! drives mouth *openness*, and a coarse F1/F2 spectral-peak estimate picks
//! the dominant *vowel* shape. The result is a time-stamped timeline the
//! avatar replays in lock-step with playback.
//!
//! This is deliberately an **approximation**. Phoneme-accurate visemes need
//! forced alignment against a recogniser; we only have the speech waveform.
//! Energy + a two-formant guess is the standard cheap audio-only technique
//! and is enough for believable mouth motion synchronised to the voice.

#![forbid(unsafe_code)]
// Bounded short-time DSP: frame indices, Hz bins and millisecond timestamps
// are all small and range-safe, so the `usize`/`u32`→`f32` casts are exact
// for the magnitudes involved and the index arithmetic cannot realistically
// overflow. Checked-arithmetic / FMA lints add noise without value to this
// derivation (same precedent as the capture-path decimator).
#![expect(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    reason = "bounded short-time DSP; frame/bin/ms magnitudes are range-safe"
)]

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Coarse mouth shape for a single analysis frame. `Sil` is a closed,
/// resting mouth (silence or sub-threshold energy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Vowel {
    /// Closed / resting mouth (silence).
    #[default]
    Sil,
    /// Open, central — /a/.
    A,
    /// Mid-front, lips spread — /e/, /ɛ/.
    E,
    /// Close-front, lips spread — /i/.
    I,
    /// Mid-back, lips rounded — /o/, /ɔ/.
    O,
    /// Close-back, lips tightly rounded — /u/, /y/.
    U,
}

/// One analysed frame of the utterance.
#[derive(Debug, Clone, Copy)]
pub struct VisemeFrame {
    /// Offset from the start of the utterance, milliseconds.
    pub t_ms: u32,
    /// Mouth openness, `0.0` (closed) … `1.0` (wide). Energy-derived.
    pub openness: f32,
    /// Dominant vowel shape for this frame.
    pub vowel: Vowel,
}

/// Analysis window length (seconds) — ~one pitch-stable speech frame.
const WIN_SECS: f32 = 0.025;
/// Hop between successive frames (seconds) — 10 ms ≈ 100 fps timeline.
const HOP_SECS: f32 = 0.010;

/// Absolute RMS noise gate: below this the mouth is closed regardless of
/// the per-utterance maximum (kills hiss / DC on near-silent buffers).
const ABS_FLOOR: f32 = 0.004;
/// Relative noise gate as a fraction of the loudest frame in the utterance.
const REL_FLOOR: f32 = 0.06;
/// Below this openness the frame is treated as a closed (`Sil`) mouth.
const OPEN_MIN: f32 = 0.06;

/// Formant-1 search band (Hz): jaw / mouth aperture.
const F1_LO: f32 = 200.0;
const F1_HI: f32 = 1000.0;
/// Formant-2 search band (Hz): tongue front/back & lip rounding.
const F2_LO: f32 = 900.0;
const F2_HI: f32 = 2900.0;
/// Minimum F2−F1 separation so the two peaks are distinct formants.
const FORMANT_GAP: f32 = 150.0;

/// `(F1, F2)` reference centroids (Hz) for a neutral speaker. Coarse on
/// purpose — enough to separate the five mouth shapes, not a phonetic model.
const CENTROIDS: [(Vowel, f32, f32); 5] = [
    (Vowel::A, 730.0, 1090.0),
    (Vowel::E, 530.0, 1840.0),
    (Vowel::I, 270.0, 2290.0),
    (Vowel::O, 570.0, 840.0),
    (Vowel::U, 300.0, 870.0),
];

/// Analyse a finished mono PCM utterance into a viseme timeline.
///
/// `pcm` is mono `f32` samples in roughly `[-1, 1]` at `sample_rate` Hz.
/// Returns one [`VisemeFrame`] every ~10 ms (ascending `t_ms`). An empty
/// input, or `sample_rate == 0`, yields an empty timeline.
#[must_use]
pub fn analyze(pcm: &[f32], sample_rate: u32) -> Vec<VisemeFrame> {
    if pcm.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let sr = sample_rate as f32;
    let win = ((WIN_SECS * sr).round() as usize).max(1);
    let hop = ((HOP_SECS * sr).round() as usize).max(1);
    let fft_size = win.next_power_of_two();

    let hann = hann_window(win);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Frame start offsets: every `hop` samples while any sample remains.
    let starts: Vec<usize> = (0..pcm.len()).step_by(hop).collect();

    // Pass 1: per-frame RMS + the loudest frame (for the relative gate).
    let rms: Vec<f32> = starts.iter().map(|&s| frame_rms(pcm, s, win)).collect();
    let max_rms = rms.iter().copied().fold(0.0_f32, f32::max);
    let floor = ABS_FLOOR.max(max_rms * REL_FLOOR);
    let span = (max_rms - floor).max(1e-6);

    // Pass 2: openness + vowel per frame.
    let mut scratch = vec![Complex::<f32>::new(0.0, 0.0); fft_size];
    let mut out = Vec::with_capacity(starts.len());
    for (idx, &start) in starts.iter().enumerate() {
        let t_ms = ((start as u64).saturating_mul(1000) / u64::from(sample_rate)) as u32;
        let r = rms.get(idx).copied().unwrap_or(0.0);
        let openness = if r <= floor {
            0.0
        } else {
            (((r - floor) / span).clamp(0.0, 1.0)).sqrt()
        };

        let vowel = if openness < OPEN_MIN {
            Vowel::Sil
        } else {
            classify_vowel(pcm, start, win, &hann, &fft, &mut scratch, sr, fft_size)
        };

        out.push(VisemeFrame {
            t_ms,
            openness,
            vowel,
        });
    }
    out
}

/// RMS of the `win`-sample frame starting at `start` (zero past the end).
fn frame_rms(pcm: &[f32], start: usize, win: usize) -> f32 {
    let end = start.saturating_add(win).min(pcm.len());
    let slice = pcm.get(start..end).unwrap_or(&[]);
    if slice.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = slice.iter().map(|&s| s * s).sum();
    (sum_sq / win as f32).sqrt()
}

/// Symmetric Hann window of length `n` (unit-less analysis taper).
fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n.max(1)];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 - 0.5 * (core::f32::consts::TAU * i as f32 / denom).cos())
        .collect()
}

/// Estimate F1/F2 by spectral-peak picking and snap to the nearest vowel
/// centroid in log-frequency space. Falls back to `Sil` when the spectrum
/// has no usable peak in the formant bands.
#[expect(
    clippy::too_many_arguments,
    reason = "hot inner DSP helper; bundling args into a struct would only \
              move the noise and add an allocation per frame"
)]
fn classify_vowel(
    pcm: &[f32],
    start: usize,
    win: usize,
    hann: &[f32],
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    scratch: &mut [Complex<f32>],
    sr: f32,
    fft_size: usize,
) -> Vowel {
    for c in scratch.iter_mut() {
        *c = Complex::new(0.0, 0.0);
    }
    let end = start.saturating_add(win).min(pcm.len());
    for (i, &s) in pcm.get(start..end).unwrap_or(&[]).iter().enumerate() {
        let w = hann.get(i).copied().unwrap_or(1.0);
        if let Some(slot) = scratch.get_mut(i) {
            *slot = Complex::new(s * w, 0.0);
        }
    }
    fft.process(scratch);

    let bin_hz = sr / fft_size as f32;
    let half = fft_size / 2;
    let peak_in = |lo: f32, hi: f32, min_hz: f32| -> Option<f32> {
        let mut best_mag = 0.0_f32;
        let mut best_hz = None;
        for bin in 1..half {
            let hz = bin as f32 * bin_hz;
            if hz < lo || hz > hi || hz < min_hz {
                continue;
            }
            let mag = scratch.get(bin).map_or(0.0, Complex::norm_sqr);
            if mag > best_mag {
                best_mag = mag;
                best_hz = Some(hz);
            }
        }
        best_hz
    };

    let Some(f1) = peak_in(F1_LO, F1_HI, 0.0) else {
        return Vowel::Sil;
    };
    let Some(f2) = peak_in(F2_LO, F2_HI, f1 + FORMANT_GAP) else {
        return Vowel::Sil;
    };

    let (lf1, lf2) = (f1.max(1.0).ln(), f2.max(1.0).ln());
    let mut best = (f32::MAX, Vowel::Sil);
    for &(v, c1, c2) in &CENTROIDS {
        let d1 = lf1 - c1.ln();
        let d2 = lf2 - c2.ln();
        let dist = d1 * d1 + d2 * d2;
        if dist < best.0 {
            best = (dist, v);
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, amp: f32, secs: f32, sr: u32) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| amp * (core::f32::consts::TAU * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn empty_or_zero_rate_is_empty() {
        assert!(analyze(&[], 24_000).is_empty());
        assert!(analyze(&[0.1, -0.1], 0).is_empty());
    }

    #[test]
    fn silence_is_closed_mouth() {
        let frames = analyze(&vec![0.0_f32; 24_000], 24_000);
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(f.openness <= 0.0, "silence must be fully closed");
            assert_eq!(f.vowel, Vowel::Sil);
        }
    }

    #[test]
    fn timestamps_are_ascending_and_bounded() {
        let frames = analyze(&sine(220.0, 0.5, 1.0, 24_000), 24_000);
        assert!(frames.len() > 50);
        let mut prev = None;
        for f in &frames {
            if let Some(p) = prev {
                assert!(f.t_ms > p, "t_ms must strictly increase");
            }
            assert!((0.0..=1.0).contains(&f.openness));
            prev = Some(f.t_ms);
        }
    }

    #[test]
    fn voiced_audio_opens_the_mouth() {
        let frames = analyze(&sine(300.0, 0.8, 0.5, 24_000), 24_000);
        let open = frames
            .iter()
            .any(|f| f.openness > 0.5 && f.vowel != Vowel::Sil);
        assert!(open, "a loud tone should open the mouth on some frame");
    }

    #[test]
    fn louder_is_not_quieter() {
        // A near-silent buffer must never out-open a loud one.
        let quiet = analyze(&sine(300.0, 0.002, 0.4, 24_000), 24_000);
        let loud = analyze(&sine(300.0, 0.9, 0.4, 24_000), 24_000);
        let qmax = quiet.iter().map(|f| f.openness).fold(0.0_f32, f32::max);
        let lmax = loud.iter().map(|f| f.openness).fold(0.0_f32, f32::max);
        assert!(lmax >= qmax);
        assert!(qmax < 0.1, "sub-floor signal stays closed");
    }
}
