//! Fixed 3:1 decimator (48 kHz → 16 kHz) for the capture path.
//!
//! The validated input device delivers 48 kHz; speech STT/VAD want 16 kHz.
//! That is an exact integer ratio, so the canonical minimal solution is a
//! windowed-sinc low-pass (anti-alias) followed by keeping every third
//! sample. The filter taps are derived from first principles at
//! construction (sinc × Hann, unit DC gain) — no opaque constants — and the
//! filter retains history across calls so block boundaries introduce no
//! discontinuity.

// This module is bounded DSP math: every index is < `NTAPS` (49) so the
// `usize`→`f32` casts are exact, and the coefficient/index arithmetic
// cannot realistically overflow. Checked-arithmetic and FMA lints add
// noise without value to the derivation; the hot convolution path still
// uses `mul_add` explicitly.
#![expect(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "bounded windowed-sinc coefficient math; indices < NTAPS"
)]

/// Decimation factor (48 000 / 16 000).
pub(crate) const DECIM: usize = 3;

/// FIR length. Odd for a symmetric linear-phase filter; 49 taps give a
/// sharp enough transition for speech while staying cheap on the audio
/// worker thread.
const NTAPS: usize = 49;

/// Anti-alias cutoff in Hz. Must sit below the 8 kHz Nyquist of the 16 kHz
/// output, with margin for the transition band.
const CUTOFF_HZ: f32 = 7_200.0;

/// Streaming 3:1 decimator with continuous filter state.
#[derive(Debug)]
pub(crate) struct Decimator3 {
    taps: [f32; NTAPS],
    /// Carry of the last `NTAPS - 1` input samples so convolution windows
    /// straddle call boundaries seamlessly.
    history: Vec<f32>,
    /// Index (in the virtual concatenated stream) of the next sample whose
    /// filtered value is emitted. Steps by `DECIM`, carried across calls.
    phase: usize,
}

impl Decimator3 {
    pub(crate) fn new(input_rate: u32) -> Self {
        // Build a Hann-windowed sinc low-pass. fc is normalised to the
        // input rate; the ideal impulse response is sinc(2*fc*n) sampled
        // around the centre tap.
        let fc = CUTOFF_HZ / input_rate as f32;
        let mut taps = [0.0_f32; NTAPS];
        let center = (NTAPS - 1) / 2;
        let mut sum = 0.0_f32;
        for (n, tap) in taps.iter_mut().enumerate() {
            let k = n as f32 - center as f32;
            let sinc = if k == 0.0 {
                2.0 * fc
            } else {
                let x = core::f32::consts::PI * 2.0 * fc * k;
                (2.0 * fc) * (x.sin() / x)
            };
            let hann = 0.5 - 0.5 * (core::f32::consts::TAU * n as f32 / (NTAPS as f32 - 1.0)).cos();
            let v = sinc * hann;
            *tap = v;
            sum += v;
        }
        // Normalise to unit DC gain so loudness is preserved.
        for tap in &mut taps {
            *tap /= sum;
        }

        Self {
            taps,
            history: vec![0.0; NTAPS - 1],
            phase: center,
        }
    }

    /// Filter and decimate `input`, appending 16 kHz mono samples to `out`.
    pub(crate) fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        // Virtual stream = history ++ input. Outputs are the filtered value
        // centred on indices `phase, phase + DECIM, ...` while a full
        // NTAPS-wide window fits.
        let work_len = self.history.len().saturating_add(input.len());
        if work_len < NTAPS {
            self.history.extend_from_slice(input);
            return;
        }

        let sample = |idx: usize| -> f32 {
            if idx < self.history.len() {
                self.history[idx]
            } else {
                input[idx - self.history.len()]
            }
        };

        let last_start = work_len - NTAPS;
        while self.phase <= last_start {
            let mut acc = 0.0_f32;
            for (k, &tap) in self.taps.iter().enumerate() {
                acc = tap.mul_add(sample(self.phase + k), acc);
            }
            out.push(acc);
            self.phase = self.phase.saturating_add(DECIM);
        }

        // Retain the tail needed for the next call's overlap and rebase
        // `phase` so it stays valid relative to the new history window.
        let keep_from = work_len.saturating_sub(NTAPS - 1);
        let mut new_history = Vec::with_capacity(NTAPS - 1);
        for idx in keep_from..work_len {
            new_history.push(sample(idx));
        }
        self.history = new_history;
        self.phase = self.phase.saturating_sub(keep_from);
    }
}

#[cfg(test)]
mod tests {
    use super::{DECIM, Decimator3};

    #[test]
    fn constant_signal_stays_constant() {
        let mut d = Decimator3::new(48_000);
        let mut out = Vec::new();
        // Feed enough constant samples for the filter to settle.
        for _ in 0..20 {
            d.process(&[1.0_f32; 480], &mut out);
        }
        // After warm-up the DC level must be ~1.0 (unit-gain filter).
        let tail = &out[out.len() - 100..];
        let mean: f32 = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!((mean - 1.0).abs() < 1e-3, "dc gain off: {mean}");
    }

    #[test]
    fn output_rate_is_one_third() {
        let mut d = Decimator3::new(48_000);
        let mut out = Vec::new();
        d.process(&vec![0.0_f32; 48_000], &mut out);
        // ~48000 in → ~16000 out (±a few for filter edges).
        let expected = 48_000 / DECIM;
        assert!(
            out.len().abs_diff(expected) < 64,
            "got {} expected ~{expected}",
            out.len()
        );
    }
}
