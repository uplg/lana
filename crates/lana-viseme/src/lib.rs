//! Real-time audio-to-viseme analysis for Lana's lip-sync.
//!
//! Performs short-time FFT, extracts formants F1/F2 and detects bilabial /
//! labio-dental onsets, then maps to 12 `ARKit` visemes (Preston Blair set).
//! Output drives blendshape interpolation in `lana-avatar`. Implemented in
//! Phase 6.

#![forbid(unsafe_code)]
