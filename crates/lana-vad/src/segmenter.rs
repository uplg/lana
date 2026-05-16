//! Utterance segmentation: turn a stream of per-chunk voice/silence flags
//! into start/end-of-utterance events.
//!
//! Pure state machine, no audio dependencies, fully unit-tested. The
//! orchestrator owns the audio buffer; this only signals *when* a turn
//! begins and ends.

/// What the segmenter decided for the chunk just fed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// Still in silence; nothing to do.
    Idle,
    /// First voiced chunk of a new utterance.
    SpeechStarted,
    /// Mid-utterance voiced chunk.
    SpeechContinues,
    /// Trailing-silence chunk while an utterance is still open (audio should
    /// keep being buffered — words can end quietly).
    SpeechTrailing,
    /// Enough trailing silence observed: the utterance is complete.
    SpeechEnded,
}

/// Tunable segmentation behaviour.
#[derive(Debug, Clone, Copy)]
pub struct SegmenterConfig {
    /// Consecutive silent chunks required to close an utterance. At 256 ms
    /// per chunk, 4 ≈ 1.0 s of trailing silence — long enough to ride over
    /// natural mid-sentence pauses without clipping word endings.
    pub silence_chunks_to_end: u32,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            silence_chunks_to_end: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silence,
    Speech,
}

/// Drives [`VadEvent`]s from a sequence of voice-active booleans.
#[derive(Debug)]
#[expect(
    missing_copy_implementations,
    reason = "stateful; making it Copy would invite silent state loss on accidental copies"
)]
pub struct UtteranceSegmenter {
    config: SegmenterConfig,
    state: State,
    trailing_silence: u32,
}

impl UtteranceSegmenter {
    #[must_use]
    pub const fn new(config: SegmenterConfig) -> Self {
        Self {
            config,
            state: State::Silence,
            trailing_silence: 0,
        }
    }

    /// Feed the voice-active flag for one chunk and get the decision.
    pub const fn push(&mut self, voice_active: bool) -> VadEvent {
        match self.state {
            State::Silence => {
                if voice_active {
                    self.state = State::Speech;
                    self.trailing_silence = 0;
                    VadEvent::SpeechStarted
                } else {
                    VadEvent::Idle
                }
            }
            State::Speech => {
                if voice_active {
                    self.trailing_silence = 0;
                    VadEvent::SpeechContinues
                } else {
                    self.trailing_silence = self.trailing_silence.saturating_add(1);
                    if self.trailing_silence >= self.config.silence_chunks_to_end {
                        self.state = State::Silence;
                        self.trailing_silence = 0;
                        VadEvent::SpeechEnded
                    } else {
                        VadEvent::SpeechTrailing
                    }
                }
            }
        }
    }

    /// Force-close any open utterance (e.g. on shutdown). Returns
    /// [`VadEvent::SpeechEnded`] if one was open, else [`VadEvent::Idle`].
    pub fn flush(&mut self) -> VadEvent {
        if self.state == State::Speech {
            self.state = State::Silence;
            self.trailing_silence = 0;
            VadEvent::SpeechEnded
        } else {
            VadEvent::Idle
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SegmenterConfig, UtteranceSegmenter, VadEvent};

    fn seg() -> UtteranceSegmenter {
        UtteranceSegmenter::new(SegmenterConfig {
            silence_chunks_to_end: 3,
        })
    }

    #[test]
    fn silence_stays_idle() {
        let mut s = seg();
        assert_eq!(s.push(false), VadEvent::Idle);
        assert_eq!(s.push(false), VadEvent::Idle);
    }

    #[test]
    fn full_utterance_lifecycle() {
        let mut s = seg();
        assert_eq!(s.push(true), VadEvent::SpeechStarted);
        assert_eq!(s.push(true), VadEvent::SpeechContinues);
        assert_eq!(s.push(false), VadEvent::SpeechTrailing);
        assert_eq!(s.push(false), VadEvent::SpeechTrailing);
        assert_eq!(s.push(false), VadEvent::SpeechEnded);
        // Back to idle afterwards.
        assert_eq!(s.push(false), VadEvent::Idle);
    }

    #[test]
    fn brief_silence_does_not_end_utterance() {
        let mut s = seg();
        assert_eq!(s.push(true), VadEvent::SpeechStarted);
        assert_eq!(s.push(false), VadEvent::SpeechTrailing);
        assert_eq!(s.push(false), VadEvent::SpeechTrailing);
        // Voice returns before the 3-chunk threshold: utterance survives,
        // trailing counter resets.
        assert_eq!(s.push(true), VadEvent::SpeechContinues);
        assert_eq!(s.push(false), VadEvent::SpeechTrailing);
        assert_eq!(s.push(false), VadEvent::SpeechTrailing);
        assert_eq!(s.push(false), VadEvent::SpeechEnded);
    }

    #[test]
    fn flush_closes_open_utterance() {
        let mut s = seg();
        assert_eq!(s.push(true), VadEvent::SpeechStarted);
        assert_eq!(s.flush(), VadEvent::SpeechEnded);
        assert_eq!(s.flush(), VadEvent::Idle);
    }
}
