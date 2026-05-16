//! Conversational orchestrator: the state machine that turns a live
//! microphone into a spoken dialogue with Lana.
//!
//! Flow: mic chunks → VAD → utterance segmentation → STT → LLM → TTS →
//! speaker. While Lana is speaking the loop keeps running VAD so the user
//! can interrupt (barge-in): detected speech cancels playback and starts a
//! fresh turn immediately.
//!
//! Engines are injected pre-built (each owns its own model instance) so
//! the orchestrator stays free of model-loading concerns and VAD can run
//! concurrently with TTS for barge-in.

#![forbid(unsafe_code)]

mod error;

use std::time::{Duration, Instant};

use lana_audio::AudioOutput;
use lana_llm::{LlmEngine, Message};
use lana_stt::SttEngine;
use lana_tts::TtsEngine;
use lana_vad::{SegmenterConfig, UtteranceSegmenter, VadEngine, VadEvent};
use tokio::sync::mpsc;
use tracing::{info, warn};

pub use error::OrchestratorError;

/// Conversation turns kept as LLM memory. Bounded so a long session does
/// not grow the prompt without limit (the system prompt is always kept,
/// separately, by the engine).
const MAX_HISTORY_MSGS: usize = 24;

/// Coarse conversational phase, surfaced to the UI/CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Waiting for the user to start speaking.
    Idle,
    /// Capturing the user's utterance.
    Listening,
    /// Running STT + LLM.
    Thinking,
    /// Playing Lana's reply (interruptible).
    Speaking,
}

/// Events emitted to the front-end (CLI for now, avatar later).
#[derive(Debug, Clone)]
pub enum OrchestratorEvent {
    /// The conversational phase changed.
    Phase(Phase),
    /// Final transcription of the user's turn.
    UserSaid(String),
    /// Lana's textual reply (before/while it is spoken).
    LanaSaid(String),
    /// Lip-sync timeline for the sentence about to play (one entry per
    /// `speak_chunk`); the avatar concatenates successive timelines so the
    /// mouth tracks the continuous speaker output.
    Visemes(Vec<lana_viseme::VisemeFrame>),
    /// Non-fatal notice (empty transcript, barge-in, ...).
    Notice(String),
    /// A per-turn error; the loop continues.
    Error(String),
}

/// Tunables for the loop.
#[derive(Debug, Clone, Copy)]
pub struct OrchestratorConfig {
    /// Utterance segmentation behaviour.
    pub segmenter: SegmenterConfig,
    /// Playback poll interval.
    pub bargein_poll: Duration,
    /// Enable barge-in (interrupt Lana by speaking). Default **off**:
    /// without acoustic echo cancellation, speaker output leaks into the
    /// mic and the VAD would treat Lana's own voice as an interruption.
    /// Only sensible with headphones or a future AEC stage.
    pub allow_bargein: bool,
    /// When barge-in is enabled, how many *consecutive* voiced chunks are
    /// required to trip it (debounces brief noise blips).
    pub bargein_consecutive: u32,
    /// After playback finishes, keep discarding mic input for this long so
    /// the residual acoustic echo tail does not seed a phantom turn.
    pub post_speech_grace: Duration,
    /// Max words buffered before a streaming-TTS chunk is flushed when no
    /// Safety cap: if the model produces this many words with no sentence
    /// terminator, flush anyway so the buffer cannot grow unbounded. Set
    /// high — normal flushing is per sentence (good prosody); this only
    /// catches pathological unpunctuated output.
    pub tts_max_words: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            segmenter: SegmenterConfig::default(),
            bargein_poll: Duration::from_millis(40),
            allow_bargein: false,
            bargein_consecutive: 3,
            post_speech_grace: Duration::from_millis(300),
            tts_max_words: 40,
        }
    }
}

/// Split off a speakable chunk: a full sentence.
///
/// Flushing only on sentence terminators keeps each TTS call prosodically
/// whole (the choppy/word-level approach reset intonation per chunk and,
/// with a cloned voice, re-ran the voice prefill every few words —
/// unusable). Streaming is still preserved: the first sentence plays while
/// later sentences are generated. `max_words` is only a safety valve for
/// pathological unpunctuated model output.
fn take_speakable_chunk(buf: &mut String, max_words: usize) -> Option<String> {
    const TERMINATORS: [char; 5] = ['.', '!', '?', '\n', '…'];

    if let Some(idx) = buf.rfind(TERMINATORS) {
        // Flush up to and including the terminator.
        let end = idx
            .checked_add(buf[idx..].chars().next().map_or(1, char::len_utf8))
            .unwrap_or(buf.len())
            .min(buf.len());
        let chunk = buf[..end].trim().to_owned();
        buf.drain(..end);
        return (!chunk.is_empty()).then_some(chunk);
    }

    if buf.split_whitespace().count() >= max_words {
        // No terminator yet but enough words: flush whole words, keep any
        // trailing partial token for the next round.
        let trimmed_end = buf.trim_end();
        let split_at = trimmed_end.rfind(char::is_whitespace)?;
        let chunk = buf[..split_at].trim().to_owned();
        buf.drain(..split_at);
        return (!chunk.is_empty()).then_some(chunk);
    }

    None
}

/// Owns the engines and runs the conversational loop.
#[derive(Debug)]
pub struct Orchestrator {
    stt: SttEngine,
    llm: LlmEngine,
    tts: TtsEngine,
    vad: VadEngine,
    out: AudioOutput,
}

/// How a speaking turn finished.
enum TurnEnd {
    /// Playback drained normally (or mic closed).
    Completed,
    /// The user interrupted; carries the mic chunk that triggered it so it
    /// seeds the next utterance.
    BargedIn(Vec<f32>),
}

impl Orchestrator {
    #[must_use]
    pub const fn new(
        stt: SttEngine,
        llm: LlmEngine,
        tts: TtsEngine,
        vad: VadEngine,
        out: AudioOutput,
    ) -> Self {
        Self {
            stt,
            llm,
            tts,
            vad,
            out,
        }
    }

    /// Run until the microphone stream closes.
    ///
    /// `mic` yields 16 kHz mono chunks (see `lana_audio::MicCapture`).
    /// `events` receives [`OrchestratorEvent`]s for display.
    ///
    /// # Errors
    ///
    /// Returns [`OrchestratorError::MicClosed`] when the mic stream ends.
    pub async fn run(
        &self,
        mut mic: mpsc::Receiver<Vec<f32>>,
        events: mpsc::Sender<OrchestratorEvent>,
        config: OrchestratorConfig,
    ) -> Result<(), OrchestratorError> {
        let mut segmenter = UtteranceSegmenter::new(config.segmenter);
        let mut utterance: Vec<f32> = Vec::new();
        let mut history: Vec<Message> = Vec::new();
        emit(&events, OrchestratorEvent::Phase(Phase::Idle)).await;

        // A barge-in hands back the chunk that interrupted Lana; it seeds
        // the next utterance instead of being dropped.
        let mut seed: Option<Vec<f32>> = None;

        loop {
            let chunk = if let Some(seed_chunk) = seed.take() {
                seed_chunk
            } else {
                match mic.recv().await {
                    Some(c) => c,
                    None => return Err(OrchestratorError::MicClosed),
                }
            };

            let voiced = self.vad.voice_active(chunk.clone()).await.unwrap_or(false);

            match segmenter.push(voiced) {
                VadEvent::Idle => {}
                VadEvent::SpeechStarted => {
                    utterance.clear();
                    utterance.extend_from_slice(&chunk);
                    emit(&events, OrchestratorEvent::Phase(Phase::Listening)).await;
                }
                VadEvent::SpeechContinues | VadEvent::SpeechTrailing => {
                    utterance.extend_from_slice(&chunk);
                }
                VadEvent::SpeechEnded => {
                    utterance.extend_from_slice(&chunk);
                    let audio = std::mem::take(&mut utterance);
                    match self
                        .handle_turn(audio, &mut mic, &events, &config, &mut history)
                        .await
                    {
                        TurnEnd::Completed => {
                            emit(&events, OrchestratorEvent::Phase(Phase::Idle)).await;
                        }
                        TurnEnd::BargedIn(barge_chunk) => {
                            segmenter.flush();
                            emit(&events, OrchestratorEvent::Notice("barge-in".to_owned())).await;
                            seed = Some(barge_chunk);
                        }
                    }
                }
            }
        }
    }

    /// STT → LLM → TTS → speak, watching for barge-in while speaking.
    async fn handle_turn(
        &self,
        audio: Vec<f32>,
        mic: &mut mpsc::Receiver<Vec<f32>>,
        events: &mpsc::Sender<OrchestratorEvent>,
        cfg: &OrchestratorConfig,
        history: &mut Vec<Message>,
    ) -> TurnEnd {
        emit(events, OrchestratorEvent::Phase(Phase::Thinking)).await;

        // The capture pipeline delivers 16 kHz mono.
        let transcript = match self.stt.transcribe_samples(audio, 16_000, 1).await {
            Ok(t) => t.text.trim().to_owned(),
            Err(e) => {
                emit(events, OrchestratorEvent::Error(format!("stt: {e}"))).await;
                return TurnEnd::Completed;
            }
        };
        if transcript.is_empty() {
            emit(
                events,
                OrchestratorEvent::Notice("empty transcript".to_owned()),
            )
            .await;
            return TurnEnd::Completed;
        }
        emit(events, OrchestratorEvent::UserSaid(transcript.clone())).await;

        // Record the user turn so Lana has conversational memory.
        history.push(Message::user(transcript));

        // Stream the LLM over the whole history, and synthesise+enqueue
        // each speakable chunk as soon as it forms so the first audio
        // starts long before the full reply is generated.
        let mut rx = self.llm.stream(history);
        let mut buf = String::new();
        let mut full = String::new();
        let mut spoke = false;

        while let Some(tok) = rx.recv().await {
            if tok.is_final {
                break;
            }
            buf.push_str(&tok.text);
            full.push_str(&tok.text);
            while let Some(chunk) = take_speakable_chunk(&mut buf, cfg.tts_max_words) {
                if !spoke {
                    emit(events, OrchestratorEvent::Phase(Phase::Speaking)).await;
                    spoke = true;
                }
                self.speak_chunk(&chunk, events).await;
            }
        }
        // Flush whatever is left after the stream ends.
        let tail = buf.trim().to_owned();
        if !tail.is_empty() {
            if !spoke {
                emit(events, OrchestratorEvent::Phase(Phase::Speaking)).await;
                spoke = true;
            }
            self.speak_chunk(&tail, events).await;
        }

        let full = full.trim().to_owned();
        if full.is_empty() || !spoke {
            emit(events, OrchestratorEvent::Notice("empty reply".to_owned())).await;
            return TurnEnd::Completed;
        }

        // Record Lana's reply (even if playback is later interrupted: she
        // did "say" it textually) and bound the running history.
        history.push(Message::assistant(full.clone()));
        let overflow = history.len().saturating_sub(MAX_HISTORY_MSGS);
        if overflow > 0 {
            history.drain(0..overflow);
        }

        emit(events, OrchestratorEvent::LanaSaid(full)).await;

        if cfg.allow_bargein {
            self.speak_with_bargein(mic, cfg).await
        } else {
            self.speak_half_duplex(mic, cfg).await
        }
    }

    /// Synthesise one full sentence and enqueue it as one contiguous block.
    /// Per-sentence (not per-frame): the first sentence still plays while
    /// later sentences generate, but a sentence never starves mid-playback
    /// — frame-level streaming made the audio cut constantly. A failure is
    /// reported as a notice and skipped so the rest of the reply still plays.
    async fn speak_chunk(&self, chunk: &str, events: &mpsc::Sender<OrchestratorEvent>) {
        match self.tts.synthesize(chunk).await {
            Ok(speech) => {
                // Derive the lip-sync timeline from the exact PCM about to
                // play and hand it to the avatar *before* enqueuing the
                // audio, so the mouth starts with the sound.
                let visemes = lana_viseme::analyze(&speech.pcm, speech.sample_rate);
                if !visemes.is_empty() {
                    emit(events, OrchestratorEvent::Visemes(visemes)).await;
                }
                if let Err(e) = self.out.enqueue_wav(&speech.wav) {
                    emit(events, OrchestratorEvent::Error(format!("audio: {e}"))).await;
                }
            }
            Err(e) => {
                emit(events, OrchestratorEvent::Notice(format!("tts skip: {e}"))).await;
            }
        }
    }

    /// Default: Lana does not listen while she speaks. Mic input is drained
    /// and discarded (so the bounded channel never stalls capture and
    /// Lana's own echo never becomes a phantom turn), then discarded for a
    /// short grace window after playback to swallow the echo tail.
    async fn speak_half_duplex(
        &self,
        mic: &mut mpsc::Receiver<Vec<f32>>,
        cfg: &OrchestratorConfig,
    ) -> TurnEnd {
        while self.out.is_playing() {
            while mic.try_recv().is_ok() {}
            tokio::time::sleep(cfg.bargein_poll).await;
        }
        let grace_end = Instant::now()
            .checked_add(cfg.post_speech_grace)
            .unwrap_or_else(Instant::now);
        while Instant::now() < grace_end {
            while mic.try_recv().is_ok() {}
            tokio::time::sleep(cfg.bargein_poll).await;
        }
        TurnEnd::Completed
    }

    /// Barge-in mode (headphones / future AEC): interrupt only after
    /// `bargein_consecutive` *consecutive* voiced chunks so a single noise
    /// blip does not cut Lana off.
    async fn speak_with_bargein(
        &self,
        mic: &mut mpsc::Receiver<Vec<f32>>,
        cfg: &OrchestratorConfig,
    ) -> TurnEnd {
        let mut consecutive: u32 = 0;
        loop {
            tokio::select! {
                biased;
                maybe = mic.recv() => {
                    let Some(chunk) = maybe else {
                        return TurnEnd::Completed;
                    };
                    if self.vad.voice_active(chunk.clone()).await.unwrap_or(false) {
                        consecutive = consecutive.saturating_add(1);
                        if consecutive >= cfg.bargein_consecutive {
                            self.out.cancel();
                            info!(consecutive, "barge-in: user interrupted playback");
                            return TurnEnd::BargedIn(chunk);
                        }
                    } else {
                        consecutive = 0;
                    }
                }
                () = tokio::time::sleep(cfg.bargein_poll) => {
                    if !self.out.is_playing() {
                        return TurnEnd::Completed;
                    }
                }
            }
        }
    }
}

async fn emit(events: &mpsc::Sender<OrchestratorEvent>, ev: OrchestratorEvent) {
    if events.send(ev).await.is_err() {
        warn!("event receiver dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::take_speakable_chunk;

    #[test]
    fn flushes_on_terminator() {
        let mut b = String::from("Bonjour, je suis Lana. Et toi");
        assert_eq!(
            take_speakable_chunk(&mut b, 6).as_deref(),
            Some("Bonjour, je suis Lana.")
        );
        assert_eq!(b, " Et toi");
    }

    #[test]
    fn flushes_on_word_budget_without_terminator() {
        let mut b = String::from("un deux trois quatre cinq six sept hu");
        assert_eq!(
            take_speakable_chunk(&mut b, 6).as_deref(),
            Some("un deux trois quatre cinq six sept")
        );
        assert_eq!(b, " hu");
    }

    #[test]
    fn holds_until_enough() {
        let mut b = String::from("un deux trois");
        assert_eq!(take_speakable_chunk(&mut b, 6), None);
        assert_eq!(b, "un deux trois");
    }
}
