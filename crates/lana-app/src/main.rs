//! Lana — entry point.
//!
//! Two subcommands for now:
//!
//! ```text
//! lana                       # interactive chat REPL (Phase 1)
//! lana transcribe <path.wav> # one-shot STT smoke test (Phase 2)
//! ```
//!
//! Chat mode reads the model from `LANA_MODEL_PATH` and tokenizer from
//! `LANA_TOKENIZER_PATH`. STT reads its ONNX model dir from
//! `LANA_STT_MODEL_DIR`, TTS its voice from `LANA_TTS_VOICE_PROMPT` or
//! `LANA_TTS_CLONE_WAV`. Entirely Rust: candle (LLM/TTS) + ONNX Runtime
//! (STT) + earshot (VAD). No Swift, no `CoreML`.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use lana_audio::{AudioOutput, MicCapture};
use lana_llm::{EngineConfig, GenerationConfig, LlmEngine, Message};
use lana_orchestrator::{Orchestrator, OrchestratorConfig, OrchestratorEvent, Phase};
use lana_stt::{SttConfig, SttEngine};
use lana_tts::{TtsConfig, TtsEngine, VoiceSource};
use lana_vad::{VadConfig, VadEngine};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use tracing::info;
use tracing_subscriber::EnvFilter;

// The system prompt steers Lana for the voice loop:
//
// - `/no_think` suppresses Qwen3's `<think>...</think>` chain-of-thought
//   prefix; the `ThinkFilter` in `lana-llm` is a defensive net should the
//   model ignore the directive on some prompts.
// - The "plain spoken text" instruction discourages markdown formatting
//   (code fences, bullet lists, asterisks) that would read terribly through
//   the TTS layer. A markdown-to-voice post-processor will live in
//   `lana-orchestrator` later for the cases the model ignores this.
// Written in French on purpose: a small model (Qwen3-1.7B) anchors its
// output language to the prompt language, so a French prompt sharply
// reduces the English drift the user reported.
const DEFAULT_SYSTEM_PROMPT: &str = "Tu es Lana, une assistante vocale \
    locale et amicale. Réponds TOUJOURS en français, jamais en anglais, \
    de façon concise et naturelle à l'oral : pas de markdown, pas de \
    listes, pas d'astérisques, pas de blocs de code. Tutoie toujours \
    l'utilisateur, ne le vouvoie jamais.";

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("transcribe") => {
            let path = args
                .next()
                .context("usage: lana transcribe <path-to-audio-file>")?;
            run_transcribe(&PathBuf::from(path)).await
        }
        Some("synth") => {
            let text = args
                .next()
                .context("usage: lana synth <text> <output.wav>")?;
            let out = args
                .next()
                .context("usage: lana synth <text> <output.wav>")?;
            run_synth(&text, &PathBuf::from(out)).await
        }
        // Boxed: builds every engine inline, so the future is large; keep
        // it off the stack.
        Some("converse") => Box::pin(run_converse()).await,
        Some(other) if !["chat", ""].contains(&other) => {
            bail!(
                "unknown subcommand `{other}` \
                 (expected `chat`, `transcribe`, `synth` or `converse`)"
            );
        }
        _ => run_chat().await,
    }
}

fn init_tracing() {
    // Default (no RUST_LOG): keep Lana's own INFO but silence the very
    // chatty ONNX Runtime memory/arena logs (bridged by `ort` under the
    // `ort::*` target) and hf-hub download chatter. An explicit RUST_LOG
    // still wins.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=warn,hf_hub=warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}

// ---- chat (phase 1) -------------------------------------------------------

async fn run_chat() -> Result<()> {
    info!("lana starting (chat repl)");

    let config = load_llm_config_from_env()?;
    let engine = LlmEngine::new(config).await?;
    chat_repl(&engine).await
}

fn load_llm_config_from_env() -> Result<EngineConfig> {
    let model_path = std::env::var("LANA_MODEL_PATH")
        .map(PathBuf::from)
        .context("LANA_MODEL_PATH not set (point it at the Qwen3 GGUF file)")?;
    let tokenizer_path = std::env::var("LANA_TOKENIZER_PATH")
        .map(PathBuf::from)
        .context("LANA_TOKENIZER_PATH not set (point it at the tokenizer.json)")?;

    Ok(EngineConfig {
        model_path,
        tokenizer_path,
        system_prompt: DEFAULT_SYSTEM_PROMPT.to_owned(),
        generation: GenerationConfig::default(),
    })
}

/// The real Kyutai "Estelle" French voice, as a predefined-voice embedding
/// on the (ungated) without-voice-cloning repo. Loaded token-free.
const DEFAULT_FRENCH_ESTELLE_EMBEDDING: &str = "hf://kyutai/pocket-tts-without-voice-cloning/languages/french_24l/embeddings/estelle.safetensors@e041936c75475d350b405bc870bcf7c22da4e9e6";

/// Build the TTS config from the environment. Voice source resolution, in
/// priority order:
///
/// - `LANA_TTS_VOICE_EMBEDDING`: a Kyutai predefined-voice embedding (local
///   path or `hf://repo/path@rev`), e.g. Estelle. Token-free.
/// - `LANA_TTS_VOICE_PROMPT`: path to a `.safetensors` `audio_prompt`.
/// - `LANA_TTS_CLONE_WAV`: path to a reference WAV to clone (needs the
///   voice-cloning checkpoint).
/// - none set: defaults to the real French Estelle embedding.
///
/// `LANA_TTS_EOS_THRESHOLD` (a float) overrides the end-of-speech
/// threshold: more negative = the model keeps speaking longer before it
/// may stop, which reduces sentences being cut off early.
fn tts_config_from_env() -> TtsConfig {
    let voice = std::env::var("LANA_TTS_VOICE_EMBEDDING")
        .ok()
        .map(|embedding| {
            info!(embedding = %embedding, "TTS voice from LANA_TTS_VOICE_EMBEDDING");
            VoiceSource::KyutaiEmbedding(embedding)
        })
        .or_else(|| {
            std::env::var("LANA_TTS_VOICE_PROMPT").ok().map(|prompt| {
                info!(prompt = %prompt, "TTS voice from LANA_TTS_VOICE_PROMPT");
                VoiceSource::PromptFile(PathBuf::from(prompt))
            })
        })
        .or_else(|| {
            std::env::var("LANA_TTS_CLONE_WAV").ok().map(|wav| {
                info!(reference = %wav, "TTS voice cloned from LANA_TTS_CLONE_WAV");
                VoiceSource::CloneWav(PathBuf::from(wav))
            })
        })
        .unwrap_or_else(|| {
            info!(embedding = %DEFAULT_FRENCH_ESTELLE_EMBEDDING, "TTS voice: default French Estelle");
            VoiceSource::KyutaiEmbedding(DEFAULT_FRENCH_ESTELLE_EMBEDDING.to_owned())
        });

    let mut config = TtsConfig::new(voice);
    if let Some(eos) = std::env::var("LANA_TTS_EOS_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        info!(eos_threshold = eos, "TTS eos_threshold overridden");
        config.eos_threshold = eos;
    }
    config
}

async fn chat_repl(engine: &LlmEngine) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    stdout
        .write_all(b"\nLana ready. Type a prompt and press enter (Ctrl-D to quit).\n")
        .await?;

    let mut history: Vec<Message> = Vec::new();

    loop {
        stdout.write_all(b"\n> ").await?;
        stdout.flush().await?;

        let Some(line) = reader.next_line().await? else {
            stdout.write_all(b"\nbye\n").await?;
            return Ok(());
        };
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }

        chat_turn(engine, prompt, &mut stdout, &mut history).await?;
    }
}

async fn chat_turn(
    engine: &LlmEngine,
    prompt: &str,
    stdout: &mut tokio::io::Stdout,
    history: &mut Vec<Message>,
) -> Result<()> {
    let started = Instant::now();
    history.push(Message::user(prompt));
    let mut rx = engine.stream(history);

    let mut ttft: Option<Duration> = None;
    let mut output_chars: usize = 0;
    let mut reply = String::new();

    while let Some(chunk) = rx.recv().await {
        if chunk.is_final {
            break;
        }
        if ttft.is_none() {
            ttft = Some(started.elapsed());
        }
        stdout.write_all(chunk.text.as_bytes()).await?;
        stdout.flush().await?;
        output_chars = output_chars.saturating_add(chunk.text.len());
        reply.push_str(&chunk.text);
    }

    let reply = reply.trim();
    if !reply.is_empty() {
        history.push(Message::assistant(reply));
    }

    let elapsed = started.elapsed();
    let ttft_ms = ttft.unwrap_or(elapsed).as_millis();
    let elapsed_ms = elapsed.as_millis();
    let line = format!("\n[ttft={ttft_ms}ms total={elapsed_ms}ms chars={output_chars}]\n");
    stdout.write_all(line.as_bytes()).await?;
    stdout.flush().await?;
    Ok(())
}

// ---- transcribe (phase 2 smoke test) --------------------------------------

fn stt_config_from_env() -> Result<SttConfig> {
    let model_dir = std::env::var("LANA_STT_MODEL_DIR")
        .map(PathBuf::from)
        .context(
            "LANA_STT_MODEL_DIR not set (dir with Parakeet-TDT v3 ONNX files: \
             encoder-model.onnx, encoder-model.onnx.data, \
             decoder_joint-model.onnx, vocab.txt — from HF \
             istupakov/parakeet-tdt-0.6b-v3-onnx)",
        )?;
    Ok(SttConfig { model_dir })
}

/// Decode any hound-readable WAV to mono `f32`, returning samples + the
/// original sample rate (parakeet-rs resamples internally).
fn decode_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    let channel_count = spec.channels.max(1);
    let channels = usize::from(channel_count);
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0_f32 / f32::from(i16::MAX);
            reader
                .samples::<i16>()
                .map(|s| s.map(|v| f32::from(v) * scale))
                .collect::<std::result::Result<_, _>>()
                .context("read wav samples")?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .context("read wav samples")?,
    };
    let mono = if channels <= 1 {
        raw
    } else {
        // `u16 -> f32` is lossless (no cast lint); channel count is tiny.
        let inv = 1.0_f32 / f32::from(channel_count);
        raw.chunks_exact(channels)
            .map(|f| f.iter().sum::<f32>() * inv)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

async fn run_transcribe(path: &Path) -> Result<()> {
    info!(audio = %path.display(), "lana starting (transcribe one-shot)");

    if !path.exists() {
        bail!("audio file not found: {}", path.display());
    }

    let (samples, rate) = decode_wav(path)?;
    let engine = SttEngine::new(stt_config_from_env()?).await?;
    let started = Instant::now();
    let transcript = engine.transcribe_samples(samples, rate, 1).await?;
    let wall = started.elapsed();

    let mut stdout = tokio::io::stdout();
    let line = format!(
        "\n{}\n[wall={:.2}s]\n",
        transcript.text.trim(),
        wall.as_secs_f64(),
    );
    stdout.write_all(line.as_bytes()).await?;
    stdout.flush().await?;
    Ok(())
}

// ---- synth (phase 3 smoke test) -------------------------------------------

async fn run_synth(text: &str, out: &Path) -> Result<()> {
    info!(out = %out.display(), "lana starting (synth one-shot)");

    let engine = TtsEngine::new(tts_config_from_env()).await?;
    let started = Instant::now();
    let speech = engine.synthesize(text).await?;
    let wall = started.elapsed();

    tokio::fs::write(out, &speech.wav)
        .await
        .with_context(|| format!("writing wav to {}", out.display()))?;

    let bytes = speech.wav.len();
    let mut stdout = tokio::io::stdout();
    let line = format!(
        "\nwrote {} ({bytes} bytes) in {:.2}s\nplay it with: afplay {}\n",
        out.display(),
        wall.as_secs_f64(),
        out.display(),
    );
    stdout.write_all(line.as_bytes()).await?;
    stdout.flush().await?;
    Ok(())
}

// ---- converse (phase 4: full voice loop) ----------------------------------

async fn run_converse() -> Result<()> {
    info!("lana starting (converse — full voice loop)");

    // Build every engine first (each loads its own model). STT/TTS/VAD pull
    // their CoreML models on first run; the LLM reads the GGUF from env.
    let llm = LlmEngine::new(load_llm_config_from_env()?).await?;
    let stt = SttEngine::new(stt_config_from_env()?).await?;
    let tts = TtsEngine::new(tts_config_from_env()).await?;
    let vad = VadEngine::new(VadConfig::default()).await?;
    let out = AudioOutput::start().map_err(|e| anyhow::anyhow!("audio output: {e}"))?;

    let (mic, mic_rx) = MicCapture::start().map_err(|e| anyhow::anyhow!("mic: {e}"))?;

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<OrchestratorEvent>(64);

    // Print events as they arrive.
    let printer = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(ev) = event_rx.recv().await {
            let line = match ev {
                OrchestratorEvent::Phase(p) => format!("[{}]\n", phase_label(p)),
                OrchestratorEvent::UserSaid(t) => format!("you: {t}\n"),
                OrchestratorEvent::LanaSaid(t) => format!("lana: {t}\n"),
                OrchestratorEvent::Notice(n) => format!("· {n}\n"),
                OrchestratorEvent::Error(e) => format!("! {e}\n"),
            };
            let _ = stdout.write_all(line.as_bytes()).await;
            let _ = stdout.flush().await;
        }
    });

    {
        let mut stdout = tokio::io::stdout();
        stdout
            .write_all(b"\nLana is listening. Speak (Ctrl-C to quit).\n")
            .await?;
        stdout.flush().await?;
    }

    let mut orch_cfg = OrchestratorConfig::default();
    if std::env::var("LANA_BARGEIN").is_ok_and(|v| v == "1") {
        info!("barge-in enabled (use headphones — speaker echo will false-trigger it)");
        orch_cfg.allow_bargein = true;
    }
    let orchestrator = Orchestrator::new(stt, llm, tts, vad, out);
    let result = orchestrator.run(mic_rx, event_tx, orch_cfg).await;

    drop(mic);
    printer.abort();
    result.map_err(|e| anyhow::anyhow!("orchestrator stopped: {e}"))
}

const fn phase_label(phase: Phase) -> &'static str {
    match phase {
        Phase::Idle => "idle",
        Phase::Listening => "listening…",
        Phase::Thinking => "thinking…",
        Phase::Speaking => "speaking…",
    }
}
