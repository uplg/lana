# Lana

Local-only conversational voice agent. You speak, Lana answers out loud, in
French (primary) or English. A real-time lip-syncing 3D avatar and local
tool-calling (e.g. driving your own home API to switch the lights — still
100 % local, just an HTTP call on your network) are planned milestones; see
[PLAN.md](./PLAN.md) §8. Nothing leaves the machine — no cloud, no
telemetry, no Python.

Target hardware: MacBook Pro M1 Max 32 GB. The machine stays fully usable
for other work while Lana runs (runtime footprint ≈ 2 GB).

## Stack (current)

| Layer | Choice |
|---|---|
| Capture | `cpal` (CoreAudio) + custom windowed-sinc FIR decimator 48→16 kHz |
| VAD | `earshot` — pure-Rust NN VAD, no ONNX, no model download |
| STT | Parakeet-TDT-0.6B-v3 via `parakeet-rs` (ONNX Runtime / `ort`, CPU EP, pure Rust — **no Swift**) |
| LLM | Luth-LFM2-1.2B (French-specialised Liquid LFM2) Q8_0 GGUF via `candle` (Metal) |
| TTS | Kyutai Pocket TTS, native Rust port (vendored `babybirdprd/pocket-tts` on `candle`/Metal), French `french_24l` + real **Estelle** voice |
| Lip-sync | Real-time FFT + formants → 12 ARKit visemes *(not started)* |
| Avatar | Bevy + `bevy_vrm` *(not started)* |
| UI | `bevy_egui` overlay *(not started)* |
| Orchestrator | Tokio state machine: streaming TTS, conversation memory, barge-in |

No Python. No Swift. No cloud. No telemetry.

## Workspace layout

```
crates/
├── lana-audio          # mic capture, FIR decimator, cancellable playback
├── lana-vad            # voice activity detection (earshot) + utterance segmenter
├── lana-stt            # speech-to-text (Parakeet via parakeet-rs / ort)
├── lana-llm            # local LLM (candle + Qwen3 GGUF), streaming, memory
├── lana-tts            # text-to-speech (native Pocket TTS), streaming
├── lana-viseme         # audio-to-viseme analysis (stub)
├── lana-avatar         # VRM rendering, blendshape control (stub)
├── lana-ui             # in-app egui overlay (stub)
├── lana-orchestrator   # state machine, channels, barge-in
└── lana-app            # binary: wires everything

vendor/
└── pocket-tts          # vendored babybirdprd/pocket-tts, patched to Kyutai
                         # upstream parity (#155 multilingual + voice-embedding
                         # bridge); its own Cargo workspace, path dependency
```

## Build

Requires Rust 1.85+ (Edition 2024).

```sh
cargo build --release
```

## Run

The LLM (Luth-LFM2-1.2B) and TTS (Pocket TTS `french_24l` + the Estelle
voice) are **downloaded from Hugging Face on first launch and cached** —
nothing to fetch by hand, no HF token needed (public repos). Only the STT
model dir is a required local path for now.

```sh
export LANA_STT_MODEL_DIR="$HOME/Library/Application Support/Lana/stt"   # Parakeet ONNX dir

# Optional: override the LLM with a local GGUF + tokenizer.json instead of
# the auto-downloaded Luth-LFM2-1.2B (kurakurai/Luth-LFM2-1.2B-GGUF, Q8_0):
# export LANA_MODEL_PATH=/path/to/model.gguf
# export LANA_TOKENIZER_PATH=/path/to/tokenizer.json

# Full voice loop (mic → STT → LLM → TTS → speaker), run in release for realtime:
cargo run --release --bin lana -- converse

# One-shots:
cargo run --release --bin lana -- chat                       # text REPL
cargo run --release --bin lana -- transcribe <in.wav>        # STT
cargo run --release --bin lana -- synth "Bonjour" out.wav    # TTS
```

Voice override (optional): `LANA_TTS_VOICE_EMBEDDING` (Kyutai predefined
embedding, path or `hf://…`), `LANA_TTS_VOICE_PROMPT` (an `audio_prompt`
safetensors), or `LANA_TTS_CLONE_WAV` (clone from a WAV — needs the gated
voice-cloning weights). Default is the real French Estelle. `LANA_BARGEIN=1`
enables barge-in (headphones / AEC only).

## Development

Strict lints (Edition 2024, clippy pedantic + nursery, `unwrap_used`/`panic`
denied). The vendored `pocket-tts` crate is third-party and keeps its own
allowance, so the workspace gate scopes to the `lana-*` crates (a plain
`cargo clippy --workspace --all-targets` would cross into the vendored
workspace and try to build `pocket-tts-cli`, whose `build.rs` needs web
assets Lana never uses):

```sh
cargo fmt --all --check
cargo clippy -p lana-audio -p lana-vad -p lana-stt -p lana-llm -p lana-tts \
             -p lana-viseme -p lana-avatar -p lana-ui -p lana-orchestrator \
             -p lana-app --all-targets --no-deps -- -D warnings
cargo test --workspace
cargo deny check
```

## License

Dual-licensed under MIT or Apache-2.0 at your option.
