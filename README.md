# Lana

Local-only conversational AI agent with a 3D avatar that lip-syncs in real time. French (primary) and English.

Target hardware: MacBook Pro M1 Max 32 GB. The machine must remain fully usable for other work while Lana is running. Total runtime memory footprint: ~2 GB.

See [PLAN.md](./PLAN.md) for the full architecture, model choices and phased delivery.

## Stack

| Layer | Choice |
|---|---|
| Capture | `cpal` (CoreAudio) |
| VAD | Silero v5 (ONNX via `ort`) |
| STT | Parakeet-TDT-0.6B-v3 (CoreML / Neural Engine via Swift bridge) |
| LLM | Gemma 4 E2B UQFF Q5K (Metal via `mistral.rs`) |
| TTS | Kyutai Pocket TTS (MLX int4 via `mlx-rs`) |
| Lip-sync | Real-time FFT + formants → 12 ARKit visemes |
| Avatar | Bevy + `bevy_vrm` |
| UI | `bevy_egui` overlay |

No Python. No cloud. No telemetry.

## Workspace layout

```
crates/
├── lana-audio          # microphone capture, ring buffers
├── lana-vad            # voice activity detection
├── lana-stt            # speech-to-text (Parakeet via Swift FFI)
├── lana-llm            # local LLM inference
├── lana-tts            # text-to-speech (Kyutai Pocket TTS)
├── lana-viseme         # audio-to-viseme analysis
├── lana-avatar         # VRM rendering, blendshape control
├── lana-ui             # in-app egui overlay
├── lana-orchestrator   # state machine, channels, barge-in
└── lana-app            # binary: wires everything
```

## Build

Requires Rust 1.85+ (Edition 2024).

```sh
cargo build --release
```

## Run

```sh
cargo run --release --bin lana
```

Models are downloaded to `~/Library/Application Support/Lana/models/` on first launch.

## Development

The workspace enforces strict lints. Before pushing:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo deny check
cargo audit
```

CI runs all of the above on `macos-latest`.

## License

Dual-licensed under MIT or Apache-2.0 at your option.
