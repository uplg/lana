# Lana

Local-only conversational voice agent. You speak, Lana answers out loud, in
French (primary) or English, as a dark, glowing **point-cloud avatar** in a
native window — a Cyberpunk-2077-braindance-style "scan hologram": the
vertices of a 3D model (drop a `.glb`/`.vrm`/`.pcd` in the folder) rendered
as glowing points with a scan sweep, flicker and audio-driven lip-sync.
Planned next:
local tool-calling (Phase 8 — e.g. driving your own home API to switch the
lights, still 100 % local: just an HTTP call on your network) and
cross-session memory (Phase 9). See
[PLAN.md](./PLAN.md). Nothing leaves the machine — no cloud, no telemetry,
no Python.

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
| Lip-sync | `lana-viseme`: short-time FFT energy + F1/F2 formants → vowel/openness; the mouth-region points spread open with the spoken audio |
| Avatar | Bevy 0.18 — **braindance point-cloud**: a model's vertices+normals (`.glb`/`.vrm`/`.pcd`, no skeleton/morphs) as one `PointList` mesh + a custom embedded-WGSL material: colour = the exact vertex normal, HDR→bloom, back-cull, per-point jitter, GPU scan/flicker + glowing mouth-band lip-sync. Orbit camera. Procedural fallback if no model |
| UI | `bevy_egui` 0.39 overlay — phase, rolling transcript, mic-mute toggle |
| Orchestrator | Tokio state machine: streaming TTS, conversation memory, barge-in |

No Python. No Swift. No cloud. No telemetry.

## Workspace layout

```
crates/
├── lana-audio          # mic capture, FIR decimator, cancellable playback
├── lana-vad            # voice activity detection (earshot) + utterance segmenter
├── lana-stt            # speech-to-text (Parakeet via parakeet-rs / ort)
├── lana-llm            # local LLM (candle + Luth-LFM2 GGUF), streaming, memory
├── lana-tts            # text-to-speech (native Pocket TTS), streaming
├── lana-viseme         # audio→viseme DSP (FFT energy + F1/F2), unit-tested
├── lana-avatar         # Bevy procedural point-cloud avatar + egui overlay
├── lana-ui             # in-app egui overlay (stub — lives in lana-avatar)
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

The LLM (Luth-LFM2-1.2B), STT (Parakeet-TDT-0.6B-v3) and TTS (Pocket TTS
`french_24l` + the Estelle voice) are **all downloaded from Hugging Face on
first launch and cached** — nothing to fetch by hand, no HF token needed
(public repos). First run pulls ≈ 1.25 GB (LLM) + ≈ 2.5 GB (STT) + the TTS
model/voice; subsequent runs are instant from cache. **Zero setup:**

`converse` opens the avatar window. The avatar is the **point-cloud scan**
of a 3D model: drop a `.glb`/`.vrm`/`.pcd` anywhere in the working
directory (or set `LANA_AVATAR_MODEL=/path/model.glb`) and its vertices
become the glowing braindance hologram — only positions are read, no rig
or materials. If no model is found it falls back to a procedural cloud.

```sh
# Full voice loop + avatar window (mic → STT → LLM → TTS → speaker + 3D
# point-cloud avatar with a live transcript overlay). Realtime needs an
# optimised build; for fast iteration use `release-fast` (optimised but
# no fat-LTO — links in seconds instead of minutes):
cargo run --profile release-fast --bin lana -- converse
# Ship/benchmark with full optimisation (slow fat-LTO link):
cargo run --release --bin lana -- converse

# One-shots (no window):
cargo run --release --bin lana -- chat                       # text REPL
cargo run --release --bin lana -- transcribe <in.wav>        # STT
cargo run --release --bin lana -- synth "Bonjour" out.wav    # TTS
```

`LANA_AVATAR_MODEL` picks the model file (else the first
`.glb`/`.vrm`/`.pcd` in the folder). It starts face-framed;
**the camera is an interactive orbit**: left-drag to orbit, mouse wheel to
zoom, ↑/↓ to pan the look-at height, and **press `L` to log the current
camera pose** (so you can pin it via `LANA_AVATAR_CAM_DIST` /
`LANA_AVATAR_CAM_Y`). The colour is the exact vertex normal, **glowing
via HDR bloom** over a near-black scene; back-facing points are culled and
a per-point jitter keeps it a living (not uncanny-rigid) cloud. The
lower-face band opens *and* glows with the spoken audio (lip-sync), tunable
on-device: `LANA_MOUTH_Y` / `LANA_MOUTH_H` / `LANA_MOUTH_AMP` and
`LANA_PT_JITTER`. The shader is embedded in the binary (no `assets/` dir).
Optional local overrides (power users):
`LANA_MODEL_PATH` /
`LANA_TOKENIZER_PATH` (LLM GGUF + tokenizer.json), `LANA_STT_MODEL_DIR`
(directory of Parakeet ONNX files). Voice override: `LANA_TTS_VOICE_EMBEDDING`
(Kyutai predefined embedding, path or `hf://…`), `LANA_TTS_VOICE_PROMPT`
(an `audio_prompt` safetensors), or `LANA_TTS_CLONE_WAV` (clone from a WAV —
needs the gated voice-cloning weights). Default voice is the real French
Estelle. `LANA_BARGEIN=1` enables barge-in (headphones / AEC only).

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
