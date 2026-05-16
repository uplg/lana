# Lana

Local-only conversational voice agent. You speak, Lana answers out loud, in
French (primary) or English, as a 3D avatar in a native window with
audio-driven lip-sync, idle blink and a portrait framing. Planned next:
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
| Lip-sync | `lana-viseme`: short-time FFT energy + F1/F2 formants → vowel visemes, smoothed onto the avatar's mouth morphs — VRM via its `blendShapeMaster` a/i/u/e/o presets, glTF via morph-target name |
| Avatar | Bevy 0.18 — native window, camera/light, idle sway, audio lip-sync. `LANA_AVATAR_PATH`: realistic `.glb`/`.gltf` (e.g. [Avaturn](https://avaturn.me)) or a `.vrm` (`bevy_vrm` 0.3) |
| UI | `bevy_egui` 0.39 overlay — phase + rolling transcript |
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

The LLM (Luth-LFM2-1.2B), STT (Parakeet-TDT-0.6B-v3) and TTS (Pocket TTS
`french_24l` + the Estelle voice) are **all downloaded from Hugging Face on
first launch and cached** — nothing to fetch by hand, no HF token needed
(public repos). First run pulls ≈ 1.25 GB (LLM) + ≈ 2.5 GB (STT) + the TTS
model/voice; subsequent runs are instant from cache. **Zero setup:**

`converse` opens the avatar window, so it needs an avatar model — set
`LANA_AVATAR_PATH`. For a **realistic human**, export a `.glb` from
[Avaturn](https://avaturn.me) (realistic, rigged, with ARKit
blendshapes/visemes — feeds Phase 6 lip-sync). A `.vrm` (stylised, VRoid)
also works.

```sh
# Full voice loop + avatar window (mic → STT → LLM → TTS → speaker + 3D
# avatar with a live transcript overlay). Run in release for realtime:
LANA_AVATAR_PATH=/path/to/avatar.glb cargo run --release --bin lana -- converse

# One-shots (no window, no avatar needed):
cargo run --release --bin lana -- chat                       # text REPL
cargo run --release --bin lana -- transcribe <in.wav>        # STT
cargo run --release --bin lana -- synth "Bonjour" out.wav    # TTS
```

`LANA_AVATAR_PATH` (required for `converse`): a `.glb`/`.gltf` realistic
avatar or a `.vrm`. `LANA_AVATAR_ROT_Y` (degrees, default `180`): corrective
yaw — VRM 0.x faces away from the camera; set `0` (or another value) if your
model then faces backwards. For a `.vrm`, the camera auto-frames a level
head-and-shoulders portrait on the head bone (the standard talking-avatar
shot) so the bind-pose arms fall out of frame — VRM 0.0 ships no idle
animation and hand-posing the skeleton is not reliable across rigs, so this
is the deterministic choice. Two knobs (the head-bone-to-crown distance
varies per model, so tune to taste): `LANA_AVATAR_CAM_DIST` (metres,
default `1.15`) — raise to pull back / shrink her, lower to come closer;
`LANA_AVATAR_CAM_HEIGHT` (metres, default `0.08`) — vertical aim above the
head bone, raise it if the crown is clipped / to drop her down in frame.
A glTF avatar auto-plays its
first embedded animation clip if it has one and keeps the default camera.
The avatar also blinks on an irregular idle timer.
Lip-sync is automatic from the spoken audio. A `.vrm` is resolved from its
own `blendShapeMaster` (the VRM spec's a/i/u/e/o presets — deterministic,
no per-rig guessing); a glTF avatar is resolved by ARKit/VRoid morph-target
name. Either way, if the mouth can't be resolved the reason is logged (it
is never a silent failure).
Optional local overrides (power users): `LANA_MODEL_PATH` /
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
