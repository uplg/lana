# Lana — Architecture Plan

A **local-only** virtual AI agent, French-first (English supported), wrapped
in a desktop app with a realistic 3D avatar that lip-syncs in real time.

Target hardware: **MacBook Pro M1 Max 32 GB**. Hard constraint: the machine
must stay fully usable for other work while Lana runs.

> **Status (2026-05-16)**: phases 0→4 delivered and validated. Full French
> voice loop works (`lana converse`), real **Estelle** voice, native
> streaming TTS, conversation memory (in-RAM), LLM = Luth-LFM2-1.2B.
> LLM/STT/TTS all auto-download from Hugging Face (zero manual setup).
> Avatar / lip-sync (phases 5+) not started. 100 % Rust stack: no Swift,
> no Python, no cloud.

---

## 1. Product vision

- 3D avatar named **Lana**, rendered in real time
- Natural voice conversation (FR primary, EN supported, transparent switch)
- Lip-sync at 60 fps over 12 ARKit visemes
- Target voice-to-voice latency: **< 1 s end-to-end** (release build)
- 100 % local: no data leaves the machine
- Total memory footprint **< 3 GB**, machine stays responsive

---

## 2. Conversational pipeline (current)

```
microphone (CoreAudio, cpal) — custom FIR decimator 48→16 kHz
  → VAD (earshot, pure-Rust NN) → utterance segmenter
    → STT (Parakeet-TDT v3, parakeet-rs / ONNX Runtime, CPU EP)
      → LLM (Luth-LFM2-1.2B, candle/Metal) — multi-turn history
        → TTS (Kyutai Pocket TTS native Rust, candle/Metal, french_24l + Estelle)
          → streaming per Mimi frame (~80 ms) → speaker (cpal, cancellable)
          └─ (planned) real-time analysis → visemes → VRM blendshapes 60 fps
```

Everything flows through **Tokio channels** between stages. Barge-in
(interrupting Lana by speaking) is implemented but **off by default**
(half-duplex): without acoustic echo cancellation the mic picks up the
speakers. `LANA_BARGEIN=1` enables it (headphones / future AEC).

---

## 3. Technical stack (current)

| Layer | Choice | Why | RAM |
|---|---|---|---|
| **Audio capture** | `cpal` (CoreAudio) + custom windowed-sinc FIR decimator 48→16 kHz | Low latency, pure Rust | — |
| **VAD** | `earshot` (pure-Rust NN, 256-sample / 16 ms frames) | No ONNX, no model download, ~110 KiB | ~0 |
| **STT** | **Parakeet-TDT-0.6B-v3** via `parakeet-rs` (ONNX Runtime `ort`, **CPU** EP) | SOTA multilingual FR STT, pure Rust via a C lib (no Swift). CoreML unstable for this model; CPU is fast on Apple Silicon | ~600 MB (ORT arena) |
| **LLM** | **Luth-LFM2-1.2B** Q8_0 GGUF via `candle` (Metal) — `quantized_lfm2` | Liquid LFM2 French-specialised (SOTA French at this size), tiny (~1.25 GB), no thinking. candle 0.10.2 already ships the `lfm2` arch | ~1.3 GB |
| **TTS** | **Kyutai Pocket TTS** — native Rust port (vendored `babybirdprd/pocket-tts`, candle/Metal), brought to upstream parity (#155) | Real **Estelle** FR voice via a predefined-voice embedding (token-free, ungated repo). Native per-Mimi-frame streaming | ~300 MB |
| **Visemes** | FFT + F1/F2 formants + bilabial onsets → 12 ARKit visemes | *(not started)* pure Rust, ~10 ms latency | — |
| **Avatar** | **Bevy** + `bevy_vrm` (full Rust, wgpu, native window) | *(not started)* 100 % Rust, free VRM models | ~600 MB |
| **UI** | `bevy_egui` overlay | *(not started)* no Tauri/webview | ~0 |
| **Orchestrator** | Tokio, channels, state machine | Idle / Listening / Thinking / Speaking + barge-in + memory | ~100 MB |

Locked-in rationale for each choice: see project memory `lana-stack`,
`lana-phase{1,2,3,4}-baseline`.

---

## 4. Cargo workspace (current)

```
lana/
├── Cargo.toml                 # workspace, strict lints, exclude = ["vendor"]
├── deny.toml                  # cargo-deny (licenses/advisories/sources)
├── crates/
│   ├── lana-audio/            # cpal capture + FIR decimator + cancellable playback (jitter buffer)
│   ├── lana-vad/              # earshot + UtteranceSegmenter
│   ├── lana-stt/              # parakeet-rs (ort), HF auto-download, no Swift
│   ├── lana-llm/              # candle + Luth-LFM2 GGUF (quantized_lfm2), HF auto-download, memory (Message/Role)
│   ├── lana-tts/              # vendored pocket-tts, streaming, Estelle voice
│   ├── lana-viseme/           # (stub)
│   ├── lana-avatar/           # (stub)
│   ├── lana-ui/               # (stub)
│   ├── lana-orchestrator/     # state machine, channels, barge-in, history
│   └── lana-app/              # binary: chat / transcribe / synth / converse
└── vendor/
    └── pocket-tts/            # vendored babybirdprd fork, patched to Kyutai
                               # parity (#155 + voice-embedding bridge). Its
                               # own Cargo workspace, `path` dependency, .git
                               # stripped.
```

Models: **LLM (Luth-LFM2), STT (Parakeet-TDT v3) and TTS (`french_24l` +
Estelle voice) are all downloaded from Hugging Face on first launch
(ungated, no token) and cached** — zero setup, nothing to fetch by hand.
Optional local overrides: `LANA_MODEL_PATH`/`LANA_TOKENIZER_PATH` (LLM),
`LANA_STT_MODEL_DIR` (Parakeet ONNX dir).

---

## 5. Quality gate (Rust 2024, perfect clippy)

Strict workspace lints: `clippy::pedantic` + `nursery`, `unwrap_used` /
`panic` = deny, `arithmetic_side_effects` / `expect_used` = warn, etc.
`#[expect]`/`#[allow]` are a last resort — restructure first.

The vendored `pocket-tts` crate is third-party and keeps its own clippy
allowance, so the workspace gate explicitly targets the `lana-*` crates
(a plain `cargo clippy --workspace --all-targets` crosses into the vendored
workspace and fails on `pocket-tts-cli`, whose `build.rs` needs web assets
Lana never uses).

```sh
cargo fmt --all --check
cargo clippy -p lana-audio -p lana-vad -p lana-stt -p lana-llm -p lana-tts \
             -p lana-viseme -p lana-avatar -p lana-ui -p lana-orchestrator \
             -p lana-app --all-targets --no-deps -- -D warnings
cargo test --workspace
cargo deny check
```

`cargo deny`: `all-features = false` (avoids `intel-mkl-src` via the `mkl`
feature; candle is built with `metal`). `intel-mkl-src` is clarified as
`LicenseRef-Intel-Simplified-Software-License`.

---

## 6. Delivery phases

### ✅ Phase 0 — Skeleton
Workspace, strict lints, clean `cargo deny`, CI with no model downloads.

### ✅ Phase 1 — Text loop
`candle` (Metal) + LLM GGUF, CLI `chat`. **Luth-LFM2-1.2B**
(`quantized_lfm2`) — French-specialised Liquid LFM2, ChatML template
(BOS `<|startoftext|>`), no thinking, implicit state reset via
`index_pos == 0` (no `clear_kv_cache` API). (History: started on a generic
small model that looped/was incoherent in French; the French-specialised
LFM2 fixed it — validated noticeably better in real use.)

### ✅ Phase 2 — Voice in
`cpal` capture 48 kHz → FIR 16 kHz; `earshot` VAD; `parakeet-rs` STT
(ONNX/ort, CPU). CLI `transcribe`.

### ✅ Phase 3 — Voice out
`lana-tts` on the native Rust port of Kyutai Pocket TTS (vendored
babybirdprd). Ported #155 (FR multilingual `french_24l`) + a voice-embedding
bridge (babybirdprd PR #17) → real **Estelle** voice, token-free. Native
per-Mimi-frame streaming. CLI `synth`.

### ✅ Phase 4 — Full loop, no avatar
`lana-orchestrator`: Tokio state machine, barge-in (opt-in), streaming TTS,
**multi-turn conversation memory** (in-RAM, bounded). CLI `converse`.

### ⬜ Phase 5 — Static avatar
`lana-avatar`: Bevy + `bevy_vrm`, load a `.vrm`, camera/lighting, idle.
`bevy_egui` overlay (transcript + settings). Single window.

### ⬜ Phase 6 — Lip-sync
`lana-viseme`: FFT (rustfft) ~50 Hz on the TTS stream, F1/F2 formants,
bilabial onsets → 12 ARKit visemes. Bevy interpolates blendshapes at
60 fps. Hook onto the PCM stream lana-tts already produces.

### ⬜ Phase 7 — Polish
FR/EN voice picker in the UI, VRM hot-reload, idle mode (breathing,
blinking, gaze at camera).

### ⬜ Phase 8 — Tools (function-calling)

Let Lana call local tools (e.g. a home-automation API: lights on/off).
**Stays 100 % local**: a tool is just an HTTP call to an API on the user's
network, no cloud.

- **Good news**: the Luth-LFM2 template has a *native* tools format — a
  tool list injected into the system message between
  `<|tool_list_start|>[…]<|tool_list_end|>`, and tool replies wrapped in
  `<|tool_response_start|>…<|tool_response_end|>` for the `tool` role. No
  protocol to invent.
- The model emits a tool call; the orchestrator **detects this stream**
  (does not send it to TTS — a targeted stream filter), runs the local
  HTTP call, feeds the result back as a `role: tool` message, then the LLM
  runs again to produce the final spoken sentence. Turn = LLM → tool → LLM.
- Building blocks already there: multi-turn history (`Message`/`Role` — add
  a `Tool` role + the `<|tool_*|>` wrapping in the prompt builder).
- **Risk**: Luth-LFM2-1.2B is very small; function-calling (valid JSON,
  right tool choice, no hallucinated args) is fragile on small models even
  with a native format. Mitigation: few tools, tight schemas; fallback =
  a minimal intent router upstream, or a bigger model (tension with the
  "small models" decision locked in §9).
- Approach: MVP with a single tool (light on/off), validate reliability
  before generalising.

### ⬜ Phase 9 — Cross-session memory

Today the conversation history lives only in RAM (`Vec<Message>` in the
orchestrator, bounded to 24 turns) — everything is lost when the app
restarts. Persist it so Lana remembers across runs.

- **Conversation persistence**: append turns to a local store (e.g. JSON
  lines or SQLite under `~/Library/Application Support/Lana/`), reload the
  recent window at startup so a new `converse` continues the prior chat.
- **Durable user facts**: distill stable facts (the user's name,
  preferences) into a small profile re-injected into the system prompt, so
  identity/preferences survive beyond the bounded window.
- **Bounded growth**: keep a recent verbatim window + a rolling summary of
  older turns (summarised by the LLM itself between turns) so the prompt
  stays small without losing long-term context.
- 100 % local, plain files, user-wipeable. Pairs with the Phase 7 UI
  (show/clear memory).

---

## 7. Non-priority refinements (debt / nice-to-have)

For when the avatar work progresses; none blocks the voice loop.

| Topic | Detail | Risk if ignored |
|---|---|---|
| **Remaining upstream ports** | #143 (comma split) **done** in vendored `pocket-tts`. #165 (non-CPU device) and #181 (clone + quantization) are **N/A to the Rust port** (Python-only bugs: `device.type` string; torch.ao dynamic quant — the Rust port has neither). Documented, not ported. | None (resolved) |
| **TTS truncation ("clipped sentences")** | Text is complete, audio cut short: early model EOS. `LANA_TTS_EOS_THRESHOLD` lever exists; default -4.0 kept (lowering it makes truncation *worse* — measured). Residual fix = raise toward 0 on a repro sentence. | Medium: perceived quality |
| **Upstream the port** | Send the #155 port + embedding bridge as a PR to `babybirdprd/pocket-tts`, then move from `path` back to a git dep | Debt: maintaining a vendored fork |
| **WAV voice cloning** | `LANA_TTS_CLONE_WAV` needs the gated voice-cloning weights (`kyutai/pocket-tts`, HF token). The Estelle embedding covers normal use | Low |
| **TTS parity tests** | Vendored fixtures are frozen pre-#155; not used as the oracle. Regenerating would need a Python env (excluded) | Low: oracle = gates + ear |
| **CI** | Verify `.github/workflows/ci.yml` matches the real gate (`lana-*` crates, not `--workspace --all-targets`) | Medium: misleadingly green CI |

---

## 8. Remaining technical risks

| Risk | Mitigation |
|---|---|
| TTS below real time while streaming (debug builds) | Always run `converse` in `--release` (LTO). The audio path now has a ~250 ms jitter buffer that re-arms on underrun |
| Early EOS → clipped sentences | Tune `eos_threshold` toward 0 (§7) |
| `bevy_vrm` maturity (ARKit blendshapes, hair physics) | Plan B: VRM via raw `gltf` + custom shader |
| Large ONNX Runtime memory arena (STT) | Acceptable on 32 GB; `ort` logs set to `warn` by default |
| Maintaining a vendored pocket-tts fork | Upstream as a PR (§7) |

---

## 9. Locked-in decisions

- **No Python** in the runtime stack (no sidecar, no dependency).
- **No Swift / CoreML bridge**: all STT/TTS is pure Rust (ONNX `ort` / candle).
- **No Tauri / webview**: Bevy owns the window, overlay UI and avatar.
- **No cloud**, no external API, no telemetry.
- **Rust Edition 2024**, latest stable deps, clippy pedantic + nursery.
- **Models**: LLM + STT + TTS auto-downloaded from HF on first run (cached,
  no token), not embedded in the binary; optional local overrides.
- **Prefer small & Apple-Silicon-native** over large & more capable
  (voice-first latency > maximum accuracy).
- **Bilingual by construction**: Parakeet FR/EN, Pocket TTS FR (Estelle),
  Luth-LFM2 (FR-specialised, EN retained via cross-lingual transfer).

---

## 10. Immediate next steps

Recently delivered: conversation memory (in-RAM), native streaming TTS,
Estelle voice, default `tu` (informal) form, **LLM = Luth-LFM2-1.2B**
(FR-specialised, `quantized_lfm2`), #143 (comma split), continuous
streaming resampler, ~250 ms audio jitter buffer (crackle fix),
LLM/STT/TTS HF auto-download (zero setup). `LANA_TTS_EOS_THRESHOLD` lever
(default -4.0 kept — lowering worsens truncation).

1. Re-test `lana converse` (`--release`): memory, no empty replies, clean
   audio (no crackle), zero-setup downloads.
2. **Phase 5**: start `lana-avatar` (Bevy + bevy_vrm), load a VRM, window +
   idle, egui transcript overlay.
3. **Phase 9**: cross-session memory — persist conversation + user profile
   to disk so nothing is lost on restart (can land before/after the avatar).
4. Phase 6: hook `lana-viseme` onto lana-tts's streaming PCM.
5. **Phase 8** (tools): MVP function-calling via the native Luth-LFM2 tools
   format (`<|tool_list_start|>`/`<|tool_response_start|>`) on a single
   tool (light on/off via a local API).
