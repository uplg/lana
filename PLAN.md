# Lana — Architecture Plan

A **local-only** virtual AI agent, French-first (English supported), wrapped
in a desktop app with a realistic 3D avatar that lip-syncs in real time.

Target hardware: **MacBook Pro M1 Max 32 GB**. Hard constraint: the machine
must stay fully usable for other work while Lana runs.

> **Status (2026-05-16)**: phases 0→4 delivered and validated. Full French
> voice loop works (`lana converse`), real **Estelle** voice, native
> streaming TTS, conversation memory (in-RAM), LLM = Luth-LFM2-1.2B.
> LLM/STT/TTS all auto-download from Hugging Face (zero manual setup).
> Avatar: **Cyberpunk-2077-braindance point-cloud** (user's call) — the
> whole VRM/glTF rig stack (`bevy_vrm`, morphs, posing) was deleted; a
> model's *vertices* (`.glb`/`.vrm`/`.pcd`, positions only) are sampled and
> rendered as glowing scanned points (bloom + scan sweep + flicker);
> lip-sync (`lana-viseme`/`VisemeSchedule`) opens the lower-head band.
> 100 % Rust stack: no Swift, no Python, no cloud.

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
| **Visemes** | short-time FFT energy + F1/F2 formants → 5 vowel shapes + `Sil` | *(done, Phase 6)* pure-Rust DSP, unit-tested, no Bevy dep | — |
| **Avatar** | **Bevy 0.18** (wgpu, native window); glTF scene or `bevy_vrm 0.3` | `LANA_AVATAR_PATH`: realistic `.glb`/`.gltf` (Avaturn) or `.vrm`. Camera/light/idle. Main thread (winit) | ~600 MB |
| **UI** | `bevy_egui 0.39` overlay | Bottom panel: phase + rolling transcript. No Tauri/webview | ~0 |
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
│   ├── lana-viseme/           # pure-DSP audio→viseme (FFT energy + F1/F2), unit-tested
│   ├── lana-avatar/           # Bevy window: VRM/glTF + camera/light + idle + lip-sync + auto-pose + egui overlay
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

### ✅ Phase 5 — Static avatar
`lana-avatar`: Bevy 0.18 + `bevy_egui` 0.39 + `bevy_vrm` 0.3 (versions
verified mutually compatible). Single native window: 3D camera, directional
light, the avatar from `LANA_AVATAR_PATH`, gentle idle (sway + breathing),
and a `bevy_egui` bottom panel showing the live phase + rolling transcript.
**Two avatar paths, dispatched by extension** (the asset root is pointed at
the model's own directory so any absolute path works): `.glb`/`.gltf` →
native Bevy scene for a **realistic human** (e.g. an Avaturn export — rig +
ARKit blendshapes/visemes ready for Phase 6 lip-sync; no spring bones);
`.vrm` → `bevy_vrm` (stylised VTuber, spring bones for free). The VRM
ecosystem is anime-centric by origin, so realistic = the glTF path.
`lana-app` restructured: dropped `#[tokio::main]`; `converse` runs the
orchestrator on a side thread (own Tokio runtime) and Bevy on the main
thread (winit requirement), bridged by a crossbeam `AvatarUpdate` channel
(stdout transcript kept too). *Not GUI-tested in CI — render verified by
the user on-device (like the audio ear-check).*

### ✅ Phase 6 — Lip-sync (confirmed on-device 2026-05-17)
`lana-viseme` (pure-DSP, no Bevy dep, unit-tested): short-time Hann/FFT
(rustfft) ~100 fps over the synthesised sentence PCM — per-frame RMS →
mouth *openness*, F1/F2 spectral-peak guess → nearest of five vowel shapes
(A/E/I/O/U) or `Sil`. Honest approximation (no forced alignment), enough
for believable motion. `TtsEngine::Speech` now also carries the raw mono
PCM; the orchestrator analyses it and emits `OrchestratorEvent::Visemes`
*before* enqueuing the WAV, so the mouth starts with the sound. The avatar
schedules successive per-sentence timelines back-to-back on the wall clock
(so they concatenate with the continuous speaker queue) and drives the
face's morph targets, smoothed per frame, on Bevy's canonical
`MorphWeights`. **Resolution is loader-correct, not guessed**: the user's
VRM exposed `MorphWeights` but `bevy_vrm`/`bevy_gltf_kun` drops glTF
`extras.targetNames`, so name lookup returned `None` (and v1 only logged on
*success* — a silent-miss bug, now fixed: every outcome logs the ground
truth once). Fix: for a `.vrm` we parse the file's own
`extensions.VRM.blendShapeMaster` (the VRM spec contract) for the a/i/u/e/o
presets → authoritative morph indices, and bind them to the face
`MorphWeights` (slot order is preserved by the loader even when names are
not). glTF avatars keep ARKit/VRoid name resolution. Verified the indices
against the actual model (`a→39 … o→43`, mesh 0); on-device confirmation
that the mouth visibly moves still pending. (`bevy_vrm1` was evaluated — it
is **VRM 1.0 only**, the model is VRM 0.0, so not applicable.) **Posture
deferred to Phase 7**: two
attempts at hand-computing the de-T-pose failed for genuine rig-dependent
reasons (blind local-axis = arms up; world-space arc = arms vanish on
mirror-scaled VRM rigs where the `GlobalTransform` decomposition is not a
pure rotation). Skeletal posing of arbitrary VRMs without per-rig data is
not deterministic — the clean fix is a real idle/retargeted animation, not
bone math. Reverted to the visible bind pose; `LANA_AVATAR_ARM_DOWN`
removed (no silent knob). *Not GUI-tested in CI — verified on-device.*

### ✅ Phase 7 — Avatar pivot: Cyberpunk-braindance point-cloud
**The VRM/glTF rig stack was scrapped for a point-cloud "scan hologram"**
(user's idea; reference = CP2077 braindance). Honest finding (SIGGRAPH
talk + the snowman attempt): the *shape* can't be procedural — it must be
a real model sampled to points; the *look* is the renderer. So:
- `glb.rs` — vertex sampler: parses a `.glb`/`.vrm` (binary glTF
  JSON+BIN, every primitive's FLOAT/VEC3 `POSITION`, honoring
  bufferView stride) or ASCII `.pcd`; **positions only**, no rig/morphs;
  graceful `None` → procedural fallback. Subsamples to ~7000.
- `cloud.rs` — model auto-detected (`LANA_AVATAR_MODEL` or first
  `.glb`/`.vrm`/`.pcd` in CWD; default = `test_woman.glb` the user
  dropped at root), normalised (feet y=0, centred, height `TARGET_H`),
  role-tagged by height band (lower-head band = lip-sync). Rendered as one
  shared tiny emissive `Sphere` auto-instanced (no WGSL), `Bloom::NATURAL`
  + `TonyMcMapface`, near-black. **Braindance animation**: slow turn, a
  vertical **scan-plane sweep** that flares the band it crosses (scale →
  more bloom), per-point **flicker** dropout ("bad scan"), shimmer; the
  lip-sync band opens with `VisemeSchedule.openness`. Procedural cloud
  kept only as the no-model fallback.
- **Deleted**: `bevy_vrm`, `vrm.rs`, `mouth.rs` morph code, glTF
  `SceneRoot`, camera-framing, `error.rs`/`AvatarError`. Kept: window,
  egui overlay, **mic-mute**, `Updates`, `AvatarUpdate`, `lana-viseme`.
**Honest limits**: mouth/eye localisation on an arbitrary mesh is a
height-band heuristic (not exact lips); "sexy" quality = the input model;
not GUI-tested in CI — user is the visual oracle. Knobs: `LANA_AVATAR_MODEL`,
`LANA_AVATAR_CAM_DIST`; point size/colours/scan-flicker are `cloud.rs`
consts. Still open: FR/EN voice picker.

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

### ⬜ Phase 10 — Photorealistic avatar (FLAME-rigged 3DGS)

The glTF/VRM mesh path (Phases 5–6) caps at "realistic stylised". True
photorealism today is **3D Gaussian Splatting**, not textured meshes
(hence the lack of glTF photoreal sources). Researched, viable path that
**reconciles photoreal with the locked constraints**:

- **Representation**: a *FLAME-rigged* 3DGS head avatar, à la
  **GaussianAvatars** (Qian et al., CVPR 2024 Highlight): Gaussians bound
  to FLAME mesh triangles. Animation = deform the FLAME mesh by
  shape/expr/jaw params; each Gaussian rigidly follows its parent triangle.
  **Pure linear algebra at runtime — no neural network, no Python.** Also
  see *3D Gaussian Blendshapes* / *RGBAvatar* (CVPR'25): blendshape-driven
  Gaussians = exactly the Phase-6 viseme→blendshape model, but photoreal.
- **Offline (one-time, outside the runtime — allowed)**: train the
  per-subject avatar from a short face video (research tooling, Python/
  CUDA). Produces a static Gaussian cloud + FLAME-triangle bindings.
- **Runtime (Rust, lean, local, real-time)**: implement FLAME forward
  (params → 5023 verts; LBS + corrective blendshapes — bounded, linear,
  ~no Rust impl exists, to write) + per-Gaussian rigid transform from
  parent triangles + render via a wgpu splat pipeline
  (`bevy_gaussian_splatting`, or a custom pass). Drive FLAME jaw/expr from
  the Phase-6 viseme analyser. Zero net, zero Python at runtime.
- **Hard ceiling — licensing, not tech**: GaussianAvatars code is
  CC-BY-NC-SA + a Toyota proprietary clause (**non-commercial**); FLAME
  pre-2023 is research-NC, but **FLAME 2023 "Open" is CC-BY-4.0** (OK for
  a personal/local project). The whole 3DGS-avatar research ecosystem is
  NC → personal use fine, future commercialisation capped. We'd
  re-implement only the (trivial, non-secret) runtime rigging math in
  Rust, not reuse NC code.
- **Effort**: multi-week research-grade subsystem; depends on the user
  producing the avatar asset offline. De-risk with a spike first (Rust
  FLAME forward + render a pretrained sample statically) before committing.

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

Recently delivered: **Avatar pivot — Cyberpunk-braindance point-cloud**
(`glb.rs` samples a model's vertices → `cloud.rs` auto-instanced glowing
points + bloom + scan-sweep/flicker; VRM/glTF rig stack deleted; lip-sync
reuses `lana-viseme`/`VisemeSchedule`; auto-detects a `.glb`/`.vrm`/`.pcd`),
**mic-mute toggle** (egui button → shared flag → orchestrator drops mic),
**Phase 6 lip-sync** confirmed on-device 2026-05-17,
**Phase 5 avatar window** (Bevy + egui overlay, Bevy on main thread),
conversation memory (in-RAM), Estelle voice, default `tu` form,
**LLM = Luth-LFM2-1.2B**, #143 (comma split), LLM/STT/TTS HF auto-download
(zero setup). Audio path is the proven Phase-4 whole-sentence
synth+enqueue (the streaming/jitter-buffer experiment was reverted; TTS
`eos_threshold` default −2.0 to stop end-of-sentence truncation).
Photoreal path researched and specified as **Phase 10** (FLAME-rigged
3DGS) — deferred until the mesh pipeline is complete (user's call).

1. On-device run of `lana converse` (`--release`, no env needed): judge
   the point-cloud avatar — does it read as a 3D glowing bust, do the
   mouth points open with speech, eyes blink? Tune `LANA_AVATAR_CAM_DIST`
   for zoom; geometry/colors/density are constants in `cloud.rs` to
   iterate on from the user's visual feedback (the only oracle here).
2. STT robustness: parakeet-rs (v3 multilingual) has no language lock, so
   heavily FR-accented English can decode as Cyrillic. Consider a
   script-sanity guard (drop/re-listen on a predominantly non-Latin
   transcript while the agent is FR/EN) rather than feeding garbage to the
   LLM. Decide with the user before implementing.
3. **Phase 8** (tools): MVP function-calling via the native Luth-LFM2 tools
   format (`<|tool_list_start|>`/`<|tool_response_start|>`) on a single
   tool (light on/off via a local API).
4. **Phase 9**: cross-session memory — persist conversation + user profile
   to disk so nothing is lost on restart.
5. Phase 7 leftovers: FR/EN voice picker, VRM hot-reload, gaze-at-camera.
