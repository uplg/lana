# Lana — Plan d'architecture

Agent IA virtuel **local-only**, francophone (et anglophone), wrappé dans
une application desktop avec avatar 3D réaliste qui bouge les lèvres en
temps réel.

Cible matérielle : **MacBook Pro M1 Max 32 Go**. Contrainte forte : la
machine doit rester pleinement utilisable pour d'autres tâches pendant que
Lana tourne.

> **État (2026-05-16)** : phases 0→4 livrées et validées. Boucle vocale
> complète FR fonctionnelle (`lana converse`), voix **Estelle** réelle,
> TTS en streaming natif, mémoire de conversation. Avatar / lip-sync (phases
> 5+) pas encore commencés. Stack 100 % Rust : zéro Swift, zéro Python,
> zéro cloud.

---

## 1. Vision produit

- Avatar 3D nommé **Lana**, rendu en temps réel
- Conversation vocale naturelle (FR principal, EN supporté, switch transparent)
- Lip-sync à 60 fps sur 12 visèmes ARKit
- Latence voice-to-voice cible : **< 1 s end-to-end** (release)
- 100 % local : aucune donnée ne quitte la machine
- Empreinte mémoire totale **< 3 Go**, machine reste fluide

---

## 2. Pipeline conversationnelle (actuel)

```
microphone (CoreAudio, cpal) — décimateur FIR maison 48→16 kHz
  → VAD (earshot, NN Rust pur) → segmenteur d'énoncé
    → STT (Parakeet-TDT v3, parakeet-rs / ONNX Runtime, EP CPU)
      → LLM (Luth-LFM2-1.2B, candle/Metal) — historique multi-tours
        → TTS (Kyutai Pocket TTS natif Rust, candle/Metal, french_24l + Estelle)
          → streaming par frame Mimi (~80 ms) → speaker (cpal, annulable)
          └─ (à venir) analyse temps réel → visèmes → blendshapes VRM 60 fps
```

Tout passe par des **channels Tokio** entre étages. Le barge-in
(interrompre Lana en parlant) est implémenté ; **désactivé par défaut**
(half-duplex) car sans annulation d'écho acoustique le micro capte les
haut-parleurs. `LANA_BARGEIN=1` l'active (casque / futur AEC).

---

## 3. Stack technique (actuel)

| Couche | Choix | Pourquoi | RAM |
|---|---|---|---|
| **Capture audio** | `cpal` (CoreAudio) + décimateur FIR windowed-sinc maison 48→16 kHz | Faible latence, Rust pur | — |
| **VAD** | `earshot` (NN Rust pur, frames 256 éch. / 16 ms) | Aucun ONNX, aucun modèle à télécharger, ~110 Kio | ~0 |
| **STT** | **Parakeet-TDT-0.6B-v3** via `parakeet-rs` (ONNX Runtime `ort`, EP **CPU**) | SOTA STT multilingue FR, Rust pur via lib C (pas de Swift). CoreML instable pour ce modèle, CPU rapide sur Apple Silicon | ~600 Mo (arène ORT) |
| **LLM** | **Luth-LFM2-1.2B** Q8_0 GGUF via `candle` (Metal) — `quantized_lfm2` | Liquid LFM2 spécialisé FR (SOTA français à cette taille), très petit (~1,25 Go), pas de thinking. candle 0.10.2 a déjà l'archi `lfm2` | ~1,3 Go |
| **TTS** | **Kyutai Pocket TTS** — portage Rust natif (vendored `babybirdprd/pocket-tts`, candle/Metal), remonté à la parité upstream (#155) | Vraie voix **Estelle** FR via embedding voix prédéfini (token-free, repo non-gated). Streaming natif par frame Mimi | ~300 Mo |
| **Visèmes** | FFT + formants F1/F2 + onsets bilabiaux → 12 visèmes ARKit | *(non commencé)* Rust pur, latence ~10 ms | — |
| **Avatar** | **Bevy** + `bevy_vrm` (full Rust, wgpu, fenêtre native) | *(non commencé)* 100 % Rust, VRM gratuits | ~600 Mo |
| **UI** | `bevy_egui` overlay | *(non commencé)* pas de Tauri/webview | ~0 |
| **Orchestrateur** | Tokio, channels, machine d'états | Idle / Listening / Thinking / Speaking + barge-in + mémoire | ~100 Mo |

Détails verrouillés des choix : voir mémoire projet `lana-stack`,
`lana-phase{1,2,3,4}-baseline`.

---

## 4. Workspace Cargo (actuel)

```
lana/
├── Cargo.toml                 # workspace, lints stricts, exclude = ["vendor"]
├── deny.toml                  # cargo-deny (licences/advisories/sources)
├── crates/
│   ├── lana-audio/            # cpal capture + décimateur FIR + playback annulable
│   ├── lana-vad/              # earshot + UtteranceSegmenter
│   ├── lana-stt/              # parakeet-rs (ort), pas de Swift
│   ├── lana-llm/              # candle + Luth-LFM2 GGUF (quantized_lfm2), mémoire (Message/Role)
│   ├── lana-tts/              # pocket-tts vendored, streaming, voix Estelle
│   ├── lana-viseme/           # (stub)
│   ├── lana-avatar/           # (stub)
│   ├── lana-ui/               # (stub)
│   ├── lana-orchestrator/     # machine d'états, channels, barge-in, historique
│   └── lana-app/              # binaire : chat / transcribe / synth / converse
└── vendor/
    └── pocket-tts/            # fork babybirdprd vendored, patché parité Kyutai
                               # (#155 + pont embedding voix). Workspace Cargo
                               # propre, dépendance par `path`. .git retiré.
```

Modèles : **LLM (Luth-LFM2) + TTS (`french_24l` + voix Estelle) téléchargés
au premier lancement depuis HF (non-gated, sans token) et cachés** — rien à
récupérer à la main. `LANA_MODEL_PATH`/`LANA_TOKENIZER_PATH` permettent un
override local du LLM. Seul `LANA_STT_MODEL_DIR` (ONNX Parakeet) reste un
chemin local requis.

---

## 5. Garde-fous qualité (Rust 2024, clippy parfait)

Lints stricts workspace : `clippy::pedantic` + `nursery`, `unwrap_used` /
`panic` = deny, `arithmetic_side_effects` / `expect_used` = warn, etc.
`#[expect]`/`#[allow]` = dernier recours, restructurer d'abord.

La crate vendored `pocket-tts` est tierce et garde sa propre tolérance
clippy ; le gate workspace cible donc explicitement les crates `lana-*`
(un `cargo clippy --workspace --all-targets` traverse le workspace vendored
et casse sur `pocket-tts-cli` dont le `build.rs` exige des assets web
inutiles à Lana).

```sh
cargo fmt --all --check
cargo clippy -p lana-audio -p lana-vad -p lana-stt -p lana-llm -p lana-tts \
             -p lana-viseme -p lana-avatar -p lana-ui -p lana-orchestrator \
             -p lana-app --all-targets --no-deps -- -D warnings
cargo test --workspace
cargo deny check
```

`cargo deny` : `all-features = false` (évite `intel-mkl-src` via la feature
`mkl` ; on build candle en `metal`). `intel-mkl-src` clarifié en
`LicenseRef-Intel-Simplified-Software-License`.

---

## 6. Phases de livraison

### ✅ Phase 0 — Squelette
Workspace, lints stricts, `cargo deny` clean, CI sans téléchargement de modèles.

### ✅ Phase 1 — Boucle texte
`candle` (Metal) + LLM GGUF, CLI `chat`. Démarré sur Qwen3-1.7B
(220-270 ms TTFT, ~60 tok/s) ; basculé sur **Luth-LFM2-1.2B**
(`quantized_lfm2`) — Liquid LFM2 spécialisé français, template ChatML
(BOS `<|startoftext|>`), pas de thinking, reset d'état implicite via
`index_pos == 0` (pas d'API `clear_kv_cache`).

### ✅ Phase 2 — Voix in
`cpal` capture 48 kHz → FIR 16 kHz ; `earshot` VAD ; `parakeet-rs` STT
(ONNX/ort, CPU). CLI `transcribe`.

### ✅ Phase 3 — Voix out
`lana-tts` sur portage Rust natif de Kyutai Pocket TTS (vendored
babybirdprd). Port #155 (multilingue FR `french_24l`), pont d'embedding voix
(PR babybirdprd #17) → voix **Estelle** réelle, token-free. Streaming natif
par frame Mimi. CLI `synth`.

### ✅ Phase 4 — Boucle complète sans avatar
`lana-orchestrator` : machine d'états Tokio, barge-in (opt-in), TTS
streaming au fil de l'eau, **mémoire de conversation** multi-tours bornée.
CLI `converse`.

### ⬜ Phase 5 — Avatar statique
`lana-avatar` : Bevy + `bevy_vrm`, charge un `.vrm`, caméra/éclairage,
idle. `bevy_egui` overlay (transcript + réglages). Une seule fenêtre.

### ⬜ Phase 6 — Lip-sync
`lana-viseme` : FFT (rustfft) ~50 Hz sur le flux TTS, formants F1/F2,
onsets bilabiaux → 12 visèmes ARKit. Bevy interpole les blendshapes à
60 fps. Brancher sur le flux PCM déjà produit en streaming par lana-tts.

### ⬜ Phase 7 — Polish
Choix de voix FR/EN dans l'UI, persistance conversation, hot-reload VRM,
mode veille (respiration, clignements, regard caméra).

### ⬜ Phase 8 — Outils (function-calling)

Permettre à Lana d'appeler des outils locaux (ex. API maison domotique :
allumer/éteindre une lampe). **Reste 100 % local** : l'outil n'est qu'un
appel HTTP vers une API sur le réseau de l'utilisateur, aucun cloud.

- **Bonne nouvelle** : le template Luth-LFM2 a un format outils *natif* —
  liste d'outils injectée en system entre `<|tool_list_start|>[…]<|tool_list_end|>`,
  et réponses d'outil enveloppées `<|tool_response_start|>…<|tool_response_end|>`
  pour le rôle `tool`. Pas besoin d'inventer un protocole.
- Le modèle émet un appel d'outil ; l'orchestrateur **détecte ce flux**
  (ne l'envoie pas au TTS — filtre de flux ciblé), exécute l'appel HTTP
  local, réinjecte le résultat comme message `role: tool`, puis le LLM
  repasse pour produire la phrase parlée finale. Tour LLM → outil → LLM.
- Briques déjà là : historique multi-tours (`Message`/`Role` — ajouter un
  rôle `Tool` + le wrapping `<|tool_*|>` dans le builder de prompt).
- **Risque** : Luth-LFM2-1.2B est très petit ; le function-calling (JSON
  correct, bon choix d'outil, pas d'arguments hallucinés) est fragile sur
  les petits modèles, même avec un format natif. Mitigation : peu d'outils,
  schémas serrés ; plan B = routeur d'intention minimal en amont, ou monter
  d'un cran de modèle (tension avec le choix « petits modèles » verrouillé
  en §9).
- Approche : MVP un seul outil (lampe on/off), valider la fiabilité avant
  de généraliser.

---

## 7. Raffinements non-prioritaires (dette / nice-to-have)

À faire quand l'avatar avance ; aucun ne bloque la boucle vocale.

| Sujet | Détail | Risque si ignoré |
|---|---|---|
| **Ports upstream restants** | Porter Kyutai #143 (split sur virgules → moins de mots sautés sur longues phrases), #165 (fix device non-CPU), #181 (fix clonage + quantization) dans le `pocket-tts` vendored | Faible : qualité/robustesse marginale ; #143 améliorerait la prosodie des longues phrases |
| **Troncature TTS (« phrases coupées »)** | Le texte est complet, l'audio est coupé : EOS précoce du modèle Pocket TTS. Leviers : `eos_threshold` (-4.0 → ~-6.0) ou `model_recommended_frames_after_eos`. À régler avec une phrase repro à l'oreille | Moyen : qualité perçue |
| **Buffer anti-underrun streaming** | En release le TTS doit tenir le temps réel ; sinon pré-bufferiser quelques frames avant lecture | Faible (release rapide) |
| **Upstreamer le portage** | Remonter le port #155 + pont embedding en PR sur `babybirdprd/pocket-tts`, puis repasser de `path` à git-dep | Dette : on maintient un vendored |
| **Clonage voix WAV** | `LANA_TTS_CLONE_WAV` nécessite les poids voice-cloning gated (`kyutai/pocket-tts`, token HF). Embedding Estelle suffit pour l'usage courant | Faible |
| **Tests parité TTS** | Les fixtures vendored sont gelées pré-#155 ; non utilisées comme oracle. Régénérer demanderait un env Python (exclu) | Faible : oracle = gates + oreille |
| **CI** | Vérifier que `.github/workflows/ci.yml` reflète le gate réel (crates `lana-*`, pas `--workspace --all-targets`) | Moyen : CI verte trompeuse |
| **Mémoire conversation** | Bornée à 24 messages (drop des plus vieux). Pas de résumé / fenêtre par tokens | Faible : sessions très longues perdent le contexte ancien |

---

## 8. Risques techniques restants

| Risque | Mitigation |
|---|---|
| TTS sous le temps réel en streaming (debug) | Toujours lancer `converse` en `--release` (LTO). Pré-buffer si besoin |
| EOS précoce → phrases coupées | Tuning `eos_threshold` / frames-after-eos (§7) |
| `bevy_vrm` maturité (blendshapes ARKit, physique cheveux) | Plan B : VRM via `gltf` brut + shader custom |
| Arène mémoire ONNX Runtime volumineuse (STT) | Acceptable sur 32 Go ; logs `ort` mis en `warn` par défaut |
| Maintien d'un fork vendored pocket-tts | Upstreamer en PR (§7) |

---

## 9. Décisions verrouillées

- **Pas de Python** dans la stack runtime (ni sidecar, ni dépendance).
- **Pas de Swift / CoreML bridge** : tout STT/TTS est Rust pur (ONNX `ort` / candle).
- **Pas de Tauri / webview** : Bevy gère fenêtre, UI overlay et avatar.
- **Pas de cloud**, pas d'API externe, pas de télémétrie.
- **Rust Edition 2024**, dernières versions stables, clippy pedantic + nursery.
- **Modèles** : LLM + TTS auto-téléchargés depuis HF au 1er lancement
  (cache, sans token), pas embarqués dans le binaire ; STT en chemin local.
- **Préférer petit & natif Apple Silicon** à gros & plus capable
  (latence voice-first > exactitude maximale).
- **Bilingue par construction** : Parakeet FR/EN, Pocket TTS FR (Estelle),
  Luth-LFM2 (spécialisé FR, EN conservé par cross-lingual transfer).

---

## 10. Prochaines étapes immédiates

Livré récemment : mémoire de conversation, streaming TTS natif, voix
Estelle, tutoiement par défaut, **LLM basculé Qwen3-1.7B → Luth-LFM2-1.2B**
(spécialisé FR, `quantized_lfm2`, `ThinkFilter` supprimé), #143 (split
virgules), resampler streaming continu (fix grésillement). Levier
`LANA_TTS_EOS_THRESHOLD` (défaut -4.0 conservé : le baisser aggrave la
troncature).

1. Re-tester `lana converse` (`--release`) : mémoire, plus de réponses
   vides, audio sans grésillement.
2. **Phase 5** : démarrer `lana-avatar` (Bevy + bevy_vrm), charger un VRM,
   fenêtre + idle, overlay egui transcript.
3. Phase 6 : brancher `lana-viseme` sur le flux PCM streaming de lana-tts.
4. **Phase 8** (outils) : MVP function-calling via le format outils natif
   Luth-LFM2 (`<|tool_list_start|>`/`<|tool_response_start|>`) sur un seul
   outil (lampe on/off via API locale) — peut passer avant ou après l'avatar.
