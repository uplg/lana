//! Avatar window for Lana (Bevy).
//!
//! Owns a native window on the **main thread** (winit requirement on
//! macOS). The avatar is a procedural, dark, glowing **point-cloud bust**
//! (see [`cloud`]) — no mesh, skeleton or morph targets, so no model file
//! is needed. Conversation updates arrive over a channel from the
//! orchestrator (another thread): they drive the egui transcript overlay
//! and the lip-sync (the mouth points open with the spoken audio).

#![forbid(unsafe_code)]
// Bevy system functions must take their `SystemParam`s (`Res`, `Query`,
// `Commands`, …) *by value*: `&T` does not implement `SystemParam`, so
// clippy's "pass by reference" suggestion does not compile. This is
// intrinsic to every Bevy system, hence a single crate-level expect
// rather than per-system noise.
#![expect(
    clippy::needless_pass_by_value,
    reason = "Bevy systems require owned SystemParam; &T is not a SystemParam"
)]

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use bevy::prelude::*;
use crossbeam_channel::Receiver;

mod cloud;
mod glb;
mod overlay;

/// A conversational update pushed to the avatar window. Kept independent
/// of `lana-orchestrator` so this crate has no engine dependencies; the
/// app maps `OrchestratorEvent` → `AvatarUpdate`.
#[derive(Debug, Clone)]
pub enum AvatarUpdate {
    /// Coarse phase label (e.g. "listening", "thinking", "speaking").
    Phase(String),
    /// The user's transcribed utterance.
    UserSaid(String),
    /// Lana's reply.
    LanaSaid(String),
    /// Lip-sync timeline for the sentence now playing (appended to any
    /// still-playing one so the mouth tracks continuous speech).
    Visemes(Vec<lana_viseme::VisemeFrame>),
    /// A non-fatal notice.
    Notice(String),
}

/// Channel end the Bevy app drains each frame.
#[derive(Resource)]
struct Updates(Receiver<AvatarUpdate>);

/// User-side mic mute, shared with the orchestrator. The overlay button
/// toggles it; when `true` the orchestrator discards mic input so Lana
/// does not listen.
#[derive(Resource, Clone)]
struct MicMute(Arc<AtomicBool>);

/// Run the avatar window. **Blocks the calling thread** until the window
/// closes, and must be called on the process main thread.
///
/// `updates` streams conversation state in from the orchestrator thread;
/// `mute` is the shared mic-mute the overlay button toggles. No avatar
/// model is needed — the avatar is generated procedurally.
pub fn run(updates: Receiver<AvatarUpdate>, mute: Arc<AtomicBool>) {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(overlay::OverlayPlugin)
        .add_plugins(cloud::CloudPlugin)
        .insert_resource(Updates(updates))
        .insert_resource(MicMute(mute))
        .run();
}
