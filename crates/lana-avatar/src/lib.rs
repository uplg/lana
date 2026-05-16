//! Avatar window for Lana (Bevy).
//!
//! Owns a native window on the **main thread** (winit requirement on
//! macOS): a 3D camera, a light, the loaded avatar and a gentle idle
//! motion. Two avatar formats are supported, dispatched by file
//! extension:
//!
//! - **`.glb` / `.gltf`** — a realistic human (e.g. exported from
//!   Avaturn): loaded as a native Bevy scene. Its `ARKit` blendshapes /
//!   visemes feed Phase 6 lip-sync. No spring-bone physics.
//! - **`.vrm`** — a VTuber-style avatar via [`bevy_vrm`] (spring bones +
//!   VRM blendshapes for free, but stylised only).
//!
//! Conversation updates arrive over a channel from the orchestrator
//! (running on another thread); they drive the on-screen transcript
//! overlay and, later (Phase 6), visemes.

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

use std::path::PathBuf;

use bevy::gltf::GltfAssetLabel;
use bevy::prelude::*;
use bevy_vrm::VrmInstance;
use bevy_vrm::VrmPlugins;
use bevy_vrm::mtoon::MtoonSun;
use crossbeam_channel::Receiver;

mod error;
mod overlay;

pub use error::AvatarError;

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
    /// A non-fatal notice.
    Notice(String),
}

/// Channel end the Bevy app drains each frame.
#[derive(Resource)]
struct Updates(Receiver<AvatarUpdate>);

/// Which loader to use for the avatar asset.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AvatarKind {
    /// Native Bevy glTF scene (realistic human).
    Gltf,
    /// `bevy_vrm` (VTuber-style).
    Vrm,
}

/// The avatar asset, addressed by file name relative to the asset root
/// (which `run` points at the model's own directory).
#[derive(Resource)]
struct AvatarAsset {
    file: String,
    kind: AvatarKind,
}

/// Marker for the avatar root entity (so the idle system can find it).
#[derive(Component)]
struct Avatar;

/// Run the avatar window. **Blocks the calling thread** until the window
/// closes, and must be called on the process main thread.
///
/// `avatar_path` is a `.glb`/`.gltf` (realistic) or `.vrm` model;
/// `updates` streams conversation state in from the orchestrator thread.
///
/// # Errors
///
/// Returns [`AvatarError::Vrm`] if the path does not exist or has an
/// unsupported extension.
pub fn run(avatar_path: PathBuf, updates: Receiver<AvatarUpdate>) -> Result<(), AvatarError> {
    if !avatar_path.exists() {
        return Err(AvatarError::Vrm(format!(
            "avatar file not found: {} (set LANA_AVATAR_PATH to a .glb/.gltf \
             realistic model — e.g. exported from avaturn.me — or a .vrm)",
            avatar_path.display()
        )));
    }

    let ext = avatar_path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let kind = match ext.as_str() {
        "glb" | "gltf" => AvatarKind::Gltf,
        "vrm" => AvatarKind::Vrm,
        other => {
            return Err(AvatarError::Vrm(format!(
                "unsupported avatar extension '.{other}' (use .glb, .gltf or .vrm)"
            )));
        }
    };

    // Bevy's asset root resolves names relative to a base directory; point
    // it at the model's own folder so an arbitrary absolute path works.
    let dir = avatar_path
        .parent()
        .map_or_else(|| ".".to_owned(), |p| p.to_string_lossy().into_owned());
    let file = avatar_path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .ok_or_else(|| AvatarError::Vrm("avatar path has no file name".to_owned()))?;

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(AssetPlugin {
        file_path: dir,
        ..default()
    }));
    if kind == AvatarKind::Vrm {
        app.add_plugins(VrmPlugins);
    }
    app.add_plugins(overlay::OverlayPlugin)
        .insert_resource(Updates(updates))
        .insert_resource(AvatarAsset { file, kind })
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .add_systems(Startup, setup)
        .add_systems(Update, idle_motion)
        .run();

    Ok(())
}

/// Camera, light and the avatar itself.
fn setup(mut commands: Commands, asset_server: Res<AssetServer>, avatar: Res<AvatarAsset>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.35, 2.4).looking_at(Vec3::new(0.0, 1.25, 0.0), Vec3::Y),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
        )),
        MtoonSun,
    ));

    match avatar.kind {
        AvatarKind::Gltf => {
            commands.spawn((
                Avatar,
                Transform::default(),
                SceneRoot(
                    asset_server.load(GltfAssetLabel::Scene(0).from_asset(avatar.file.clone())),
                ),
            ));
        }
        AvatarKind::Vrm => {
            commands.spawn((
                Avatar,
                Transform::default(),
                VrmInstance(asset_server.load(avatar.file.clone())),
            ));
        }
    }
}

/// Subtle idle: a slow sway plus a faint breathing scale, so the avatar
/// never looks frozen between turns.
fn idle_motion(time: Res<Time>, mut avatars: Query<&mut Transform, With<Avatar>>) {
    let t = time.elapsed_secs();
    for mut tf in &mut avatars {
        tf.rotation = Quat::from_rotation_y((t * 0.35).sin() * 0.06);
        let breathe = (t * 1.6).sin().mul_add(0.004, 1.0);
        tf.scale = Vec3::splat(breathe);
    }
}
