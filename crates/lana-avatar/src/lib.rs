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
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use bevy::gltf::GltfAssetLabel;
use bevy::prelude::*;
use bevy::scene::SceneInstanceReady;
use bevy_vrm::BoneName;
use bevy_vrm::VrmInstance;
use bevy_vrm::VrmPlugins;
use bevy_vrm::mtoon::MtoonSun;
use crossbeam_channel::Receiver;

mod error;
mod mouth;
mod overlay;
mod vrm;

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

/// Authoritative viseme→morph-slot map read from a `.vrm`'s
/// `blendShapeMaster` (present only for the VRM path; the glTF path
/// resolves the mouth by morph-target name instead).
#[derive(Resource)]
struct VrmMouth(vrm::VrmVisemeMap);

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
    /// Corrective base yaw (radians). VRM 0.x models face −Z, so they show
    /// their back to a +Z camera unless rotated 180°; glTF exports vary.
    /// Tunable via `LANA_AVATAR_ROT_Y` (degrees).
    yaw: f32,
    /// VRM only: camera distance (metres) in front of the head bone for the
    /// portrait/bust framing. Tunable via `LANA_AVATAR_CAM_DIST`.
    cam_dist: f32,
    /// VRM only: vertical aim offset (metres) above the head *bone* — the
    /// bone sits ≈ear level, so a positive offset frames the whole skull
    /// with headroom. Tunable via `LANA_AVATAR_CAM_HEIGHT`.
    cam_height: f32,
}

/// Marker for the avatar root entity (so the idle system can find it).
#[derive(Component)]
struct Avatar;

/// The avatar's base transform (corrective orientation). `idle_motion`
/// composes the sway/breathing on top of this instead of overwriting it.
#[derive(Component)]
struct Base(Transform);

/// The embedded idle clip to play once the glTF scene is spawned. Without
/// it a rigged humanoid renders in its bind pose (the "starfish" T-pose).
/// If the model has no animation, the graph node has no clip and nothing
/// plays (graceful — still T-pose; export the avatar with an idle clip).
#[derive(Component)]
struct IdleAnimation {
    graph: Handle<AnimationGraph>,
    index: AnimationNodeIndex,
}

/// Run the avatar window. **Blocks the calling thread** until the window
/// closes, and must be called on the process main thread.
///
/// `avatar_path` is a `.glb`/`.gltf` (realistic) or `.vrm` model;
/// `updates` streams conversation state in from the orchestrator thread;
/// `mute` is the shared mic-mute the overlay button toggles.
///
/// # Errors
///
/// Returns [`AvatarError::Vrm`] if the path does not exist or has an
/// unsupported extension.
pub fn run(
    avatar_path: PathBuf,
    updates: Receiver<AvatarUpdate>,
    mute: Arc<AtomicBool>,
) -> Result<(), AvatarError> {
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

    // Corrective yaw: VRM 0.x faces away from a +Z camera, so default to a
    // 180° turn. Override per-model (degrees) if it then faces away.
    let yaw = std::env::var("LANA_AVATAR_ROT_Y")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(180.0)
        .to_radians();
    // Portrait/bust framing distance in front of the head bone. VRM 0.0
    // ships no idle, so rather than hand-pose the skeleton (rig-dependent,
    // proven unreliable) we frame head+shoulders so the bind-pose arms fall
    // out of frame — the standard talking-avatar shot.
    let cam_dist = std::env::var("LANA_AVATAR_CAM_DIST")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|d| *d > 0.05)
        .unwrap_or(1.15);
    // Vertical aim above the head bone (≈ear level) so the whole skull has
    // headroom; level (untilted) shot avoids cropping the crown.
    let cam_height = std::env::var("LANA_AVATAR_CAM_HEIGHT")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.08);

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(AssetPlugin {
        file_path: dir,
        ..default()
    }));
    if kind == AvatarKind::Vrm {
        app.add_plugins(VrmPlugins);
        // bevy_vrm drops glTF morph-target names, so resolve the mouth
        // from the VRM's own authoritative blendShapeMaster instead.
        if let Some(map) = vrm::parse_vrm_visemes(&avatar_path) {
            info!(
                a = ?map.a, e = ?map.e, i = ?map.i, o = ?map.o, u = ?map.u,
                "lip-sync: VRM blendShapeMaster viseme map loaded"
            );
            app.insert_resource(VrmMouth(map));
        } else {
            warn!(
                "lip-sync: this VRM has no blendShapeMaster viseme presets — \
                 the mouth cannot be resolved (use a glTF/ARKit avatar)"
            );
        }
    }
    app.add_plugins(overlay::OverlayPlugin)
        .insert_resource(Updates(updates))
        .insert_resource(MicMute(mute))
        .insert_resource(AvatarAsset {
            file,
            kind,
            yaw,
            cam_dist,
            cam_height,
        })
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .init_resource::<mouth::VisemeSchedule>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                idle_motion,
                frame_camera_vrm,
                mouth::resolve_mouth,
                mouth::drive_mouth,
            ),
        )
        .run();

    Ok(())
}

/// Camera, light and the avatar itself.
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    avatar: Res<AvatarAsset>,
) {
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

    let base = Transform::from_rotation(Quat::from_rotation_y(avatar.yaw));
    match avatar.kind {
        AvatarKind::Gltf => {
            // Play the model's first embedded clip (idle) once the scene
            // spawns, so a rigged human isn't stuck in its bind pose.
            let (graph, index) = AnimationGraph::from_clip(
                asset_server.load(GltfAssetLabel::Animation(0).from_asset(avatar.file.clone())),
            );
            let graph = graphs.add(graph);
            commands
                .spawn((
                    Avatar,
                    base,
                    Base(base),
                    IdleAnimation { graph, index },
                    SceneRoot(
                        asset_server.load(GltfAssetLabel::Scene(0).from_asset(avatar.file.clone())),
                    ),
                ))
                .observe(play_idle_when_ready);
        }
        AvatarKind::Vrm => {
            commands.spawn((
                Avatar,
                base,
                Base(base),
                VrmInstance(asset_server.load(avatar.file.clone())),
            ));
        }
    }
}

/// Subtle idle: a slow sway plus a faint breathing scale, so the avatar
/// never looks frozen between turns.
fn idle_motion(time: Res<Time>, mut avatars: Query<(&mut Transform, &Base), With<Avatar>>) {
    let t = time.elapsed_secs();
    for (mut tf, base) in &mut avatars {
        let sway = Quat::from_rotation_y((t * 0.35).sin() * 0.06);
        let breathe = (t * 1.6).sin().mul_add(0.004, 1.0);
        // `base` only carries the corrective orientation (scale = ONE,
        // translation = ZERO), so compose with method calls (avoids the
        // operator-overload arithmetic the lint flags on glam types).
        tf.rotation = base.0.rotation.mul_quat(sway);
        tf.translation = base.0.translation;
        tf.scale = Vec3::splat(breathe);
    }
}

/// VRM path: frame the camera on the head bone (portrait/bust shot) so the
/// unsolved bind-pose arms fall out of frame — the standard talking-avatar
/// shot, and a deterministic alternative to hand-posing the skeleton.
/// One-shot once the skeleton's world transforms exist; glTF keeps the
/// default camera (no `BoneName`). Uses the head's *world position* only
/// (translation is robust even on the mirror-scaled rigs that broke the
/// earlier rotation-based pose attempts).
fn frame_camera_vrm(
    mut done: Local<bool>,
    avatar: Res<AvatarAsset>,
    bones: Query<(&BoneName, &GlobalTransform)>,
    mut cam: Query<&mut Transform, With<Camera3d>>,
) {
    if *done || avatar.kind != AvatarKind::Vrm {
        return;
    }
    let mut head = None;
    for (name, gt) in &bones {
        if matches!(name, BoneName::Head) {
            head = Some(gt.translation());
            break;
        }
    }
    let Some(h) = head else {
        return;
    };
    if h.length_squared() < 1e-4 {
        return; // global transforms not propagated yet — retry.
    }
    let Ok(mut tf) = cam.single_mut() else {
        return;
    };
    // Level (untilted) shot: eye and aim share the same height, `cam_dist`
    // in front (+Z, the side the corrective yaw makes the avatar face).
    // Aim a touch above the head bone so the whole skull has headroom and
    // the crown is not clipped; shoulders sit near the bottom edge so the
    // T-pose arms stay out of frame.
    let aim_y = h.y + avatar.cam_height;
    let eye = Vec3::new(h.x, aim_y, h.z + avatar.cam_dist);
    let target = Vec3::new(h.x, aim_y, h.z);
    *tf = Transform::from_translation(eye).looking_at(target, Vec3::Y);
    *done = true;
    info!(
        dist = avatar.cam_dist,
        height = avatar.cam_height,
        "camera: framed portrait on VRM head"
    );
}

/// glTF path: once the scene is spawned, start its first embedded clip
/// looping on the model's `AnimationPlayer`. If the model has no animation
/// the graph node has no clip and nothing plays (graceful — still bind
/// pose; export the avatar with an idle clip).
fn play_idle_when_ready(
    ready: On<SceneInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    idle: Query<&IdleAnimation>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let Ok(idle) = idle.get(ready.entity) else {
        return;
    };
    for child in children.iter_descendants(ready.entity) {
        if let Ok(mut player) = players.get_mut(child) {
            player.play(idle.index).repeat();
            commands
                .entity(child)
                .insert(AnimationGraphHandle(idle.graph.clone()));
        }
    }
}
