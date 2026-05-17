//! Procedural point-cloud avatar — a dark, glowing "hologram bust".
//!
//! No mesh, no skeleton, no morph targets (the three VRM headaches): the
//! avatar is a few thousand emissive points generated procedurally (a
//! fibonacci-sampled head ellipsoid + a bust shell + eye/mouth clusters),
//! drawn as one shared tiny emissive sphere auto-instanced by Bevy, glowing
//! via an HDR camera + bloom over a near-black scene. Lip-sync reuses the
//! Phase-6 [`VisemeSchedule`]: the mouth-region points spread open with the
//! spoken `openness`. Idle = a slow turn/breathe + per-point shimmer; the
//! eye points blink on an irregular timer.

// Bounded procedural geometry: point counts, ring indices and trig phases
// are all small and range-safe, so the `usize`/`u32`→`f32` casts are exact
// for the magnitudes involved, and explicit-FMA lints only obscure the
// position formulas. Same precedent as the capture-path DSP decimator.
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    reason = "bounded geometry + normal→colour buckets clamped to 0..LEVELS; \
              FMA rewrites obscure the position formulas, counts are small"
)]

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use lana_viseme::{VisemeFrame, Vowel};

/// One scheduled mouth pose, timed on the wall clock so it lines up with
/// the speaker output the orchestrator enqueued at the same instant.
#[derive(Debug, Clone, Copy)]
struct Scheduled {
    at: Instant,
    openness: f32,
    vowel: Vowel,
}

/// Pending lip-sync poses. Successive per-sentence timelines are appended
/// back-to-back so the mouth tracks the continuous speaker stream.
#[derive(Resource, Default)]
pub(crate) struct VisemeSchedule {
    queue: VecDeque<Scheduled>,
    /// Wall-clock end of the last queued frame, so the next timeline starts
    /// where this one ends instead of overlapping it.
    tail: Option<Instant>,
}

impl VisemeSchedule {
    /// Append a freshly-analysed sentence timeline.
    pub(crate) fn push(&mut self, frames: &[VisemeFrame]) {
        let now = Instant::now();
        let base = match self.tail {
            Some(t) if t > now => t,
            _ => now,
        };
        for f in frames {
            let at = base
                .checked_add(Duration::from_millis(u64::from(f.t_ms)))
                .unwrap_or(base);
            self.queue.push_back(Scheduled {
                at,
                openness: f.openness,
                vowel: f.vowel,
            });
        }
        if let Some(last) = frames.last() {
            self.tail = base.checked_add(Duration::from_millis(
                u64::from(last.t_ms).saturating_add(40),
            ));
        }
    }

    /// Drop everything (barge-in / new user turn).
    pub(crate) fn clear(&mut self) {
        self.queue.clear();
        self.tail = None;
    }

    /// Pose due at `now`, advancing past stale frames. `None` once the
    /// schedule is exhausted (mouth returns to rest).
    fn current(&mut self, now: Instant) -> Option<(f32, Vowel)> {
        while self.queue.len() > 1 {
            if self.queue.get(1).is_some_and(|s| s.at <= now) {
                self.queue.pop_front();
            } else {
                break;
            }
        }
        match self.queue.front() {
            Some(s) if s.at <= now => Some((s.openness, s.vowel)),
            Some(_) => Some((0.0, Vowel::Sil)),
            None => {
                self.tail = None;
                None
            }
        }
    }
}

/// Which part of the avatar a point belongs to (drives its animation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Skull,
    Bust,
    Eye,
    Mouth,
}

/// A single point: rest position, surface normal (for back-cull + facing
/// shade so the form reads), role, and a per-point phase seed for shimmer.
#[derive(Component)]
struct Pt {
    home: Vec3,
    normal: Vec3,
    role: Role,
    seed: f32,
}

/// The cloud's parent entity; rotated/scaled for the idle turn + breathing.
#[derive(Component)]
struct CloudRoot;

/// Normalised vertical extent (feet `0` → head `max_y`), for the scan sweep.
#[derive(Resource, Clone, Copy)]
struct Bounds {
    max_y: f32,
}

/// Interactive orbit camera. Left-drag orbits, wheel zooms, ↑/↓ pans the
/// look-at height, `L` logs the current values so they can be pinned.
#[derive(Resource, Clone, Copy)]
struct OrbitCam {
    target_y: f32,
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl OrbitCam {
    fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(
            self.dist * cp * sy,
            self.target_y + self.dist * sp,
            self.dist * cp * cy,
        )
    }
}

/// Idle-blink bookkeeping (system-local), timed on `Time::elapsed_secs`.
#[derive(Debug, Default)]
pub(crate) struct BlinkClock {
    next_at: f32,
    end_at: f32,
}

// Procedural-fallback layout (only used if no model file is found).
const HEAD_C: Vec3 = Vec3::new(0.0, 0.60, 0.0);
const HEAD_R: Vec3 = Vec3::new(0.46, 0.55, 0.46);
const EYE_Y: f32 = 0.68;
const MOUTH_Y: f32 = 0.42;
const FACE_Z: f32 = 0.40;
const BLINK_DUR: f32 = 0.12;

/// Golden-angle in radians, for fibonacci-sphere / disc sampling.
const GOLDEN_ANGLE: f32 = 2.399_963_2;

/// All clouds are normalised to this height (feet at y=0), so the camera
/// and the animation bands are model-independent.
const TARGET_H: f32 = 1.7;
/// Max points kept (auto-instanced; subsampled from the model).
const TARGET_PTS: usize = 16_000;
/// Normalised-space role bands: the head's lower ring drives lip-sync; the
/// top cap is treated as skull/hair. Tuned for a full-body humanoid.
const MOUTH_CY: f32 = TARGET_H * 0.895;
const EYE_CY: f32 = TARGET_H * 0.945;
const HEAD_LO: f32 = TARGET_H * 0.86;
const HEAD_HI: f32 = TARGET_H * 0.93;

/// Per-channel quantisation of the normal→colour map (`LEVELS³` distinct
/// emissive materials max → still auto-instanced). Higher = smoother.
const NORMAL_LEVELS: u8 = 6;
/// HDR scale of the normal colour so bloom glows without washing to white.
const EMISSIVE_K: f32 = 3.0;

/// Quantise a unit normal to a small RGB bucket via `n*0.5 + 0.5`
/// (the classic "vertex-normals" visualisation), so close orientations
/// share one material.
fn normal_key(n: Vec3) -> [u8; 3] {
    let lf = f32::from(NORMAL_LEVELS - 1);
    let q = |c: f32| (((c * 0.5) + 0.5).clamp(0.0, 1.0) * lf).round() as u8;
    [q(n.x), q(n.y), q(n.z)]
}

/// The HDR emissive colour for a normal bucket.
fn key_color(k: [u8; 3]) -> LinearRgba {
    let lf = f32::from(NORMAL_LEVELS - 1);
    let f = |b: u8| f32::from(b) / lf * EMISSIVE_K;
    LinearRgba::rgb(f(k[0]), f(k[1]), f(k[2]))
}

/// Plugin: resources + the spawn and animation systems.
pub(crate) struct CloudPlugin;

impl Plugin for CloudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VisemeSchedule>()
            .insert_resource(ClearColor(Color::srgb(0.010, 0.012, 0.020)))
            .add_systems(Startup, setup)
            .add_systems(Update, (orbit_camera, animate_cloud));
    }
}

/// Fibonacci point `i` of `n` on the unit sphere.
fn fib_sphere(i: usize, n: usize) -> Vec3 {
    let y = 1.0 - (i as f32 + 0.5) / n as f32 * 2.0;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = i as f32 * GOLDEN_ANGLE;
    Vec3::new(theta.cos() * r, y, theta.sin() * r)
}

/// Build the full procedural cloud: `(position, role)` per point.
fn build_points() -> Vec<(Vec3, Role)> {
    let mut pts = Vec::with_capacity(2200);

    // Skull: a fibonacci-sampled ellipsoid.
    let skull_n = 1300;
    for i in 0..skull_n {
        let u = fib_sphere(i, skull_n);
        pts.push((
            Vec3::new(
                u.x.mul_add(HEAD_R.x, HEAD_C.x),
                u.y.mul_add(HEAD_R.y, HEAD_C.y),
                u.z.mul_add(HEAD_R.z, HEAD_C.z),
            ),
            Role::Skull,
        ));
    }

    // Bust: stacked elliptical rings widening downward.
    let rings = 22;
    for row in 0..rings {
        let f = row as f32 / (rings as f32 - 1.0);
        let y = 0.10_f32.mul_add(1.0 - f, -0.55 * f) + 0.10; // 0.20 → -0.45
        let rx = 0.30_f32.mul_add(f, 0.20);
        let rz = 0.18_f32.mul_add(f, 0.12);
        let per = 26_usize.saturating_add(row);
        for k in 0..per {
            let a = k as f32 / per as f32 * std::f32::consts::TAU;
            pts.push((Vec3::new(a.cos() * rx, y, a.sin() * rz + 0.02), Role::Bust));
        }
    }

    // Eyes: two small fibonacci discs on the face.
    for &sx in &[-0.155_f32, 0.155] {
        for i in 0..46 {
            let r = ((i as f32 + 0.5) / 46.0).sqrt() * 0.058;
            let a = i as f32 * GOLDEN_ANGLE;
            pts.push((
                Vec3::new(a.cos() * r + sx, a.sin() * r + EYE_Y, FACE_Z),
                Role::Eye,
            ));
        }
    }

    // Mouth: a thin ellipse band (animated open by `openness`).
    let mouth_n = 80;
    for i in 0..mouth_n {
        let a = i as f32 / mouth_n as f32 * std::f32::consts::TAU;
        pts.push((
            Vec3::new(a.cos() * 0.135, a.sin() * 0.028 + MOUTH_Y, FACE_Z + 0.01),
            Role::Mouth,
        ));
    }

    pts
}

/// Model file: `LANA_AVATAR_MODEL`, else the first `.glb`/`.vrm`/`.pcd`
/// in the working directory (so dropping one at the repo root just works).
fn model_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("LANA_AVATAR_MODEL") {
        return Some(PathBuf::from(p));
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(".")
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_ascii_lowercase)
                    .as_deref(),
                Some("glb" | "vrm" | "pcd")
            )
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// Axis-aligned bounds of a point set.
fn aabb(v: &[Vec3]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for p in v {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    (lo, hi)
}

/// Centre on X/Z, drop feet to `y = 0`, scale to [`TARGET_H`].
fn normalize(p: Vec3, lo: Vec3, hi: Vec3) -> Vec3 {
    let s = TARGET_H / (hi.y - lo.y).max(1e-3);
    Vec3::new(
        (p.x - (lo.x + hi.x) * 0.5) * s,
        (p.y - lo.y) * s,
        (p.z - (lo.z + hi.z) * 0.5) * s,
    )
}

/// One ready-to-spawn point: normalised position, normal, role.
type Spawn = (Vec3, Vec3, Role);

/// Load the avatar cloud: a sampled model (positions + normals) if one is
/// present, else the procedural fallback. Output is normalised + tagged.
fn load_cloud() -> Vec<Spawn> {
    if let Some(path) = model_path() {
        if let Some(v) = crate::glb::sample_points(&path, TARGET_PTS) {
            info!(model = %path.display(), points = v.len(), "avatar: model point cloud");
            let verts: Vec<Vec3> = v.iter().map(|(p, _)| *p).collect();
            let (lo, hi) = aabb(&verts);
            return v
                .into_iter()
                .map(|(p, nrm)| {
                    let n = normalize(p, lo, hi);
                    let role = if n.y > HEAD_HI {
                        Role::Skull
                    } else if n.y > HEAD_LO {
                        Role::Mouth
                    } else {
                        Role::Bust
                    };
                    // Uniform scale + translation preserves normal direction.
                    (n, nrm.normalize_or_zero(), role)
                })
                .collect();
        }
        warn!(model = %path.display(), "avatar: model sampling failed — procedural fallback");
    }
    let pts = build_points();
    let verts: Vec<Vec3> = pts.iter().map(|(p, _)| *p).collect();
    let (lo, hi) = aabb(&verts);
    let ctr = Vec3::new(
        (lo.x + hi.x) * 0.5,
        (lo.y + hi.y) * 0.5,
        (lo.z + hi.z) * 0.5,
    );
    pts.into_iter()
        .map(|(p, r)| {
            // No mesh normals in the fallback: use the outward radial.
            let nrm = Vec3::new(p.x - ctr.x, p.y - ctr.y, p.z - ctr.z).normalize_or_zero();
            (normalize(p, lo, hi), nrm, r)
        })
        .collect()
}

/// HDR + bloom camera, near-black scene, and the spawned point cloud
/// (one shared sphere mesh + one emissive material per role → auto-instanced).
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cloud = load_cloud();
    let max_y = cloud.iter().map(|(p, _, _)| p.y).fold(0.0_f32, f32::max);
    // Face-focused start; the user then orbits/zooms live (left-drag,
    // wheel, ↑/↓ pan) and presses `L` to log values to pin as defaults.
    let cam_dist = std::env::var("LANA_AVATAR_CAM_DIST")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|d| *d > 0.05)
        .unwrap_or(max_y * 0.40);
    let cam_y_frac = std::env::var("LANA_AVATAR_CAM_Y")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|f| (0.0..=1.5).contains(f))
        .unwrap_or(0.92);
    let orbit = OrbitCam {
        target_y: max_y * cam_y_frac,
        yaw: 0.0,
        pitch: 0.08,
        dist: cam_dist,
    };

    // `Bloom` requires the HDR pipeline (a required component in Bevy
    // 0.18); `TonyMcMapface` tames the glow. `orbit_camera` repositions
    // this camera from `OrbitCam` every frame.
    commands.spawn((
        Camera3d::default(),
        Tonemapping::TonyMcMapface,
        Bloom::NATURAL,
        Transform::from_translation(orbit.eye())
            .looking_at(Vec3::new(0.0, orbit.target_y, 0.0), Vec3::Y),
    ));

    // Small points so the face doesn't fuse; colour = the vertex normal
    // (the "vertex-normals" visualiser look) so the relief reads, glowing
    // via bloom. One emissive material per quantised normal bucket,
    // built up-front (≤ NORMAL_LEVELS³ ≈ 216 → still auto-instanced).
    let dot = meshes.add(Mesh::from(Sphere::new(0.0032)));
    let mut mats: HashMap<[u8; 3], Handle<StandardMaterial>> = HashMap::new();
    for (_, normal, _) in &cloud {
        let key = normal_key(*normal);
        mats.entry(key).or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::BLACK,
                emissive: key_color(key),
                ..default()
            })
        });
    }

    info!(
        points = cloud.len(),
        materials = mats.len(),
        dist = orbit.dist,
        target_y = orbit.target_y,
        "point-cloud avatar ready (left-drag orbit · wheel zoom · ↑/↓ pan · L logs)"
    );
    commands
        .spawn((CloudRoot, Transform::default(), Visibility::default()))
        .with_children(|root| {
            for (idx, (home, normal, role)) in cloud.into_iter().enumerate() {
                let Some(mat) = mats.get(&normal_key(normal)) else {
                    continue;
                };
                root.spawn((
                    Pt {
                        home,
                        normal,
                        role,
                        seed: (idx as f32 * 0.618_034).fract(),
                    },
                    Mesh3d(dot.clone()),
                    MeshMaterial3d(mat.clone()),
                    Transform::from_translation(home),
                ));
            }
        });
    commands.insert_resource(Bounds { max_y });
    commands.insert_resource(orbit);
}

/// Interactive orbit camera: left-drag orbits, wheel zooms, ↑/↓ pans the
/// look-at height, `L` logs the current pose so it can be pinned as the
/// `LANA_AVATAR_CAM_*` defaults.
fn orbit_camera(
    time: Res<Time>,
    mut orbit: ResMut<OrbitCam>,
    mouse_btn: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut cam: Query<&mut Transform, With<Camera3d>>,
) {
    // Accumulate input as plain f32 (avoids glam-operator lint / float-eq).
    let (mut dx, mut dy) = (0.0_f32, 0.0_f32);
    if mouse_btn.pressed(MouseButton::Left) {
        for m in motion.read() {
            dx += m.delta.x;
            dy += m.delta.y;
        }
    } else {
        motion.clear();
    }
    let mut scroll = 0.0_f32;
    for w in wheel.read() {
        scroll += w.y;
    }
    let dt = time.delta_secs();
    let mut pan = 0.0_f32;
    if keys.pressed(KeyCode::ArrowUp) {
        pan += 1.0;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        pan -= 1.0;
    }

    // Always apply (zero input → no-op), so no equality tests are needed.
    orbit.yaw -= dx * 0.006;
    orbit.pitch = (orbit.pitch - dy * 0.006).clamp(-1.4, 1.4);
    orbit.dist = (orbit.dist * (-scroll * 0.12).exp()).clamp(0.05, 50.0);
    orbit.target_y += pan * dt * 0.6;

    if let Ok(mut tf) = cam.single_mut() {
        let target = Vec3::new(0.0, orbit.target_y, 0.0);
        *tf = Transform::from_translation(orbit.eye()).looking_at(target, Vec3::Y);
    }

    if keys.just_pressed(KeyCode::KeyL) {
        info!(
            dist = orbit.dist,
            target_y = orbit.target_y,
            yaw_deg = orbit.yaw.to_degrees(),
            pitch_deg = orbit.pitch.to_degrees(),
            "orbit cam (pin via LANA_AVATAR_CAM_DIST / LANA_AVATAR_CAM_Y)"
        );
    }
}

/// Pseudo-random next-blink interval (≈2.5–5.5 s, no RNG dep).
fn blink_interval(t: f32) -> f32 {
    1.5_f32.mul_add((t * 1.37).sin(), 4.0)
}

/// Braindance-style animation: a slow turn, a vertical scan sweep that
/// brightens the band it crosses, per-point flicker/jitter ("imperfect
/// scan"), the mouth band opening with the viseme schedule, and an
/// irregular eye blink (procedural fallback only).
fn animate_cloud(
    time: Res<Time>,
    bounds: Option<Res<Bounds>>,
    orbit: Option<Res<OrbitCam>>,
    mut schedule: ResMut<VisemeSchedule>,
    mut blink: Local<BlinkClock>,
    mut roots: Query<&mut Transform, With<CloudRoot>>,
    mut pts: Query<(&Pt, &mut Transform), Without<CloudRoot>>,
) {
    let t = time.elapsed_secs();
    let top = bounds.map_or(TARGET_H, |b| b.max_y);
    // Camera position from the orbit pose (the root barely rotates, so a
    // point's animated position ≈ its world position).
    let cam_pos = orbit.as_deref().map(OrbitCam::eye);

    // Root: a very slow drift only — the user orbits the camera, so the
    // cloud itself stays nearly still for precise inspection. The same
    // yaw is applied to normals below so the facing shade stays correct.
    let root_yaw = (t * 0.15).sin() * 0.08;
    let yaw_q = Quat::from_rotation_y(root_yaw);
    for mut tf in &mut roots {
        tf.rotation = yaw_q;
        tf.scale = Vec3::splat((t * 1.4).sin().mul_add(0.006, 1.0));
    }

    let openness = schedule
        .current(Instant::now())
        .map_or(0.0, |(o, _)| o)
        .clamp(0.0, 1.0);

    // Vertical scan plane sweeping the figure (period ≈ 7.8 s).
    let scan_y = ((t * 0.8).sin() * 0.5 + 0.5) * top;
    let scan_band = top * 0.045;

    // Irregular blink envelope (0 → 1 → 0 over BLINK_DUR).
    if blink.next_at <= 0.0 {
        blink.next_at = t + blink_interval(t);
    }
    if t >= blink.next_at && t >= blink.end_at {
        blink.end_at = t + BLINK_DUR;
        blink.next_at = t + BLINK_DUR + blink_interval(t);
    }
    let blink_w = if t < blink.end_at {
        let p = 1.0 - ((blink.end_at - t) / BLINK_DUR).clamp(0.0, 1.0);
        (p * std::f32::consts::PI).sin()
    } else {
        0.0
    };

    for (pt, mut tf) in &mut pts {
        let ph = pt.seed * std::f32::consts::TAU;
        let shimmer = Vec3::new(
            (t * 1.7 + ph).sin() * 0.004,
            (t * 1.5 + ph).cos() * 0.004,
            (t * 1.3 + ph).sin() * 0.004,
        );
        let mut p = Vec3::new(
            pt.home.x + shimmer.x,
            pt.home.y + shimmer.y,
            pt.home.z + shimmer.z,
        );
        match pt.role {
            Role::Mouth => {
                // The lower-head band opens with the spoken `openness`.
                let spread = (pt.home.y - MOUTH_CY) * (openness * 3.0);
                p.y += spread;
            }
            Role::Eye => {
                let f = 1.0 - blink_w;
                p.y = EYE_CY + (pt.home.y - EYE_CY) * f + shimmer.y;
            }
            Role::Skull | Role::Bust => {}
        }
        tf.translation = p;

        // Scan sweep: points the plane crosses flare brighter (bigger →
        // more bloom). Flicker: occasional brief dropout ("bad scan").
        let scan = (1.0 - (pt.home.y - scan_y).abs() / scan_band).clamp(0.0, 1.0);
        let flick = (t * 19.0 + pt.seed * 53.0).sin();
        let dim = if flick > 0.93 { 0.12 } else { 1.0 };

        // Normal-based reveal: cull points whose normal faces away from
        // the camera (no back-of-head bleed → clean shell) and shade the
        // rest by how squarely they face you, so cheeks/nose/brow model
        // instead of fusing into a white blob.
        let nw = yaw_q.mul_vec3(pt.normal);
        let view = cam_pos.map_or(Vec3::Z, |c| {
            Vec3::new(c.x - p.x, c.y - p.y, c.z - p.z).normalize_or_zero()
        });
        let facing = nw.dot(view);
        // Cull the back (clean front shell); keep the front fairly *flat*
        // in brightness so the per-point normal *colour* carries the form
        // (the vertex-normals look) instead of a white front gradient.
        let shade = if facing < 0.06 {
            0.0
        } else {
            0.55 + facing * 0.7
        };
        tf.scale = Vec3::splat(scan.mul_add(0.4, 1.0) * dim * shade);
    }
}
