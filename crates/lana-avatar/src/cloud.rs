//! Point-cloud avatar — a Cyberpunk-braindance "scan hologram".
//!
//! A model's vertices (positions + normals, sampled by [`crate::glb`]) are
//! one `PointList` mesh drawn by a custom WGSL [`PointMaterial`] embedded
//! in the binary (`point.wgsl`): the colour IS the vertex normal
//! (`n*0.5+0.5`, exact/unquantised), HDR-scaled so the camera bloom makes
//! it glow; back-facing points are discarded for a clean shell; the mouth
//! cluster drops with the spoken `openness` (lip-sync); the eyes get an
//! auto-centred iris; a scan/flicker + per-point jitter play — all on the
//! GPU. No model file ⇒ a procedural fallback cloud.

// Bounded procedural geometry (fallback only): small counts, range-safe
// casts; FMA rewrites only obscure the position formulas. resample.rs
// precedent.
#![expect(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "bounded procedural/anim geometry; small counts, FMA rewrites \
              only obscure the position/idle formulas (resample.rs precedent)"
)]

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::mesh::{MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use lana_viseme::VisemeFrame;

/// One scheduled mouth pose, timed on the wall clock so it lines up with
/// the speaker output the orchestrator enqueued at the same instant.
#[derive(Debug, Clone, Copy)]
struct Scheduled {
    at: Instant,
    openness: f32,
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

    /// Openness due at `now`, advancing past stale frames. `0.0` once the
    /// schedule is exhausted (mouth closes).
    fn openness(&mut self, now: Instant) -> f32 {
        while self.queue.len() > 1 {
            if self.queue.get(1).is_some_and(|s| s.at <= now) {
                self.queue.pop_front();
            } else {
                break;
            }
        }
        let Some(s) = self.queue.front() else {
            self.tail = None;
            return 0.0;
        };
        if s.at <= now { s.openness } else { 0.0 }
    }
}

/// The cloud's root entity; a slow idle drift composes with the orbit cam.
#[derive(Component)]
struct CloudRoot;

/// Handle to the live point material (its uniform is updated each frame).
#[derive(Resource)]
struct PointMat(Handle<PointMaterial>);

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

/// Custom point material — three `Vec4` uniforms mirroring the WGSL
/// `b0`/`b1`/`b2` (mouth / jitter / iris params; see fields).
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct PointMaterial {
    /// x openness · y emissive K · z back-cull · w mouth-band centre.
    #[uniform(0)]
    p: Vec4,
    /// x mouth-band half-height · y open amplitude · z jitter · w mouth
    /// X half-width (isolates the mouth cluster).
    #[uniform(1)]
    q: Vec4,
    /// x eye-centre Y · y eye-centre |X| · z eye radius · w pupil radius
    /// (the iris that de-uncannies the eyes).
    #[uniform(2)]
    r: Vec4,
    /// x eye-centre Z (front of the eyeball) · yzw spare — gives the iris
    /// a 3D mask so it's a clean round patch, not a smeared XY disc.
    #[uniform(3)]
    s: Vec4,
}

/// Embedded shader path (see `embedded_asset!` in [`CloudPlugin::build`]).
const POINT_SHADER: &str = "embedded://lana_avatar/point.wgsl";

impl Material for PointMaterial {
    fn vertex_shader() -> ShaderRef {
        POINT_SHADER.into()
    }
    fn fragment_shader() -> ShaderRef {
        POINT_SHADER.into()
    }
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.topology = PrimitiveTopology::PointList;
        let vbl = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
        ])?;
        descriptor.vertex.buffers = vec![vbl];
        Ok(())
    }
}

/// Golden-angle in radians, for fibonacci-sphere sampling (fallback only).
const GOLDEN_ANGLE: f32 = 2.399_963_2;
/// All clouds are normalised to this height (feet at y=0).
const TARGET_H: f32 = 1.7;
/// Max points kept (one draw call — keep it fine).
const TARGET_PTS: usize = 120_000;
/// Mouth-band centre in normalised Y (≈ lower face of a full-body model).
const MOUTH_CY: f32 = TARGET_H * 0.89;
/// HDR scale of the normal colour (bloom glow without washing to white).
const EMISSIVE_K: f32 = 2.6;
/// Back-cull threshold: keep points whose normal faces the camera.
const CULL: f32 = 0.05;
/// Procedural-fallback head ellipsoid (only if no model file is found).
const HEAD_C: Vec3 = Vec3::new(0.0, 0.60, 0.0);
const HEAD_R: Vec3 = Vec3::new(0.46, 0.62, 0.46);

/// Plugin: the material, resources and systems.
pub(crate) struct CloudPlugin;

impl Plugin for CloudPlugin {
    fn build(&self, app: &mut App) {
        // Ship the shader inside the binary (resolves to
        // `embedded://lana_avatar/point.wgsl`) — no `assets/` dir, immune
        // to the cwd/crate-manifest path resolution that broke before.
        embedded_asset!(app, "point.wgsl");
        app.init_resource::<VisemeSchedule>()
            .insert_resource(ClearColor(Color::srgb(0.010, 0.012, 0.020)))
            .add_plugins(MaterialPlugin::<PointMaterial>::default())
            .add_systems(Startup, setup)
            .add_systems(Update, (orbit_camera, animate));
    }
}

/// Fibonacci point `i` of `n` on the unit sphere (fallback geometry).
fn fib_sphere(i: usize, n: usize) -> Vec3 {
    let y = 1.0 - (i as f32 + 0.5) / n as f32 * 2.0;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = i as f32 * GOLDEN_ANGLE;
    Vec3::new(theta.cos() * r, y, theta.sin() * r)
}

/// Procedural fallback: a head ellipsoid + a bust shell, with outward
/// normals. Only used when no model file is present.
fn build_points() -> Vec<(Vec3, Vec3)> {
    let mut pts = Vec::with_capacity(4000);
    let skull_n = 2600;
    for i in 0..skull_n {
        let u = fib_sphere(i, skull_n);
        let p = Vec3::new(
            u.x.mul_add(HEAD_R.x, HEAD_C.x),
            u.y.mul_add(HEAD_R.y, HEAD_C.y),
            u.z.mul_add(HEAD_R.z, HEAD_C.z),
        );
        pts.push((p, u));
    }
    let rings = 26;
    for row in 0..rings {
        let f = row as f32 / (rings as f32 - 1.0);
        let y = 0.10_f32.mul_add(1.0 - f, -0.55 * f) + 0.10;
        let rx = 0.30_f32.mul_add(f, 0.20);
        let rz = 0.18_f32.mul_add(f, 0.12);
        let per = 40_usize.saturating_add(row);
        for k in 0..per {
            let a = k as f32 / per as f32 * std::f32::consts::TAU;
            let p = Vec3::new(a.cos() * rx, y, a.sin() * rz + 0.02);
            pts.push((p, Vec3::new(a.cos(), 0.2, a.sin()).normalize_or_zero()));
        }
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

/// Raw `(position, normal)` pairs: the sampled model, else the procedural
/// fallback (logged either way).
fn load_raw() -> Vec<(Vec3, Vec3)> {
    let Some(path) = model_path() else {
        return build_points();
    };
    let Some(v) = crate::glb::sample_points(&path, TARGET_PTS) else {
        warn!(model = %path.display(), "avatar: model sampling failed — procedural fallback");
        return build_points();
    };
    info!(model = %path.display(), points = v.len(), "avatar: model point cloud");
    v
}

/// Load `(position, normal)` pairs, normalised to [`TARGET_H`].
fn load_cloud() -> Vec<(Vec3, Vec3)> {
    let raw = load_raw();
    let verts: Vec<Vec3> = raw.iter().map(|(p, _)| *p).collect();
    let (lo, hi) = aabb(&verts);
    raw.into_iter()
        .map(|(p, n)| (normalize(p, lo, hi), n.normalize_or_zero()))
        .collect()
}

/// Auto-detect the eye centroids. Eyes are the forward-facing points (a
/// convex cornea: `n.z > 0`, front half) in a Y band around eye level,
/// split left/right in X. Returns the mean `(|x|, y, z)` of both eyes, or
/// `None` if either side has no points (caller falls back to env coords).
fn detect_eyes(cloud: &[(Vec3, Vec3)], y0: f32, band: f32) -> Option<(f32, f32, f32)> {
    let (mut lx, mut ly, mut lz, mut ln) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let (mut rx, mut ry, mut rz, mut rn) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for (p, n) in cloud {
        if (p.y - y0).abs() >= band || p.z <= 0.0 || n.z <= 0.2 {
            continue;
        }
        if p.x < 0.0 {
            lx += p.x;
            ly += p.y;
            lz += p.z;
            ln += 1.0;
        } else {
            rx += p.x;
            ry += p.y;
            rz += p.z;
            rn += 1.0;
        }
    }
    if ln <= 0.0 || rn <= 0.0 {
        return None;
    }
    Some((
        ((lx / ln).abs() + rx / rn) * 0.5,
        (ly / ln + ry / rn) * 0.5,
        (lz / ln + rz / rn) * 0.5,
    ))
}

/// Camera (orbit + bloom), the point-cloud mesh and its material.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PointMaterial>>,
) {
    let cloud = load_cloud();
    let max_y = cloud.iter().map(|(p, _)| p.y).fold(0.0_f32, f32::max);

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(cloud.len());
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(cloud.len());
    for (p, n) in &cloud {
        positions.push([p.x, p.y, p.z]);
        normals.push([n.x, n.y, n.z]);
    }
    let mesh = Mesh::new(PrimitiveTopology::PointList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    let mesh = meshes.add(mesh);

    // Mouth band + jitter are env-tunable (the user dials them on-device
    // since the render can't be seen here; the band also glows so it's
    // visible where the lip-sync acts). All in normalised-Y units.
    let envf = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(d)
    };
    let mouth_y = envf("LANA_MOUTH_Y", MOUTH_CY);
    // Thin band, subtle asymmetric jaw drop (was head-splitting at 0.045).
    let mouth_h = envf("LANA_MOUTH_H", 0.010);
    let mouth_amp = envf("LANA_MOUTH_AMP", 0.008);
    // X half-width so only the central mouth cluster reacts, not the slice.
    let mouth_w = envf("LANA_MOUTH_W", 0.03);
    // A touch more jitter → organic nebula, less uncanny rigid scan.
    let jitter = envf("LANA_PT_JITTER", 0.006);
    // Iris auto-centred on the real eye cluster (see `detect_eyes`);
    // `LANA_EYE_Y0`/`LANA_EYE_BAND` seed the search, fixed env coords are
    // the fallback if the cluster isn't found.
    let eye_y0 = envf("LANA_EYE_Y0", TARGET_H * 0.92);
    let eye_band = envf("LANA_EYE_BAND", 0.11);
    let (eye_x, eye_y, eye_z) = detect_eyes(&cloud, eye_y0, eye_band).unwrap_or_else(|| {
        warn!("avatar: eye cluster not found — using fixed LANA_EYE_X/Y/Z");
        (
            envf("LANA_EYE_X", 0.07),
            envf("LANA_EYE_Y", eye_y0),
            envf("LANA_EYE_Z", 0.4),
        )
    });
    // Geometric eye = two concentric particle rings: outer ring radius
    // (LANA_EYE_R) and inner ring radius (LANA_PUPIL_R). Small vs the
    // ~0.07 inter-eye spacing.
    let eye_r = envf("LANA_EYE_R", 0.018);
    let pupil_r = envf("LANA_PUPIL_R", 0.008);
    // Anti-uncanny: coherent flow drift, animated dissolve, breath+hue.
    let flow = envf("LANA_FLOW", 0.010);
    let dissolve = envf("LANA_DISSOLVE", 0.15);
    let life = envf("LANA_LIFE", 1.0);
    info!(
        eye_x,
        eye_y, eye_z, "avatar: iris centred on eye cluster (3D)"
    );
    let mat = materials.add(PointMaterial {
        p: Vec4::new(0.0, EMISSIVE_K, CULL, mouth_y),
        q: Vec4::new(mouth_h, mouth_amp, jitter, mouth_w),
        r: Vec4::new(eye_y, eye_x, eye_r, pupil_r),
        s: Vec4::new(eye_z, flow, dissolve, life),
    });

    // Pinned from the user's logged "perfect" pose (orbit-tunable live).
    let cam_dist = std::env::var("LANA_AVATAR_CAM_DIST")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|d| *d > 0.05)
        .unwrap_or(0.535);
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

    // `Bloom` pulls in the HDR pipeline (required component in Bevy 0.18);
    // `TonyMcMapface` tames the glow.
    commands.spawn((
        Camera3d::default(),
        Tonemapping::TonyMcMapface,
        Bloom::NATURAL,
        Transform::from_translation(orbit.eye())
            .looking_at(Vec3::new(0.0, orbit.target_y, 0.0), Vec3::Y),
    ));
    commands.spawn((
        CloudRoot,
        Mesh3d(mesh),
        MeshMaterial3d(mat.clone()),
        Transform::default(),
        Visibility::default(),
    ));
    info!(
        points = cloud.len(),
        dist = orbit.dist,
        target_y = orbit.target_y,
        "point-cloud avatar ready (left-drag orbit · wheel zoom · ↑/↓ pan · L logs)"
    );
    commands.insert_resource(orbit);
    commands.insert_resource(PointMat(mat));
}

/// Interactive orbit camera: left-drag orbits, wheel zooms, ↑/↓ pans the
/// look-at height, `L` logs the pose to pin as the `LANA_AVATAR_CAM_*`
/// defaults.
fn orbit_camera(
    time: Res<Time>,
    mut orbit: ResMut<OrbitCam>,
    mouse_btn: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut cam: Query<&mut Transform, With<Camera3d>>,
) {
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

/// Feed `openness` into the material uniform (the shader does the mouth
/// open + scan/flicker), and a faint idle drift on the cloud root.
fn animate(
    time: Res<Time>,
    mut schedule: ResMut<VisemeSchedule>,
    pm: Option<Res<PointMat>>,
    mut materials: ResMut<Assets<PointMaterial>>,
    mut roots: Query<&mut Transform, With<CloudRoot>>,
) {
    let openness = schedule.openness(Instant::now()).clamp(0.0, 1.0);
    if let Some(m) = pm.as_ref().and_then(|pm| materials.get_mut(&pm.0)) {
        m.p.x = openness;
    }

    let t = time.elapsed_secs();
    for mut tf in &mut roots {
        tf.rotation = Quat::from_rotation_y((t * 0.15).sin() * 0.06);
        tf.scale = Vec3::splat((t * 1.4).sin().mul_add(0.005, 1.0));
    }
}
