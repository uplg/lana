// Lana avatar point-cloud shader (embedded — no assets/ dir needed).
//
// One PointList mesh of the model's vertices+normals. Colour IS the
// vertex normal (`n*0.5+0.5`, exact/unquantised), HDR-scaled so the
// camera bloom glows; back-facing points are discarded for a clean
// shell. Anti-uncanny: a *coherent flow field* makes the cloud drift
// like living energy (not a rigid scan); the iris wanders + blinks; the
// surface dissolves/recomposes; a slow breath + hue drift give life. A
// thin mouth cluster drops + glows with the spoken `openness`. All GPU.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::{view, globals}

// b0: x openness · y emissive K (HDR) · z back-cull · w mouth-band centre Y
struct U0 { v: vec4<f32> };
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> b0: U0;
// b1: x mouth-band half-height · y open amplitude · z fine jitter
//     w mouth X half-width (isolates the mouth cluster)
struct U1 { v: vec4<f32> };
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> b1: U1;
// b2: x eye-centre Y · y eye-centre |X| · z eye radius · w pupil radius
struct U2 { v: vec4<f32> };
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> b2: U2;
// b3: x eye-centre Z · y flow amplitude · z dissolve amount · w life amount
struct U3 { v: vec4<f32> };
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> b3: U3;

const SPAN: f32 = 1.7;

fn hash13(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

// Smooth, spatially-coherent vector field: nearby points drift together
// (like smoke / a data stream), not independent vibration.
fn flow(p: vec3<f32>, t: f32) -> vec3<f32> {
    return vec3<f32>(
        sin(p.y * 2.3 + t * 0.6) + cos(p.z * 1.9 - t * 0.4),
        sin(p.z * 2.1 + t * 0.5) + cos(p.x * 2.4 + t * 0.35),
        sin(p.x * 2.0 - t * 0.45) + cos(p.y * 1.7 + t * 0.55),
    );
}

// Rotate an RGB colour about the grey axis (a hue shift, luminance kept).
fn hue_rot(c: vec3<f32>, a: f32) -> vec3<f32> {
    let k = vec3<f32>(0.57735, 0.57735, 0.57735);
    let ct = cos(a);
    return c * ct + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ct);
}

struct Vertex {
    @builtin(instance_index) ii: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wnormal: vec3<f32>,
    @location(1) wpos: vec3<f32>,
    @location(2) ny: f32,
    @location(3) h: f32,
    @location(4) nx: f32,
    @location(5) nz: f32,
};

@vertex
fn vertex(v: Vertex) -> VOut {
    var pos = v.position;
    let h = hash13(v.position);
    let ph = h * 6.2831853;

    // Coherent flow drift (the big anti-uncanny: living energy, not a
    // frozen scan) + a touch of fine per-point jitter on top.
    pos = pos + flow(v.position, globals.time) * b3.v.y;
    let j = b1.v.z;
    pos.x = pos.x + sin(globals.time * 1.7 + ph) * j;
    pos.y = pos.y + cos(globals.time * 1.5 + ph) * j;
    pos.z = pos.z + sin(globals.time * 1.3 + ph) * j;

    // Lip-sync: a thin band just below the mouth line drops (jaw / lower
    // lip) by `openness`, gated in X to the mouth cluster — asymmetric &
    // small, never splits the head.
    let below = clamp((b0.v.w - pos.y) / b1.v.x, 0.0, 1.0);
    let drop = below * (1.0 - below) * 4.0;
    let xw = 1.0 - smoothstep(0.0, b1.v.w, abs(v.position.x));
    pos.y = pos.y - drop * xw * b0.v.x * b1.v.y;

    let m = get_world_from_local(v.ii);
    var out: VOut;
    out.clip = mesh_position_local_to_clip(m, vec4<f32>(pos, 1.0));
    out.wpos = (m * vec4<f32>(pos, 1.0)).xyz;
    out.wnormal = normalize((m * vec4<f32>(v.normal, 0.0)).xyz);
    out.ny = v.position.y;
    out.nx = v.position.x;
    out.nz = v.position.z;
    out.h = h;
    return out;
}

@fragment
fn fragment(in: VOut) -> @location(0) vec4<f32> {
    let life = b3.v.w;

    // Dissolve / recompose: each point's hash + time picks an on/off
    // phase that scrolls, so zones materialise and fade (a hologram, not
    // a complete corpse).
    if (fract(in.h * 7.0 + globals.time * 0.08) < b3.v.z) {
        discard;
    }

    let n = normalize(in.wnormal);
    let viewdir = normalize(view.world_position - in.wpos);
    if (dot(n, viewdir) < b0.v.z) {
        discard;
    }

    var col = n * 0.5 + 0.5;

    // Synthetic geometric eye: TWO concentric glowing particle rings (a
    // HUD/optic), NOT a human iris+eyelid — the half-closed pink-lid look
    // was the core uncanny trigger, so it's gone entirely. The rings
    // slowly wander (a scanning optic reads as *alive*, not creepy).
    // Outer/inner ring radii are env-tunable (LANA_EYE_R / LANA_PUPIL_R).
    let gz = life * 0.010;
    let gaze = vec2<f32>(sin(globals.time * 0.37) * gz,
                         sin(globals.time * 0.53 + 1.3) * gz * 0.6);
    let eye_c = vec3<f32>(b2.v.y + gaze.x, b2.v.x + gaze.y, b3.v.x);
    let d_eye = distance(vec3<f32>(abs(in.nx), in.ny, in.nz), eye_c);
    let r_out = b2.v.z;
    if (d_eye < r_out) {
        let w = r_out * 0.16;
        let on = max(
            1.0 - smoothstep(0.0, w, abs(d_eye - r_out * 0.92)),
            1.0 - smoothstep(0.0, w, abs(d_eye - b2.v.w)),
        );
        if (on < 0.2) {
            discard; // hollow between/around → crisp particle rings
        }
        col = vec3<f32>(0.30, 0.95, 1.20) * (0.7 + on * 0.8);
    }

    // Braindance scan sweep.
    let scan_y = (sin(globals.time * 0.8) * 0.5 + 0.5) * SPAN;
    let scan = clamp(1.0 - abs(in.ny - scan_y) / (SPAN * 0.05), 0.0, 1.0);

    // The lower-mouth cluster glows a touch while speaking.
    let mlo = clamp((b0.v.w - in.ny) / b1.v.x, 0.0, 1.0);
    let mxw = 1.0 - smoothstep(0.0, b1.v.w, abs(in.nx));
    let speak = mlo * (1.0 - mlo) * 4.0 * mxw * b0.v.x;

    // Life: slow breath in brightness + a gentle hue drift.
    let breath = 1.0 + sin(globals.time * 0.9) * 0.05 * life;
    col = hue_rot(col, sin(globals.time * 0.05) * 0.5 * life);

    let bright = (0.8 + scan * 0.5 + speak * 0.45) * breath;
    return vec4<f32>(col * b0.v.y * bright, 1.0);
}
