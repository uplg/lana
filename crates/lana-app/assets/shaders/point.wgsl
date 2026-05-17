// Lana avatar point-cloud shader.
//
// One PointList mesh of the model's vertices (positions + normals). The
// colour IS the vertex normal (`n*0.5+0.5`, the classic "vertex-normals"
// visualiser) so the relief reads exactly; HDR-scaled so the camera's
// bloom makes it glow. Back-facing points are discarded for a clean
// shell. A thin Y band opens with the spoken `openness` (lip-sync); a
// scan plane sweeps and points flicker for the braindance feel — all on
// the GPU from the time `globals`, no CPU per-point work.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::{view, globals}

struct PointData {
    // x: openness 0..1 · y: emissive K (HDR) · z: back-cull threshold
    // w: mouth-band centre (normalised Y)
    p: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> data: PointData;

// Mouth band half-height and open amplitude (normalised-Y units).
const MOUTH_HALF: f32 = 0.022;
const MOUTH_AMP: f32 = 0.055;
// Figure height the cloud is normalised to (matches Rust `TARGET_H`).
const SPAN: f32 = 1.7;

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
};

@vertex
fn vertex(v: Vertex) -> VOut {
    var pos = v.position;

    // Lip-sync: only a thin band around the mouth centre moves, pushed
    // away from the centre line by `openness` — never the whole head.
    let dy = pos.y - data.p.w;
    let band = 1.0 - smoothstep(0.0, MOUTH_HALF, abs(dy));
    pos.y = pos.y + sign(dy) * band * data.p.x * MOUTH_AMP;

    let m = get_world_from_local(v.ii);
    var out: VOut;
    out.clip = mesh_position_local_to_clip(m, vec4<f32>(pos, 1.0));
    out.wpos = (m * vec4<f32>(pos, 1.0)).xyz;
    out.wnormal = normalize((m * vec4<f32>(v.normal, 0.0)).xyz);
    out.ny = v.position.y;
    return out;
}

@fragment
fn fragment(in: VOut) -> @location(0) vec4<f32> {
    let n = normalize(in.wnormal);
    let viewdir = normalize(view.world_position - in.wpos);
    // Discard the back so we see a clean front shell, not a solid mass.
    if (dot(n, viewdir) < data.p.z) {
        discard;
    }

    // Exact, unquantised vertex-normal colour.
    let col = n * 0.5 + 0.5;

    // Braindance scan sweep + sparse flicker, driven by global time.
    let scan_y = (sin(globals.time * 0.8) * 0.5 + 0.5) * SPAN;
    let scan = clamp(1.0 - abs(in.ny - scan_y) / (SPAN * 0.05), 0.0, 1.0);
    let flick = step(0.93, sin(globals.time * 19.0 + in.ny * 53.0));
    let bright = (0.85 + scan * 0.55) * (1.0 - flick * 0.8);

    return vec4<f32>(col * data.p.y * bright, 1.0);
}
