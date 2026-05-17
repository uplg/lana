//! Vertex point-sampler for `.glb`/`.vrm` (binary glTF) and `.pcd`.
//!
//! We only want a *shape*, rendered as a glowing point cloud — no
//! skeleton, no morphs, no materials. So this reads raw vertex positions
//! **and normals** (every primitive's `POSITION`/`NORMAL` accessor for
//! glTF, or the XYZ fields for PCD). The normal lets the renderer cull
//! back-facing points and shade the surface so a face reads. Anything
//! malformed yields `None` (caller falls back to procedural) — no panic.

use std::path::Path;

use bevy::log::warn;
use bevy::prelude::Vec3;

/// A sampled vertex: world-ish position and its (model-space) normal.
pub(crate) type Point = (Vec3, Vec3);

/// Sample up to `target` `(position, normal)` pairs from the model at
/// `path`. The normal lets the renderer cull back-facing points and shade
/// the surface so a face actually reads. `None` on unsupported/malformed.
pub(crate) fn sample_points(path: &Path, target: usize) -> Option<Vec<Point>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let bytes = std::fs::read(path)
        .map_err(|e| warn!(error = %e, "model read failed"))
        .ok()?;

    let pts = match ext.as_str() {
        "glb" | "vrm" => glb_points(&bytes),
        "pcd" => pcd_points(&bytes),
        other => {
            warn!(ext = other, "unsupported avatar model extension");
            None
        }
    }?;
    if pts.is_empty() {
        return None;
    }
    Some(subsample(pts, target))
}

/// Keep at most `target` points by uniform stride (preserves overall form).
fn subsample(pts: Vec<Point>, target: usize) -> Vec<Point> {
    if target == 0 || pts.len() <= target {
        return pts;
    }
    let step = pts.len().div_ceil(target).max(1);
    pts.into_iter().step_by(step).collect()
}

fn u32_le(b: &[u8], off: usize) -> Option<u32> {
    let s = b.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn f32_le(b: &[u8], off: usize) -> Option<f32> {
    let s = b.get(off..off.checked_add(4)?)?;
    Some(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn json_u(v: &serde_json::Value, key: &str) -> Option<usize> {
    usize::try_from(v.get(key)?.as_u64()?).ok()
}

/// Read one FLOAT/VEC3 accessor (by index) fully into a `Vec<Vec3>`,
/// honoring its `bufferView` offset/stride. `None` if the accessor is not
/// FLOAT/VEC3 or any byte is out of range.
fn read_vec3(
    accessors: &[serde_json::Value],
    views: &[serde_json::Value],
    bin: &[u8],
    idx: usize,
) -> Option<Vec<Vec3>> {
    let acc = accessors.get(idx)?;
    if acc.get("componentType").and_then(serde_json::Value::as_u64) != Some(5126)
        || acc.get("type").and_then(serde_json::Value::as_str) != Some("VEC3")
    {
        return None;
    }
    let count = json_u(acc, "count")?;
    let acc_off = json_u(acc, "byteOffset").unwrap_or(0);
    let view = views.get(json_u(acc, "bufferView")?)?;
    let view_off = json_u(view, "byteOffset").unwrap_or(0);
    let stride = json_u(view, "byteStride").unwrap_or(12).max(12);
    let base = view_off.checked_add(acc_off)?;

    let mut out = Vec::with_capacity(count);
    for vi in 0..count {
        let off = base.checked_add(vi.checked_mul(stride)?)?;
        let (Some(fx), Some(fy), Some(fz)) = (
            f32_le(bin, off),
            f32_le(bin, off.checked_add(4)?),
            f32_le(bin, off.checked_add(8)?),
        ) else {
            break;
        };
        let vec = Vec3::new(fx, fy, fz);
        out.push(if vec.is_finite() { vec } else { Vec3::ZERO });
    }
    Some(out)
}

/// Collect every primitive's `(POSITION, NORMAL)` from a binary glTF.
/// Missing/invalid `NORMAL` falls back to a +Z normal for that primitive.
fn glb_points(bytes: &[u8]) -> Option<Vec<Point>> {
    if bytes.get(0..4)? != b"glTF" || u32_le(bytes, 4)? != 2 {
        return None;
    }
    let json_len = u32_le(bytes, 12)? as usize;
    let json_end = 20usize.checked_add(json_len)?;
    let root: serde_json::Value = serde_json::from_slice(bytes.get(20..json_end)?)
        .map_err(|e| warn!(error = %e, "glTF JSON parse failed"))
        .ok()?;

    // Chunk 1 is the BIN (`buffer` 0). Header: [u32 len, u32 type=0x004E4942].
    if u32_le(bytes, json_end.checked_add(4)?)? != 0x004E_4942 {
        return None;
    }
    let bin = bytes.get(json_end.checked_add(8)?..)?;

    let accessors = root.get("accessors")?.as_array()?;
    let views = root.get("bufferViews")?.as_array()?;
    let mut out = Vec::new();

    for mesh in root.get("meshes")?.as_array()? {
        for prim in mesh.get("primitives")?.as_array()? {
            let attrs = prim.get("attributes")?;
            let Some(pi) = json_u(attrs, "POSITION") else {
                continue;
            };
            let Some(pos) = read_vec3(accessors, views, bin, pi) else {
                continue;
            };
            let normals =
                json_u(attrs, "NORMAL").and_then(|ni| read_vec3(accessors, views, bin, ni));
            for (i, p) in pos.iter().enumerate() {
                let n = normals
                    .as_ref()
                    .and_then(|ns| ns.get(i))
                    .copied()
                    .unwrap_or(Vec3::Z)
                    .normalize_or_zero();
                let n = if n.length_squared() < 1e-9 {
                    Vec3::Z
                } else {
                    n
                };
                out.push((*p, n));
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Minimal PCD reader: the common `x y z` (FLOAT) case, ASCII or
/// little-endian binary. Extra fields (rgb, normals) are skipped.
fn pcd_points(bytes: &[u8]) -> Option<Vec<Point>> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut fields: Vec<String> = Vec::new();
    let mut points = 0usize;
    let mut ascii = true;
    let mut header_len = 0usize;

    for line in text.lines() {
        header_len = header_len.checked_add(line.len())?.checked_add(1)?;
        let mut it = line.split_whitespace();
        match it.next() {
            Some("FIELDS") => fields = it.map(str::to_owned).collect(),
            Some("POINTS") => points = it.next()?.parse().ok()?,
            Some("DATA") => {
                ascii = it.next()? == "ascii";
                break;
            }
            _ => {}
        }
    }
    let (xi, yi, zi) = (
        fields.iter().position(|f| f.as_str() == "x")?,
        fields.iter().position(|f| f.as_str() == "y")?,
        fields.iter().position(|f| f.as_str() == "z")?,
    );
    if !ascii {
        return None; // binary PCD: not needed yet (glTF is the path)
    }
    let body = text.get(header_len..)?;
    let out: Vec<Point> = body
        .lines()
        .filter_map(|l| {
            let c: Vec<&str> = l.split_whitespace().collect();
            let p = Vec3::new(
                c.get(xi)?.parse().ok()?,
                c.get(yi)?.parse().ok()?,
                c.get(zi)?.parse().ok()?,
            );
            // PCD has no normals here; +Z keeps everything front-facing.
            Some((p, Vec3::Z))
        })
        .take(points.max(1))
        .collect();
    (!out.is_empty()).then_some(out)
}
