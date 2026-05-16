//! VRM 0.0 viseme extraction.
//!
//! `bevy_vrm`'s glTF importer (`bevy_gltf_kun`) does not copy
//! `mesh.extras.targetNames` into Bevy, so [`Mesh::morph_target_names`] is
//! `None` for a VRM and the mouth cannot be resolved by name. The morph
//! *weight slot order* is preserved, though, and a VRM 0.0 file carries an
//! authoritative `extensions.VRM.blendShapeMaster` that maps the standard
//! preset names (`a`/`i`/`u`/`e`/`o`) to exact morph indices. Reading that
//! table is the VRM spec contract — deterministic, not a per-rig guess.

use std::path::Path;

use lana_viseme::Vowel;
use tracing::warn;

/// Morph-target slot index per vowel, as declared by the VRM's own
/// `blendShapeMaster`. `None` for a preset the model does not define.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VrmVisemeMap {
    pub(crate) a: Option<usize>,
    pub(crate) e: Option<usize>,
    pub(crate) i: Option<usize>,
    pub(crate) o: Option<usize>,
    pub(crate) u: Option<usize>,
    /// `blink` preset (both eyes), for the idle blink — not a viseme.
    pub(crate) blink: Option<usize>,
}

impl VrmVisemeMap {
    pub(crate) const fn slot(&self, v: Vowel) -> Option<usize> {
        match v {
            Vowel::A => self.a,
            Vowel::E => self.e,
            Vowel::I => self.i,
            Vowel::O => self.o,
            Vowel::U => self.u,
            Vowel::Sil => None,
        }
    }

    pub(crate) const fn any(&self) -> bool {
        self.a.is_some()
            || self.e.is_some()
            || self.i.is_some()
            || self.o.is_some()
            || self.u.is_some()
    }

    /// Highest slot index referenced (so the caller can sanity-check it
    /// against the actual `MorphWeights` length).
    pub(crate) fn max_slot(&self) -> usize {
        [self.a, self.e, self.i, self.o, self.u, self.blink]
            .into_iter()
            .flatten()
            .max()
            .unwrap_or(0)
    }
}

/// Parse the `blendShapeMaster` viseme presets out of a `.vrm` (binary
/// glTF). Returns `None` if the file is not a VRM 0.0 with viseme presets
/// (the caller then falls back to the glTF name-based path).
pub(crate) fn parse_vrm_visemes(path: &Path) -> Option<VrmVisemeMap> {
    let bytes = std::fs::read(path)
        .map_err(|e| warn!(error = %e, "VRM read failed"))
        .ok()?;

    // glb: [magic "glTF" | u32 version | u32 total] then chunk 0:
    // [u32 len | u32 type 0x4E4F534A "JSON" | json bytes].
    if bytes.len() < 20 || bytes.get(0..4) != Some(b"glTF") {
        return None;
    }
    let json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let json_end = 20usize.checked_add(json_len)?;
    let json = bytes.get(20..json_end)?;
    let root: serde_json::Value = serde_json::from_slice(json)
        .map_err(|e| warn!(error = %e, "VRM JSON parse failed"))
        .ok()?;

    let groups = root
        .get("extensions")?
        .get("VRM")?
        .get("blendShapeMaster")?
        .get("blendShapeGroups")?
        .as_array()?;

    let mut map = VrmVisemeMap::default();
    for g in groups {
        let preset = g.get("presetName").and_then(serde_json::Value::as_str);
        // First bind's morph index (visemes are single-bind, weight 100).
        let idx = g
            .get("binds")
            .and_then(serde_json::Value::as_array)
            .and_then(|b| b.first())
            .and_then(|b| b.get("index"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok());
        match (preset, idx) {
            (Some("a"), Some(n)) => map.a = Some(n),
            (Some("e"), Some(n)) => map.e = Some(n),
            (Some("i"), Some(n)) => map.i = Some(n),
            (Some("o"), Some(n)) => map.o = Some(n),
            (Some("u"), Some(n)) => map.u = Some(n),
            (Some("blink"), Some(n)) => map.blink = Some(n),
            _ => {}
        }
    }
    map.any().then_some(map)
}
