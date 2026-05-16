//! Lip-sync: turn the orchestrator's viseme timeline into morph-target
//! weights on the avatar's face mesh.
//!
//! The approach is format-agnostic. `bevy_vrm` loads a VRM as a plain glTF,
//! so VRM and realistic-human glTF avatars both end up as Bevy meshes with
//! named morph targets and a canonical [`MorphWeights`] on the parent node.
//! At load we resolve the mouth morphs by *name* against the conventions in
//! the wild (`VRoid` `Fcl_MTH_*`, Oculus/RPM `viseme_*`, `ARKit` `jawOpen`) and
//! log what we matched — if a rig uses none of them the mouth stays still
//! but the available names are logged so a mapping can be added (an
//! observable miss, never a silent one).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use bevy::mesh::morph::MorphWeights;
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
            let next_due = self.queue.get(1).is_some_and(|s| s.at <= now);
            if next_due {
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

/// Resolved mouth morph-target slots on the face mesh's parent node.
#[derive(Resource)]
pub(crate) struct Mouth {
    /// Entity carrying the canonical `MorphWeights`.
    node: Entity,
    /// Per-vowel morph weight index (absent if the rig has no such shape).
    vowels: [(Vowel, Option<usize>); 5],
    /// Aperture morph (`jawOpen`/`mouthOpen`), driven by openness — also
    /// the sole driver on rigs that expose no vowel shapes.
    jaw: Option<usize>,
}

/// How responsive the mouth is (seconds to ~63 % of a step). Small enough
/// to articulate syllables, large enough to avoid buzzing.
const SMOOTH_TAU: f32 = 0.045;
/// Aperture contribution from openness when a vowel shape is also active.
const JAW_GAIN: f32 = 0.55;
/// Below this openness the mouth is treated as closed.
const REST_EPS: f32 = 0.04;

fn name_index(names: &[String], candidates: &[&str]) -> Option<usize> {
    names
        .iter()
        .position(|n| candidates.iter().any(|c| n.eq_ignore_ascii_case(c)))
}

/// System-local bookkeeping for the one-shot mouth scan.
#[derive(Debug, Default)]
pub(crate) struct MouthScan {
    frames: u32,
    done: bool,
}

/// Resolve the mouth morphs once the face mesh has loaded, insert the
/// [`Mouth`] resource and stop. VRM avatars use the file's authoritative
/// `blendShapeMaster` slots; glTF avatars resolve by morph-target name and
/// log the ground truth on a miss (never a silent failure).
pub(crate) fn resolve_mouth(
    mut commands: Commands,
    existing: Option<Res<Mouth>>,
    vrm: Option<Res<crate::VrmMouth>>,
    mut scan: Local<MouthScan>,
    nodes: Query<(Entity, &MorphWeights)>,
    meshes: Res<Assets<Mesh>>,
) {
    if existing.is_some() || scan.done {
        return;
    }
    scan.frames = scan.frames.saturating_add(1);

    if let Some(vrm) = vrm.as_deref() {
        bind_vrm_mouth(&mut commands, &mut scan, &nodes, &vrm.0);
    } else {
        scan_named_mouth(&mut commands, &mut scan, &nodes, &meshes);
    }
}

/// VRM path: `bevy_vrm` drops glTF morph names but preserves the weight
/// slot order, so bind the file's own `blendShapeMaster` indices to the
/// `MorphWeights` node (the one with enough slots).
fn bind_vrm_mouth(
    commands: &mut Commands,
    scan: &mut MouthScan,
    nodes: &Query<(Entity, &MorphWeights)>,
    vrm: &crate::vrm::VrmVisemeMap,
) {
    let need = vrm.max_slot();
    for (entity, weights) in nodes {
        if weights.weights().len() > need {
            let vowels = [
                (Vowel::A, vrm.slot(Vowel::A)),
                (Vowel::E, vrm.slot(Vowel::E)),
                (Vowel::I, vrm.slot(Vowel::I)),
                (Vowel::O, vrm.slot(Vowel::O)),
                (Vowel::U, vrm.slot(Vowel::U)),
            ];
            info!(
                slots = weights.weights().len(),
                "lip-sync: bound VRM blendShapeMaster visemes to MorphWeights"
            );
            commands.insert_resource(Mouth {
                node: entity,
                vowels,
                jaw: None,
            });
            scan.done = true;
            return;
        }
    }
    if scan.frames > 600 {
        warn!(
            need = need.saturating_add(1),
            "lip-sync: VRM viseme map ready but no MorphWeights node has \
             enough morph slots — mouth disabled"
        );
        scan.done = true;
    }
}

/// glTF path: resolve the mouth by morph-target name (the native loader
/// imports them). On a miss, log the ground truth once and give up.
fn scan_named_mouth(
    commands: &mut Commands,
    scan: &mut MouthScan,
    nodes: &Query<(Entity, &MorphWeights)>,
    meshes: &Res<Assets<Mesh>>,
) {
    let mut node_count = 0_usize;
    let mut loaded_unmatched: Vec<(Entity, Vec<String>)> = Vec::new();

    for (entity, weights) in nodes {
        node_count = node_count.saturating_add(1);
        let Some(handle) = weights.first_mesh() else {
            continue;
        };
        let Some(mesh) = meshes.get(handle) else {
            continue; // mesh asset still loading — retry next frame.
        };
        let Some(names) = mesh.morph_target_names() else {
            loaded_unmatched.push((entity, Vec::new()));
            continue;
        };
        let vowels = [
            (
                Vowel::A,
                name_index(names, &["Fcl_MTH_A", "viseme_aa", "aa", "A"]),
            ),
            (Vowel::E, name_index(names, &["Fcl_MTH_E", "viseme_E", "E"])),
            (Vowel::I, name_index(names, &["Fcl_MTH_I", "viseme_I", "I"])),
            (Vowel::O, name_index(names, &["Fcl_MTH_O", "viseme_O", "O"])),
            (Vowel::U, name_index(names, &["Fcl_MTH_U", "viseme_U", "U"])),
        ];
        let jaw = name_index(names, &["jawOpen", "mouthOpen", "Fcl_MTH_Open"]);

        if vowels.iter().any(|(_, i)| i.is_some()) || jaw.is_some() {
            info!(
                matched_vowels = vowels.iter().filter(|(_, i)| i.is_some()).count(),
                has_jaw = jaw.is_some(),
                morphs = names.len(),
                "lip-sync: resolved mouth morphs"
            );
            commands.insert_resource(Mouth {
                node: entity,
                vowels,
                jaw,
            });
            scan.done = true;
            return;
        }
        loaded_unmatched.push((entity, names.to_vec()));
    }

    // A loaded mesh's names are stable, so retrying cannot help: log the
    // truth once and give up (drive_mouth then no-ops, mouth stays still).
    if !loaded_unmatched.is_empty() {
        for (entity, names) in &loaded_unmatched {
            if names.is_empty() {
                warn!(
                    ?entity,
                    "lip-sync: node has MorphWeights but the loader imported \
                     no morph-target names — cannot name-resolve the mouth"
                );
            } else {
                warn!(
                    ?entity,
                    count = names.len(),
                    ?names,
                    "lip-sync: no known mouth shape among these morph names \
                     — add the convention to mouth.rs::scan_named_mouth"
                );
            }
        }
        scan.done = true;
        return;
    }

    if node_count == 0 && scan.frames > 600 {
        warn!(
            "lip-sync: no MorphWeights found ~10 s after start — this avatar \
             exposes no morph targets via the loader; lip-sync needs a \
             glTF/ARKit model (e.g. an Avaturn export)"
        );
        scan.done = true;
    }
}

/// Drive the resolved mouth morphs from the schedule, smoothed per frame.
pub(crate) fn drive_mouth(
    mouth: Option<Res<Mouth>>,
    mut schedule: ResMut<VisemeSchedule>,
    time: Res<Time>,
    mut weights: Query<&mut MorphWeights>,
) {
    let Some(mouth) = mouth else {
        return;
    };
    let Ok(mut mw) = weights.get_mut(mouth.node) else {
        return;
    };

    let (openness, vowel) = schedule
        .current(Instant::now())
        .unwrap_or((0.0, Vowel::Sil));
    let speaking = openness >= REST_EPS && vowel != Vowel::Sil;
    let has_vowel_shapes = mouth.vowels.iter().any(|(_, i)| i.is_some());

    // Frame-rate-independent exponential smoothing toward the target.
    let dt = time.delta_secs();
    let k = if dt > 0.0 {
        1.0 - (-dt / SMOOTH_TAU).exp()
    } else {
        1.0
    };

    let slots = mw.weights_mut();
    let mut set = |idx: Option<usize>, target: f32| {
        if let Some(w) = idx.and_then(|i| slots.get_mut(i)) {
            *w += (target - *w) * k;
        }
    };

    for (v, idx) in mouth.vowels {
        let target = if speaking && v == vowel {
            openness
        } else {
            0.0
        };
        set(idx, target);
    }
    // Jaw adds aperture under a vowel; on vowel-less rigs it *is* the mouth.
    let jaw_target = if speaking {
        if has_vowel_shapes {
            openness * JAW_GAIN
        } else {
            openness
        }
    } else {
        0.0
    };
    set(mouth.jaw, jaw_target);
}
