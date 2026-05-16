//! Conversation overlay: drains [`AvatarUpdate`]s into a [`Transcript`]
//! resource and renders it as a `bevy_egui` panel over the avatar.

use std::sync::atomic::Ordering;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use crate::mouth::VisemeSchedule;
use crate::{AvatarUpdate, MicMute, Updates};

/// Most recent conversation lines kept for display.
const MAX_LINES: usize = 12;

/// Rolling transcript + current phase, rendered by the overlay UI.
#[derive(Resource, Default)]
pub(crate) struct Transcript {
    /// Current coarse phase label.
    pub(crate) phase: String,
    /// Recent `"you: …"` / `"lana: …"` / `"· …"` lines, newest last.
    pub(crate) lines: Vec<String>,
}

impl Transcript {
    fn push(&mut self, line: String) {
        self.lines.push(line);
        let overflow = self.lines.len().saturating_sub(MAX_LINES);
        if overflow > 0 {
            self.lines.drain(0..overflow);
        }
    }
}

/// Wires egui, the transcript resource and the channel-drain + render systems.
pub(crate) struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            .init_resource::<Transcript>()
            .add_systems(Update, pump_updates)
            .add_systems(EguiPrimaryContextPass, render_overlay);
    }
}

/// Drain every pending [`AvatarUpdate`] into the [`Transcript`] and the
/// lip-sync [`VisemeSchedule`] (non-blocking).
fn pump_updates(
    updates: Res<Updates>,
    mut transcript: ResMut<Transcript>,
    mut visemes: ResMut<VisemeSchedule>,
) {
    while let Ok(update) = updates.0.try_recv() {
        match update {
            AvatarUpdate::Phase(p) => transcript.phase = p,
            AvatarUpdate::UserSaid(t) => {
                // A new user turn: drop any lip-sync still queued from a
                // reply that was cut short (barge-in).
                visemes.clear();
                transcript.push(format!("you: {t}"));
            }
            AvatarUpdate::LanaSaid(t) => transcript.push(format!("lana: {t}")),
            AvatarUpdate::Visemes(frames) => visemes.push(&frames),
            AvatarUpdate::Notice(n) => transcript.push(format!("· {n}")),
        }
    }
}

/// Draw the phase badge + mic-mute toggle + rolling transcript.
fn render_overlay(
    mut contexts: EguiContexts,
    transcript: Res<Transcript>,
    mute: Res<MicMute>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::TopBottomPanel::bottom("lana_transcript")
        .resizable(false)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.strong("Lana");
                if !transcript.phase.is_empty() {
                    ui.label(
                        egui::RichText::new(format!("· {}", transcript.phase))
                            .weak()
                            .italics(),
                    );
                }
                let muted = mute.0.load(Ordering::Relaxed);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if muted {
                        ui.colored_label(egui::Color32::from_rgb(0xE0, 0x50, 0x50), "● muted");
                    }
                    let label = if muted { "Unmute mic" } else { "Mute mic" };
                    if ui.button(label).clicked() {
                        mute.0.store(!muted, Ordering::Relaxed);
                    }
                });
            });
            ui.separator();
            egui::ScrollArea::vertical()
                .max_height(160.0)
                .stick_to_bottom(true)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for line in &transcript.lines {
                        ui.label(line);
                    }
                });
            ui.add_space(4.0);
        });
    Ok(())
}
