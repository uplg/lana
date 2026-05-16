//! Conversation overlay: drains [`AvatarUpdate`]s into a [`Transcript`]
//! resource and renders it as a `bevy_egui` panel over the avatar.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use crate::{AvatarUpdate, Updates};

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

/// Drain every pending [`AvatarUpdate`] into the [`Transcript`] (non-blocking).
fn pump_updates(updates: Res<Updates>, mut transcript: ResMut<Transcript>) {
    while let Ok(update) = updates.0.try_recv() {
        match update {
            AvatarUpdate::Phase(p) => transcript.phase = p,
            AvatarUpdate::UserSaid(t) => transcript.push(format!("you: {t}")),
            AvatarUpdate::LanaSaid(t) => transcript.push(format!("lana: {t}")),
            AvatarUpdate::Notice(n) => transcript.push(format!("· {n}")),
        }
    }
}

/// Draw the phase badge + rolling transcript as a bottom-anchored panel.
fn render_overlay(mut contexts: EguiContexts, transcript: Res<Transcript>) -> Result {
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
