//! Error type for the avatar window.

use thiserror::Error;

/// Errors returned by [`crate::run`].
#[derive(Debug, Error)]
pub enum AvatarError {
    /// The VRM model could not be found or loaded.
    #[error("vrm error: {0}")]
    Vrm(String),
}
