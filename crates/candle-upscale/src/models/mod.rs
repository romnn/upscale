//! Concrete upscaling models. Each model is feature-gated, so a build compiles
//! only the models it enables.

#[cfg(feature = "sdx4")]
pub mod sdx4;

#[cfg(feature = "tvt")]
pub mod tvt;

#[cfg(feature = "vosr")]
pub mod vosr;
