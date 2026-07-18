//! Device-synchronized phase timing for VOSR performance diagnostics.

use std::ffi::OsStr;
use std::time::Instant;

use candle_core::{Device, Result};

const PROFILE_ENV: &str = "VOSR_PROFILE";

pub(super) struct PhaseTimer<'a> {
    device: &'a Device,
    enabled: bool,
    started: Instant,
}

impl<'a> PhaseTimer<'a> {
    pub(super) fn start(device: &'a Device, attention: &str) -> Result<Self> {
        let enabled = std::env::var_os(PROFILE_ENV).is_some_and(|value| value != OsStr::new("0"));
        if enabled {
            device.synchronize()?;
            eprintln!("[vosr] attention: {attention}");
        }
        Ok(Self {
            device,
            enabled,
            started: Instant::now(),
        })
    }

    pub(super) fn finish(&mut self, phase: &str) -> Result<()> {
        if self.enabled {
            self.device.synchronize()?;
            eprintln!(
                "[vosr] {phase}: {:.3}s",
                self.started.elapsed().as_secs_f64()
            );
            self.started = Instant::now();
        }
        Ok(())
    }
}
