//! Deterministic host-side Gaussian noise.
//!
//! The parity comparison against burn requires feeding *both* frameworks the
//! identical initial latents and low-res noise, since each framework's own
//! `randn` uses a different RNG. Generating the noise on the host (a fixed seed →
//! f32 array) and loading it into each backend's tensor removes that difference.
//! Box-Muller over a splitmix64 stream gives a portable, reproducible N(0,1).

use std::f64::consts::PI;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `[0, 1)` from the top 53 bits of a 64-bit word.
fn u01(bits: u64) -> f64 {
    (bits >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// `n` standard-normal `f32` samples, deterministic in `seed`.
pub fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        // Guard the log against an exact 0 draw (undefined at u1 == 0).
        let u1 = u01(splitmix64(&mut state)).max(f64::MIN_POSITIVE);
        let u2 = u01(splitmix64(&mut state));
        let radius = (-2.0 * u1.ln()).sqrt();
        out.push((radius * (2.0 * PI * u2).cos()) as f32);
        if out.len() < n {
            out.push((radius * (2.0 * PI * u2).sin()) as f32);
        }
    }
    out
}
