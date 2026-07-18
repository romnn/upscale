//! Host-side bicubic resampling, matching PyTorch's `F.interpolate(mode="bicubic",
//! align_corners=False)` (cubic convolution with `a = -0.75`, edge-clamped taps).
//!
//! Several models need bicubic where the candle tensor API does not cover it: the
//! ×4 low-res upscale that forms the conditioning image (VOSR and TVT), the 448²
//! DINOv2 resize, and the DINOv2 positional-embedding grid interpolation. One
//! faithful CPU implementation serves them all; the tensors involved are small
//! enough that a host round-trip is cheap relative to the diffusion loop.

/// Cubic convolution kernel with `a = -0.75` (PyTorch's default).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.75;
    let x = x.abs();
    if x <= 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per-output-sample resampling plan: the leftmost source index and the four tap
/// weights, precomputed once per axis and reused across channels/rows.
struct AxisPlan {
    base: Vec<isize>,
    weights: Vec<[f64; 4]>,
}

impl AxisPlan {
    /// Build the tap plan resampling `in_len` samples to `out_len` under
    /// `align_corners=False` (source coord `= (o + 0.5) * in/out - 0.5`).
    fn new(in_len: usize, out_len: usize) -> Self {
        let scale = in_len as f64 / out_len as f64;
        let mut base = Vec::with_capacity(out_len);
        let mut weights = Vec::with_capacity(out_len);
        for o in 0..out_len {
            let src = (o as f64 + 0.5) * scale - 0.5;
            let floor = src.floor();
            let frac = src - floor;
            base.push(floor as isize - 1);
            weights.push([
                cubic(frac + 1.0),
                cubic(frac),
                cubic(frac - 1.0),
                cubic(frac - 2.0),
            ]);
        }
        Self { base, weights }
    }
}

/// Clamp a tap index to `[0, len)` (edge replication, matching PyTorch's bicubic
/// boundary handling).
fn clamp_idx(i: isize, len: usize) -> usize {
    i.clamp(0, len as isize - 1) as usize
}

/// Bicubic-resize a planar `[c, ih, iw]` buffer to `[c, oh, ow]`.
///
/// Separable: a horizontal pass to `ow`, then a vertical pass to `oh`. The tap
/// weights of the cubic convolution sum to one analytically, so no per-sample
/// renormalization is applied (again matching PyTorch).
pub(crate) fn bicubic_chw(
    src: &[f32],
    c: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let hplan = AxisPlan::new(iw, ow);
    let vplan = AxisPlan::new(ih, oh);

    let mut horiz = vec![0f32; c * ih * ow];
    for ch in 0..c {
        let src_ch = &src[ch * ih * iw..(ch + 1) * ih * iw];
        let dst_ch = &mut horiz[ch * ih * ow..(ch + 1) * ih * ow];
        for row in 0..ih {
            let srow = &src_ch[row * iw..(row + 1) * iw];
            for ox in 0..ow {
                let b = hplan.base[ox];
                let w = &hplan.weights[ox];
                let mut acc = 0f64;
                for (t, &wt) in w.iter().enumerate() {
                    acc += wt * f64::from(srow[clamp_idx(b + t as isize, iw)]);
                }
                dst_ch[row * ow + ox] = acc as f32;
            }
        }
    }

    let mut out = vec![0f32; c * oh * ow];
    for ch in 0..c {
        let src_ch = &horiz[ch * ih * ow..(ch + 1) * ih * ow];
        let dst_ch = &mut out[ch * oh * ow..(ch + 1) * oh * ow];
        for oy in 0..oh {
            let b = vplan.base[oy];
            let w = &vplan.weights[oy];
            for ox in 0..ow {
                let mut acc = 0f64;
                for (t, &wt) in w.iter().enumerate() {
                    acc += wt * f64::from(src_ch[clamp_idx(b + t as isize, ih) * ow + ox]);
                }
                dst_ch[oy * ow + ox] = acc as f32;
            }
        }
    }
    out
}
