//! Tiled image I/O shared by the candle upscaler models: RGBA↔tensor
//! conversion, the tile-geometry planner, and seam-blended accumulation of
//! decoded output tiles.

use candle_core::{DType, Device, Result, Tensor};

/// RGBA8 `[h*w*4]` → `[1, 3, h, w]` in `[0, 1]` at the compute dtype (drops alpha).
pub(crate) fn rgba_to_tensor(
    rgba: &[u8],
    width: usize,
    height: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let hw = width * height;
    let mut v = vec![0f32; 3 * hw];
    for i in 0..hw {
        for c in 0..3 {
            v[c * hw + i] = f32::from(rgba[i * 4 + c]) / 255.0;
        }
    }
    Tensor::from_vec(v, (1, 3, height, width), device)?.to_dtype(dtype)
}

/// Plan the low-res tile grid for a `width`×`height` image.
///
/// Tiles step by `tile - overlap` and are clamped so a near-edge tile slides back
/// to end at the image border (never reading out of bounds). Returns the tile
/// top-left `origins`, the per-tile height/width (`th`, `tw`), and the tile
/// `total` used to drive progress. `tile`/`overlap` are the raw options; the
/// clamping matches the historical inline geometry exactly.
pub(crate) fn tile_origins(
    width: usize,
    height: usize,
    tile: usize,
    overlap: usize,
) -> (Vec<(usize, usize)>, usize, usize, usize) {
    let tile = tile.clamp(8, width.max(height).max(8));
    let overlap = overlap.min(tile / 2);
    let stride = (tile - overlap).max(1);
    let ys: Vec<usize> = (0..height).step_by(stride).collect();
    let xs: Vec<usize> = (0..width).step_by(stride).collect();
    let total = (ys.len() * xs.len()).max(1);

    let (th, tw) = (tile.min(height), tile.min(width));
    let origins: Vec<(usize, usize)> = ys
        .iter()
        .flat_map(|&y| {
            let y0 = (y + tile).min(height).saturating_sub(tile);
            xs.iter()
                .map(move |&x| (y0, (x + tile).min(width).saturating_sub(tile)))
        })
        .collect();
    (origins, th, tw, total)
}

/// Add a decoded output tile's CHW `vals` (`[3, th, tw]`) into the accumulation
/// buffers at pixel offset `(ox, oy)`, one weight unit per covered pixel.
pub(crate) fn accumulate(
    out: &mut [f32],
    weight: &mut [f32],
    vals: &[f32],
    th: usize,
    tw: usize,
    out_width: usize,
    ox: usize,
    oy: usize,
) {
    let plane = th * tw;
    for ty in 0..th {
        for tx in 0..tw {
            let dst_px = (oy + ty) * out_width + (ox + tx);
            for c in 0..3 {
                out[dst_px * 3 + c] += vals[c * plane + ty * tw + tx];
            }
            weight[dst_px] += 1.0;
        }
    }
}

/// Divide the accumulated `out` by the per-pixel `weight`, clamp to `[0, 1]`, and
/// pack into an opaque RGBA8 buffer.
pub(crate) fn normalize_to_rgba(out: &[f32], weight: &[f32], width: usize, height: usize) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];
    for px in 0..width * height {
        let w = weight[px].max(1.0);
        for c in 0..3 {
            let v = (out[px * 3 + c] / w).clamp(0.0, 1.0);
            rgba[px * 4 + c] = (v * 255.0 + 0.5) as u8;
        }
        rgba[px * 4 + 3] = 255;
    }
    rgba
}
