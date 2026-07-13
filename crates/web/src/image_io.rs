//! Image <-> pixel-buffer plumbing via the `<canvas>` 2D API.
//!
//! Decoding: `File`/`Blob` -> `ImageBitmap` (browser-native decode of
//! whatever format the user uploaded) -> drawn onto an offscreen canvas ->
//! `getImageData` for raw RGBA8. Encoding for preview/download is the
//! reverse: RGBA8 -> `putImageData` -> `canvas.toDataURL("image/png")`.

use wasm_bindgen::{Clamped, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageBitmap, ImageData};

/// A decoded/produced image kept alongside a ready-to-display PNG data URL so
/// the UI doesn't have to re-encode on every render.
#[derive(Clone)]
pub struct ImageBuf {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub data_url: String,
}

fn js_err(e: JsValue) -> String {
    e.as_string()
        .or_else(|| js_sys::Error::from(e.clone()).message().as_string())
        .unwrap_or_else(|| "unknown JS error".to_string())
}

fn new_canvas(
    width: u32,
    height: u32,
) -> Result<(HtmlCanvasElement, CanvasRenderingContext2d), String> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let canvas: HtmlCanvasElement = document
        .create_element("canvas")
        .map_err(js_err)?
        .dyn_into()
        .map_err(|_| "created element is not a canvas".to_string())?;
    canvas.set_width(width);
    canvas.set_height(height);
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .map_err(js_err)?
        .ok_or("no 2d context")?
        .dyn_into()
        .map_err(|_| "context is not CanvasRenderingContext2d".to_string())?;
    Ok((canvas, ctx))
}

/// Decode an uploaded file (any format the browser understands) into RGBA8.
pub async fn decode_blob_to_rgba(blob: &web_sys::Blob) -> Result<ImageBuf, String> {
    let window = web_sys::window().ok_or("no window")?;
    let bitmap_promise = window.create_image_bitmap_with_blob(blob).map_err(js_err)?;
    let bitmap: ImageBitmap = JsFuture::from(bitmap_promise)
        .await
        .map_err(js_err)?
        .dyn_into()
        .map_err(|_| "createImageBitmap did not resolve to an ImageBitmap".to_string())?;

    let width = bitmap.width();
    let height = bitmap.height();
    let (_canvas, ctx) = new_canvas(width, height)?;
    ctx.draw_image_with_image_bitmap(&bitmap, 0.0, 0.0)
        .map_err(js_err)?;
    bitmap.close();

    let image_data = ctx
        .get_image_data(0.0, 0.0, width as f64, height as f64)
        .map_err(js_err)?;
    let rgba = image_data.data().0;
    let data_url = rgba_to_data_url(&rgba, width, height)?;
    Ok(ImageBuf {
        rgba,
        width,
        height,
        data_url,
    })
}

/// Encode raw RGBA8 as a `data:image/png;base64,...` URL, suitable for both
/// `<img src>` preview and an `<a download>` link.
pub fn rgba_to_data_url(rgba: &[u8], width: u32, height: u32) -> Result<String, String> {
    let (canvas, ctx) = new_canvas(width, height)?;
    let data = rgba.to_vec();
    let image_data = ImageData::new_with_u8_clamped_array_and_sh(Clamped(&data), width, height)
        .map_err(js_err)?;
    ctx.put_image_data(&image_data, 0.0, 0.0).map_err(js_err)?;
    canvas.to_data_url_with_type("image/png").map_err(js_err)
}
