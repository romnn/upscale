//! Model weight download with the browser Cache Storage API.
//!
//! On first load, streams the response body (reporting byte progress as it
//! goes) and stores an unread clone of the response in `caches`. On later
//! loads, serves straight from that cache with no network request. Bumping
//! [`CACHE_NAME`] busts the cache (old entries just become unreachable and
//! get evicted by the browser under storage pressure — we don't bother
//! explicitly deleting them, this is a dev tool, not a production PWA).

use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Cache, CacheStorage, Response};

/// Bump this when the model files served at the configured URL change
/// incompatibly, so stale bytes under the old key are simply orphaned rather
/// than being (wrongly) served to a build that expects something else.
pub const CACHE_NAME: &str = "sd-x4-v1";

#[derive(Clone, Debug)]
pub enum ModelStatus {
    CheckingCache,
    Downloading { received: u64, total: Option<u64> },
    Saving,
    Ready,
}

fn js_err(e: JsValue) -> String {
    e.as_string()
        .or_else(|| js_sys::Error::from(e.clone()).message().as_string())
        .unwrap_or_else(|| "unknown JS error".to_string())
}

async fn open_cache() -> Result<Cache, String> {
    let window = web_sys::window().ok_or("no window")?;
    let caches: CacheStorage = window.caches().map_err(js_err)?;
    let cache = JsFuture::from(caches.open(CACHE_NAME))
        .await
        .map_err(js_err)?;
    cache
        .dyn_into()
        .map_err(|_| "caches.open() did not resolve to a Cache".to_string())
}

/// Fetch `url`'s bytes, preferring the Cache Storage API entry if present.
/// `on_status` is called throughout so the caller can drive a progress UI.
pub async fn load_model_bytes(
    url: &str,
    on_status: impl Fn(ModelStatus),
) -> Result<Vec<u8>, String> {
    on_status(ModelStatus::CheckingCache);
    let cache = open_cache().await?;

    let matched = JsFuture::from(cache.match_with_str(url))
        .await
        .map_err(js_err)?;
    if !matched.is_undefined() {
        let resp: Response = matched
            .dyn_into()
            .map_err(|_| "cache match() did not resolve to a Response".to_string())?;
        let buf = JsFuture::from(resp.array_buffer().map_err(js_err)?)
            .await
            .map_err(js_err)?;
        let bytes = js_sys::Uint8Array::new(&buf).to_vec();
        on_status(ModelStatus::Ready);
        return Ok(bytes);
    }

    let window = web_sys::window().ok_or("no window")?;
    let resp_val = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(js_err)?;
    let resp: Response = resp_val
        .dyn_into()
        .map_err(|_| "fetch() did not resolve to a Response".to_string())?;
    if !resp.ok() {
        return Err(format!(
            "failed to download model: HTTP {} ({url})",
            resp.status()
        ));
    }

    let total = resp
        .headers()
        .get("content-length")
        .ok()
        .flatten()
        .and_then(|s| s.parse::<u64>().ok());

    // Clone *before* reading the body: Response bodies can only be consumed
    // once, so this tees the underlying stream — one copy we read here for
    // progress, one untouched copy we hand to `cache.put` below.
    let resp_for_cache = resp.clone().map_err(js_err)?;

    on_status(ModelStatus::Downloading { received: 0, total });
    let body = resp.body().ok_or("response has no body")?;
    let reader = body
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
        .map_err(|_| "getReader() did not resolve to a default reader".to_string())?;

    let mut received: u64 = 0;
    let mut bytes: Vec<u8> = Vec::with_capacity(total.unwrap_or(0) as usize);
    loop {
        let chunk_result = JsFuture::from(reader.read()).await.map_err(js_err)?;
        let done = js_sys::Reflect::get(&chunk_result, &"done".into())
            .map_err(js_err)?
            .as_bool()
            .unwrap_or(true);
        if done {
            break;
        }
        let value = js_sys::Reflect::get(&chunk_result, &"value".into()).map_err(js_err)?;
        let chunk = js_sys::Uint8Array::new(&value).to_vec();
        received += chunk.len() as u64;
        bytes.extend_from_slice(&chunk);
        on_status(ModelStatus::Downloading { received, total });
    }

    // Caching is an optimization, not a requirement: some browsers reject a
    // single multi-hundred-MB `Cache.put` ("Unexpected internal error") even
    // with ample disk. If it fails, warn and carry on with the bytes we already
    // downloaded rather than failing the whole upscale — next run just
    // re-downloads instead of serving from cache.
    on_status(ModelStatus::Saving);
    if let Err(e) = JsFuture::from(cache.put_with_str(url, &resp_for_cache)).await {
        web_sys::console::warn_1(
            &format!("cache.put failed for {url} (continuing without caching): {}", js_err(e))
                .into(),
        );
    }

    on_status(ModelStatus::Ready);
    Ok(bytes)
}
