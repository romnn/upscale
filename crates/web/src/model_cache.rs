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

/// Cheap structural sniff for a safetensors file. The format is an 8-byte
/// little-endian header length `n`, immediately followed by `n` bytes of JSON
/// metadata — so byte 8 is always `{`, and `n` must fit inside the file.
///
/// A wrong "download" — an HTML error page, a directory listing, a git-LFS
/// pointer, a truncated body — fails at least one of these checks. Rejecting it
/// here, rather than handing it to the parser, both prevents a poisoned cache
/// entry and turns the parser's cryptic `HeaderTooLarge` into an actionable
/// error naming the URL.
fn looks_like_safetensors(bytes: &[u8]) -> bool {
    let Some(prefix) = bytes.get(0..8) else {
        return false;
    };
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(prefix);
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    // The header must be non-empty JSON that fits within the remaining bytes.
    header_len >= 2 && header_len <= bytes.len() - 8 && bytes.get(8) == Some(&b'{')
}

/// Render the leading bytes of a bad response for an error message, so the user
/// can recognize e.g. an HTML page (`<!DOCTYPE ...`) or a plain-text 404.
fn describe_head(bytes: &[u8]) -> String {
    let head = &bytes[..bytes.len().min(32)];
    let printable: String = head
        .iter()
        .map(|&b| {
            if (0x20..0x7f).contains(&b) {
                b as char
            } else {
                '.'
            }
        })
        .collect();
    format!("{} bytes, starts with {printable:?}", bytes.len())
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
        if looks_like_safetensors(&bytes) {
            on_status(ModelStatus::Ready);
            return Ok(bytes);
        }
        // A poisoned entry — an earlier misconfigured run cached a 200 response
        // whose body was not the model (e.g. a dev-server SPA `index.html`
        // fallback or a wrong-root 404 page). Evict it and fall through to a
        // fresh download so the cache self-heals instead of serving these bytes
        // forever (the parser would otherwise fail with `HeaderTooLarge`).
        web_sys::console::warn_1(
            &format!(
                "evicting stale cache entry for {url}: not a safetensors file ({})",
                describe_head(&bytes)
            )
            .into(),
        );
        let _ = JsFuture::from(cache.delete_with_str(url)).await;
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

    // A 200 response is not proof it's the model: a misconfigured file server or
    // dev-server SPA fallback can return an HTML page or a 404 body with status
    // 200. Reject it before caching so the bad bytes never get persisted, and
    // give an error that names the URL instead of the parser's `HeaderTooLarge`.
    if !looks_like_safetensors(&bytes) {
        return Err(format!(
            "{url} did not return a safetensors file ({}) — check the model \
             server is serving the real weights at this URL",
            describe_head(&bytes)
        ));
    }

    // Caching is an optimization, not a requirement: some browsers reject a
    // single multi-hundred-MB `Cache.put` ("Unexpected internal error") even
    // with ample disk. If it fails, warn and carry on with the bytes we already
    // downloaded rather than failing the whole upscale — next run just
    // re-downloads instead of serving from cache.
    on_status(ModelStatus::Saving);
    if let Err(e) = JsFuture::from(cache.put_with_str(url, &resp_for_cache)).await {
        web_sys::console::warn_1(
            &format!(
                "cache.put failed for {url} (continuing without caching): {}",
                js_err(e)
            )
            .into(),
        );
    }

    on_status(ModelStatus::Ready);
    Ok(bytes)
}
