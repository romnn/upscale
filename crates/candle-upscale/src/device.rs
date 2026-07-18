//! Device selection for the compiled candle backends.

/// Fastest available device for the compiled backends: CUDA, else Metal, else
/// CPU.
///
/// # Errors
/// Only the final CPU fallback can fail; the CUDA/Metal probes fall through on
/// error rather than propagating.
pub fn select_device() -> candle_core::Result<candle_core::Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = candle_core::Device::new_cuda(0) {
            return Ok(d);
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = candle_core::Device::new_metal(0) {
            return Ok(d);
        }
    }
    Ok(candle_core::Device::Cpu)
}

/// Open CUDA device 0. Only available with the `cuda` feature.
///
/// # Errors
/// Fails if no CUDA device is present or the driver cannot be initialized.
#[cfg(feature = "cuda")]
pub fn cuda_device() -> candle_core::Result<candle_core::Device> {
    candle_core::Device::new_cuda(0)
}
