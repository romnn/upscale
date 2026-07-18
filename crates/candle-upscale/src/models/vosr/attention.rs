//! Attention backend selection for VOSR's diffusion transformer.
//!
//! CUDA half-precision inference uses FlashAttention by default, while Metal
//! uses Candle's fused SDPA kernel. The compact matmul/softmax implementation
//! remains the portable CPU path and the troubleshooting escape hatch through
//! `VOSR_ATTENTION=candle`.

use std::ffi::OsStr;

#[cfg(feature = "cuda")]
use candle_core::Error;
use candle_core::{DType, Device, Result, Tensor, D};
#[cfg(feature = "metal")]
use candle_nn::ops::sdpa;
use candle_nn::ops::softmax;

const BACKEND_ENV: &str = "VOSR_ATTENTION";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttentionBackend {
    Candle,
    #[cfg(feature = "cuda")]
    Flash,
    #[cfg(feature = "metal")]
    Metal,
}

impl AttentionBackend {
    pub(super) fn select(device: &Device, dtype: DType) -> Result<Self> {
        Self::from_setting(std::env::var_os(BACKEND_ENV).as_deref(), device, dtype)
    }

    fn from_setting(setting: Option<&OsStr>, device: &Device, dtype: DType) -> Result<Self> {
        match setting.and_then(OsStr::to_str) {
            None | Some("" | "auto") => {
                #[cfg(feature = "cuda")]
                if let Some(backend) = Self::automatic_cuda(device, dtype)? {
                    return Ok(backend);
                }
                #[cfg(feature = "metal")]
                if matches!(device, Device::Metal(_))
                    && matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
                {
                    return Ok(Self::Metal);
                }
                let _ = (device, dtype);
                Ok(Self::Candle)
            }
            Some("candle" | "baseline") => Ok(Self::Candle),
            Some("flash") => Self::explicit_flash(device, dtype),
            Some("metal") => Self::explicit_metal(device, dtype),
            Some(value) => candle_core::bail!(
                "{BACKEND_ENV} must be auto, candle, baseline, flash, or metal (got {value:?})"
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn automatic_cuda(device: &Device, dtype: DType) -> Result<Option<Self>> {
        if matches!(dtype, DType::F16 | DType::BF16)
            && flash_capability(device)?.is_some_and(|(major, _)| major >= 8)
        {
            return Ok(Some(Self::Flash));
        }
        Ok(None)
    }

    #[cfg(feature = "cuda")]
    fn explicit_flash(device: &Device, dtype: DType) -> Result<Self> {
        let Some((major, minor)) = flash_capability(device)? else {
            candle_core::bail!("{BACKEND_ENV}=flash requires a CUDA device")
        };
        if !matches!(dtype, DType::F16 | DType::BF16) {
            candle_core::bail!("{BACKEND_ENV}=flash requires f16 or bf16, got {dtype:?}")
        }
        if major < 8 {
            candle_core::bail!(
                "{BACKEND_ENV}=flash requires CUDA compute capability 8.0 or newer, got {major}.{minor}"
            )
        }
        Ok(Self::Flash)
    }

    #[cfg(not(feature = "cuda"))]
    fn explicit_flash(_device: &Device, _dtype: DType) -> Result<Self> {
        candle_core::bail!("{BACKEND_ENV}=flash requires building with the cuda feature")
    }

    #[cfg(feature = "metal")]
    fn explicit_metal(device: &Device, dtype: DType) -> Result<Self> {
        if !matches!(device, Device::Metal(_)) {
            candle_core::bail!("{BACKEND_ENV}=metal requires a Metal device")
        }
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
            candle_core::bail!("{BACKEND_ENV}=metal requires f16, bf16, or f32, got {dtype:?}")
        }
        Ok(Self::Metal)
    }

    #[cfg(not(feature = "metal"))]
    fn explicit_metal(_device: &Device, _dtype: DType) -> Result<Self> {
        candle_core::bail!("{BACKEND_ENV}=metal requires building with the metal feature")
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            #[cfg(feature = "cuda")]
            Self::Flash => "flash",
            #[cfg(feature = "metal")]
            Self::Metal => "metal-sdpa",
        }
    }

    /// Applies scaled dot-product attention to `[B, heads, N, head_dim]`
    /// tensors and preserves that layout in the result.
    pub(super) fn forward(self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let head_dim = q.dim(D::Minus1)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        match self {
            Self::Candle => candle_attention(q, k, v, scale),
            #[cfg(feature = "cuda")]
            Self::Flash => flash_attention(q, k, v, scale as f32),
            #[cfg(feature = "metal")]
            Self::Metal => metal_attention(q, k, v, scale as f32),
        }
    }
}

#[cfg(feature = "cuda")]
fn flash_capability(device: &Device) -> Result<Option<(i32, i32)>> {
    let Device::Cuda(device) = device else {
        return Ok(None);
    };
    device
        .cuda_stream()
        .context()
        .compute_capability()
        .map(Some)
        .map_err(Error::wrap)
}

fn candle_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let scores = (q
        .contiguous()?
        .matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?
        * scale)?;
    softmax(&scores, D::Minus1)?.matmul(&v.contiguous()?)
}

#[cfg(feature = "cuda")]
fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let q = q.transpose(1, 2)?;
    let k = k.transpose(1, 2)?;
    let v = v.transpose(1, 2)?;
    candle_flash_attn::flash_attn(&q, &k, &v, scale, false)?.transpose(1, 2)
}

#[cfg(feature = "metal")]
fn metal_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    sdpa(q, k, v, None, false, scale, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_devices_choose_candle_attention() -> Result<()> {
        let backend = AttentionBackend::from_setting(None, &Device::Cpu, DType::BF16)?;
        assert_eq!(backend, AttentionBackend::Candle);
        Ok(())
    }

    #[test]
    fn candle_override_is_backend_independent() -> Result<()> {
        let backend =
            AttentionBackend::from_setting(Some(OsStr::new("candle")), &Device::Cpu, DType::F32)?;
        assert_eq!(backend, AttentionBackend::Candle);
        Ok(())
    }

    #[test]
    fn baseline_alias_selects_candle_attention() -> Result<()> {
        let backend =
            AttentionBackend::from_setting(Some(OsStr::new("baseline")), &Device::Cpu, DType::F32)?;
        assert_eq!(backend, AttentionBackend::Candle);
        Ok(())
    }

    #[test]
    fn unknown_override_is_rejected() {
        let result =
            AttentionBackend::from_setting(Some(OsStr::new("unknown")), &Device::Cpu, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn metal_override_rejects_non_metal_devices() {
        let result =
            AttentionBackend::from_setting(Some(OsStr::new("metal")), &Device::Cpu, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn candle_attention_preserves_shape() -> Result<()> {
        let q = Tensor::new(
            &[[[[1f32, 0.], [0., 1.]], [[1., 1.], [0.5, 0.5]]]],
            &Device::Cpu,
        )?;
        let output = AttentionBackend::Candle.forward(&q, &q, &q)?;
        assert_eq!(output.dims(), q.dims());
        Ok(())
    }
}
