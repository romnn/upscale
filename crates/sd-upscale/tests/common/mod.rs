//! Shared parity-test support: a feature-selected backend and fixture helpers.
//!
//! Default is CPU (`NdArray`) so `cargo test` works everywhere; pass `--features
//! wgpu` (our deployment backend) or `--features cuda` to run the same tests on
//! the GPU — orders of magnitude faster than CPU conv, and for wgpu it also
//! proves the target backend actually supports every op we use.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test/example code fails loudly by design"
)]
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor, TensorData};
use safetensors::SafeTensors;

#[cfg(feature = "wgpu")]
pub type TestBackend = burn::backend::Wgpu;
#[cfg(feature = "wgpu")]
pub type TestDevice = burn::backend::wgpu::WgpuDevice;

#[cfg(all(feature = "cuda", not(feature = "wgpu")))]
pub type TestBackend = burn::backend::Cuda;
#[cfg(all(feature = "cuda", not(feature = "wgpu")))]
pub type TestDevice = burn::backend::cuda::CudaDevice;

#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
pub type TestBackend = burn::backend::NdArray;
#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
pub type TestDevice = burn::backend::ndarray::NdArrayDevice;

pub fn test_device() -> TestDevice {
    TestDevice::default()
}

/// Read an f32 tensor from a fixture safetensors into a burn tensor.
pub fn tensor_from<B: Backend, const D: usize>(
    st: &SafeTensors,
    name: &str,
    device: &B::Device,
) -> Tensor<B, D> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|_| panic!("fixture missing tensor {name}"));
    let shape: [usize; D] = view.shape().try_into().expect("rank mismatch");
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Tensor::<B, D>::from_data(TensorData::new(data, shape), device)
}

/// Relative L2: `||actual - expected|| / ||expected||`. Scale-invariant, which
/// matters because activations span many orders of magnitude across a deep stack.
pub fn rel_l2<B: Backend, const D: usize>(actual: Tensor<B, D>, expected: Tensor<B, D>) -> f32 {
    let diff: f32 = (actual - expected.clone())
        .powi_scalar(2)
        .sum()
        .into_scalar()
        .elem();
    let denom: f32 = expected.powi_scalar(2).sum().into_scalar().elem();
    (diff / denom.max(1e-20)).sqrt()
}
