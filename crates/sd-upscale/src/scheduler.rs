//! DDIM (v-prediction) denoising scheduler + DDPM low-res noising for the
//! SD x4 upscaler.
//!
//! Both schedulers share the same `beta_schedule="scaled_linear"` beta/alpha
//! schedule (`beta_start=0.0001`, `beta_end=0.02`, `num_train_timesteps=1000`)
//! but differ in prediction type and step math, matching the reference
//! diffusers `DDIMScheduler`/`DDPMScheduler` configs loaded from the x4
//! upscaler checkpoint's `scheduler`/`low_res_scheduler` subfolders. See
//! `tests/scheduler_parity.rs` for the numerical-parity proof.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const BETA_START: f64 = 0.0001;
const BETA_END: f64 = 0.02;

/// `betas = linspace(sqrt(beta_start), sqrt(beta_end), num_train_timesteps)**2`
/// (`beta_schedule="scaled_linear"`), then `alphas_cumprod = cumprod(1 - betas)`.
fn alphas_cumprod() -> Vec<f32> {
    let start = BETA_START.sqrt();
    let end = BETA_END.sqrt();
    let n = NUM_TRAIN_TIMESTEPS;
    let mut acp = Vec::with_capacity(n);
    let mut running = 1.0f64;
    for i in 0..n {
        let t = start + (end - start) * (i as f64) / ((n - 1) as f64);
        let beta = t * t;
        let alpha = 1.0 - beta;
        running *= alpha;
        acp.push(running as f32);
    }
    acp
}

/// DDIM (v-prediction) denoising scheduler, `eta=0`, matching diffusers
/// `DDIMScheduler` with `prediction_type="v_prediction"`,
/// `set_alpha_to_one=false`, `steps_offset=1`, `clip_sample=false`.
pub struct DdimScheduler {
    alphas_cumprod: Vec<f32>,
    final_alpha_cumprod: f32,
    num_train_timesteps: usize,
    timesteps: Vec<i64>,
    step_ratio: usize,
}

impl DdimScheduler {
    /// Builds the scheduler and defaults to the full `NUM_TRAIN_TIMESTEPS`
    /// schedule; call [`set_timesteps`](Self::set_timesteps) to pick the
    /// inference step count.
    pub fn new() -> Self {
        let alphas_cumprod = alphas_cumprod();
        // `set_alpha_to_one=false`: final_alpha_cumprod = alphas_cumprod[0].
        let final_alpha_cumprod = alphas_cumprod[0];
        let mut this = Self {
            alphas_cumprod,
            final_alpha_cumprod,
            num_train_timesteps: NUM_TRAIN_TIMESTEPS,
            timesteps: Vec::new(),
            step_ratio: 1,
        };
        this.set_timesteps(NUM_TRAIN_TIMESTEPS);
        this
    }

    /// `step_ratio = num_train_timesteps / num_inference_steps` (integer div);
    /// `timesteps = (arange(num_inference_steps) * step_ratio).round()[::-1] + 1`
    /// (`steps_offset=1`). Stored descending, matching `self.timesteps()`.
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        self.step_ratio = step_ratio;
        self.timesteps = (0..num_inference_steps)
            .rev()
            .map(|i| (i * step_ratio) as i64 + 1)
            .collect();
    }

    /// The descending diffusion timesteps to iterate over, as set by the most
    /// recent [`set_timesteps`](Self::set_timesteps).
    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    /// Initial-latent scaling factor (always `1.0` for this scheduler), applied
    /// to the sampled Gaussian noise before the first step.
    pub fn init_noise_sigma(&self) -> f32 {
        1.0
    }

    fn alpha_cumprod_at(&self, t: i64) -> f32 {
        self.alphas_cumprod[t as usize]
    }

    /// v-prediction DDIM update (`eta=0`, `clip_sample=false`).
    pub fn step<B: Backend>(
        &self,
        model_output: Tensor<B, 4>,
        timestep: i64,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let prev_t = timestep - self.step_ratio as i64;

        let alpha_prod_t = self.alpha_cumprod_at(timestep);
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alpha_cumprod_at(prev_t)
        } else {
            self.final_alpha_cumprod
        };
        let beta_prod_t = 1.0 - alpha_prod_t;

        let sqrt_alpha_prod_t = alpha_prod_t.sqrt();
        let sqrt_beta_prod_t = beta_prod_t.sqrt();

        let pred_original = sample
            .clone()
            .mul_scalar(sqrt_alpha_prod_t)
            .sub(model_output.clone().mul_scalar(sqrt_beta_prod_t));
        let pred_epsilon = model_output
            .mul_scalar(sqrt_alpha_prod_t)
            .add(sample.mul_scalar(sqrt_beta_prod_t));

        pred_original
            .mul_scalar(alpha_prod_t_prev.sqrt())
            .add(pred_epsilon.mul_scalar((1.0 - alpha_prod_t_prev).sqrt()))
    }
}

impl Default for DdimScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// DDPM low-res noising, matching diffusers `DDPMScheduler` with
/// `prediction_type="epsilon"`, `clip_sample=true`, `variance_type="fixed_small"`
/// — only `add_noise` is needed here, so clip/variance don't matter.
pub struct LowResNoiser {
    alphas_cumprod: Vec<f32>,
}

impl LowResNoiser {
    /// Builds the noiser over the shared `scaled_linear` alpha schedule.
    pub fn new() -> Self {
        Self {
            alphas_cumprod: alphas_cumprod(),
        }
    }

    /// `noisy = sqrt(acp[t]) * original + sqrt(1 - acp[t]) * noise`
    pub fn add_noise<B: Backend>(
        &self,
        original: Tensor<B, 4>,
        noise: Tensor<B, 4>,
        timestep: i64,
    ) -> Tensor<B, 4> {
        let acp = self.alphas_cumprod[timestep as usize];
        original
            .mul_scalar(acp.sqrt())
            .add(noise.mul_scalar((1.0 - acp).sqrt()))
    }
}

impl Default for LowResNoiser {
    fn default() -> Self {
        Self::new()
    }
}
