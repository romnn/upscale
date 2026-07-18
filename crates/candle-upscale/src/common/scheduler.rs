//! DDIM (v-prediction) denoising scheduler + DDPM low-res noising (candle port).
//!
//! Byte-for-byte the same schedule math as `sd-upscale/src/scheduler.rs`; only
//! the tensor ops differ (candle instead of burn). The scalar schedule is
//! computed on the host and applied to tensors via affine ops.

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const BETA_START: f64 = 0.0001;
const BETA_END: f64 = 0.02;

use candle_core::{Result, Tensor};

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
pub(crate) struct DdimScheduler {
    alphas_cumprod: Vec<f32>,
    final_alpha_cumprod: f32,
    num_train_timesteps: usize,
    timesteps: Vec<i64>,
    step_ratio: usize,
}

impl DdimScheduler {
    pub(crate) fn new() -> Self {
        let alphas_cumprod = alphas_cumprod();
        // `set_alpha_to_one=false`: final_alpha_cumprod = alphas_cumprod[0].
        let final_alpha_cumprod = alphas_cumprod[0];
        Self {
            alphas_cumprod,
            final_alpha_cumprod,
            num_train_timesteps: NUM_TRAIN_TIMESTEPS,
            timesteps: Vec::new(),
            step_ratio: 1,
        }
    }

    /// `step_ratio = num_train_timesteps / num_inference_steps` (integer div);
    /// `timesteps = (arange(num_inference_steps) * step_ratio)[::-1] + 1`.
    pub(crate) fn set_timesteps(&mut self, num_inference_steps: usize) {
        let step_ratio = self.num_train_timesteps / num_inference_steps.max(1);
        self.step_ratio = step_ratio;
        self.timesteps = (0..num_inference_steps)
            .rev()
            .map(|i| (i * step_ratio) as i64 + 1)
            .collect();
    }

    pub(crate) fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    fn alpha_cumprod_at(&self, t: i64) -> f32 {
        self.alphas_cumprod[t as usize]
    }

    /// v-prediction DDIM update (`eta=0`, `clip_sample=false`).
    pub(crate) fn step(
        &self,
        model_output: &Tensor,
        timestep: i64,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let prev_t = timestep - self.step_ratio as i64;

        let alpha_prod_t = self.alpha_cumprod_at(timestep);
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alpha_cumprod_at(prev_t)
        } else {
            self.final_alpha_cumprod
        };
        let beta_prod_t = 1.0 - alpha_prod_t;

        let sqrt_alpha_prod_t = f64::from(alpha_prod_t.sqrt());
        let sqrt_beta_prod_t = f64::from(beta_prod_t.sqrt());

        let pred_original = ((sample * sqrt_alpha_prod_t)? - (model_output * sqrt_beta_prod_t)?)?;
        let pred_epsilon = ((model_output * sqrt_alpha_prod_t)? + (sample * sqrt_beta_prod_t)?)?;

        let out = ((pred_original * f64::from(alpha_prod_t_prev.sqrt()))?
            + (pred_epsilon * f64::from((1.0 - alpha_prod_t_prev).sqrt()))?)?;
        Ok(out)
    }
}

/// DDPM low-res noising, matching diffusers `DDPMScheduler` — only `add_noise`
/// is needed here.
pub(crate) struct LowResNoiser {
    alphas_cumprod: Vec<f32>,
}

impl LowResNoiser {
    pub(crate) fn new() -> Self {
        Self {
            alphas_cumprod: alphas_cumprod(),
        }
    }

    /// `noisy = sqrt(acp[t]) * original + sqrt(1 - acp[t]) * noise`
    pub(crate) fn add_noise(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timestep: i64,
    ) -> Result<Tensor> {
        let acp = self.alphas_cumprod[timestep as usize];
        let out = ((original * f64::from(acp.sqrt()))? + (noise * f64::from((1.0 - acp).sqrt()))?)?;
        Ok(out)
    }
}
