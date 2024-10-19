import numpy as np
import torch

from diffusers import StableDiffusionPipeline
from accelerate.logging import get_logger

MAX_SEQ_LENGTH = 77
logger = get_logger(__name__)


# UTILS FN

# ---------------------------------------------------------------------
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
def predicted_origin(model_output,
                     timesteps,
                     boundary_timesteps,
                     sample,
                     prediction_type,
                     alphas,
                     sigmas,
                     pred_x_0=None):
    sigmas_s = extract_into_tensor(sigmas, boundary_timesteps, sample.shape)
    alphas_s = extract_into_tensor(alphas, boundary_timesteps, sample.shape)

    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)

    # Set hard boundaries to ensure equivalence with forward (direct) CD
    alphas_s[boundary_timesteps == 0] = 1.0
    sigmas_s[boundary_timesteps == 0] = 0.0

    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas if pred_x_0 is None else pred_x_0  # x0 prediction
        pred_x_0 = alphas_s * pred_x_0 + sigmas_s * model_output  # Euler step to the boundary step
    elif prediction_type == "v_prediction":
        assert boundary_timesteps == 0, "v_prediction does not support multiple endpoints at the moment"
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
# ---------------------------------------------------------------------


# DDIM SOLVER CLASS
# ---------------------------------------------------------------------
class DDIMSolver:
    def __init__(
            self, alpha_cumprods, timesteps=1000, ddim_timesteps=50,
            num_boundaries=1,
            num_inverse_boundaries=1,
            max_inverse_timestep_index=49
    ):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(
            np.int64) - 1  # [19, ..., 999]
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_alpha_cumprods_next = np.asarray(
            alpha_cumprods[self.ddim_timesteps[1:]].tolist() + [0.0]
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)
        self.ddim_alpha_cumprods_next = torch.from_numpy(self.ddim_alpha_cumprods_next)

        # Set endpoints for direct CTM
        timestep_interval = ddim_timesteps // num_boundaries + int(ddim_timesteps % num_boundaries > 0)
        endpoint_idxs = torch.arange(timestep_interval, ddim_timesteps, timestep_interval) - 1
        self.endpoints = torch.tensor([0] + self.ddim_timesteps[endpoint_idxs].tolist())

        # Set endpoints for inverse CTM
        timestep_interval = ddim_timesteps // num_inverse_boundaries + int(ddim_timesteps % num_inverse_boundaries > 0)
        inverse_endpoint_idxs = torch.arange(timestep_interval, ddim_timesteps, timestep_interval) - 1
        inverse_endpoint_idxs = torch.tensor(inverse_endpoint_idxs.tolist() + [max_inverse_timestep_index])
        self.inverse_endpoints = self.ddim_timesteps[inverse_endpoint_idxs]

    def to(self, device):
        self.endpoints = self.endpoints.to(device)
        self.inverse_endpoints = self.inverse_endpoints.to(device)

        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        self.ddim_alpha_cumprods_next = self.ddim_alpha_cumprods_next.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev
# ---------------------------------------------------------------------


# SAMPLING FN FOR MULTIBOUDARY CD
# ---------------------------------------------------------------------
@torch.no_grad()
def sample_deterministic(
        pipe,
        prompt,
        unet,
        latents=None,
        generator=None,
        num_scales=50,
        num_inference_steps=1,
        start_timestep=19,
        max_inverse_timestep_index=49,
        return_latent=False,
):
    assert isinstance(pipe, StableDiffusionPipeline), f"Does not support the pipeline {type(pipe)}"
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(prompt, height, width, 1, None, None, None)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    device = pipe._execution_device

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt, device, 1, False)
    assert prompt_embeds.dtype == torch.float16

    # Prepare the DDIM solver
    solver = DDIMSolver(
        pipe.scheduler.alphas_cumprod.numpy(),
        timesteps=pipe.scheduler.num_train_timesteps,
        ddim_timesteps=num_scales,
        num_boundaries=num_inference_steps,
        num_inverse_boundaries=num_inference_steps,
        max_inverse_timestep_index=max_inverse_timestep_index
    ).to(device)

    timesteps = solver.inverse_endpoints.flip(0)
    boundary_timesteps = solver.endpoints.flip(0)

    alpha_schedule = torch.sqrt(pipe.scheduler.alphas_cumprod).to(device)
    sigma_schedule = torch.sqrt(1 - pipe.scheduler.alphas_cumprod).to(device)

    # 5. Prepare latent variables
    if latents is None:
        num_channels_latents = pipe.unet.config.in_channels
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        assert latents.dtype == torch.float16
    else:
        latents = latents.to(prompt_embeds.dtype)

    for i, (t, s) in enumerate(zip(timesteps, boundary_timesteps)):
        noise_pred = unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        latents = predicted_origin(
            noise_pred,
            torch.tensor([t] * len(noise_pred)).to(device),
            torch.tensor([s] * len(noise_pred)).to(device),
            latents,
            pipe.scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        ).half()

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * image.shape[0]
    image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)

    if return_latent:
        return image, latents
    else:
        return image
# ---------------------------------------------------------------------