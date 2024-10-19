import copy
import functools
import logging
import os
import torch
import torch.utils.checkpoint
import transformers
import diffusers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from pathlib import Path

from utils import encode_prompt
from solver import DDIMSolver
from utils import log_validation
from train_utils import multiboundary_cd_loss, cd_loss
from dataset import get_coco_loader


MAX_SEQ_LENGTH = 77
torch.set_num_threads(40)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------------------------
def train(args):

    # LOGGING
    accelerator, logging_dir = prepare_accelerators(args)

    # MODELS
    # ----------------------------------------
    unet, teacher_unet, text_encoder, tokenizer, vae, weight_dtype = prepare_models(args, accelerator)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model, subfolder="scheduler",
    )

    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
        num_boundaries=args.num_boundaries,
        num_inverse_boundaries=args.num_boundaries,
    )
    solver = solver.to(accelerator.device)
    # ----------------------------------------

    # OPTIMIZERS
    # ----------------------------------------
    optimizer = torch.optim.AdamW(
                        unet.parameters(),
                        lr=args.learning_rate,
                        betas=(args.adam_beta1, args.adam_beta2),
                        weight_decay=args.adam_weight_decay,
                        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
                        args.lr_scheduler,
                        optimizer=optimizer,
                        num_warmup_steps=args.lr_warmup_steps,
                        num_training_steps=args.max_train_steps)

    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    # ----------------------------------------

    # DATASET
    # ----------------------------------------
    train_dataloader, uncond_prompt_embeds = prepare_data(args, accelerator, tokenizer, text_encoder)
    train_dataloader = iter(train_dataloader)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )
    # ----------------------------------------

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    ##################################################################################
    # TRAINING LOOP
    ##################################################################################

    for step in range(args.max_train_steps):

        # Train step
        # ----------------------------------------------------
        batch = next(train_dataloader)
        latents, encoded_text, prompt_embeds, noise, bsz = sample_batch(batch, accelerator, vae,
                                                                        compute_embeddings_fn, weight_dtype)

        generator_fn = multiboundary_cd_loss if args.task_type == 'multi_cd' else cd_loss
        loss = generator_fn(
                        args, accelerator, latents, noise,
                        prompt_embeds, uncond_prompt_embeds, encoded_text,
                        unet, teacher_unet,
                        solver, args.w,
                        noise_scheduler, optimizer, lr_scheduler,
                        weight_dtype,
            )
        # ----------------------------------------------------

        # Validation
        # ----------------------------------------------------
        if accelerator.local_process_index == 0:
            with torch.no_grad():
                if global_step % args.validation_steps == 0:
                    log_validation(vae, unet, args, accelerator, weight_dtype, global_step)

            progress_bar.update(1)
            global_step += 1

        logs = {
            "lr": lr_scheduler.get_last_lr()[0],
            "loss": loss.detach().item(),
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break
        # ----------------------------------------------------

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        torch.save(unet.state_dict(), f'{args.output_dir}/model_weights_{args.task_type}.pth')

    accelerator.end_training()

    ##################################################################################

# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def sample_batch(batch, accelerator, vae, compute_embeddings_fn, weight_dtype):
    image, text = batch['image'], batch['text']
    image = image.to(accelerator.device, non_blocking=True)
    encoded_text = compute_embeddings_fn(text)
    pixel_values = image.to(dtype=weight_dtype)
    if vae.dtype != weight_dtype:
        vae.to(dtype=weight_dtype)
    prompt_embeds = encoded_text.pop("prompt_embeds")

    # encode pixel values with batch size of at most 32
    latents = []
    for i in range(0, pixel_values.shape[0], 32):
        latents.append(vae.encode(pixel_values[i: i + 32]).latent_dist.sample())
    latents = torch.cat(latents, dim=0)
    latents = latents * vae.config.scaling_factor
    latents = latents.to(weight_dtype)
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    return latents, encoded_text, prompt_embeds, noise, bsz
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def prepare_models(args, accelerator):

    # TEACHER INIT
    # ----------------------------------------
    # 2. Load tokenizers from SD1.5 checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model, subfolder="tokenizer", use_fast=False
    )

    # 3. Load text encoders from SD-1.5 checkpoint.
    text_encoder = StableDiffusionPipeline.from_pretrained(
        args.pretrained_teacher_model,
        torch_dtype=torch.float16,
        variant="fp16",
    ).text_encoder

    # 4. Load VAE from SD1.5 checkpoint
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
    )

    # 5. Load teacher U-Net from SD1.5 checkpoint
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet",
        torch_dtype = torch.float16,
        variant = "fp16"
    )

    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    # ----------------------------------------

    # STUDENT INIT
    # ----------------------------------------
    # 7. Load student U-net
    unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_teacher_model, subfolder="unet",
        )
    teacher_unet = copy.deepcopy(unet)
    unet.train()

    # 9. Cast to weight type and move to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)

    teacher_unet.to(accelerator.device)
    teacher_unet.to(dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    # ----------------------------------------

    return unet, teacher_unet, text_encoder, tokenizer, vae, weight_dtype
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def prepare_data(args, accelerator, tokenizer, text_encoder):
    train_dataloader = get_coco_loader(args, batch_size=args.train_batch_size, is_train=True)

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    uncond_input_ids = tokenizer(
        [""] * args.train_batch_size, return_tensors="pt", padding="max_length", max_length=77
    ).input_ids.to(accelerator.device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]

    return train_dataloader, uncond_prompt_embeds
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def prepare_accelerators(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    mixed_precision=args.mixed_precision,
                    log_with=args.report_to,
                    project_config=accelerator_project_config,
                    split_batches=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    return accelerator, logging_dir
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
    prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
    return {"prompt_embeds": prompt_embeds}
# ---------------------------------------------------------------------------------------------