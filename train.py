import copy
import functools
import logging
import math
import os
import random
import shutil
import torch
import torch.utils.checkpoint
import transformers
import diffusers
import yaml
import types
import torch.distributed as dist

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
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

from src.utils import recover_resume_step, get_module_kohya_state_dict_2, encode_prompt
from src.lcm import guidance_scale_embedding, append_dims, predicted_origin, extract_into_tensor, DDIMSolver, update_ema
from src.eval import log_validation, distributed_sampling, calculate_scores
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
        args.pretrained_teacher_model, subfolder="scheduler", revision=args.teacher_revision
    )

    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
        num_endpoints=args.num_endpoints,
        num_inverse_endpoints=args.num_endpoints,
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
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
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
        w_embedding, w_embedding_fake, w = sample_w(args, len(latents), latents)

        generator_fn = multiboundary_cd_loss if args.task_type == 'baseline_dmd2' else multiboundary_cd_loss
        loss = generator_fn(
                        args, accelerator, latents, noise,
                        prompt_embeds, uncond_prompt_embeds, encoded_text,
                        unet, teacher_unet,
                        solver, w, w_embedding,
                        noise_scheduler, optimizer, lr_scheduler,
                        weight_dtype, global_step,
            )
        # ----------------------------------------------------

        # Validation
        # ----------------------------------------------------
        if accelerator.local_process_index == 0:
            with torch.no_grad():
                if global_step % args.validation_steps == 0:
                    w_list = [int(x) for x in args.w_list.split(",")]
                    for w in w_list:
                        log_validation(vae, unet, args, accelerator, weight_dtype, global_step, w_guidance=w)

            progress_bar.update(1)
            global_step += 1

        logs = {
            "lr": lr_scheduler.get_last_lr()[0],
            "fake_diffusion_loss": loss.detach().item(),
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
        unet.save_pretrained(args.output_dir)
        lora_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
        StableDiffusionPipeline.save_lora_weights(os.path.join(args.output_dir, "unet_lora"), lora_state_dict)

    accelerator.end_training()

    ##################################################################################

# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def sample_batch(batch, accelerators, vae, compute_embeddings_fn, weight_dtype):
    image, text = batch['image'], batch['text']
    image = image.to(accelerators[0].device, non_blocking=True)
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
def sample_w(args, bsz, latents):
    w_list = [int(x) for x in args.w_list.split(",")]
    w = torch.tensor(random.choices(w_list, k=bsz))
    if args.embed_guidance:
        w_embedding = guidance_scale_embedding(w, embedding_dim=512)
        w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
    else:
        w_embedding = None
    w = w.reshape(bsz, 1, 1, 1)
    w = w.to(device=latents.device, dtype=latents.dtype)
    if args.task_type == 'baseline_dmd2':
        w_fake = torch.tensor(random.choices([int(x) for x in args.w_list_fake.split(",")], k=bsz))
        w_embedding_fake = guidance_scale_embedding(w_fake, embedding_dim=512)
        w_embedding_fake = w_embedding_fake.to(device=latents.device, dtype=latents.dtype)
    else:
        w_embedding_fake = w_embedding
    return w_embedding, w_embedding_fake, w
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def prepare_models(args, accelerator):

    # TEACHER INIT
    # ----------------------------------------
    # 2. Load tokenizers from SD1.5 checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model, subfolder="tokenizer", revision=args.teacher_revision, use_fast=False
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
        revision=args.teacher_revision,
    )

    # 5. Load teacher U-Net from SD1.5 checkpoint
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision,
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
            args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision,
            time_cond_proj_dim=512, low_cpu_mem_usage=False, device_map=None
        )
    teacher_unet = copy.deepcopy(unet)
    teacher_unet.load_state_dict(torch.load(args.teacher_checkpoint))
    unet.load_state_dict(torch.load(args.teacher_checkpoint))
    unet.train()

    # 8. Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.
    modules = ["to_q",
               "to_k",
               "to_v",
               "to_out.0",
               "proj_in",
               "proj_out",
               "ff.net.0.proj",
               "ff.net.2",
               "conv1",
               "conv2",
               "conv_shortcut",
               "downsamplers.0.conv",
               "upsamplers.0.conv",
               "time_emb_proj"]

    lora_config = LoraConfig(
            r=args.lora_rank,
            target_modules=modules,
        )
    unet = get_peft_model(unet, lora_config)

    # 9. Cast to weight type and move to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
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

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
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