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
from safetensors.torch import load_file
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from pathlib import Path
from omegaconf import OmegaConf
from yt_tools.nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot, copy_logs_to_logs_path
from yt_tools.utils import instantiate_from_config

from src.utils import recover_resume_step, get_module_kohya_state_dict_2, encode_prompt
from src.lcm import guidance_scale_embedding, append_dims, predicted_origin, extract_into_tensor, DDIMSolver, update_ema
from src.eval import log_validation, distributed_sampling, calculate_scores
from src.train import multiboundary_cd_loss, diffusion_loss, dmd_loss
from src.sd_fake_with_gan import FakeUnetCls, classify_forward


MAX_SEQ_LENGTH = 77
torch.set_num_threads(40)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------------------------
def train(args):

    # LOGGING
    accelerators, accelerator_fake, logging_dir = prepare_accelerators(args)

    # MODELS
    # ----------------------------------------
    unets, teacher_unet, fake_unet, text_encoder, tokenizer, vae, weight_dtype = prepare_models(args, accelerators)
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
    solver = solver.to(accelerators[0].device)
    # ----------------------------------------

    # OPTIMIZERS
    # ----------------------------------------
    optimizers = [torch.optim.AdamW(
                        unet.parameters(),
                        lr=args.learning_rate,
                        betas=(args.adam_beta1, args.adam_beta2),
                        weight_decay=args.adam_weight_decay,
                        eps=args.adam_epsilon,
                ) for unet in unets]
    optimizer_fake = torch.optim.AdamW(
                        fake_unet.parameters(),
                        lr=args.learning_rate,
                        betas=(args.adam_beta1, args.adam_beta2),
                        weight_decay=args.adam_weight_decay,
                        eps=args.adam_epsilon,
                     ) if args.task_type == 'baseline_dmd2' else None
    lr_schedulers = [
                    get_scheduler(
                        args.lr_scheduler,
                        optimizer=optimizer,
                        num_warmup_steps=args.lr_warmup_steps,
                        num_training_steps=args.max_train_steps,
                    ) for optimizer in optimizers]
    lr_scheduler_fake = get_scheduler(
                        args.lr_scheduler,
                        optimizer=optimizer_fake,
                        num_warmup_steps=args.lr_warmup_steps,
                        num_training_steps=args.max_train_steps,
                    ) if args.task_type == 'baseline_dmd2' else None

    for i in range(len(accelerators)):
        unets[i], optimizers[i], lr_schedulers[i] = accelerators[i].prepare(unets[i], optimizers[i], lr_schedulers[i])
    if args.task_type == 'baseline_dmd2':
        fake_unet, optimizer_fake, lr_scheduler_fake = accelerator_fake.prepare(fake_unet, optimizer_fake, lr_scheduler_fake)
    # ----------------------------------------

    # DATASET
    # ----------------------------------------
    train_dataloader, uncond_prompt_embeds = prepare_data(args, accelerators[0], tokenizer, text_encoder)
    train_dataloader = iter(train_dataloader)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )
    # ----------------------------------------

    # Train!
    total_batch_size = args.train_batch_size * accelerators[0].num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
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
        disable=not accelerators[0].is_local_main_process,
    )

    ##################################################################################
    # TRAINING LOOP
    ##################################################################################

    for step in range(args.max_train_steps):

        # Train step
        # ----------------------------------------------------
        # Fake score/discriminator update if needed
        if accelerator_fake is not None:
            with accelerator_fake.accumulate(fake_unet):
                for i in range(len(unets)):
                    # Timestep index
                    idx_of_endpoints = [torch.where(solver.ddim_timesteps == i)[0] for i in solver.endpoints]
                    idx_of_endpoints[0] = torch.tensor([0])
                    additional_idx_min = 0 if i == 0 else 1
                    idx_min = idx_of_endpoints[i].item() + additional_idx_min
                    idx_max = idx_of_endpoints[i + 1].item() + 1 if i < len(accelerators) - 1 else args.num_ddim_timesteps
                    for _ in range(args.n_fake_updates_per_iter):
                        # Sample
                        batch = next(train_dataloader)
                        latents, encoded_text, prompt_embeds, noise, bsz = sample_batch(batch,
                                                                                        accelerators,
                                                                                        vae,
                                                                                        compute_embeddings_fn,
                                                                                        weight_dtype)
                        _, w_embedding_fake, w = sample_w(args, len(latents), latents)

                        fake_net_loss = diffusion_loss(
                            args, accelerator_fake, latents, noise,
                            prompt_embeds, uncond_prompt_embeds, encoded_text,
                            unets[i], [teacher_unet, fake_unet],
                            solver, w, w_embedding_fake,
                            noise_scheduler, optimizer_fake, lr_scheduler_fake,
                            weight_dtype, global_step,
                            min_idx=idx_min, max_idx=idx_max,
                        ).detach().item()
                        torch.cuda.empty_cache()
        else:
            fake_net_loss = 0.0

        # Generator update
        losses = {}
        for i in range(len(accelerators)):
            # Sample
            batch = next(train_dataloader)
            latents, encoded_text, prompt_embeds, noise, bsz = sample_batch(batch, accelerators, vae,
                                                                            compute_embeddings_fn, weight_dtype)
            w_embedding, w_embedding_fake, w = sample_w(args, len(latents), latents)

            # Timestep index
            idx_of_endpoints = [torch.where(solver.ddim_timesteps == i)[0] for i in solver.endpoints]
            idx_of_endpoints[0] = torch.tensor([0])
            additional_idx_min = 0 if i == 0 else 1
            idx_min = idx_of_endpoints[i].item() + additional_idx_min
            idx_max = idx_of_endpoints[i + 1].item() + 1 if i < len(accelerators) - 1 else args.num_ddim_timesteps

            with accelerators[i].accumulate(unets[i]):
                generator_fn = dmd_loss if args.task_type == 'baseline_dmd2' else multiboundary_cd_loss
                fixed_unets = [teacher_unet, fake_unet] if args.task_type == 'baseline_dmd2' else teacher_unet
                w_embedding = [w_embedding, w_embedding_fake] if args.task_type == 'baseline_dmd2' else w_embedding
                loss = generator_fn(
                        args, accelerators[i], latents, noise,
                        prompt_embeds, uncond_prompt_embeds, encoded_text,
                        unets[i], fixed_unets,
                        solver, w, w_embedding,
                        noise_scheduler, optimizers[i], lr_schedulers[i],
                        weight_dtype, global_step,
                        min_idx=idx_min, max_idx=idx_max,
                )
                losses[str(i)] = loss.detach().item()
                torch.cuda.empty_cache()
        # ----------------------------------------------------

        # Validation
        # ----------------------------------------------------
        if accelerators[0].sync_gradients:  # TODO

            if accelerators[0].local_process_index == 0:
                # Checkpoint
                #saving(args, global_step, accelerator, inverse_accelerator, logging_dir)

                # Snapshot
                with torch.no_grad():
                    if global_step % args.validation_steps == 0:
                        w_list = [int(x) for x in args.w_list.split(",")]
                        for w in w_list:
                            log_validation(vae, unets,
                                           args, accelerators[0], weight_dtype, global_step, w_guidance=w)
                        copy_out_to_snapshot(args.output_dir)
                        copy_logs_to_logs_path(logging_dir)

            # Sampling and metrics
            if global_step % args.evaluation_steps == 0 and args.evaluation_steps >= 0:
                w_list = [int(x) for x in args.w_list.split(",")]
                for w in w_list:
                    images, prompts = distributed_sampling(vae, unets,
                                                           args, accelerators[0], weight_dtype, step,
                                                           num_inference_steps=4, batch_size=16, max_cnt=5000,
                                                           w_guidance=w)
                    additional_images = []
                    if args.calc_diversity_score:
                        for i in range(4):
                            imgs, _ = distributed_sampling(vae, unets,
                                                           args, accelerators[0], weight_dtype, step,
                                                           num_inference_steps=4, batch_size=16, max_cnt=5000,
                                                           w_guidance=w, seed=i)
                            additional_images.append(imgs)


                    if accelerators[0].is_main_process:
                        image_reward, pick_score, clip_score, fid_score, div_score = calculate_scores(args,
                                                                                                      images,
                                                                                                      prompts,
                                                                                                      additional_images)
                        logs = {
                                f"fid_{w}": fid_score.item(),
                                f"pick_score_{w}": pick_score.item(),
                                f"clip_score_{w}": clip_score.item(),
                                f"image_reward_{w}": image_reward.item(),
                                f"diversity_score_{w}": div_score.item(),
                        }
                        print(logs)
                        accelerators[0].log(logs, step=global_step)
                        copy_logs_to_logs_path(logging_dir)

            progress_bar.update(1)
            global_step += 1

        logs = {
            "lr": lr_schedulers[0].get_last_lr()[0],
            "fake_diffusion_loss": fake_net_loss,
        }
        logs.update(losses)
        progress_bar.set_postfix(**logs)
        accelerators[0].log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break
        # ----------------------------------------------------

    # Create the pipeline using the trained modules and save it.
    accelerators[0].wait_for_everyone()
    if accelerators[0].is_main_process:
        for i in range(len(accelerators)):
            unet = accelerators[i].unwrap_model(unets[i])
            unet.save_pretrained(args.output_dir)
            lora_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
            StableDiffusionPipeline.save_lora_weights(os.path.join(args.output_dir, "unet_lora"), lora_state_dict)

    for i in range(len(accelerators)):
        accelerators[i].end_training()

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
def prepare_models(args, accelerators):

    # TEACHER INIT
    # ----------------------------------------
    # 2. Load tokenizers from SD-XL checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model, subfolder="tokenizer", revision=args.teacher_revision, use_fast=False
    )

    # 3. Load text encoders from SD-1.5 checkpoint.
    text_encoder = StableDiffusionPipeline.from_pretrained(
        args.pretrained_teacher_model,
        torch_dtype=torch.float16,
        variant="fp16",
    ).text_encoder

    # 4. Load VAE from SD-XL checkpoint (or more stable VAE)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.teacher_revision,
    )

    # 5. Load teacher U-Net from SD-XL checkpoint
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

    # STUDENTS INIT
    # ----------------------------------------
    unets = []
    if args.embed_guidance:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision,
            time_cond_proj_dim=512, low_cpu_mem_usage=False, device_map=None
        )
        if len(args.teacher_checkpoint) > 0:
            print(f'Loading teacher from {args.teacher_checkpoint}')
            teacher_unet = copy.deepcopy(unet)
            teacher_unet.load_state_dict(torch.load(args.teacher_checkpoint))
            unet.load_state_dict(torch.load(args.teacher_checkpoint))
        if len(args.forward_checkpoint) > 0:
            tmp_pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_teacher_model,
                vae=vae,
                unet=unet,
                scheduler=LCMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler"),
                revision=args.revision,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            lora_weight = load_file(args.forward_checkpoint)
            lora_state_dict = get_module_kohya_state_dict_2(lora_weight, "lora_unet", torch.float16)
            tmp_pipeline.load_lora_weights(lora_state_dict)
            tmp_pipeline.fuse_lora()
            unet = tmp_pipeline.unet
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
        )
    unet.train()

    unets.append(unet)
    for _ in range(args.num_students - 1):
        unets.append(copy.deepcopy(unet))

    # 8. Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.
    if args.do_lora:
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

        for i, unet in enumerate(unets):
            unets[i] = get_peft_model(unet, lora_config)

    weight_dtype = torch.float32
    if accelerators[0].mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerators[0].mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerators[0].device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerators[0].device, dtype=weight_dtype)
    for i, unet in enumerate(unets):
        unet.to(accelerators[i].device)

    # Move teacher_unet to device, optionally cast to weight_dtype
    teacher_unet.to(accelerators[0].device)
    fake_unet = copy.deepcopy(teacher_unet) if args.task_type == 'baseline_dmd2' else None
    teacher_unet.to(dtype=weight_dtype)

    if args.gradient_checkpointing:
        for i, unet in enumerate(unets):
            unet.enable_gradient_checkpointing()
        if args.task_type == 'baseline_dmd2':
            fake_unet.enable_gradient_checkpointing()

    if args.do_gan:
        fake_unet.forward = types.MethodType(
            classify_forward, fake_unet
        )
        fake_unet = FakeUnetCls(args, accelerators, fake_unet)
        fake_unet.to(accelerators[0].device)
    # ----------------------------------------

    return unets, teacher_unet, fake_unet, text_encoder, tokenizer, vae, weight_dtype
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def prepare_data(args, accelerator, tokenizer, text_encoder):
    global_batch_size = args.train_batch_size * accelerator.num_processes
    if dist.get_rank() == 0:
        copy_snapshot_to_out(args.output_dir)
    dist.barrier()
    current_step = recover_resume_step(args.output_dir)
    logger.info(f"Resume the LAION training from {global_batch_size * current_step}")

    with open(args.laion_config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config = OmegaConf.create(config)
        config['train_dataloader'][0]['params']['batch_size'] = args.train_batch_size

    train_dataloader = instantiate_from_config(config['train_dataloader'][0],
                                               skip_rows=global_batch_size * current_step
                                               )

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

    accelerators = [Accelerator(
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    mixed_precision=args.mixed_precision,
                    log_with=args.report_to,
                    project_config=accelerator_project_config,
                    split_batches=True,
                    ) for _ in range(args.num_students)]
    accelerator_fake = Accelerator(
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    mixed_precision=args.mixed_precision,
                    log_with=args.report_to,
                    project_config=accelerator_project_config,
                    split_batches=True,
                    ) if args.task_type == 'baseline_dmd2' else None

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerators[0].state, main_process_only=False)
    if accelerators[0].is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerators[0].is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    return accelerators, accelerator_fake, logging_dir
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def saving(args, global_step, accelerator, inverse_accelerator, logging_dir):
    if (global_step + 10) % args.checkpointing_steps == 0:
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)

        save_path = os.path.join(args.output_dir, f"inverse-checkpoint-{global_step}")
        inverse_accelerator.save_state(save_path)

        logger.info(f"Saved state to {save_path}")
        copy_out_to_snapshot(args.output_dir)
        copy_logs_to_logs_path(logging_dir)
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
    prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
    return {"prompt_embeds": prompt_embeds}
# ---------------------------------------------------------------------------------------------