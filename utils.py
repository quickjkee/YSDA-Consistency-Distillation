import os
import torch
import numpy as np
import random
import gc
import copy
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, LCMScheduler
from solver import sample_deterministic

logger = get_logger(__name__)

# ---------------------------------------------------------------------
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds
# ---------------------------------------------------------------------


@torch.no_grad()
# ---------------------------------------------------------------------
def log_validation(vae, unet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)
    pipeline = StableDiffusionPipeline.from_pretrained(
           args.pretrained_teacher_model,
           vae=vae,
           unet=copy.deepcopy(unet).eval(),
           scheduler=LCMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler"),
           torch_dtype=weight_dtype,
           safety_checker=None,
        )
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        'A sad puppy with large eyes',
        'A girl with pale blue hair and a cami tank top',
        'cute girl, Kyoto animation, 4k, high resolution',
        "A person laying on a surfboard holding his dog",
        "Green commercial building with refrigerator and refrigeration units outside",
        "An airplane with two propellor engines flying in the sky",
        "Four cows in a pen on a sunny day",
        "Three dogs sleeping together on an unmade bed",
        "a deer with bird feathers, highly detailed, full body"
    ]

    image_logs = []

    for j, prompt in enumerate(validation_prompts):
        with torch.autocast("cuda", dtype=weight_dtype):
            if args.task_type == 'multi_cd':
                images = sample_deterministic(
                    pipeline,
                    [prompt] * 4,
                    unet=unet,
                    num_inference_steps=args.num_boundaries,
                    generator=generator,
                )
            else:
                images = pipeline(
                    prompt=prompt,
                    num_inference_steps=4,
                    num_images_per_prompt=4,
                    generator=generator,
                    guidance_scale=0.0,
                ).images
            os.makedirs(f'{args.output_dir}/snapshots_{args.task_type}', exist_ok=True)
            for u, image in enumerate(images):
                image.save(f'{args.output_dir}/snapshots_{args.task_type}/{step}_{j}_{u}.jpg')

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return image_logs
# ---------------------------------------------------------------------