import torch
import torch.nn.functional as F
from src.lcm import guidance_scale_embedding, append_dims, predicted_origin, extract_into_tensor


# Consistency Distillation
# ---------------------------------------------------------------------
def cd_loss(
        args, accelerator, latents, noise,
        prompt_embeds, encoded_text,
        unet, teacher_unet,
        solver, w_embedding, noise_scheduler,
        optimizer, lr_scheduler,
        weight_dtype
):
    optimizer.zero_grad(set_to_none=True)

    # Sample a random timestep for each image t_n ~ U[0, N - k - 1]
    # ---------------------------------------------------------------------------
    index = torch.randint(0, args.num_ddim_timesteps, (len(latents),), device=latents.device).long()  # [0, 49]
    topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
    start_timesteps = solver.ddim_timesteps[index]
    timesteps = torch.clamp(start_timesteps - topk, 0, solver.ddim_timesteps[-1])
    # ---------------------------------------------------------------------------

    # Add noise and make prediction
    # ---------------------------------------------------------------------------
    noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    noise_pred = unet(
        noisy_model_input,
        start_timesteps,
        timestep_cond=w_embedding if args.embed_guidance else None,
        encoder_hidden_states=prompt_embeds.float(),
        added_cond_kwargs=encoded_text,
    ).sample

    model_pred = predicted_origin(
        noise_pred,
        start_timesteps,
        torch.zeros_like(start_timesteps),
        noisy_model_input,
        noise_scheduler.config.prediction_type,
        alpha_schedule,
        sigma_schedule,
    )
    # ---------------------------------------------------------------------------

    # Teacher pred and solver step
    # ---------------------------------------------------------------------------
    with torch.no_grad():
        with torch.autocast("cuda", dtype=weight_dtype):
            cond_teacher_output = teacher_unet(
                noisy_model_input.to(weight_dtype),
                start_timesteps,
                timestep_cond=w_embedding if args.embed_guidance else None,
                encoder_hidden_states=prompt_embeds.to(weight_dtype),
            ).sample
        cond_pred_x0 = predicted_origin(
                cond_teacher_output,
                start_timesteps,
                torch.zeros_like(start_timesteps),
                noisy_model_input,
                noise_scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
        pred_x0 = cond_pred_x0
        pred_noise = cond_teacher_output
        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

    # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
    with torch.no_grad():
        with torch.autocast("cuda", dtype=weight_dtype):
            target_noise_pred = unet(
                x_prev.float(),
                timesteps,
                timestep_cond=w_embedding if args.embed_guidance else None,
                encoder_hidden_states=prompt_embeds.float(),
            ).sample

        target = predicted_origin(
            target_noise_pred,
            timesteps,
            torch.zeros_like(start_timesteps),
            x_prev,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )
    # ---------------------------------------------------------------------------

    # Calculate loss
    # ---------------------------------------------------------------------------
    if args.loss_type == "l2":
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    elif args.loss_type == "huber":
        # c = 0.00054 * np.prod(target.shape[1:]) ** 0.5
        c = args.huber_c
        loss = torch.sqrt((model_pred.float() - target.float()) ** 2 + c ** 2) - c
        loss = torch.mean(loss)

    # 20.4.14. Backpropagate on the online student model (`unet`)
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    # ---------------------------------------------------------------------------

    return loss
# ---------------------------------------------------------------------


# Multiboundary Consistency Distillation
# ---------------------------------------------------------------------
def multiboundary_cd_loss(
        args, accelerator, latents, noise,
        prompt_embeds, uncond_prompt_embeds, encoded_text,
        unet, teacher_unet,
        solver, w,  w_embedding, noise_scheduler,
        optimizer, lr_scheduler,
        weight_dtype, step, min_idx=0, max_idx=None, device='cuda',
):
    optimizer.zero_grad(set_to_none=True)

    # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
    # ---------------------------------------------------------------------------
    max_idx = args.num_ddim_timesteps if max_idx is None else max_idx
    index = torch.randint(min_idx, max_idx, (len(latents),), device=latents.device).long()  # [0, 49]
    topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
    start_timesteps = solver.ddim_timesteps[index]
    timesteps = torch.clamp(start_timesteps - topk, 0, solver.ddim_timesteps[-1])
    assert (start_timesteps > 0).all()
    mask = (timesteps[None, :] >= solver.endpoints[:, None]).to(int)
    mask[:-1] = mask[:-1] - mask[1:]
    boundary_timesteps = (mask * solver.endpoints[:, None]).sum(0)
    # ---------------------------------------------------------------------------

    # Noise and make pred
    # ---------------------------------------------------------------------------
    noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    noise_pred = unet(
        noisy_model_input,
        start_timesteps,
        timestep_cond=w_embedding if args.embed_guidance else None,
        encoder_hidden_states=prompt_embeds.float(),
        added_cond_kwargs=encoded_text,
    ).sample

    model_pred = predicted_origin(
        noise_pred,
        start_timesteps,
        boundary_timesteps,
        noisy_model_input,
        noise_scheduler.config.prediction_type,
        alpha_schedule,
        sigma_schedule,
    )
    # ---------------------------------------------------------------------------

    # Teacher pred and solver step
    # ---------------------------------------------------------------------------
    with torch.no_grad():
        with torch.autocast("cuda", dtype=weight_dtype):
            cond_teacher_output = teacher_unet(
                noisy_model_input.to(weight_dtype),
                start_timesteps,
                timestep_cond=w_embedding if args.embed_guidance else None,
                encoder_hidden_states=prompt_embeds.to(weight_dtype),
            ).sample
        cond_pred_x0 = predicted_origin(
                cond_teacher_output,
                start_timesteps,
                torch.zeros_like(start_timesteps),
                noisy_model_input,
                noise_scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
        pred_x0 = cond_pred_x0
        pred_noise = cond_teacher_output
        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

    # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
    with torch.no_grad():
        with torch.autocast("cuda", dtype=weight_dtype):
            target_noise_pred = unet(
                x_prev.float(),
                timesteps,
                timestep_cond=w_embedding if args.embed_guidance else None,
                encoder_hidden_states=prompt_embeds.float(),
            ).sample

        target = predicted_origin(
            target_noise_pred,
            timesteps,
            boundary_timesteps,
            x_prev,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )

        # Apply boundary condition
        boundary_mask = (append_dims(timesteps == boundary_timesteps, x_prev.ndim)).to(int)
        target = boundary_mask * x_prev + (1 - boundary_mask) * target
    # ---------------------------------------------------------------------------

    # Calculate loss
    # ---------------------------------------------------------------------------
    if args.loss_type == "l2":
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    elif args.loss_type == "huber":
        # c = 0.00054 * np.prod(target.shape[1:]) ** 0.5
        c = args.huber_c
        loss = torch.sqrt((model_pred.float() - target.float()) ** 2 + c ** 2) - c
        loss = torch.mean(loss)

    # 20.4.14. Backpropagate on the online student model (`unet`)
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    # ---------------------------------------------------------------------------

    return loss
# ---------------------------------------------------------------------
