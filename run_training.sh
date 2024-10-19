ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))

accelerate launch --config_file $ACCELERATE_CONFIG --num_processes=1 --main_process_port $PORT main.py \
    --pretrained_teacher_model="sd-legacy/stable-diffusion-v1-5" \
    --output_dir="results" \
    --train_path="data" \
    --task_type='multi_cd' \
    --num_boundaries=4 \
    --mixed_precision=fp16 \
    --resolution=512 \
    --w=8 \
    --learning_rate=3e-5 \
    --loss_type="huber" \
    --adam_weight_decay=0.02 \
    --lr_scheduler=constant \
    --max_train_steps=200 \
    --validation_steps=50 \
    --train_batch_size=32 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=1 \
    --report_to=tensorboard \
    --seed=453645634 \
    --num_ddim_timesteps 50