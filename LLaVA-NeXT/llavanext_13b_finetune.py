import os
os.environ["MASTER_PORT"] = "29501"

import subprocess

command = [
    "deepspeed",
    "--include=localhost:0",  # Explicitly specify GPUs
    "--master_port", "29501",
    "llava/train/train_mem.py",
    "--deepspeed", "scripts/zero3.json",
    "--lora_enable", "True",
    "--lora_r", "64",
    "--lora_alpha", "128",
    "--mm_projector_lr", "5e-5",
    "--model_name_or_path", "llava-hf/llava-v1.6-vicuna-13b-hf",
    "--version", "v1",
    "--data_path", "/path/to/llavanext_13b_metadata.json", # TODO: Replace with the path to this file 
    "--image_folder", "path/to/train/images", # TODO: Replace with actual path to your train/images directory here
    "--vision_tower", "openai/clip-vit-large-patch14-336",
    "--mm_projector_type", "mlp2x_gelu",
    "--mm_vision_select_layer", "-2",
    "--mm_use_im_start_end", "False",
    "--mm_use_im_patch_token", "False",
    "--image_aspect_ratio", "pad",
    "--group_by_modality_length", "True",
    "--bf16", "True",
    "--output_dir", "checkpoints/finetuned_llavanext_13b",
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "8",
    "--per_device_eval_batch_size", "4",
    "--gradient_accumulation_steps", "2",
    "--evaluation_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "1000",
    "--save_total_limit", "1",
    "--learning_rate", "1e-4",
    "--weight_decay", "0.01",
    "--warmup_ratio", "0.1",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--tf32", "True",
    "--model_max_length", "2048",
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", "4",
    "--lazy_preprocess", "True",
    "--report_to", "wandb"
]

subprocess.run(command, check=True)
