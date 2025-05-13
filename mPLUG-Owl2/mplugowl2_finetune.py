import subprocess

# Define paths
LOAD = "MAGAer13/mplug-owl2-llama2-7b"
DATA_FILE = "/path/to/mplugowl2_metadata.json" # TODO: Replace with the path to this file 
DEEPSPEED_CONFIG = "scripts/zero3.json"
OUTPUT_DIR = "checkpoints/finetuned_mplugowl2"

command = [
    "deepspeed",
    "--include=localhost:0",  # Explicitly specify GPUs
    "mplug_owl2/train/train_mem.py",
    "--lora_enable", "True",
    "--lora_r", "128",
    "--lora_alpha", "256",
    "--visual_abstractor_lr", "2e-5",
    "--deepspeed", DEEPSPEED_CONFIG,
    "--model_name_or_path", LOAD,
    "--version", "v1",
    "--data_path", DATA_FILE,
    "--image_folder", "path/to/train/images", # TODO: Replace with actual path to your train/images directory here
    "--image_aspect_ratio", "pad",
    "--group_by_modality_length", "True",
    "--bf16", "True",
    "--output_dir", OUTPUT_DIR,
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "2",
    "--per_device_eval_batch_size", "4",
    "--gradient_accumulation_steps", "8",
    "--evaluation_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "10",
    "--save_total_limit", "1",
    "--learning_rate", "1e-4",
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--tf32", "True",
    "--model_max_length", "2048",
    "--gradient_checkpointing", "True",
    "--tune_visual_abstractor", "True",
    "--freeze_vision_model", "True",
    "--dataloader_num_workers", "4",
    "--lazy_preprocess", "True",
    "--report_to", "wandb",
]

subprocess.run(command, check=True)
