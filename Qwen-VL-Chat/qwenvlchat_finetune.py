import os
import subprocess

# Set environment variables
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# Define paths
CURRENT_DIR = os.getcwd()
MODEL = "Qwen/Qwen-VL-Chat"  # or "Qwen/Qwen-VL"
DATA = "/path/to/qwenvlchat_metadata.json" # TODO: Replace with the path to this file 

# Construct the finetune command
command = [
    "python", "qwenvlchat_finetune.py",
    "--model_name_or_path", MODEL,
    "--data_path", DATA,
    "--bf16", "True",
    "--fix_vit", "True",
    "--output_dir", "checkpoints/finetuned_qwenvlchat",
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--gradient_accumulation_steps", "8",
    "--save_strategy", "steps",
    "--save_steps", "1000",
    "--save_total_limit", "10",
    "--learning_rate", "1e-5",
    "--weight_decay", "0.1",
    "--adam_beta2", "0.95",
    "--warmup_ratio", "0.01",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--report_to", "none",
    "--model_max_length", "2048",
    "--lazy_preprocess", "True",
    "--gradient_checkpointing",
    "--use_lora"
]

# Execute the command
subprocess.run(command, check=True)
