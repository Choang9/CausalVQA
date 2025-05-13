from unsloth import FastVisionModel
import torch
import pandas as pd
from PIL import Image
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
# from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

model, tokenizer = FastVisionModel.from_pretrained(
    "Xkev/Llama-3.2V-11B-cot",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, 
    finetune_language_layers   = True, 
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,           
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3443,
    use_rslora = False,
    loftq_config = None,
)

# TODO: Replace with actual path to your train/metadata.csv file here
input_csv = 'path/to/train/metadata.csv' 
df = pd.read_csv(input_csv)

def convert_to_conversation(index, row):
    # TODO: Replace with actual path to your train/images directory here
    path_to_train_images = "path/to/train/images"
    image_path = path_to_train_images + row['file_name']
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": row["question"]},
                {"type": "image", "image": Image.open(image_path).convert('RGB')},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": row["answer"]}],
        },
    ]
    return {"messages": conversation}


pass


converted_dataset = [convert_to_conversation(index, row) for index, row in df.iterrows()]

FastVisionModel.for_training(model)  # Enable for training!

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=3000,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="finetuned_llava_cot",
        report_to="none",  # For Weights and Biases
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,
    ),
)

trainer_stats = trainer.train()

# TODO: Replace with your Hugging Face token here
login(token="your-hf-token")

model.save_pretrained("finetuned_llava_cot") # Local saving
tokenizer.save_pretrained("finetuned_llava_cot")

# TODO: Replace with your Hugging Face finetuned model
model.push_to_hub(
    "your-huggingface-finetuned-model"
)  # Online saving
tokenizer.push_to_hub(
    "your-huggingface-finetuned-model"
)  # Online saving
