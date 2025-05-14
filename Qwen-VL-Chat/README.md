# Qwen-VL-Chat

Follow the following steps to set up the enviromment and run the inference and fine-tuning scripts for Qwen-VL-Chat. 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 45GB to run and fine-tune this model.

## Getting Started

1. Clone the GitHub repo

``git clone https://github.com/QwenLM/Qwen-VL.git``

2. Go to the ``Qwen-VL`` directory
   
``cd Qwen-VL``

3. Download the ``qwenvl.yml`` file and move it to the ``Qwen-VL`` directory

4. Create and activate the environment

``conda env create -f qwenvl.yml``

``conda activate qwenvl``

## Inference (Baseline)

1. Download the ``qwenvlchat_baseline_causal_vqa.py`` script and move it to ``Qwen-VL``

2. Open ``qwenvlchat_baseline_causal_vqa.py`` and modify the code under 3 TODOs in this script

- Line 14: Determine whether you are running Experiment A or Experiment B
- Line 27: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 82: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``Qwen-VL`` and run the code there

``python qwenvlchat_baseline_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and Qwen-VL-Chat's responses to these questions. There will be two CSV files if you run both experiments A and B.

## Fine-tuning Data Preparation

1. Download the ``qwenvlchat_create_metadata.py`` script and move it to ``Qwen-VL``

2. Open ``qwenvlchat_create_metadata.py`` and modify the code under 2 TODOs in this script

- Line 7: Replace with actual path to your train/metadata.csv file (downloaded from our dataset)
- Line 18: Replace with actual path to your train/images directory (downloaded from our dataset)

This will create ``qwenvlchat_metadata.json``, which can be used for fine-tuning.

## Fine-tuning

1. Download the ``qwenvlchat_finetune.py`` script and move it to ``Qwen-VL/finetune``

2. Open ``qwenvlchat_finetune.py`` and modify the code under 1 TODO in this script

- Line 10: Replace with the path to the file ``qwenvlchat_metadata.json`` created in the previous section

3. Save the code, go to ``Qwen-VL/finetune`` and run the code there

``python qwenvlchat_finetune.py``

This will then create a model checkpoint in ``Qwen-VL/finetune/checkpoints/finetuned_qwenvlchat``. You can then use it to do inference shown in the next section.

Alternatively, if you want to skip the fine-tuning part, you are welcome to use our fine-tuned model checkpoint [here](https://drive.google.com/drive/folders/1VZVwCqkZ2GndYxjyZCnVta_ZOfLXldA5?usp=sharing).

## Inference (Fine-tuned)

1. Download the ``qwenvlchat_finetuned_causal_vqa.py`` script and move it to ``Qwen-VL``

2. Open ``qwenvlchat_finetuned_causal_vqa.py`` and modify the code under 4 TODOs in this script

- Line 14: Replace with the path to your fine-tuned model's checkpoint or you can use our fine-tuned model checkpoint provided from the previous section
- Line 20: Determine whether you are running Experiment A or Experiment B
- Line 33: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 88: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``Qwen-VL`` and run the code there

``python qwenvlchat_finetuned_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and fine-tuned Qwen-VL-Chat's responses to these questions. There will be two CSV files if you run both experiments A and B.
