# MiniGPT-4

Follow the following steps to set up the enviromment and run the inference and fine-tuning scripts for MiniGPT-4. 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 45GB to run and fine-tune this model.

## Getting Started

1. Clone the GitHub repo

``git clone https://github.com/Vision-CAIR/MiniGPT-4.git``

2. Go to the ``MiniGPT-4`` directory
   
``cd MiniGPT-4``

3. Download the ``minigpt4.yml`` file and move it to the ``MiniGPT-4`` directory

4. Create and activate the environment

``conda env create -f minigpt4.yml``

``conda activate minigpt4``

## Inference (Baseline)

1. Download the pre-trained checkpoint of MiniGPT-4 13B [here](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view) and move it to ``MiniGPT-4``

2. Set the path to the pre-trained checkpoint in the evaluation config file in ``eval_configs/minigpt4_eval.yaml`` at Line 8

3. Download the ``minigpt4_causal_vqa.py`` script and move it to ``MiniGPT-4``

4. Open ``minigpt4_causal_vqa.py`` and modify the code under 3 TODOs in this script

- Line 151: Determine whether you are running Experiment A or Experiment B
- Line 164: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 220: Replace with actual path to your test/images directory (downloaded from our dataset)

5. Save the code, go to ``MiniGPT-4`` and run the code there

``python minigpt4_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and MiniGPT-4's responses to these questions. There will be two CSV files if you run both experiments A and B.

## Fine-tuning Data Preparation

1. Download the ``minigpt4_metadata.json`` file and move it to ``MiniGPT-4``

2. Go to this directory ``MiniGPT-4/minigpt4/configs/datasets/multitask_conversation`` and delete the ``default.yaml`` file already existed there

3. Download the ``default.yaml`` file from this GitHub repo and move it to ``MiniGPT-4/minigpt4/configs/datasets/multitask_conversation`` (basically replacing this with the original ``default.yaml`` file)

4. Open ``default.yaml`` and modify the code under 2 TODOs in this script

- Line 6: Replace with actual path to your train/images directory (downloaded from our dataset)
- Line 7: Replace with the path to the file ``minigpt4_metadata.json``

5. Save the file

## Fine-tuning

1. Go to this directory ``MiniGPT-4/train_configs`` and delete the ``minigpt4_stage2_finetune.yaml`` file already existed there

2. Download the ``minigpt4_stage2_finetune.yaml`` file from this GitHub repo and move it to ``MiniGPT-4/train_configs`` (basically replacing this with the original ``minigpt4_stage2_finetune.yaml`` file)

3. Open ``minigpt4_stage2_finetune.yaml`` and modify the code under 1 TODO in this script

- Line 8: Replace with the path to the pre-trained checkpoint of MiniGPT-4 (downloaded in the first step of Baseline Inference)

4. Save the code, go to ``MiniGPT-4`` and run the following command to fine-tune the model

``torchrun --nproc-per-node 1 train.py --cfg-path train_configs/minigptv2_finetune.yaml``

This will then create a model checkpoint in ``MiniGPT-4/minigpt4/output/finetuned_minigpt4``. You can then use it to do inference shown in the next section.

Alternatively, if you want to skip the fine-tuning part, you are welcome to use our fine-tuned model checkpoint [here](https://drive.google.com/file/d/1SoYFRz4mPxwk_UzIXfhe0pMFSjXV4wN_/view?usp=sharing).

## Inference (Fine-tuned)

1. Set the path to the model checkpoint in the evaluation config file in ``eval_configs/minigpt4_eval.yaml`` at Line 8

2. Open ``minigpt4_causal_vqa.py`` and modify the code under 3 TODOs in this script

- Line 151: Determine whether you are running Experiment A or Experiment B
- Line 164: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 220: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``MiniGPT-4`` and run the code there

``python minigpt4_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and fine-tuned MiniGPT-4's responses to these questions. There will be two CSV files if you run both experiments A and B.
