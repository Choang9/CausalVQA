# LLaVA-NeXT

Follow the following steps to set up the enviromment and run the inference and fine-tuning scripts for LLaVA-NeXT. 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 45GB to run and fine-tune this model.

## Getting Started

1. Clone the GitHub repo

``git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git``

2. Go to the ``LLaVA-NeXT`` directory
   
``cd LLaVA-NeXT``

3. Download the ``llava_next.yml`` file and move it to the ``LLaVA-NeXT`` directory

4. Create and activate the environment

``conda env create -f llava_next.yml``

``conda activate llava_next``

## Inference (Baseline)

1. Download the ``llavanext_13b_baseline_causal_vqa.py`` script and move it to ``LLaVA-NeXT``

2. Open ``llavanext_13b_baseline_causal_vqa.py`` and modify the code under 3 TODOs in this script

- Line 12: Determine whether you are running Experiment A or Experiment B
- Line 25: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 81: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``LLaVA-NeXT`` and run the code there

``python llavanext_13b_baseline_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and LLaVA-NeXT's responses to these questions. There will be two CSV files if you run both experiments A and B.

## Fine-tuning

1. Download the ``llavanext_13b_finetune.py`` script and move it to ``LLaVA-NeXT``

2. Open ``llavanext_13b_finetune.py`` and modify the code under 2 TODOs in this script

- Line 18: Replace with the path to the file ``llavanext_13b_metadata.json``.
- Line 19: Replace with actual path to your train/images directory (downloaded from our dataset)

3. Save the code, go to ``LLaVA-NeXT`` and run the code there

``python llavanext_13b_finetune.py``

This will then create a model checkpoint in ``LLaVA-NeXT/checkpoints/finetuned_llavanext_13b``. You can then use it to do inference shown in the next section.

Alternatively, if you want to skip the fine-tuning part, you are welcome to use our fine-tuned model checkpoint [here](https://drive.google.com/drive/folders/1etsh_oRnGIvrFv3_Lz2aZ0pZ5kRoeFfM?usp=sharing).

**Note:** If the fine-tuning script throws an error, try running it using the environment ``llava_rlhf`` (Check the LLaVA-RLHF directory for instructions on how to create this environment).

## Inference (Fine-tuned)

1. Download the ``llavanext_13b_finetuned_causal_vqa.py`` script and move it to ``LLaVA-NeXT``

2. Open ``llavanext_13b_finetuned_causal_vqa.py`` and modify the code under 4 TODOs in this script

- Line 15: Replace with the path to your fine-tuned model's checkpoint or you can use our fine-tuned model checkpoint provided from the previous section
- Line 21: Determine whether you are running Experiment A or Experiment B
- Line 34: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 89: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``LLaVA-NeXT`` and run the code there

``python llavanext_13b_finetuned_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and fine-tuned LLaVA-NeXT's responses to these questions. There will be two CSV files if you run both experiments A and B.
