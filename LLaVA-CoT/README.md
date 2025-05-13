# LLaVA-CoT

Follow the following steps to set up the enviromment and run the inference and fine-tuning scripts for LLaVA-CoT. 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 45GB to run and fine-tune this model.

## Getting Started

1. Create a dirctory and name it ``LLaVA_CoT``

2. Go to the ``LLaVA_CoT`` directory
   
``cd LLaVA_CoT``

3. Download the ``llava_cot.yml`` file and move it to the ``LLaVA_CoT`` directory

4. Create and activate the environment

``conda env create -f llava_cot.yml``

``conda activate llava_cot``

## Inference (Baseline)

1. Download the ``llava_cot_baseline_causal_vqa.py`` script and move it to ``LLaVA_CoT``

2. Open the ``llava_cot_baseline_causal_vqa.py`` and modify the code under 3 TODOs in this script

- Line 19: Determine whether you are running Experiment A or Experiment B
- Line 32: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 88: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``LLaVA_CoT`` and run the code there

``python llava_cot_baseline_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and LLaVA-CoT's responses to these questions. There will be two CSV files if you run both experiments A and B.

## Fine-tuning

1. Log in to/Create your Hugging Face account and create a [user access token](https://huggingface.co/docs/hub/en/security-tokens). 

2. Download the ``llava_cot_finetune.py`` script and move it to ``LLaVA_CoT``

3. Open the ``llava_cot_finetune.py`` and modify the code under 4 TODOs in this script

- Line 34: Replace with actual path to your train/metadata.csv file (downloaded from our dataset)
- Line 39: Replace with actual path to your train/images directory (downloaded from our dataset)
- Line 96: Replace with your Hugging Face user access token you created in Step 1
- Lines 103 and 106: Replace with your Hugging Face fine-tuned model (you can name them like yourusername/yourmodel)

4. Save the code, go to ``LLaVA_CoT`` and run the code there

``python llava_cot_finetune.py``

This will then create a model on your Hugging Face account. You can then use it to do inference shown in the next section.

Alternatively, if you want to skip the fine-tuning part, you are welcome to use our fine-tuned model [here](https://huggingface.co/cxhoang/ft-llava-cot-causal-qa).

## Inference (Fine-tuned)

1. Download the ``llava_cot_finetuned_causal_vqa.py`` script and move it to ``LLaVA_CoT``

2. Open the ``llava_cot_finetuned_causal_vqa.py`` and modify the code under 4 TODOs in this script

- Line 15: Replace with your/our Hugging Face finetuned model's name (yourusername/yourmodel or cxhoang/ft-llava-cot-causal-qa)
- Line 20: Determine whether you are running Experiment A or Experiment B
- Line 33: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 88: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``LLaVA_CoT`` and run the code there

``python llava_cot_finetuned_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and fine-tuned LLaVA-CoT's responses to these questions. There will be two CSV files if you run both experiments A and B.
