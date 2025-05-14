# LLaVA-RLHF

Follow the following steps to set up the enviromment and run the inference script for LLaVA-RLHF. 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 45GB to run this model.

## Getting Started

1. Clone the GitHub repo

``git clone https://github.com/haotian-liu/LLaVA.git``

2. Go to the ``LLaVA`` directory
   
``cd LLaVA``

3. Download the ``llava_rlhf.yml`` file and move it to the ``LLaVA`` directory

4. Create and activate the environment

``conda env create -f llava_rlhf.yml``

``conda activate llava_rlhf``

5. Download the ``download_from_hf.py`` script and move it to the ``LLaVA`` directory

6. Download the [LLaVA-RLHF model](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) from Hugging Face by running the ``download_from_hf.py`` script

``python download_from_hf.py``

## Inference

1. Download the ``llava_rlhf_13b_baseline_causal_vqa.py`` script and move it to ``LLaVA``

2. Open ``llava_rlhf_13b_baseline_causal_vqa.py`` and modify the code under 4 TODOs in this script

- Lines 21 and 22: Replace model_path and lora_path with the path to your sft_model and rlhf_lora_adapter_model directories
- Line 40: Determine whether you are running Experiment A or Experiment B
- Line 53: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 109: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``LLaVA`` and run the code there

``python llava_rlhf_13b_baseline_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and LLaVA-RLHF's responses to these questions. There will be two CSV files if you run both experiments A and B.
