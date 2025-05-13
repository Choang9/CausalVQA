# CogVLM2

Follow the following steps to set up the enviromment and run the fine-tuning and inference scripts for CogVLM2. 

## Getting Started

1. Clone the GitHub repo

``git clone https://github.com/THUDM/CogVLM2``

2. Go to the ``CogVLM2`` directory
   
``cd CogVLM2``

3. Download the ``cogvlm2.yml`` file and move it to the ``CogVLM2`` directory
4. Create and activate the environment

``conda env create -f cogvlm2.yml``

``conda activate cogvlm2``

## Inference

1. Download the ``cogvlm2_baseline_causal_vqa.py`` script and move it to ``CogVLM2/basic_demo``

2. Open the ``cogvlm2_baseline_causal_vqa.py`` and modify the code under 3 TODOs in this script

- Line 32: Determine whether you are running Experiment A or Experiment B
- Line 45: Replace with actual path to your test/metadata.csv file
- Line 101: Replace with actual path to your test/images directory

3. Save the code, go to ``CogVLM2/basic_demo`` and run the code there

``python cogvlm2_baseline_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the questions, and CogVLM2's response to these questions. There will be two CSV files if you run both experiments A and B.


## Minimum configuration

We used 1 NVIDIA A40 GPUs with 45GB each to run this model.
