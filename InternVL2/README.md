# InternVL2

Follow the following steps to set up the enviromment and run the inference script for InternVL2. 

## Getting Started

1. Create a dirctory and name it ``InternVL2``

2. Go to the ``InternVL2`` directory
   
``cd InternVL2``

3. Download the ``internvl2.yml`` file and move it to the ``InternVL2`` directory
4. Create and activate the environment

``conda env create -f internvl2.yml``

``conda activate internvl2``

## Inference

1. Download the ``internvl2_76b_baseline_causal_vqa.py`` script and move it to ``InternVL2``

2. Open the ``internvl2_76b_baseline_causal_vqa.py`` and modify the code under 3 TODOs in this script

- Line 138: Determine whether you are running Experiment A or Experiment B
- Line 151: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 206: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``InternVL2`` and run the code there

``python internvl2_76b_baseline_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and CogVLM2's response to these questions. There will be two CSV files if you run both experiments A and B.

## Minimum configuration

We used 2 NVIDIA A40 GPUs with 45GB each to run this model.
