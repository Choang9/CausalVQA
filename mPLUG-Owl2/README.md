# mPLUG-Owl2

Follow the following steps to set up the enviromment and run the inference and fine-tuning scripts for mPLUG-Owl2. 

## Minimum configuration

We used 1 NVIDIA A40 GPU with 45GB to run and fine-tune this model.

## Getting Started

1. Clone the GitHub repo

``git clone https://github.com/X-PLUG/mPLUG-Owl.git``

2. Go to the ``mPLUG-Owl/mPLUG-Owl2`` directory
   
``cd mPLUG-Owl/mPLUG-Owl2``

3. Download the ``mplug_owl2.yml`` file and move it to the ``mPLUG-Owl/mPLUG-Owl2`` directory

4. Create and activate the environment

``conda env create -f mplug_owl2.yml``

``conda activate mplug_owl2``

## Inference (Baseline)

1. Download the ``mplugowl2_baseline_causal_vqa.py`` script and move it to ``mPLUG-Owl/mPLUG-Owl2``

2. Open ``mplugowl2_baseline_causal_vqa.py`` and modify the code under 3 TODOs in this script

- Line 17: Determine whether you are running Experiment A or Experiment B
- Line 30: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 88: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``mPLUG-Owl/mPLUG-Owl2`` and run the code there

``python mplugowl2_baseline_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and mPLUG-Owl2's responses to these questions. There will be two CSV files if you run both experiments A and B.

## Fine-tuning

1. Download the ``mplugowl2_finetune.py`` script and move it to ``mPLUG-Owl/mPLUG-Owl2``

2. Open ``mplugowl2_finetune.py`` and modify the code under 2 TODOs in this script

- Line 5: Replace with the path to the file ``mplugowl2_metadata.json``.
- Line 21: Replace with actual path to your train/images directory (downloaded from our dataset)

3. Save the code, go to ``mPLUG-Owl/mPLUG-Owl2`` and run the code there

``python mplugowl2_finetune.py``

This will then create a model checkpoint in ``mPLUG-Owl/mPLUG-Owl2/checkpoints/finetuned_mplugowl2``. You can then use it to do inference shown in the next section.

Alternatively, if you want to skip the fine-tuning part, you are welcome to use our fine-tuned model checkpoint [here](https://drive.google.com/drive/folders/1-ylMnkeCrDl2mSYmMc8DMKsi4so_IdNT?usp=sharing).

## Inference (Fine-tuned)

1. Download the ``mplugowl2_finetuned_causal_vqa.py`` script and move it to ``mPLUG-Owl/mPLUG-Owl2``

2. Open ``mplugowl2_finetuned_causal_vqa.py`` and modify the code under 4 TODOs in this script

- Line 16: Replace with the path to your fine-tuned model's checkpoint or you can use our fine-tuned model checkpoint provided from the previous section
- Line 27: Determine whether you are running Experiment A or Experiment B
- Line 40: Replace with actual path to your test/metadata.csv file (downloaded from our dataset)
- Line 103: Replace with actual path to your test/images directory (downloaded from our dataset)

3. Save the code, go to ``mPLUG-Owl/mPLUG-Owl2`` and run the code there

``python mplugowl2_finetuned_causal_vqa.py``

This will then create a CSV file that contains the image file paths, the types of questions (Association, Intervention, and Counterfactual), the questions, and fine-tuned mPLUG-Owl2's responses to these questions. There will be two CSV files if you run both experiments A and B.
