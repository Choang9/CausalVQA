# CausalVQA: A Benchmark Exposing Causal Illusions in Vision-Language Models

This repository is the official implementation of CausalVQA: A Benchmark Exposing Causal Illusions in Vision-Language Models. 

## Getting Started

Download our CausalVQA dataset [here](https://www.kaggle.com/datasets/choang19/causalvqa).

Determine which VLM you want to experiment with. Go to that VLM's directory and follow the steps in the README file there to set up the environment and run the scripts. Most directories contain these:
- A README file
- A yml file
- Inference scripts (for baseline and fine-tuned models)
- Fine-tuning script (only for these 5 models: LLaVA-CoT, LLaVA-NeXT, MiniGPT-4, mPLUG-Owl2, and Qwen-VL-Chat)
- Fine-tuned version of the model (only for these 5 models: LLaVA-CoT, LLaVA-NeXT, MiniGPT-4, mPLUG-Owl2, and Qwen-VL-Chat). This is typically a TXT file containing a Google Drive link for downloading the fine-tuned model.

## Fine-tuning

Use the fine-tuning script in each VLM's directory to do fine-tuning. More information can be found in the README file of each VLM's own directory.

Minimum configuration: We used at most 2 NVIDIA A40 GPUs with 45GB each to fine-tune these models.

## Evaluation

Use the evaluation scripts in the Evaluation directory. We used Claude 3.5 Sonnet and GPT-4o as judges to evaluate the responses of these models. More information can be found in the README file of this directory.

Warning: This requires **paid subscriptions** to use the APIs.

## Acknowledgment

Our research used these wonderful resources:
- [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet)
- [CogVLM2](https://github.com/THUDM/CogVLM2)
- [GPT-4o](https://openai.com/index/hello-gpt-4o/)
- [InternVL2](https://github.com/OpenGVLab/InternVL)
- [Kaggle](https://www.kaggle.com/)
- [LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl)
- [Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL)
