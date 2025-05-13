from PIL import Image
import torch
from peft import PeftModel
import re
import pandas as pd

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.utils import disable_torch_init

# TODO: Replace model_path and lora_path with the path to your sft_model and rlhf_lora_adapter_model directories
model_path = "/path/to/.cache/huggingface/hub/models--zhiqings--LLaVA-RLHF-13b-v1.5-336/snapshots/09a4f65d09051f0ec2d030d4787c22d3ddce1de3/sft_model"
lora_path = "/path/to/.cache/huggingface/hub/models--zhiqings--LLaVA-RLHF-13b-v1.5-336/snapshots/09a4f65d09051f0ec2d030d4787c22d3ddce1de3/rlhf_lora_adapter_model"
model_name = "LLaVA-RLHF-13b-v1.5-224"

disable_torch_init()
load_bf16 = True

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name, 
)

model = PeftModel.from_pretrained(
    model,
    lora_path,
)

# TODO: Determine whether you are running Experiment A or Experiment B here
EXPERIMENT = "A"

if EXPERIMENT == "A":
    file_name = "cogvlm2_baseline_expA.csv"
else:
    file_name = "cogvlm2_baseline_expB.csv"

open(file_name, 'w').close()
file = open(file_name, 'a')
file.write("file_name,type,prompt,response" + '\n')
file.close()

# TODO: Replace with actual path to your test/metadata.csv file here
test_set = 'path/to/test/metadata.csv' 

df = pd.read_csv(test_set)

# Prompt template used for Experiment B
template = '''You are a vision-language model (VLM) tasked with answering a provided question about an image in a way that demonstrates causal reasoning, based on Judea Pearl’s framework. You are encouraged to construct a lightweight causal graph to represent the chain of cause-and-effect relationships relevant to the answer, using the number of nodes and arrows needed to accurately capture the causal logic (e.g., 2 nodes like A → B, 3 nodes like A → B → C, or more if appropriate), tailored to the complexity of the reasoning. However, it is reasonable to provide only a text response without a causal graph if you find it sufficient to explain the causal reasoning. If you include a graph, provide it first, followed by a concise text response that explains the reasoning based on the graph. If you omit the graph, provide only the text response, ensuring it still reflects causal reasoning.

Instructions:
- Base the response on the image and question: Ensure the answer (and graph, if provided) is grounded in the image’s content and relevant to the question.
- Optional causal graph: If you choose to include a graph, follow these steps:
    1. Identify the key cause-and-effect relationships implied by the question (e.g., ‘Rain causes wet ground, which darkens the scene’).
    2. Format these relationships as a graph using ‘→’ to connect nodes (e.g., Rain → Wet Ground → Darkened Scene).
    3. Use the fewest nodes needed for clarity, but include additional nodes if the causal chain requires multiple steps (e.g., 2, 3, or more nodes).
- Provide a concise answer: Write 1–2 sentences that explain the causal reasoning, referencing the graph’s steps if a graph is provided, or directly addressing the question’s causal logic if no graph is included.
- Avoid overly simplistic graphs: Ensure the response (and graph, if provided) captures specific, relevant causal relationships (e.g., avoid vague terms like ‘Change → Outcome’ unless fully justified).
- Format the output exactly as follows:

    If including a graph:
    Causal Graph: [Your causal graph, e.g., A → B or A → B → C → D]
    Answer: [Your answer, explaining the reasoning based on the graph]

    If omitting the graph:
    Answer: [Your answer, explaining the causal reasoning]

Example:

Question: How would the scene change if a sudden rainstorm began in a sunny park?
Causal Graph: Rainstorm → Wet Ground → Darkened Atmosphere
Answer: The rainstorm wets the ground, creating a darker, moodier atmosphere as rain falls.

Question: What would the park look like if it had been photographed at night instead of midday?
Causal Graph: Night → Reduced Natural Light
Answer: At night, the lack of natural light darkens the park, with only artificial lights visible.

Question: How would the scene differ if a month-long drought had preceded this day?
Causal Graph: Drought → Dry Soil → Wilted Vegetation → Sparse Leaves
Answer: The drought dries the soil, leading to wilted, brown grass and sparse tree leaves.

Provided Information:

Question: {}

Task:
Answer the provided question based on the image, optionally providing a lightweight causal graph to represent the causal reasoning followed by a text response, or providing only a text response that explains the causal reasoning. Follow the instructions above and output in the appropriate format:
- If including a graph:
    Causal Graph: [Your causal graph]
    Answer: [Your answer]
-If omitting the graph:
    Answer: [Your answer]'''

count = 1
for index, row in df.iterrows():
    print("Row:", count)
    conv = conv_templates["llava_v0"].copy()

    # TODO: Replace with actual path to your test/images directory here
    path_to_test_images = "path/to/test/images"
    image_path = path_to_test_images + row['file_name']

    image = Image.open(image_path).convert('RGB')

    type = row['type']

    if EXPERIMENT == "A":
        query = row['question'] + " Answer in 1-2 sentences."
    else:
        if type == "association":
            query = row['question'] + " Answer in 1-2 sentences."
        else:
            query = template.format(row['question'])

    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    conv.append_message(conv.roles[0], query)
    prompt = conv.get_prompt()

    image_tensor = process_images([image], image_processor, model.config).to("cuda", dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.2,
            top_p=0.6,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=512,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    file = open(file_name, 'a')
    file.write(image_path)
    file.write(",")
    file.write(type)
    file.write(",")
    file.write('"' + query + '"')
    file.write(",")
    file.write('"' + outputs + '"' + '\n')
    print(outputs)
    file.close() 

    conv.append_message(conv.roles[1], outputs)
    count += 1
