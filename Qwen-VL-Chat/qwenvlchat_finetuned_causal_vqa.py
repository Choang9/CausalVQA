from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
import tensorflow as tf
import pandas as pd
from peft import AutoPeftModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# TODO: Replace with the path to your finetuned model's checkpoint
# Or you can use our finetuned model's checkpoint provided in this directory
model = AutoPeftModelForCausalLM.from_pretrained(
    "/path/to/finetuned_qwenvlchat", 
    device_map="auto",
    trust_remote_code=True
).eval()

# TODO: Determine whether you are running Experiment A or Experiment B here
EXPERIMENT = "A"

if EXPERIMENT == "A":
    file_name = "qwenvlchat_finetuned_expA.csv"
else:
    file_name = "qwenvlchat_finetuned_expB.csv"

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

    # TODO: Replace with actual path to your test/images directory here
    path_to_test_images = "path/to/test/images"
    image_path = path_to_test_images + row['file_name']

    type = row['type']

    if EXPERIMENT == "A":
        query = row['question'] + " Answer in 1-2 sentences."
    else:
        if type == "association":
            query = row['question'] + " Answer in 1-2 sentences."
        else:
            query = template.format(row['question'])

    query_1 = tokenizer.from_list_format([
        {'image': str(image_path)}, # Either a local path or an url
        {'text': query},
    ])

    response, history = model.chat(tokenizer, query=query_1, history=None)

    file = open(file_name, 'a')
    file.write(image_path)
    file.write(",")
    file.write(type)
    file.write(",")
    file.write('"' + query + '"')
    file.write(",")
    file.write('"' + response + '"' + '\n')
    print(response)
    file.close()
    count += 1
