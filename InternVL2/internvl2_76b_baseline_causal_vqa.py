import math
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# Load the model
path = 'OpenGVLab/InternVL2-Llama3-76B'
device_map = split_model('InternVL2-Llama3-76B')

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map=device_map).eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

generation_config = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)

# TODO: Determine whether you are running Experiment A or Experiment B here
EXPERIMENT = "A"

if EXPERIMENT == "A":
    file_name = "internvl2_baseline_expA.csv"
else:
    file_name = "internvl2_baseline_expB.csv"

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

    image = Image.open(image_path).convert('RGB')

    type = row['type']

    if EXPERIMENT == "A":
        query = row['question'] + " Answer in 1-2 sentences."
    else:
        if type == "association":
            query = row['question'] + " Answer in 1-2 sentences."
        else:
            query = template.format(row['question'])

    pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cuda()

    question = f'<image>\n{query}'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)

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
