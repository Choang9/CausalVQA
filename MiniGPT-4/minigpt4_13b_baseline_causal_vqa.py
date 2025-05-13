import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pandas as pd

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import *
# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="")
parser.add_argument('--cfg-path', help='')
parser.add_argument('--options', nargs="+",help='')
parser.add_argument('--gpu-id', default=0, help='')
args = parser.parse_args(" --cfg-path /eval_configs/minigpt4_eval.yaml".split())
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')

class MiniGPT4Chat:
    
    def __init__(self, model, vis_processor, device='cuda'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.conv, self.img_list = None, None
        self.reset_history()
        
    def ask(self, text):
        if len(self.conv.messages) > 0 and self.conv.messages[-1][0] == self.conv.roles[0] \
                and self.conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            self.conv.messages[-1][1] = ' '.join([self.conv.messages[-1][1], text])
        else:
            self.conv.append_message(self.conv.roles[0], text)

    def answer(self, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=4000):
        self.conv.append_message(self.conv.roles[1], None)
        embs = self.get_context_emb(self.img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True if num_beams==1 else False,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        self.conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        image_emb, _ = self.model.encode_img(image)
        self.img_list.append(image_emb)
        self.conv.append_message(self.conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        return msg

    def get_context_emb(self, img_list):
        prompt = self.conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    
    def reset_history(self):
        self.conv = Conversation(
            system="Give the following image: <Img>ImageContent</Img>. "
                   "You will be able to see the image once I provide it to you. Please answer my questions.",
            roles=("Human", "Assistant"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        )
        self.img_list = []

minigpt4 = MiniGPT4Chat(model, vis_processor)
num_beams = 1
temperature = 0.1
max_new_tokens = 600

# TODO: Determine whether you are running Experiment A or Experiment B here
EXPERIMENT = "A"

if EXPERIMENT == "A":
    file_name = "minigpt4_13b_baseline_expA.csv"
else:
    file_name = "minigpt4_13b_baseline_expB.csv"

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
    minigpt4.reset_history()

    # TODO: Replace with actual path to your test/images directory here
    path_to_test_images = "path/to/test/images"
    image_path = path_to_test_images + row['file_name']

    minigpt4.upload_img(image_path)

    type = row['type']

    if EXPERIMENT == "A":
        query = row['question'] + " Answer in 1-2 sentences."
    else:
        if type == "association":
            query = row['question'] + " Answer in 1-2 sentences."
        else:
            query = template.format(row['question'])

    minigpt4.ask(query)
    response, _ = minigpt4.answer(
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

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
