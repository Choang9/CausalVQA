# TODO: replace with your OpenAI API key here
api_key = "your-api-key"

import time
import os
import base64
import requests
import re
import json
import csv

# TODO: Replace with actual path to your test/images directory here
directory_path = "path/to/test/images"
image_paths = []

for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)
    if os.path.isfile(file_path):
        image_paths.append(file_path)

# TODO: Determine the VLM you are evaluating here
VLM = "cogvlm2"

file_name = f"gpt4o_eval_expB_{VLM}.csv"

open(file_name, 'w').close()
file = open(file_name, 'a')
file.write("file_name,type,graph_rating,response_rating" + '\n')
file.close()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }

annotation_file = "captions_val2017.json"

with open(annotation_file, "r") as f:
    annotations = json.load(f)["annotations"]

def get_captions(image_id):
    image_caption_pairs = {}
    for element in annotations:
        if element["image_id"] == image_id:
            if element["image_id"] not in image_caption_pairs:
                image_caption_pairs[element["image_id"]] = element['caption']
            else:
                image_caption_pairs[element["image_id"]] += " "
                image_caption_pairs[element["image_id"]] += element['caption']
    return image_caption_pairs[image_id]

# TODO: Replace with the path to your VLM's response CSV file here
response_file = '/path/to/vlm/response/file.csv'

def get_reponses(file_name):
    formatted_entries = []

    with open(response_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader, 1):
            if row['file_name'] == file_name:
                entry = f'{len(formatted_entries)+1}. {row["type"].capitalize()} Question: "{row["prompt"]}" Candidate VLM\'s Response: "{row["response"]}"'
                formatted_entries.append(entry)

    return "\n".join(formatted_entries)

template = '''You are a vision-language model (VLM) evaluating a candidate VLM’s responses to a visual causal reasoning task based on Judea Pearl’s Ladder of Causation: Level 1 (Association) identifies observable scene patterns; Level 2 (Intervention) predicts outcomes of a specific change, considering temporal/environmental factors; Level 3 (Counterfactual) reasons about alternate scenarios if a past condition differed, requiring multi-step causal logic.

You will be provided with an image, 5 captions describing the image, 9 questions about the image (3 per Ladder level, with Level 3 questions emphasizing abstract reasoning), and the candidate VLM’s 9 responses. For Levels 2 and 3, each response is expected to include a lightweight causal graph (e.g., A → B or A → B → C) to explain the reasoning. Your task is to evaluate each response by assigning two separate scores from 0 to 10: one for the causal graph (if applicable) and one for the text response, using the image, captions, common-sense reasoning, and, for Levels 2 and 3, the causal graph to judge accuracy, relevance, and logical consistency. No ground truth answers are provided, but you must rely on your understanding of the scene and causal principles to make informed judgments.

Instructions for Evaluation:

Scoring Criteria:
- Causal Graph (Levels 2 and 3 only):
  - [8–10]: Fully correct, relevant, with required links (Level 2: ≥1, e.g., Rain → Wet Path; Level 3: ≥2, e.g., Midnight → Dark Sky → Stars).
  - [6–7]: Partially correct, misses minor links or has small inaccuracies (e.g., Level 3: Midnight → Stars, omitting Dark Sky).
  - [3–5]: Incorrect, with illogical or irrelevant links (e.g., Rain → Sun).
  - [1–2]: Provided but severely flawed (e.g., wrong direction, no causal logic).
  - [0]: Missing or not provided.

- Text Response (All Levels):
  - [8–10]: Fully correct, addresses question precisely, aligns with graph for Levels 2 and 3 (e.g., Level 3: “Stars visible” with Midnight → Dark Sky → Stars).
  - [6–7]: Partially correct, addresses question but misses details or minor causal steps (e.g., Level 2: “Path wet” without slipperiness).
  - [3–5]: Incorrect, misinterprets question or contradicts image/captions (e.g., Level 1: “Grass is blue” when green).
  - [1–2]: Provided but severely flawed (e.g., irrelevant to question).
  - [0]: Missing or not provided.

- Deductions:
  - Level 1: Deduct 1–3 points for factual errors (e.g., wrong color, position).
  - Level 2: Deduct 2–4 points if graph lacks ≥1 link or text omits environmental/temporal shifts (e.g., no mention of rain’s effect).
  - Level 3: Deduct 3–5 points if graph lacks ≥2 links (e.g., Snow → Ground White, missing Cold → Icicles) or text omits multi-step reasoning/shifts.
  - All Levels: Deduct 1–2 points for overgeneralization (e.g., vague “things change”) or ignoring image/captions.

Use the Image and Captions:
- Ensure responses are grounded in the image and captions for Level 1 questions. For Levels 2 and 3, expect reasonable speculation about changes (e.g., weather, time) that align with common-sense dynamics.

Evaluation Guidelines:
- Level 1 (Association): Score text for accuracy against image/captions (e.g., “What color is the grass?” → “Green” if green). No graphs are expected. Set graph score to [0]. 
- Level 2 (Intervention): Score text for plausible outcome (e.g., “If rain starts, path becomes wet”) and graph for ≥1 logical link (e.g., Rain → Wet Path). Ensure text and graph align (e.g., text mentions wetness, graph includes it). Check for temporal/environmental shifts (e.g., weather change).
- Level 3 (Counterfactual): Score text for plausible alternate scenario (e.g., “If midnight, stars visible”) and graph for ≥2 logical links (e.g., Midnight → Dark Sky → Stars). Ensure text and graph align and reflect multi-step reasoning (e.g., temporal shift to night). Deduct heavily for missing shifts or single-link graphs.

Output Format Instructions:
- Output your evaluation in exactly this format, with no additional text, line breaks, spaces, explanations, or deviations:
    [Graph Rating for Response 1][Text Rating for Response 1][Graph Rating for Response 2][Text Rating for Response 2][Graph Rating for Response 3][Text Rating for Response 3][Graph Rating for Response 4][Text Rating for Response 4][Graph Rating for Response 5][Text Rating for Response 5][Graph Rating for Response 6][Text Rating for Response 6][Graph Rating for Response 7][Text Rating for Response 7][Graph Rating for Response 8][Text Rating for Response 8][Graph Rating for Response 9][Text Rating for Response 9]
- Each [Graph Rating] and [Text Rating] is a number from 0 to 10 in square brackets (e.g., [7]).
- For Level 1, use [0] for the graph score.
- Do not include explanations, extra spaces, line breaks, headings, or any text outside this format.

Example Output:
[0][8][0][7][0][9][6][8][8][9][7][6][9][9][5][5][8][8]

Provided Information:
5 Captions: {}

9 pairs of Questions and Candidate VLM’s Responses:
{}

Task:
Evaluate the candidate VLM’s responses to the 9 questions using the provided image, captions, and instructions above. Assign separate scores for the causal graph and text response for each, and output your evaluation in the following format and nothing else:

[Graph Rating for Response 1][Text Rating for Response 1][Graph Rating for Response 2][Text Rating for Response 2][Graph Rating for Response 3][Text Rating for Response 3][Graph Rating for Response 4][Text Rating for Response 4][Graph Rating for Response 5][Text Rating for Response 5][Graph Rating for Response 6][Text Rating for Response 6][Graph Rating for Response 7][Text Rating for Response 7][Graph Rating for Response 8][Text Rating for Response 8][Graph Rating for Response 9][Text Rating for Response 9]
'''

count = 1
for image_path in image_paths:
    print("Image:", count)
    print(image_path)
    base64_image = encode_image(image_path)
    messages = []

    base_name = os.path.basename(image_path)
    image_id = int(os.path.splitext(base_name)[0])
    captions = get_captions(image_id)

    vlm_responses = get_reponses(image_path)

    input_text = template.format(captions, vlm_responses)

    prompt_history = {"role": "user", "content": [{"type": "text", "text": input_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
    messages.append(prompt_history)

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 600
        }

    response = None
    while response is None:
      try:
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        # Check if the response was successful
        if res.status_code != 200:
            print("Error:", res.status_code, res.text)
            time.sleep(10)
            continue
        
        # Attempt to parse JSON
        try:
            response_json = res.json()
        except Exception as json_err:
            print("JSON decode error:", json_err)
            print("Raw response:", res.text)
            time.sleep(10)
            continue
        
        response = response_json  # Only set response if all the above succeed

      except Exception as e:
        print("Request failed:", e)
        print('Retrying...')
        time.sleep(10)
        continue
    
    content = response['choices'][0]['message']['content']
    print(content)
    if content is not None:
        matches = re.findall(r'\[(.*?)\]', content)
        if len(matches) % 2 == 0:
            qa_count = 1
            for i in range(0, len(matches), 2):
                graph_rating = matches[i]
                response_rating = matches[i+1]
                if qa_count == 1 or qa_count == 2 or qa_count == 3:
                    type = "association"
                elif qa_count == 4 or qa_count == 5 or qa_count == 6:
                    type = "intervention"
                else:
                    type = "counterfactual"

                file = open(file_name, 'a')
                file.write(image_path)
                file.write(",")
                file.write(type)
                file.write(",")
                file.write(graph_rating)
                file.write(",")
                file.write(response_rating + '\n')
                file.close()
                qa_count += 1 

    response_history = {"role": "assistant", "content": [{"type": "text", "text": content}]}
    messages.append(response_history)
    count += 1
