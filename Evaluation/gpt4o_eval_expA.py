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

file_name = f"gpt4o_eval_expA_{VLM}.csv"

open(file_name, 'w').close()
file = open(file_name, 'a')
file.write("file_name,type,rating" + '\n')
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

You will be provided with an image, 5 captions describing the image, 9 questions about the image (3 per Ladder level, with Level 3 questions emphasizing abstract reasoning), and the candidate VLM’s 9 responses. Your task is to evaluate the candidate VLM’s responses by assigning each a score from 0 to 10 (0 being the worst, 10 being the best). Use the image, captions, and common-sense reasoning to evaluate accuracy, relevance, and logical consistency.

Instructions for Evaluation:

Scoring Criteria:
- Assign a score from 0 to 10 based on how well the candidate VLM’s response answers the question, considering its accuracy, relevance, and logical consistency with the image, captions, and common-sense reasoning:
  - [8–10]: Fully correct, addresses question precisely.
  - [6–7]: Partially correct, addresses question but misses details (e.g., Level 2: “Path wet” without slipperiness).
  - [3–5]: Incorrect, misinterprets question or contradicts image/captions (e.g., Level 1: “Grass is blue” when green).
  - [1–2]: Provided but severely flawed (e.g., irrelevant to the question).
  - [0]: Missing or not provided.

- Deductions:
  - Level 1: Deduct 1–3 points for factual errors (e.g., wrong color, position).
  - Level 2: Deduct 2–4 points if the response omits environmental/temporal shifts (e.g., no mention of rain’s effect).
  - Level 3: Deduct 3–5 points if the response omits multi-step reasoning/shifts.
  - All Levels: Deduct 1–2 points for overgeneralization (e.g., vague “things change”) or ignoring image/captions.

Use the Image and Captions:
- Ensure responses are grounded in the image and captions for Level 1 questions. For Levels 2 and 3, expect reasonable speculation about changes (e.g., weather, time) that align with common-sense dynamics.

Evaluation Guidelines:
- Level 1 (Association): Check if the response accurately describes observable features (e.g., colors, positions) based on the image and captions. Deduct points for factual errors.
- Level 2 (Intervention): Verify that the response predicts a plausible outcome of the change (e.g., ‘If rain starts, the path becomes wet’). Deduct points if the response lacks causal reasoning or fails to simulate environmental/temporal shifts. 
- Level 3 (Counterfactual): Confirm that the response reconstructs a plausible alternate scenario (e.g., ‘If it were midnight, the stars would be visible’). Deduct points if the response lacks multi-step reasoning or fails to simulate environmental/temporal shifts.

Output Format Instructions:
- Output your evaluation in exactly this format, with no additional text, line breaks, spaces, explanations, or deviations:
    [Rating for Response 1][Rating for Response 2][Rating for Response 3][Rating for Response 4][Rating for Response 5][Rating for Response 6][Rating for Response 7][Rating for Response 8][Rating for Response 9]
- Each [Rating] is a number from 0 to 10 in square brackets (e.g., [7]).
- Do not include explanations, extra spaces, line breaks, headings, or any text outside this format.

Example Output:
[8][7][9][8][9][6][9][5][8]

Provided Information:
5 Captions: {}

9 pairs of Questions and Candidate VLM’s Responses:
{}

Task:
Evaluate the candidate VLM’s responses to the 9 questions using the provided image, captions, and instructions above. Output your evaluation in the following format and nothing else:

[Rating for Response 1][Rating for Response 2][Rating for Response 3][Rating for Response 4][Rating for Response 5][Rating for Response 6][Rating for Response 7][Rating for Response 8][Rating for Response 9]
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
        if len(matches) % 1 == 0:
            qa_count = 1
            for i in range(0, len(matches), 1):
                rating = matches[i]
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
                file.write(rating + '\n')
                file.close()
                qa_count += 1 

    response_history = {"role": "assistant", "content": [{"type": "text", "text": content}]}
    messages.append(response_history)
    count += 1
