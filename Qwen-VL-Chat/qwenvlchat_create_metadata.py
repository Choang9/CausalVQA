import csv
import json
import os
from collections import defaultdict

# TODO: Replace with actual path to your train/metadata.csv file here
input_csv = 'path/to/train/metadata.csv' 
output_json = 'qwenvlchat_metadata.json'

data = []
image_counts = defaultdict(int)

# Read the CSV and construct the JSON structure
with open(input_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_path = row['file_name']
        question = row["question"]
        image_name = os.path.basename(image_path)
        base_id = os.path.splitext(image_name)[0]

        # Increment the count for this image
        image_counts[image_path] += 1
        image_id = f"{base_id}_{image_counts[image_path]}"

        entry = {
            "id": image_id,
            "conversations": [
                {
                    "from": "user",
                    "value": f"Picture 1: <img>{image_path}</img>\n{question}"
                },
                {
                    "from": "assistant",
                    "value": row["answer"]
                }
            ]
        }

        data.append(entry)

# Write to JSON file
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"JSON written to {output_json}")
