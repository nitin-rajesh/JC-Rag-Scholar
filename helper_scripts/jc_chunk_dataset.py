import json
import os

input_file = "julius_caesar_scenes_v2.jsonl"
output_dir = "chunks_output"

# Create folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r") as f:
    dataset = [json.loads(line) for line in f]

# data is a dict: key → content
for data in dataset:
    prefix = ""
    segment = (f"Act:{data["act"]} \nScene:{data["scene"]} \n\n{data["full_text"]}")

    filename = f"act-{data["act"]}_scene-{data["scene"]}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as out:
        out.write(segment)

print("Done. Wrote", len(data), "files.")
