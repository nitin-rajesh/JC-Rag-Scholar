import json
import os

input_file = "jc_dataset.json"
output_dir = "chunks_output"

# Create folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r") as f:
    data = json.load(f)

# data is a dict: key → content
for key, value in data.items():
    filename = f"{key}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as out:
        out.write(key+"\n")
        out.write(value)

print("Done. Wrote", len(data), "files.")
