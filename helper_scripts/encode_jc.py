import sqlite3
import uuid
import numpy as np
import pickle
import re
import json
import sys

# Use Ollama for embeddings
from ollama import Client
client = Client(host="http://localhost:11434")   # default Ollama


# Load Julius Caesar text

with open("js_raw.txt", "r", encoding="utf-8") as f:
    full_text = f.read()


# Chunking Function

def split_by_act_and_scene(text):
    # Matches:
    # ACT I
    # ACT II
    # etc.
    act_pattern = r"(ACT\s+[IVX]+)"
    
    # Split text by acts
    act_parts = re.split(act_pattern, text)

    # We'll recombine like:
    # ["ACT I", "<full act content>", "ACT II", "<full act content>", ...]
    acts = []
    for i in range(1, len(act_parts), 2):
        act_name = act_parts[i].strip()
        act_text = act_parts[i+1]
        acts.append((act_name, act_text.strip()))
    
    # Now further split each act by scene
    scene_pattern = r"(SCENE\s+[IVX]+\. .+?\n)"

    chunks = []

    for act_name, act_text in acts:
        scene_parts = re.split(scene_pattern, act_text)

        # First chunk may be intro text before first SCENE — keep only if not empty
        current_scene = ""

        for part in scene_parts:
            if re.match(scene_pattern, part):
                # If we have an existing scene, save it first
                if current_scene.strip():
                    chunks.append(current_scene.strip())
                
                # Begin new scene, with ACT included
                current_scene = f"{act_name}\n{part}"
            else:
                # Content body of scene
                current_scene += part
        
        # Save trailing scene
        if current_scene.strip():
            chunks.append(current_scene.strip())
    
    return chunks


scenes = split_by_act_and_scene(full_text)

#Add initial contents
scenes.insert(0,full_text.split('ACT I')[0])

print(f"Total scenes found: {len(scenes)}")
print("\n--- First scene preview ---\n")
print(scenes[0][:500])

scene_dict = {}

for scene, i in zip(scenes,range(len(scenes))):
    tokens_est = len(scene.split(' '))
    print(f'SECTION-{i} length: {tokens_est}')
    if tokens_est < 599 or i == 0: 
        scene_dict[f'SECTION-{i}'] = scene
    else: 
        # Split scenes with >600 tokens into parts of ~500 tokens each
        num_splits = tokens_est//500+1
        split_size = len(scene)//num_splits
        for j in range(num_splits):
            split_start = j*split_size
            split_end = (j+1)*split_size+100 if (j+1)*split_size+100 < len(scene) else len(scene) # 100 char overlap per part
            scene_part = scene[split_start:split_end]
            scene_dict[f'SECTION-{i} PART-{j+1}'] = scene_part 

            print(f'SECTION-{i} PART-{j+1} length: {len(scene_part.split(' '))}')


with open('jc_dataset.json','w') as fp:
    json.dump(scene_dict, fp, indent=4)

chunks = [f"{k}:{v}" for k, v in scene_dict.items()]
print(f"Total chunks created: {len(chunks)}")
print('Sample chunk:', chunks[0][:500])

# sys.exit(0)

# Connect to SQLite Database

db_path = "jc_rag_data.db"   # your new DB file
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS iiitb_dataset (
    id INTEGER PRIMARY KEY,
    milvus_id VARCHAR(1024),
    data BLOB
);
""")
conn.commit()



# Encode + Insert into SQLite

for i, chunk in enumerate(chunks):

    # --- Generate embedding using Ollama ---
    result = client.embeddings(model="nomic-embed-text", prompt=chunk)
    emb = np.array(result["embedding"], dtype=np.float32)

    # --- Convert embedding to BLOB ---
    emb_blob = pickle.dumps(emb)

    # --- Create Milvus ID ---
    milvus_id = str(uuid.uuid4())

    # --- Insert into SQLite ---
    cursor.execute("""
        INSERT INTO iiitb_dataset (milvus_id, data)
        VALUES (?, ?)
    """, (milvus_id, emb_blob))

    if (i + 1) % 20 == 0:
        print(f"Inserted {i+1} embeddings...")

conn.commit()
conn.close()

print("Embedding generation & SQLite import complete!")
