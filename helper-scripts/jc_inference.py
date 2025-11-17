import json
import time
import requests
from tqdm import tqdm

INPUT_PATH = "jca_qna_set.json"     # your input file
OUTPUT_PATH = "jc_model_answers.json"  # output file
API_URL = "http://localhost:8000/ask"


def ask_model(question_text):
    """Send a question to the model API and return the model answer text."""
    payload = {"question": question_text}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    response.raise_for_status()  # raises error if not 200

    data = response.json()

    return data.replace("\"","'")


if __name__ == "__main__":
    # Load input JSON
    with open(INPUT_PATH, "r") as f:
        qna_pairs = json.load(f)

    updated_qna_pairs = []

    # Loop through each Q&A pair
    for idx, item in tqdm(enumerate(qna_pairs, start=1)):
        question = item["question"]
        print(f"[{idx}/{len(qna_pairs)}] Asking model: {question}")

        model_answer = json.loads(ask_model(question))
        
        print("Model answer:", model_answer)

        # Store the model answer
        item["model_answer"] = model_answer['answer']
        item["contexts"]
        updated_qna_pairs.append(item)

        # Optional: sleep to avoid hammering the server
        time.sleep(0.5)

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(updated_qna_pairs, f, indent=2)

    print("\nDone! Saved results to:", OUTPUT_PATH)

