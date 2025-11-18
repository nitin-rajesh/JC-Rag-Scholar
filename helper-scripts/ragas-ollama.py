import json
import random

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_similarity,
    answer_correctness,
)

# NEW: Local Ollama LLM + Embeddings
# Removed: from ragas import LLMFactory, EmbeddingFactory # Obsolete
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings


# CONFIG
INPUT_FILE = "jc_model_answers.json"
OUTPUT_CSV = "ragas_evaluation_results.csv"

OLLAMA_LLM_NAME = "gemma3:1b"  # Ensure this model is pulled and running in Ollama
OLLAMA_EMBED_NAME = "nomic-embed-text" # Ensure this model is pulled and running in Ollama


# 1. LOAD BASE EVALUATION DATA (Assuming evaluation-tests.json exists)
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        base_items = json.load(f)
except FileNotFoundError:
    print(f"🚨 WARNING: '{INPUT_FILE}' not found. Creating mock data for demonstration.")
    # Create mock data if the file doesn't exist to allow the script to run
    base_items = [
        {"question": "Who appears to Brutus?", "model_answer": "Caesar's ghost.", "ideal_answer": "The ghost of Julius Caesar."},
        {"question": "What does Calpurnia dream?", "model_answer": "Blood flowing from a statue.", "ideal_answer": "Caesar's statue running with blood."},
    ]

print(f"Loaded {len(base_items)} evaluation items from {INPUT_FILE} (or mock-generated)")

items_with_contexts = base_items

with open("evaluation_data_mock.json", "w", encoding="utf-8") as f:
    json.dump(items_with_contexts, f, indent=2)
print("Saved mock-augmented data to evaluation_data_mock.json")


# 2. BUILD RAGAS DATASET

questions = []
answers = []
ground_truths = []
contexts = []

for item in items_with_contexts:
    questions.append(item["question"])
    answers.append(item["model_answer"])
    ground_truths.append(item["ideal_answer"])
    contexts.append(item["contexts"])

data_dict = {
    "question": questions,
    "answer": answers,
    "ground_truth": ground_truths,
    "contexts": contexts,
}

dataset = Dataset.from_dict(data_dict)
print("RAGAS dataset schema:", dataset)


# 4. CONFIGURE OLLAMA FOR RAGAS

# Initialize LangChain LLM and Embeddings directly

# Define a longer timeout in seconds (e.g., 60 seconds)
OLLAMA_TIMEOUT = 240

# Initialize LangChain LLM and Embeddings directly
# Pass the timeout explicitly to the Ollama client
ollama_llm = Ollama(model=OLLAMA_LLM_NAME, timeout=OLLAMA_TIMEOUT)
ollama_embed = OllamaEmbeddings(model=OLLAMA_EMBED_NAME) # Embeddings are usually fast, so default is fine

print(f"Configured Ollama LLM: {OLLAMA_LLM_NAME} with timeout={OLLAMA_TIMEOUT}s")
print(f"Configured Ollama Embeddings: {OLLAMA_EMBED_NAME}")


# 5. RUN RAGAS EVALUATION (Local, Offline)

# Pass the LangChain objects directly to evaluate()
result = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_similarity,
        answer_correctness,
    ],
    llm=ollama_llm,
    embeddings=ollama_embed,
)

print("\n=== Aggregate RAGAS scores (mock contexts, local Ollama) ===")
print(result)
print()


# 6. PER-QUESTION METRICS TO CSV

df = result.to_pandas()
print("=== First few per-question rows ===")
print(df.head())

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nPer-question metrics saved to {OUTPUT_CSV}")