from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import os

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
TOKEN_DIR = Path("data/tokenized")

def main(limit: int = 20_000):
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{limit}]")
    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
        out["labels"] = out["input_ids"].copy()
        return out
    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds.save_to_disk(TOKEN_DIR)
    print("Dataset tokenized and saved to", TOKEN_DIR)

if __name__ == "__main__":
    main()