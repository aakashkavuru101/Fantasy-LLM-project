"""Data processing module for Fantasy LLM."""

from datasets import load_dataset
from transformers import AutoTokenizer

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


def main(limit: int = None) -> None:
    """Process and tokenize the dataset."""
    limit = limit or settings.dataset_limit
    
    logger.info("Starting data processing", limit=limit)
    
    settings.token_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading tokenizer", model=settings.base_model)
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading dataset")
    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{limit}]")
    
    logger.info("Tokenizing dataset", samples=len(ds))
    
    def tokenize(batch):
        out = tokenizer(
            batch["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=settings.max_length
        )
        out["labels"] = out["input_ids"].copy()
        return out
    
    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    
    logger.info("Saving tokenized dataset", path=str(settings.token_dir))
    ds.save_to_disk(settings.token_dir)
    
    logger.info("Dataset processing completed", 
                samples=len(ds), 
                output_dir=str(settings.token_dir))

if __name__ == "__main__":
    main()