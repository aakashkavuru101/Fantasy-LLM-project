import os, torch, wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from pathlib import Path

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
TOKEN_DIR  = Path("data/tokenized")
OUTPUT_DIR = Path("checkpoints/fantasy-lora")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    model = get_peft_model(model, lora)
    data = load_from_disk(TOKEN_DIR)

    args = TrainingArguments(
        str(OUTPUT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        save_steps=500,
        logging_steps=10,
        report_to="wandb",
    )
    trainer = Trainer(model=model, args=args, train_dataset=data)
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

if __name__ == "__main__":
    main()