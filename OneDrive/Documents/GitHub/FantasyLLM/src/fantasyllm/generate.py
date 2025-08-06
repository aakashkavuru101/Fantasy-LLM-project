import torch, fire
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_DIR = Path("checkpoints/fantasy-lora")

def main(prompt: str, max_new_tokens: int = 128):
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    print(result[0]["generated_text"])

if __name__ == "__main__":
    fire.Fire(main)