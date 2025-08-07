# FantasyLLM [![CI](https://github.com/<you>/FantasyLLM/workflows/CI/badge.svg)](https://github.com/<you>/FantasyLLM/actions)

LoRA fine-tuned Llama-2 7B for fantasy short stories.

## Quick start
```bash
pip install -e .
fantasy-data           # download & tokenize
accelerate launch -m fantasyllm.train
fantasy-demo           # Gradio UI