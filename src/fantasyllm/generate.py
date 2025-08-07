"""Text generation module for Fantasy LLM."""

import time
from typing import Optional

import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .config import settings
from .logging import get_logger, log_inference, log_error


logger = get_logger(__name__)


def main(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> None:
    """Generate text using the fine-tuned model."""
    start_time = time.time()
    
    # Use provided parameters or fall back to settings
    generation_params = {
        "max_new_tokens": max_new_tokens or settings.max_new_tokens,
        "temperature": temperature or settings.temperature,
        "top_p": top_p or settings.top_p,
        "top_k": top_k or settings.top_k,
        "do_sample": True,
    }
    
    logger.info("Starting text generation", 
                prompt_length=len(prompt),
                **generation_params)
    
    try:
        # Load tokenizer and model
        logger.info("Loading tokenizer", path=str(settings.adapter_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(settings.adapter_dir))
        
        logger.info("Loading model", model=settings.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            settings.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            return_full_text=False
        )
        
        # Generate text
        logger.info("Generating text")
        generation_params["pad_token_id"] = tokenizer.eos_token_id
        result = pipe(prompt, **generation_params)
        generated_text = result[0]["generated_text"]
        
        inference_time = time.time() - start_time
        
        # Log the inference
        log_inference(prompt, generated_text, inference_time=inference_time)
        
        # Output the result
        print(generated_text)
        
        logger.info("Generation completed", 
                    inference_time=inference_time,
                    generated_length=len(generated_text))
        
    except Exception as e:
        log_error(e, {"prompt": prompt, "params": generation_params})
        raise

if __name__ == "__main__":
    fire.Fire(main)