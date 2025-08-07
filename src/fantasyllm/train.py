"""Training module for Fantasy LLM."""

import os
from typing import Optional

import torch
import wandb
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from .config import settings
from .logging import get_logger, log_model_info, log_training_step

logger = get_logger(__name__)


def main() -> None:
    """Train the LoRA fine-tuned model."""
    logger.info("Starting training process")
    
    # Set up directories
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if configured
    if settings.wandb_mode != "offline":
        wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            name=f"fantasy-llm-{settings.environment}",
            config=dict(settings)
        )
    
    # Load tokenizer
    logger.info("Loading tokenizer", model=settings.base_model)
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("Loading base model", model=settings.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Configure LoRA
    logger.info("Setting up LoRA configuration", 
                r=settings.lora_r,
                alpha=settings.lora_alpha,
                dropout=settings.lora_dropout)
    
    lora_config = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=settings.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Log model information
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    log_model_info(settings.base_model, {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params,
        "lora_config": lora_config.__dict__,
    })
    
    # Load dataset
    logger.info("Loading tokenized dataset", path=str(settings.token_dir))
    if not settings.token_dir.exists():
        raise FileNotFoundError(
            f"Tokenized data not found at {settings.token_dir}. "
            "Run fantasy-data first to process the dataset."
        )
    
    data = load_from_disk(settings.token_dir)
    logger.info("Dataset loaded", samples=len(data))
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(settings.output_dir),
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        num_train_epochs=settings.num_train_epochs,
        learning_rate=settings.learning_rate,
        fp16=False,
        bf16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=10,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_bnb_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        group_by_length=True,
        save_total_limit=3,
        report_to="wandb" if settings.wandb_mode != "offline" else None,
        run_name=f"fantasy-llm-{settings.environment}",
        logging_dir=f"logs/training-{settings.environment}",
    )
    
    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training", 
                epochs=settings.num_train_epochs,
                batch_size=settings.batch_size,
                learning_rate=settings.learning_rate)
    
    try:
        trainer.train()
        
        # Save the model
        logger.info("Saving model", path=str(settings.output_dir))
        trainer.save_model(str(settings.output_dir))
        tokenizer.save_pretrained(str(settings.output_dir))
        
        logger.info("Training completed successfully")
        
        # Finish wandb run
        if settings.wandb_mode != "offline":
            wandb.finish()
            
    except Exception as e:
        logger.error("Training failed", error=str(e))
        if settings.wandb_mode != "offline":
            wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    main()