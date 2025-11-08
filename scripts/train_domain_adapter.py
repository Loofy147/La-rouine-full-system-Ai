# scripts/train_domain_adapter.py
"""
Script to train a domain adapter using QLoRA.

This script handles the entire domain adaptation process, including:
- Loading the base model with 4-bit quantization.
- Configuring and applying a LoRA adapter for domain pre-training (DAPT).
- Loading and tokenizing the unlabeled domain corpus.
- Running the training loop using the Hugging Face Trainer.
- Saving the trained domain adapter weights.
"""
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

import config
from utils.exceptions import ModelLoadingError, TrainingError
from utils.logger import get_logger

logger = get_logger(__name__)


def train_domain_adapter() -> None:
    """
    Trains and saves a domain adapter on the unlabeled domain corpus.

    This function orchestrates the end-to-end domain adaptation process
    using parameters defined in the central `config.py` file.

    Raises:
        ModelLoadingError: If the base model or tokenizer fails to load.
        TrainingError: If an error occurs during the training process.
    """
    try:
        logger.info("Starting domain adapter training...")

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.BNB_CONFIG["load_in_4bit"],
            bnb_4bit_quant_type=config.BNB_CONFIG["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config.BNB_CONFIG["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(
                torch, config.BNB_CONFIG["bnb_4bit_compute_dtype"].split(".")[-1]
            ),
        )

        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.TOKENIZER_NAME, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            raise ModelLoadingError(f"Failed to load model or tokenizer: {e}") from e

        # Configure LoRA for domain adaptation
        lora_config = LoraConfig(**config.DAPT_LORA_CONFIG)
        model = get_peft_model(model, lora_config)

        # Load and prepare the dataset
        dataset = load_dataset(
            "text", data_files=config.DOMAIN_CORPUS_PATH, split="train"
        )

        def tokenize_function(
            examples: Dict[str, List[str]],
        ) -> Dict[str, List[int]]:
            """Tokenizes a batch of text examples."""
            return tokenizer(examples["text"], truncation=True, max_length=512)

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        # Set up the trainer
        training_args = TrainingArguments(
            output_dir=config.DOMAIN_ADAPTER_OUTPUT_DIR, **config.DAPT_TRAINING_ARGS
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Train the model
        trainer.train()
        logger.info("Domain adapter training complete.")

        # Save the adapter
        model.save_pretrained(config.DOMAIN_ADAPTER_OUTPUT_DIR)
        logger.info(f"Domain adapter saved to {config.DOMAIN_ADAPTER_OUTPUT_DIR}")

    except (ModelLoadingError, TrainingError) as e:
        logger.error(e)
    except Exception as e:
        raise TrainingError(
            f"An unexpected error occurred during domain adapter training: {e}"
        ) from e


if __name__ == "__main__":
    train_domain_adapter()
