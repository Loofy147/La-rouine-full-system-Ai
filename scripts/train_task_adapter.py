# scripts/train_task_adapter.py
"""
Script to train a task-specific adapter using QLoRA.

This script manages the supervised fine-tuning (SFT) process for a given
task, including:
- Loading the base model with 4-bit quantization.
- Configuring and applying a LoRA adapter for the SFT task.
- Loading and formatting the structured JSON dataset.
- Running the training loop using the Hugging Face Trainer.
- Saving the trained task adapter weights.
"""
import json
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

import config
from utils.exceptions import DataPreparationError, ModelLoadingError, TrainingError
from utils.logger import get_logger

logger = get_logger(__name__)


def train_task_adapter() -> None:
    """
    Trains and saves a task adapter on a labeled dataset.

    This function orchestrates the end-to-end supervised fine-tuning process
    using parameters defined in the central `config.py` file.

    Raises:
        ModelLoadingError: If the base model or tokenizer fails to load.
        DataPreparationError: If the task data cannot be loaded or processed.
        TrainingError: If an error occurs during the training process.
    """
    try:
        logger.info("Starting task adapter training...")

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
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                config.TOKENIZER_NAME, trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            raise ModelLoadingError(f"Failed to load model or tokenizer: {e}") from e

        # Configure LoRA for the task
        lora_config = LoraConfig(**config.SFT_LORA_CONFIG)
        model = get_peft_model(model, lora_config)

        # Load and prepare the dataset
        try:
            with open(config.TASK_DATA_PATH, "r", encoding="utf-8") as f:
                data: List[Dict[str, str]] = json.load(f)
            dataset = Dataset.from_list(data)
        except Exception as e:
            raise DataPreparationError(
                f"Failed to load or process task data: {e}"
            ) from e

        def format_and_tokenize(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
            """Formats and tokenizes a batch of examples."""
            full_texts = [
                f"Context: {c}\n\nQuestion: {q}\n\nAnswer: {a}"
                for c, q, a in zip(
                    examples["context"],
                    examples["question"],
                    examples["answer"],
                    strict=True,
                )
            ]
            tokenized_outputs = tokenizer(
                full_texts, truncation=True, max_length=512, padding="max_length"
            )
            tokenized_outputs["labels"] = [x[:] for x in tokenized_outputs["input_ids"]]
            return tokenized_outputs

        tokenized_dataset = dataset.map(format_and_tokenize, batched=True)

        # Set up the trainer
        training_args = TrainingArguments(
            output_dir=config.TASK_ADAPTER_OUTPUT_DIR, **config.SFT_TRAINING_ARGS
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Train the model
        trainer.train()
        logger.info("Task adapter training complete.")

        # Save the adapter
        model.save_pretrained(config.TASK_ADAPTER_OUTPUT_DIR)
        logger.info(f"Task adapter saved to {config.TASK_ADAPTER_OUTPUT_DIR}")

    except (ModelLoadingError, TrainingError, DataPreparationError) as e:
        logger.error(e)
    except Exception as e:
        raise TrainingError(
            f"An unexpected error occurred during task adapter training: {e}"
        ) from e


if __name__ == "__main__":
    train_task_adapter()
