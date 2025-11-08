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
import argparse
import json
import os
import subprocess
import time
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

from utils.exceptions import ModelLoadingError, TrainingError
from utils.logger import get_logger

try:
    import config
except ImportError:
    config = None

logger = get_logger(__name__)


def train_domain_adapter(
    base_model: str,
    tokenizer_name: str,
    domain_corpus_path: str,
    output_dir: str,
    no_quant: bool,
    target_modules: List[str],
    dry_run: bool,
    no_fp16: bool,
) -> None:
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

        quantization_config = None
        if not no_quant:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            raise ModelLoadingError(f"Failed to load model or tokenizer: {e}") from e

        # Configure LoRA for domain adaptation
        lora_config_dict = config.DAPT_LORA_CONFIG if config else {}
        if target_modules:
            lora_config_dict["target_modules"] = target_modules
        lora_config = LoraConfig(**lora_config_dict)
        model = get_peft_model(model, lora_config)

        # Load and prepare the dataset
        dataset = load_dataset("text", data_files=domain_corpus_path, split="train")

        def tokenize_function(
            examples: Dict[str, List[str]],
        ) -> Dict[str, List[int]]:
            """Tokenizes a batch of text examples."""
            tokenized_outputs = tokenizer(
                examples["text"], truncation=True, max_length=512
            )
            tokenized_outputs["labels"] = tokenized_outputs["input_ids"][:]
            return tokenized_outputs

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        # Set up the trainer
        training_args_dict = config.DAPT_TRAINING_ARGS if config else {}
        if no_fp16:
            training_args_dict["fp16"] = False
        training_args = TrainingArguments(
            output_dir=output_dir, **training_args_dict
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        if dry_run:
            logger.info("Dry run complete. Skipping training.")
            return

        # Train the model
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        logger.info("Domain adapter training complete.")

        # Save the adapter
        model.save_pretrained(output_dir)
        logger.info(f"Domain adapter saved to {output_dir}")

        # Generate training manifest
        manifest = {
            "base_model": base_model,
            "adapter_type": "domain_adapter",
            "training_corpus": domain_corpus_path,
            "lora_config": lora_config_dict,
            "training_args": training_args.to_dict(),
            "git_commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip(),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
        }
        with open(
            os.path.join(output_dir, "training_manifest.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Training manifest saved to {output_dir}")

    except (ModelLoadingError, TrainingError) as e:
        logger.error(e)
    except Exception as e:
        raise TrainingError(
            f"An unexpected error occurred during domain adapter training: {e}"
        ) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a domain adapter for a causal language model."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=config.MODEL_NAME if config else None,
        help="The name of the base model to use.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=config.TOKENIZER_NAME if config else None,
        help="The name of the tokenizer to use.",
    )
    parser.add_argument(
        "--domain_corpus_path",
        type=str,
        default=config.DOMAIN_CORPUS_PATH if config else None,
        help="The path to the domain corpus.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.DOMAIN_ADAPTER_OUTPUT_DIR if config else None,
        help="The directory to save the trained adapter.",
    )
    parser.add_argument(
        "--no_quant", action="store_true", help="Disable quantization."
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=None,
        help="The names of the modules to apply LORA to.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Perform a dry run without training."
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help="Disable fp16 training."
    )
    args = parser.parse_args()

    train_domain_adapter(
        args.base_model,
        args.tokenizer_name,
        args.domain_corpus_path,
        args.output_dir,
        args.no_quant,
        args.target_modules,
        args.dry_run,
        args.no_fp16,
    )
