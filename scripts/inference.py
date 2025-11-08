# scripts/inference.py
"""
Inference script for the modular fine-tuning project.

This script provides functionalities to:
- Load a base model with 4-bit quantization.
- Compose trained domain and task adapters onto the base model.
- Run inference on a given prompt using the composed model.
- Merge the adapters into the base model and save the result for deployment.

The script is configurable via command-line arguments to select between
running inference or merging adapters.
"""
import argparse

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from utils.exceptions import InferenceError, ModelLoadingError
from utils.logger import get_logger
from utils.model_utils import compose_adapters, load_model_and_tokenizer

logger = get_logger(__name__)


def run_inference(
    prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> str:
    """
    Runs inference on a prompt using the provided model and tokenizer.

    Args:
        prompt: The input text to generate a response for.
        model: The composed model with adapters loaded.
        tokenizer: The tokenizer corresponding to the base model.

    Returns:
        The generated response string.
    """
    logger.info("Running inference...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")
    logger.info("Inference complete.")
    return response


def merge_and_save(
    model: PeftModel, tokenizer: PreTrainedTokenizerBase, output_path: str
) -> None:
    """
    Merges the LoRA adapters into the base model and saves the result.

    This function is useful for creating a standalone, deployable model that
    does not require the PEFT library for inference.

    Args:
        model: The PeftModel with adapters to be merged.
        tokenizer: The tokenizer to save alongside the merged model.
        output_path: The directory path to save the merged model and tokenizer.
    """
    logger.info(f"Merging adapters and saving to {output_path}...")
    model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Model merged and saved successfully.")


def main() -> None:
    """
    Main function to orchestrate the inference or merging process.

    Parses command-line arguments to determine whether to run inference on a
    prompt or to merge and save the adapters. Handles model loading, adapter
    composition, and calls the appropriate function.

    Raises:
        ModelLoadingError: If the base model, tokenizer, or adapters fail to load.
        InferenceError: If an unexpected error occurs during the process.
    """
    parser = argparse.ArgumentParser(description="Run inference or merge adapters.")
    parser.add_argument("--prompt", type=str, help="The prompt to run inference on.")
    parser.add_argument("--merge_path", type=str, help="Path to save the merged model.")
    args = parser.parse_args()

    if not args.prompt and not args.merge_path:
        logger.error("Please provide a --prompt or --merge_path.")
        return

    try:
        logger.info("Starting inference script...")
        model, tokenizer = load_model_and_tokenizer()
        model = compose_adapters(model)

        if args.prompt:
            run_inference(args.prompt, model, tokenizer)

        if args.merge_path:
            merge_and_save(model, tokenizer, args.merge_path)

    except (ModelLoadingError, InferenceError) as e:
        logger.error(e)
    except Exception as e:
        raise InferenceError(f"An unexpected error occurred: {e}") from e


if __name__ == "__main__":
    main()
