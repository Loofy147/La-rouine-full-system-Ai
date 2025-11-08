# scripts/inference.py

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from utils.logger import get_logger
from utils.exceptions import ModelLoadingError, InferenceError
import config

logger = get_logger(__name__)


def run_inference(prompt: str, model, tokenizer):
    """
    Runs inference on a given prompt using the composed model.
    """
    logger.info("Running inference...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")
    logger.info("Inference complete.")
    return response


def merge_and_save(model, tokenizer, output_path: str):
    """
    Merges the adapters into the base model and saves the merged model.
    """
    logger.info(f"Merging adapters and saving to {output_path}...")
    model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Model merged and saved successfully.")


def main():
    """
    Loads the base model, composes the domain and task adapters,
    and runs inference or merges the model based on command-line arguments.
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

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.BNB_CONFIG["load_in_4bit"],
            bnb_4bit_quant_type=config.BNB_CONFIG["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config.BNB_CONFIG["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, config.BNB_CONFIG["bnb_4bit_compute_dtype"].split('.')[-1])
        )

        # Load the base model and tokenizer
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, trust_remote_code=True)
        except Exception as e:
            raise ModelLoadingError(f"Failed to load base model or tokenizer: {e}")

        # Load and compose the adapters
        try:
            model = PeftModel.from_pretrained(model, config.DOMAIN_ADAPTER_OUTPUT_DIR, adapter_name="domain_adapter")
            logger.info("Domain adapter loaded.")
            model.load_adapter(config.TASK_ADAPTER_OUTPUT_DIR, adapter_name="task_adapter")
            logger.info("Task adapter loaded and composed.")
        except Exception as e:
            raise ModelLoadingError(f"Failed to load one or more adapters: {e}")

        if args.prompt:
            run_inference(args.prompt, model, tokenizer)

        if args.merge_path:
            merge_and_save(model, tokenizer, args.merge_path)

    except (ModelLoadingError, InferenceError) as e:
        logger.error(e)
    except Exception as e:
        raise InferenceError(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
