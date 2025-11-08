# scripts/inference.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from utils.logger import get_logger
from utils.exceptions import ModelLoadingError, InferenceError
import config

logger = get_logger(__name__)

def run_inference():
    """
    Loads the base model, composes the domain and task adapters,
    and runs inference on a sample prompt.
    """
    try:
        logger.info("Starting inference...")

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

        # Create a sample prompt
        prompt = "Context: The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris.\n\nQuestion: What is the capital of France?\n\nAnswer:"

        # Tokenize the prompt and generate a response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}")
        logger.info("Inference complete.")

    except (ModelLoadingError, InferenceError) as e:
        logger.error(e)
    except Exception as e:
        raise InferenceError(f"An unexpected error occurred during inference: {e}")

if __name__ == "__main__":
    run_inference()
