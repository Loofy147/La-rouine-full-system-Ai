# scripts/benchmark.py

import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def run_benchmark(num_runs=10):
    """
    Measures the inference latency of the final composed model.

    Args:
        num_runs (int): The number of inference runs to average over.
    """
    try:
        logger.info("Starting benchmark...")

        # Load the model and adapters (similar to inference.py)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.BNB_CONFIG["load_in_4bit"],
            bnb_4bit_quant_type=config.BNB_CONFIG["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config.BNB_CONFIG["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(
                torch, config.BNB_CONFIG["bnb_4bit_compute_dtype"].split(".")[-1]
            ),
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.TOKENIZER_NAME, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            model, config.DOMAIN_ADAPTER_OUTPUT_DIR, adapter_name="domain_adapter"
        )
        model.load_adapter(config.TASK_ADAPTER_OUTPUT_DIR, adapter_name="task_adapter")

        prompt = "Context: The capital of France is Paris.\n\nQuestion: What is the capital of France?\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Warm-up run
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50)

        # Benchmark runs
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=50)
            end_time = time.time()
            latencies.append(end_time - start_time)

        avg_latency = sum(latencies) / len(latencies)
        logger.info("--- Benchmark Results ---")
        logger.info(f"Number of runs: {num_runs}")
        logger.info(f"Average latency per generation: {avg_latency:.4f} seconds")
        logger.info("-------------------------")

    except Exception as e:
        logger.error(f"An error occurred during benchmarking: {e}")


if __name__ == "__main__":
    run_benchmark()
