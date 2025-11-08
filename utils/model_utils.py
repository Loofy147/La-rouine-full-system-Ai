# utils/model_utils.py
"""
Utility functions for model loading and adapter composition.
"""
import os

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

try:
    import config
except ImportError:
    config = None
from utils.logger import get_logger

logger = get_logger(__name__)


def load_model_and_tokenizer(
    model_name: str, tokenizer_name: str, no_quant: bool
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads the base model and tokenizer with optional quantization.

    Args:
        model_name: The name of the base model to load.
        tokenizer_name: The name of the tokenizer to load.
        no_quant: If True, disables quantization.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    quantization_config = None
    if not no_quant:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )
    return model, tokenizer


def compose_adapters(
    model: PreTrainedModel, domain_adapter_path: str, task_adapter_path: str
) -> PeftModel:
    """
    Composes the domain and task adapters onto the base model.

    Args:
        model: The base model to compose adapters onto.
        domain_adapter_path: The path to the domain adapter.
        task_adapter_path: The path to the task adapter.

    Returns:
        The composed PeftModel.
    """
    if os.path.exists(domain_adapter_path):
        model = PeftModel.from_pretrained(
            model, domain_adapter_path, adapter_name="domain_adapter"
        )
        logger.info("Domain adapter loaded.")

        if os.path.exists(task_adapter_path):
            model.load_adapter(task_adapter_path, adapter_name="task_adapter")
            logger.info("Task adapter loaded and composed.")
        else:
            logger.warning(f"Task adapter not found at {task_adapter_path}. Skipping.")

    elif os.path.exists(task_adapter_path):
        model = PeftModel.from_pretrained(
            model, task_adapter_path, adapter_name="task_adapter"
        )
        logger.info("Task adapter loaded.")
    else:
        logger.warning("No adapters found. Using base model only.")

    return model
