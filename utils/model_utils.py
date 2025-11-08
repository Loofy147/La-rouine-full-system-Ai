# utils/model_utils.py
"""
Utility functions for model loading and adapter composition.
"""
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def load_model_and_tokenizer() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads the base model and tokenizer with quantization.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.BNB_CONFIG["load_in_4bit"],
        bnb_4bit_quant_type=config.BNB_CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config.BNB_CONFIG["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(
            torch, config.BNB_CONFIG["bnb_4bit_compute_dtype"].split(".")[-1]
        ),
    )

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        config.TOKENIZER_NAME, trust_remote_code=True
    )
    return model, tokenizer


def compose_adapters(model: PreTrainedModel) -> PeftModel:
    """
    Composes the domain and task adapters onto the base model.

    Args:
        model: The base model to compose adapters onto.

    Returns:
        The composed PeftModel.
    """
    model = PeftModel.from_pretrained(
        model, config.DOMAIN_ADAPTER_OUTPUT_DIR, adapter_name="domain_adapter"
    )
    logger.info("Domain adapter loaded.")
    model.load_adapter(config.TASK_ADAPTER_OUTPUT_DIR, adapter_name="task_adapter")
    logger.info("Task adapter loaded and composed.")
    return model
