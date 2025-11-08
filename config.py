# config.py

"""
Central configuration for the modular fine-tuning project.
All parameters that are likely to change between experiments should be defined here.
"""

# --- Model Configuration ---
MODEL_NAME = "MiniMaxAI/MiniMax-M2"
TOKENIZER_NAME = "MiniMaxAI/MiniMax-M2"

# --- Data Configuration ---
DOMAIN_CORPUS_PATH = "data/domain_corpus/domain_corpus.txt"
TASK_DATA_PATH = "data/task_data/task_data.json"

# --- Output Configuration ---
DOMAIN_ADAPTER_OUTPUT_DIR = "models/domain_adapter"
TASK_ADAPTER_OUTPUT_DIR = "models/task_adapter"

# --- Quantization Configuration ---
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "torch.bfloat16" # Note: This will be converted to a torch dtype in the scripts
}

# --- Domain Adapter (DAPT) LoRA Configuration ---
DAPT_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# --- Task Adapter (SFT) LoRA Configuration ---
SFT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# --- Domain Adapter (DAPT) Training Arguments ---
DAPT_TRAINING_ARGS = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "save_steps": 1000,
    "logging_steps": 100,
    "fp16": True
}

# --- Task Adapter (SFT) Training Arguments ---
SFT_TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "save_steps": 500,
    "logging_steps": 50,
    "fp16": True
}
