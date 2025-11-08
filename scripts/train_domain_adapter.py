import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def train_domain_adapter():
    """
    Trains a domain adapter on the unlabeled domain corpus using QLoRA.
    """
    model_name = "MiniMaxAI/MiniMax-M2"
    domain_corpus_path = "data/domain_corpus/domain_corpus.txt"
    output_dir = "models/domain_adapter"

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Configure LoRA for domain adaptation
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load and prepare the dataset
    dataset = load_dataset("text", data_files=domain_corpus_path, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Set up the trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        save_steps=1000,
        logging_steps=100,
        fp16=True, # Use mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting domain adapter training...")
    trainer.train()
    print("Domain adapter training complete.")

    # Save the adapter
    model.save_pretrained(output_dir)
    print(f"Domain adapter configuration saved to {output_dir}")

if __name__ == "__main__":
    train_domain_adapter()
