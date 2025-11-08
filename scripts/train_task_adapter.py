import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import json

def train_task_adapter():
    """
    Trains a task adapter on the labeled task dataset using QLoRA.
    """
    model_name = "gpt2"
    task_data_path = "data/task_data/task_data.json"
    output_dir = "models/task_adapter"

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token for batching
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Configure LoRA for the task
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load and prepare the dataset
    with open(task_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create a Dataset object
    from datasets import Dataset
    dataset = Dataset.from_list(data)

    def format_and_tokenize(examples):
        # Create a prompt from the context, question, and answer
        full_texts = [f"Context: {c}\n\nQuestion: {q}\n\nAnswer: {a}" for c, q, a in zip(examples['context'], examples['question'], examples['answer'])]

        # Tokenize the full texts
        tokenized_outputs = tokenizer(full_texts, truncation=True, max_length=512, padding="max_length")

        # The labels are the input_ids themselves
        tokenized_outputs["labels"] = [x[:] for x in tokenized_outputs["input_ids"]]

        return tokenized_outputs

    tokenized_dataset = dataset.map(format_and_tokenize, batched=True)

    # Set up the trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        save_steps=500,
        logging_steps=50,
        fp16=True, # Use mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting task adapter training...")
    trainer.train()
    print("Task adapter training complete.")

    # Save the adapter
    model.save_pretrained(output_dir)
    print(f"Task adapter configuration saved to {output_dir}")

if __name__ == "__main__":
    train_task_adapter()
