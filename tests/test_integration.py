# tests/test_integration.py

import json
import os
import shutil
import unittest

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        """
        Set up a tiny dataset and configuration for the integration test.
        """
        self.model_name = "sshleifer/tiny-gpt2"
        self.tokenizer_name = "sshleifer/tiny-gpt2"
        self.dapt_lora_config = {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["c_attn"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        self.sft_lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["c_attn"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        self.tiny_domain_corpus = (
            "This is a test sentence. This is another test sentence."
        )
        self.tiny_task_data = [
            {
                "context": "The capital of France is Paris.",
                "question": "What is the capital of France?",
                "answer": "Paris",
            },
            {
                "context": "The sky is blue.",
                "question": "What color is the sky?",
                "answer": "blue",
            },
        ]

        # Create temporary data files
        os.makedirs("data/tiny_test", exist_ok=True)
        self.domain_corpus_path = "data/tiny_test/domain_corpus.txt"
        self.task_data_path = "data/tiny_test/task_data.json"

        with open(self.domain_corpus_path, "w") as f:
            f.write(self.tiny_domain_corpus)

        with open(self.task_data_path, "w") as f:
            json.dump(self.tiny_task_data, f)

        self.domain_adapter_output_dir = "models/tiny_test/dapt"
        self.task_adapter_output_dir = "models/tiny_test/sft"

    def test_end_to_end_pipeline(self):
        """
        Performs a minimal, end-to-end run of the entire pipeline.
        """
        # --- 1. Train Domain Adapter ---
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token

        lora_config_dapt = LoraConfig(**self.dapt_lora_config)
        model = get_peft_model(model, lora_config_dapt)

        dataset_dapt = load_dataset(
            "text", data_files=self.domain_corpus_path, split="train"
        )

        def tokenize_dapt(examples):
            tokenized_outputs = tokenizer(
                examples["text"], truncation=True, max_length=512
            )
            tokenized_outputs["labels"] = tokenized_outputs["input_ids"][:]
            return tokenized_outputs

        tokenized_dataset_dapt = dataset_dapt.map(
            tokenize_dapt, batched=True, remove_columns=["text"]
        )

        training_args_dapt = TrainingArguments(
            output_dir=self.domain_adapter_output_dir, max_steps=1
        )
        trainer_dapt = Trainer(
            model=model, args=training_args_dapt, train_dataset=tokenized_dataset_dapt
        )
        trainer_dapt.train()
        model.save_pretrained(self.domain_adapter_output_dir)

        # --- 2. Train Task Adapter ---
        lora_config_sft = LoraConfig(**self.sft_lora_config)
        model = get_peft_model(model, lora_config_sft)

        dataset_sft = Dataset.from_list(self.tiny_task_data)

        def tokenize_sft(examples):
            full_texts = [
                f"Context: {c}\n\nQuestion: {q}\n\nAnswer: {a}"
                for c, q, a in zip(
                    examples["context"],
                    examples["question"],
                    examples["answer"],
                    strict=True,
                )
            ]
            tokenized_outputs = tokenizer(
                full_texts, truncation=True, max_length=512, padding="max_length"
            )
            tokenized_outputs["labels"] = [x[:] for x in tokenized_outputs["input_ids"]]
            return tokenized_outputs

        tokenized_dataset_sft = dataset_sft.map(tokenize_sft, batched=True)

        training_args_sft = TrainingArguments(
            output_dir=self.task_adapter_output_dir, max_steps=1
        )
        trainer_sft = Trainer(
            model=model, args=training_args_sft, train_dataset=tokenized_dataset_sft
        )
        trainer_sft.train()
        model.save_pretrained(self.task_adapter_output_dir)

        # --- 3. Run Inference ---
        model = PeftModel.from_pretrained(
            model, self.domain_adapter_output_dir, adapter_name="domain_adapter"
        )
        model.load_adapter(self.task_adapter_output_dir, adapter_name="task_adapter")

        prompt = (
            "Context: The sky is blue.\n\nQuestion: What color is the sky?\n\nAnswer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.assertIn(prompt, response)
        self.assertTrue(len(response) > len(prompt))

    def tearDown(self):
        """
        Clean up the temporary data and model files.
        """
        if os.path.exists("data/tiny_test"):
            shutil.rmtree("data/tiny_test")
        if os.path.exists("models/tiny_test"):
            shutil.rmtree("models/tiny_test")


if __name__ == "__main__":
    unittest.main()
