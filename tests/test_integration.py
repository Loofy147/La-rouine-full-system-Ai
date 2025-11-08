# tests/test_integration.py

import unittest
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
import config

@unittest.skipIf(os.environ.get("RUN_INTEGRATION_TESTS") != "true", "Skipping integration tests by default.")
class TestIntegration(unittest.TestCase):

    def setUp(self):
        """
        Set up a tiny dataset for the integration test.
        """
        self.tiny_domain_corpus = "This is a test sentence. This is another test sentence."
        self.tiny_task_data = [
            {
                "context": "The capital of France is Paris.",
                "question": "What is the capital of France?",
                "answer": "Paris"
            },
            {
                "context": "The sky is blue.",
                "question": "What color is the sky?",
                "answer": "blue"
            }
        ]

        # Create temporary data files
        os.makedirs("data/tiny_test", exist_ok=True)
        self.domain_corpus_path = "data/tiny_test/domain_corpus.txt"
        self.task_data_path = "data/tiny_test/task_data.json"

        with open(self.domain_corpus_path, "w") as f:
            f.write(self.tiny_domain_corpus)

        with open(self.task_data_path, "w") as f:
            json.dump(self.tiny_task_data, f)

    def test_end_to_end_pipeline(self):
        """
        Performs a minimal, end-to-end run of the entire pipeline.
        """
        # --- 1. Train Domain Adapter ---
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)

        lora_config_dapt = LoraConfig(**config.DAPT_LORA_CONFIG)
        model = get_peft_model(model, lora_config_dapt)

        dataset_dapt = load_dataset("text", data_files=self.domain_corpus_path, split="train")
        def tokenize_dapt(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512)
        tokenized_dataset_dapt = dataset_dapt.map(tokenize_dapt, batched=True, remove_columns=["text"])

        training_args_dapt = TrainingArguments(output_dir="models/tiny_test/dapt", max_steps=1)
        trainer_dapt = Trainer(model=model, args=training_args_dapt, train_dataset=tokenized_dataset_dapt)
        trainer_dapt.train()
        model.save_pretrained("models/tiny_test/dapt")

        # --- 2. Train Task Adapter ---
        lora_config_sft = LoraConfig(**config.SFT_LORA_CONFIG)
        model = get_peft_model(model, lora_config_sft)

        dataset_sft = Dataset.from_list(self.tiny_task_data)
        def tokenize_sft(examples):
            full_texts = [f"Context: {c}\n\nQuestion: {q}\n\nAnswer: {a}" for c, q, a in zip(examples['context'], examples['question'], examples['answer'])]
            tokenized_outputs = tokenizer(full_texts, truncation=True, max_length=512, padding="max_length")
            tokenized_outputs["labels"] = [x[:] for x in tokenized_outputs["input_ids"]]
            return tokenized_outputs
        tokenized_dataset_sft = dataset_sft.map(tokenize_sft, batched=True)

        training_args_sft = TrainingArguments(output_dir="models/tiny_test/sft", max_steps=1)
        trainer_sft = Trainer(model=model, args=training_args_sft, train_dataset=tokenized_dataset_sft)
        trainer_sft.train()
        model.save_pretrained("models/tiny_test/sft")

        # --- 3. Run Inference ---
        model = PeftModel.from_pretrained(model, "models/tiny_test/dapt", adapter_name="domain_adapter")
        model.load_adapter("models/tiny_test/sft", adapter_name="task_adapter")

        prompt = "Context: The sky is blue.\n\nQuestion: What color is the sky?\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.assertIn(prompt, response)
        self.assertTrue(len(response) > len(prompt))

    def tearDown(self):
        """
        Clean up the temporary data and model files.
        """
        import shutil
        if os.path.exists("data/tiny_test"):
            shutil.rmtree("data/tiny_test")
        if os.path.exists("models/tiny_test"):
            shutil.rmtree("models/tiny_test")

if __name__ == "__main__":
    unittest.main()
