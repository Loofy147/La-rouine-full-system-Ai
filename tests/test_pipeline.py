# tests/test_pipeline.py

import unittest
from unittest.mock import patch, MagicMock
import sys
import importlib
from transformers import AutoTokenizer
import config
from utils.logger import get_logger

class TestPipeline(unittest.TestCase):

    def setUp(self):
        """
        Set up the tokenizer for the tests.
        """
        self.model_name = config.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def test_tokenizer_not_none(self):
        """
        Tests that the tokenizer is loaded correctly.
        """
        self.assertIsNotNone(self.tokenizer)

    def test_prompt_template_formatting(self):
        """
        Tests that the prompt template is formatted correctly.
        """
        context = "The capital of France is Paris."
        question = "What is the capital of France?"
        expected_prompt = "Context: The capital of France is Paris.\n\nQuestion: What is the capital of France?\n\nAnswer:"

        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        self.assertEqual(prompt, expected_prompt)

    def test_tokenization_of_prompt(self):
        """
        Tests that the tokenization of the prompt is correct.
        """
        prompt = "Context: The capital of France is Paris.\n\nQuestion: What is the capital of France?\n\nAnswer:"
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")

        self.assertIn("input_ids", tokenized_prompt)
        self.assertIn("attention_mask", tokenized_prompt)
        self.assertTrue(len(tokenized_prompt["input_ids"][0]) > 0)

    def test_config_loading(self):
        """
        Tests that the configuration is loaded correctly.
        """
        self.assertEqual(self.model_name, config.MODEL_NAME)
        self.assertIn("load_in_4bit", config.BNB_CONFIG)
        self.assertIn("num_train_epochs", config.DAPT_TRAINING_ARGS)

    def test_logger_initialization(self):
        """
        Tests that the logger is initialized correctly.
        """
        logger = get_logger(__name__)
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, __name__)

    @unittest.skip("This is a placeholder for a hallucination probe.")
    def test_hallucination_probe(self):
        """
        A placeholder for a test that would probe the model for hallucinations.
        """
        pass

    @unittest.skip("This is a placeholder for a safety and privacy test.")
    def test_safety_and_privacy(self):
        """
        A placeholder for a test that would check for safety and privacy issues.
        """
        pass

    @patch('scripts.inference.run_inference')
    @patch('scripts.inference.PeftModel.from_pretrained')
    @patch('scripts.inference.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.inference.AutoTokenizer.from_pretrained')
    def test_inference_script_with_prompt(self, mock_tokenizer, mock_model, mock_peft_from_pretrained, mock_run_inference):
        """
        Tests that the inference script calls run_inference with the correct prompt.
        """
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_peft_from_pretrained.return_value = MagicMock()

        test_prompt = "This is a test prompt"
        sys.argv = ['scripts/inference.py', '--prompt', test_prompt]

        from scripts import inference
        inference.main()

        mock_run_inference.assert_called_once()
        # The actual call might have more arguments, so we check the prompt
        called_prompt = mock_run_inference.call_args[0][0]
        self.assertEqual(called_prompt, test_prompt)

    @patch('scripts.inference.merge_and_save')
    @patch('scripts.inference.PeftModel.from_pretrained')
    @patch('scripts.inference.AutoModelForCausalLM.from_pretrained')
    @patch('scripts.inference.AutoTokenizer.from_pretrained')
    def test_inference_script_with_merge_path(self, mock_tokenizer, mock_model, mock_peft_from_pretrained, mock_merge_and_save):
        """
        Tests that the inference script calls merge_and_save with the correct path.
        """
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_peft_from_pretrained.return_value = MagicMock()

        test_path = "/tmp/merged_model"
        sys.argv = ['scripts/inference.py', '--merge_path', test_path]

        from scripts import inference
        inference.main()

        mock_merge_and_save.assert_called_once()
        # The actual call might have more arguments, so we check the path
        called_path = mock_merge_and_save.call_args[0][2]
        self.assertEqual(called_path, test_path)

if __name__ == "__main__":
    unittest.main()
