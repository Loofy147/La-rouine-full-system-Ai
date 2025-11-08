# tests/test_utils.py
import os
import unittest
from unittest.mock import MagicMock, patch

from peft import PeftModel

from utils.exceptions import (
    DataPreparationError,
    InferenceError,
    ModelLoadingError,
    ModularFineTuningError,
    TrainingError,
)
from utils.logger import get_logger
from utils.model_utils import compose_adapters, load_model_and_tokenizer


class TestUtils(unittest.TestCase):
    def test_logger_initialization(self):
        """
        Tests that the logger is initialized correctly.
        """
        logger = get_logger(__name__)
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, __name__)

    def test_custom_exceptions(self):
        """
        Tests that the custom exception hierarchy is set up correctly.
        """
        self.assertTrue(issubclass(DataPreparationError, ModularFineTuningError))
        self.assertTrue(issubclass(ModelLoadingError, ModularFineTuningError))
        self.assertTrue(issubclass(TrainingError, ModularFineTuningError))
        self.assertTrue(issubclass(InferenceError, ModularFineTuningError))

    @patch("utils.model_utils.AutoModelForCausalLM.from_pretrained")
    @patch("utils.model_utils.AutoTokenizer.from_pretrained")
    def test_load_model_and_tokenizer(
        self, mock_tokenizer_from_pretrained, mock_model_from_pretrained
    ):
        """
        Tests that the model and tokenizer are loaded correctly.
        """
        mock_model_from_pretrained.return_value = MagicMock()
        mock_tokenizer_from_pretrained.return_value = MagicMock()

        model, tokenizer = load_model_and_tokenizer("test_model", "test_tokenizer", True)

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        mock_model_from_pretrained.assert_called_once()
        mock_tokenizer_from_pretrained.assert_called_once()

    @patch("peft.PeftModel.from_pretrained")
    @patch("os.path.exists", return_value=True)
    def test_compose_adapters(self, mock_exists, mock_from_pretrained):
        """
        Tests that the adapters are composed correctly.
        """
        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_from_pretrained.return_value = mock_peft_model

        composed_model = compose_adapters(mock_base_model, "domain", "task")

        self.assertIsNotNone(composed_model)
        mock_from_pretrained.assert_called_once_with(
            mock_base_model, "domain", adapter_name="domain_adapter"
        )
        mock_peft_model.load_adapter.assert_called_once_with(
            "task", adapter_name="task_adapter"
        )

    @patch("peft.PeftModel.from_pretrained")
    @patch("os.path.exists", return_value=False)
    def test_compose_adapters_fallback(self, mock_exists, mock_from_pretrained):
        """
        Tests that the adapter composition falls back gracefully.
        """
        mock_base_model = MagicMock()
        mock_from_pretrained.return_value = MagicMock()

        composed_model = compose_adapters(mock_base_model, "domain", "task")

        self.assertIsNotNone(composed_model)
        mock_from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
