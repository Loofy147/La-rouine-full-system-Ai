# tests/test_utils.py
import unittest
from unittest.mock import MagicMock, patch

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

        model, tokenizer = load_model_and_tokenizer()

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        mock_model_from_pretrained.assert_called_once()
        mock_tokenizer_from_pretrained.assert_called_once()

    @patch("utils.model_utils.PeftModel")
    def test_compose_adapters(self, mock_peft_model):
        """
        Tests that the adapters are composed correctly.
        """
        mock_base_model = MagicMock()
        mock_peft_model.from_pretrained.return_value = MagicMock()

        composed_model = compose_adapters(mock_base_model)

        self.assertIsNotNone(composed_model)
        mock_peft_model.from_pretrained.assert_called_once()
        composed_model.load_adapter.assert_called_once()


if __name__ == "__main__":
    unittest.main()
