# utils/exceptions.py
"""
Custom exception classes for the modular fine-tuning project.

This module defines a hierarchy of custom exceptions to facilitate more specific
and robust error handling throughout the application. All custom exceptions
inherit from a common base class, ModularFineTuningError.
"""


class ModularFineTuningError(Exception):
    """Base class for all custom exceptions in this project."""

    pass


class DataPreparationError(ModularFineTuningError):
    """
    Exception raised for errors during the data preparation phase.

    This may include issues with downloading, cleaning, or formatting the
    domain corpus or the task-specific dataset.
    """

    pass


class ModelLoadingError(ModularFineTuningError):
    """
    Exception raised for errors when loading a model or tokenizer.

    This can occur if the model name is incorrect, the required libraries are not
    installed, or there are issues with the quantization configuration.
    """

    pass


class TrainingError(ModularFineTuningError):
    """
    Exception raised for errors that occur during the model training process.

    This includes issues related to the Trainer, dataset preparation, or other
    unexpected problems during the training loop.
    """

    pass


class InferenceError(ModularFineTuningError):
    """
    Exception raised for errors during the inference phase.

    This can be triggered by issues with loading adapters, generating text, or
    other problems that occur when running the model for predictions.
    """

    pass
