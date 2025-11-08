# utils/exceptions.py

class ModularFineTuningError(Exception):
    """Base class for exceptions in this project."""
    pass

class DataPreparationError(ModularFineTuningError):
    """Exception raised for errors in the data preparation process."""
    pass

class ModelLoadingError(ModularFineTuningError):
    """Exception raised for errors when loading a model or tokenizer."""
    pass

class TrainingError(ModularFineTuningError):
    """Exception raised for errors during the training process."""
    pass

class InferenceError(ModularFineTuningError):
    """Exception raised for errors during inference."""
    pass
