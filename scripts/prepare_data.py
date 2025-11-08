# scripts/prepare_data.py
"""
Data preparation script for the modular fine-tuning project.

This script provides functions to download, clean, and format the domain
corpus and the task-specific dataset. It includes a placeholder for PII
anonymization and saves the processed data to the paths specified in the
project configuration.
"""
import json
import os
import re

from datasets import load_dataset

import config
from utils.exceptions import DataPreparationError
from utils.logger import get_logger

logger = get_logger(__name__)


def anoynimize_pii(text: str) -> str:
    """
    A simple PII anonymizer to replace email addresses and phone numbers.

    Note: This is a basic placeholder and should be replaced with a more
    robust and comprehensive PII detection and anonymization solution for

    production use.

    Args:
        text: The input string to anonymize.

    Returns:
        The anonymized string.
    """
    text = re.sub(r"\S+@\S+", "[EMAIL]", text)
    text = re.sub(r"\d{3}-\d{3}-\d{4}", "[PHONE]", text)
    return text


def prepare_domain_corpus() -> None:
    """
    Downloads and prepares the domain-specific corpus.

    This function uses the wikitext dataset as a placeholder. It performs basic
    cleaning and PII anonymization before saving the corpus as a plain text
    file, with one document per line.

    Raises:
        DataPreparationError: If any issue occurs during data processing.
    """
    try:
        logger.info("Preparing domain corpus...")
        output_dir = os.path.dirname(config.DOMAIN_CORPUS_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        cleaned_texts = [
            anoynimize_pii(text) for text in dataset["text"] if text.strip()
        ]

        with open(config.DOMAIN_CORPUS_PATH, "w", encoding="utf-8") as f:
            for text in cleaned_texts:
                f.write(text + "\n")
        logger.info(f"Domain corpus saved to {config.DOMAIN_CORPUS_PATH}")
    except Exception as e:
        raise DataPreparationError(f"Failed to prepare domain corpus: {e}") from e


def prepare_task_data(num_examples: int = 1000) -> None:
    """
    Downloads and prepares the task-specific supervised fine-tuning data.

    This function uses the SQuAD dataset as a placeholder. It formats the data
    into a structured JSON file containing context, question, and answer
    triplets, after performing basic PII anonymization.

    Args:
        num_examples: The number of examples to prepare from the dataset.

    Raises:
        DataPreparationError: If any issue occurs during data processing.
    """
    try:
        logger.info("Preparing task data...")
        output_dir = os.path.dirname(config.TASK_DATA_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = load_dataset("squad", split="train")
        formatted_data = []
        for example in dataset.select(range(num_examples)):
            context = example["context"]
            question = example["question"]
            answer = example["answers"]["text"][0]
            formatted_data.append(
                {
                    "context": anoynimize_pii(context),
                    "question": anoynimize_pii(question),
                    "answer": anoynimize_pii(answer),
                }
            )

        with open(config.TASK_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=2)
        logger.info(f"Task data saved to {config.TASK_DATA_PATH}")
    except Exception as e:
        raise DataPreparationError(f"Failed to prepare task data: {e}") from e


if __name__ == "__main__":
    try:
        prepare_domain_corpus()
        prepare_task_data()
    except DataPreparationError as e:
        logger.error(e)
