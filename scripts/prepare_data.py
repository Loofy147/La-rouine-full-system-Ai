# scripts/prepare_data.py

import os
import re
import json
from datasets import load_dataset
from utils.logger import get_logger
from utils.exceptions import DataPreparationError
import config

logger = get_logger(__name__)

def anoynimize_pii(text: str) -> str:
    """
    A simple PII anonymizer that replaces email addresses and phone numbers.
    This is a placeholder and should be replaced with a more robust solution.

    Args:
        text (str): The text to anonymize.

    Returns:
        str: The anonymized text.
    """
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\d{3}-\d{3}-\d{4}', '[PHONE]', text)
    return text

def prepare_domain_corpus():
    """
    Downloads the wikitext dataset, cleans it, and saves it as a placeholder
    for the domain-specific corpus.
    """
    try:
        logger.info("Preparing domain corpus...")
        output_dir = os.path.dirname(config.DOMAIN_CORPUS_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        cleaned_texts = [anoynimize_pii(text) for text in dataset['text'] if text.strip()]

        with open(config.DOMAIN_CORPUS_PATH, "w", encoding="utf-8") as f:
            for text in cleaned_texts:
                f.write(text + "\n")
        logger.info(f"Domain corpus saved to {config.DOMAIN_CORPUS_PATH}")
    except Exception as e:
        raise DataPreparationError(f"Failed to prepare domain corpus: {e}")

def prepare_task_data(num_examples=1000):
    """
    Downloads the SQuAD dataset, formats it for question answering,
    and saves a subset as a placeholder for the task-specific data.

    Args:
        num_examples (int): The number of examples to prepare.
    """
    try:
        logger.info("Preparing task data...")
        output_dir = os.path.dirname(config.TASK_DATA_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = load_dataset("squad", split="train")
        formatted_data = []
        for example in dataset.select(range(num_examples)):
            context = example['context']
            question = example['question']
            answer = example['answers']['text'][0]
            formatted_data.append({
                "context": anoynimize_pii(context),
                "question": anoynimize_pii(question),
                "answer": anoynimize_pii(answer)
            })

        with open(config.TASK_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=2)
        logger.info(f"Task data saved to {config.TASK_DATA_PATH}")
    except Exception as e:
        raise DataPreparationError(f"Failed to prepare task data: {e}")

if __name__ == "__main__":
    try:
        prepare_domain_corpus()
        prepare_task_data()
    except DataPreparationError as e:
        logger.error(e)
