import os
from datasets import load_dataset
import re

def anoynimize_pii(text):
    """
    A simple PII anonymizer that replaces email addresses and phone numbers.
    This is a placeholder and should be replaced with a more robust solution.
    """
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\d{3}-\d{3}-\d{4}', '[PHONE]', text)
    return text

def prepare_domain_corpus(output_dir: str):
    """
    Downloads the wikitext dataset, cleans it, and saves it as a placeholder
    for the domain-specific corpus.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Clean and anonymize the data
    cleaned_texts = [anoynimize_pii(text) for text in dataset['text'] if text.strip()]

    # Save the cleaned data to a file
    output_path = os.path.join(output_dir, "domain_corpus.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for text in cleaned_texts:
            f.write(text + "\n")
    print(f"Domain corpus saved to {output_path}")

def prepare_task_data(output_dir: str, num_examples=1000):
    """
    Downloads the SQuAD dataset, formats it for question answering,
    and saves a subset as a placeholder for the task-specific data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset
    dataset = load_dataset("squad", split="train")

    # Format the data
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

    # Save the formatted data to a JSON file
    import json
    output_path = os.path.join(output_dir, "task_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2)
    print(f"Task data saved to {output_path}")

if __name__ == "__main__":
    prepare_domain_corpus("data/domain_corpus")
    prepare_task_data("data/task_data")
