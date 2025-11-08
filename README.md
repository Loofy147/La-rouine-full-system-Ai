# Modular Fine-Tuning with Domain and Task Adapters

This project provides a complete, runnable recipe for fine-tuning large language models using a modular, adapter-based approach. It follows the methodology of first training a domain adapter on an unlabeled corpus (Domain-Adaptive Pretraining or DAPT) and then training separate task adapters on labeled datasets (Parameter-Efficient Fine-Tuning or PEFT). This implementation uses the powerful `MiniMaxAI/MiniMax-M2` model. This approach is modular, cost-effective, and easy to iterate on.

## Methodology

The core pattern implemented here is:

1.  **Domain Adapter (DAPT):** A LoRA adapter is trained on an unlabeled, in-domain corpus using a causal language modeling objective. This injects domain-specific knowledge into the model without modifying the base weights.
2.  **Task Adapter (PEFT):** A separate LoRA adapter is trained on a small, labeled dataset for a specific downstream task (in this case, question answering).
3.  **Composition at Inference:** At inference time, the base model is loaded, and both the domain and task adapters are dynamically composed to generate a response. This allows for flexible combination and easy rollbacks.

## Project Structure

```
.
├── .gitignore
├── data/                  # Placeholder for data (ignored by git)
├── decision_document.md   # Initial research and planning document
├── models/                # Placeholder for trained models (ignored by git)
├── scripts/
│   ├── prepare_data.py          # Script to download and prepare data
│   ├── train_domain_adapter.py  # Script to train the domain adapter
│   ├── train_task_adapter.py    # Script to train the task adapter
│   └── inference.py             # Script to run inference with composed adapters
└── tests/
    └── test_pipeline.py       # Verification suite for the pipeline
```

## How to Run

### 1. Setup the Environment

First, create a Python virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Note: A `requirements.txt` file will be created in a subsequent step.)*

### 2. Prepare the Data

Run the data preparation script to download and process the placeholder datasets:

```bash
python scripts/prepare_data.py
```

This will download the `wikitext` and `squad` datasets and save them to the `data/` directory.

### 3. Train the Domain Adapter

To train the domain adapter, run the following command in a GPU environment:

```bash
python scripts/train_domain_adapter.py
```

This will train a domain adapter on the unlabeled corpus and save it to `models/domain_adapter`.

### 4. Train the Task Adapter

To train the task adapter, run the following command in a GPU environment:

```bash
python scripts/train_task_adapter.py
```

This will train a task adapter on the labeled dataset and save it to `models/task_adapter`.

### 5. Run Inference

To run inference with the composed adapters, execute the following script in a GPU environment:

```bash
python scripts/inference.py
```

This will load the base model, the domain adapter, and the task adapter, and then generate a response to a sample prompt.

### 6. Run Tests

To run the verification suite, use the following command:

```bash
python -m unittest tests/test_pipeline.py
```
