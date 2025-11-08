# Modular Fine-Tuning with Domain and Task Adapters

This project provides a complete, production-ready recipe for fine-tuning large language models using a modular, adapter-based approach. It follows a professional methodology of **Research -> Implement -> Verify**, ensuring that the solution is robust, maintainable, and well-documented. This implementation uses the powerful `MiniMaxAI/MiniMax-M2` model.

## Methodology

The core pattern implemented here is:

1.  **Domain Adapter (DAPT):** A LoRA adapter is trained on an unlabeled, in-domain corpus using a causal language modeling objective. This injects domain-specific knowledge into the model without modifying the base weights.
2.  **Task Adapter (PEFT):** A separate LoRA adapter is trained on a small, labeled dataset for a specific downstream task (in this case, question answering).
3.  **Composition at Inference:** At inference time, the base model is loaded, and both the domain and task adapters are dynamically composed to generate a response.

This project is grounded in established best practices and academic research. For a detailed breakdown of the methodology, tooling, and risk analysis, please see the `decision_document.md`.

## Project Structure

```
.
├── .gitignore
├── config.py              # Central configuration for the project
├── data/                  # Placeholder for data (ignored by git)
├── decision_document.md   # Research and planning document
├── models/                # Placeholder for trained models (ignored by git)
├── scripts/
│   ├── prepare_data.py          # Script to download and prepare data
│   ├── train_domain_adapter.py  # Script to train the domain adapter
│   ├── train_task_adapter.py    # Script to train the task adapter
│   ├── inference.py             # Script to run inference with composed adapters
│   └── benchmark.py           # Script to benchmark inference latency
├── tests/
│   ├── test_pipeline.py       # Unit tests for the pipeline components
│   └── test_integration.py    # Integration test for the end-to-end pipeline
└── utils/
    ├── logger.py            # Structured logger configuration
    └── exceptions.py        # Custom exception classes
```

## How to Run

### 1. Setup the Environment

First, create a Python virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure the Project

All parameters for the project are managed in the `config.py` file. Before running any scripts, review this file and adjust the parameters as needed (e.g., model names, data paths, training hyperparameters).

### 3. Prepare the Data

Run the data preparation script to download and process the placeholder datasets:

```bash
python scripts/prepare_data.py
```

### 4. Train the Adapters

To train the domain and task adapters, use `accelerate` for distributed training. First, configure `accelerate` for your environment:

```bash
accelerate config
```

Then, launch the training scripts:

```bash
# Train the domain adapter
accelerate launch scripts/train_domain_adapter.py

# Train the task adapter
accelerate launch scripts/train_task_adapter.py
```

### 5. Run Inference and Benchmarking

The inference script can be run with a custom prompt or be used to merge the adapters into a single model.

```bash
# Run a sample inference with a custom prompt
python scripts/inference.py --prompt "Your prompt here"

# Merge the adapters and save the new model
python scripts/inference.py --merge_path "models/merged_model"

# Run the performance benchmark
python scripts/benchmark.py
```

### 6. Run Verification Suite

To run the full verification suite, including unit and integration tests, use the following command:

```bash
python -m unittest discover tests
```

## Development Workflow

This project uses `black` for code formatting and `ruff` for linting. To ensure your contributions are well-formatted and free of linting errors, please run the following commands before submitting a pull request:

```bash
# Format the code
black .

# Lint the code and automatically fix issues
ruff check --fix .
```
