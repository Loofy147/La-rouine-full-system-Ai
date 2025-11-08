# Decision Document: Modular Fine-Tuning Strategy

This document outlines the decisions made for implementing a modular fine-tuning strategy using Domain-Adaptive Pretraining (DAPT) and Parameter-Efficient Fine-Tuning (PEFT) with LoRA/QLoRA.

## 1. Base Model and Tokenizer

*   **Base Model:** `MiniMaxAI/MiniMax-M2`
    *   **Reasoning:** This is a powerful, open-source Mixture-of-Experts (MoE) model with strong performance in coding and agentic tasks. It is a suitable choice for demonstrating the adapter-based fine-tuning workflow on a modern, high-performance architecture.
*   **Tokenizer:** The default `MiniMaxAI/MiniMax-M2` tokenizer will be used.
    *   **Reasoning:** For this initial implementation, we will assume the domain-specific vocabulary is sufficiently covered by the base model's tokenizer. If significant out-of-vocabulary tokens are discovered during data analysis, we will revisit the decision and consider tokenizer extension.

## 2. Data Strategy

*   **Domain-Adaptive Pretraining (DAPT) Corpus:**
    *   **Source:** A placeholder dataset will be used, simulating a large corpus of unlabeled domain-specific text. For this project, we will use the `wikitext` dataset (`wikitext-2-raw-v1`) as a stand-in to demonstrate the process.
    *   **Preprocessing:** The script will include steps for basic cleaning and normalization. PII removal will be included as a placeholder step.
*   **Task-Specific Fine-Tuning (SFT) Data:**
    *   **Source:** A small, labeled dataset for a specific task (e.g., question answering, summarization). We will use the `squad` dataset as a placeholder to demonstrate the SFT process.
    *   **Size:** We will start with a small subset of 1,000 examples to facilitate quick iteration, as recommended.
*   **Data Usage Constraints:** All data used will be from public, open-source datasets. No private or proprietary data will be used in this implementation.

## 3. Compute Budget and Training Strategy

*   **Compute:** The implementation will be designed to run on a single GPU with at least 24GB of VRAM, which is common in cloud environments (e.g., T4, V100).
*   **Training Strategy:** We will use QLoRA (4-bit quantization) via the `bitsandbytes` library to minimize memory footprint during training.
*   **Workflow:**
    1.  The base model will be loaded in 4-bit precision.
    2.  A domain adapter will be trained on the unlabeled corpus using a language modeling objective.
    3.  A separate task adapter (LoRA) will be trained on the labeled dataset.
    4.  At inference, the base model, domain adapter, and task adapter will be composed dynamically.

## 4. Rollback Policy

*   **Modular Adapters:** The use of separate adapters for domain and task allows for a simple and effective rollback policy.
*   **Procedure:** If a task adapter introduces regressions or undesirable behavior, it can be disabled at inference time, reverting the model's behavior to the base model + domain adapter. This provides a safe and immediate fallback without requiring a full redeployment. The domain adapter can be similarly rolled back. Versioning will be managed in an artifact store.
