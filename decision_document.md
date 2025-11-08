# Decision Document: Modular Fine-Tuning Strategy

This document outlines the decisions made for implementing a modular fine-tuning strategy using Domain-Adaptive Pretraining (DAPT) and Parameter-Efficient Fine-Tuning (PEFT) with LoRA/QLoRA.

## 1. Base Model and Tokenizer

*   **Base Model:** `MiniMaxAI/MiniMax-M2`
    *   **Reasoning:** This is a powerful, open-source Mixture-of-Experts (MoE) model with strong performance in coding and agentic tasks. It is a suitable choice for demonstrating the adapter-based fine-tuning workflow on a modern, high-performance architecture.
*   **Tokenizer:** The default `MiniMaxAI/MiniMax-M2` tokenizer will be used.
    *   **Reasoning:** For this initial implementation, we will assume the domain-specific vocabulary is sufficiently covered by the base model's tokenizer. If significant out-of-vocabulary tokens are discovered during data analysis, we will revisit the decision and consider tokenizer extension.

## 2. Tooling Justification

*   **`transformers`:** This library from Hugging Face is the de-facto standard for working with state-of-the-art language models. It provides a unified API for loading, configuring, and training a wide range of models.
*   **`peft`:** The Parameter-Efficient Fine-Tuning (PEFT) library from Hugging Face provides a simple and effective way to apply techniques like LoRA and QLoRA to large models. It is well-maintained and integrates seamlessly with the `transformers` library.
*   **`bitsandbytes`:** This library is essential for QLoRA, as it provides the 4-bit quantization and dequantization operations that make it possible to fine-tune large models on consumer-grade hardware.
*   **`datasets`:** This library from Hugging Face simplifies the process of downloading, processing, and iterating over large datasets.

## 3. Pattern Identification

*   **Adapter Pattern:** The core of this project is the use of the Adapter Pattern, where we "attach" new, smaller modules (the LoRA adapters) to a larger, frozen model. This allows us to modify the model's behavior without altering its original weights.
*   **Modular Design:** The separation of the domain adapter and the task adapter is a form of modular design. This allows us to reuse the domain adapter across multiple tasks and to update the task adapter independently of the domain adapter.
*   **Strategy Pattern:** The ability to dynamically compose the adapters at inference time is an example of the Strategy Pattern. We can change the model's behavior by selecting a different combination of adapters (or no adapters at all) without changing the underlying code.

## 4. Risk Analysis

*   **Catastrophic Forgetting:** Heavy domain-adaptive pretraining can lead to catastrophic forgetting, where the model forgets some of its general-purpose knowledge. The use of a lightweight LoRA adapter for DAPT helps to mitigate this risk by keeping the base model's weights frozen.
*   **Overfitting:** Fine-tuning on a small, labeled dataset can lead to overfitting. This is mitigated by using a small LoRA rank and a dropout layer in the task adapter.
*   **Model Performance:** The performance of the `MiniMaxAI/MiniMax-M2` model, while strong, may not be suitable for all tasks. A thorough evaluation is necessary to determine if it meets the specific requirements of a given use case.
*   **Dependency Risk:** The project relies on several open-source libraries. A change in any of these libraries could potentially break the project. This is a standard risk in software development and is mitigated by pinning dependency versions in a `requirements.txt` file.

## 5. Citations

*   **LoRA:** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
*   **QLoRA:** Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv preprint arXiv:2305.14314*.
*   **DAPT:** Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *arXiv preprint arXiv:2004.10964*.
*   **Hugging Face PEFT:** [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)
*   **bitsandbytes:** [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
