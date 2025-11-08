# Human Evaluation Plan

This document outlines a plan for the human evaluation of the fine-tuned models. The goal of this evaluation is to assess the quality of the model's responses and compare the performance of the different model versions.

## Evaluation Strategy

We will conduct a blind A/B/C comparison of the following three model versions:

1.  **Model A:** The base `MiniMaxAI/MiniMax-M2` model.
2.  **Model B:** The base model + the domain adapter.
3.  **Model C:** The base model + the domain adapter + the task adapter.

Evaluators will be presented with a prompt and the responses from two of the three models, without knowing which model generated which response. They will then be asked to rate the responses based on a set of criteria.

## Evaluation Rubric

For each response, evaluators will provide a score from 1 to 5 (1 = Very Poor, 5 = Excellent) for each of the following criteria:

*   **Factuality:** Is the information in the response accurate and correct?
*   **Helpfulness:** Does the response directly and effectively answer the user's question?
*   **Clarity:** Is the response well-written, easy to understand, and free of grammatical errors?
*   **Conciseness:** Is the response as short as possible while still being complete and helpful?

In addition to the numerical scores, evaluators will be asked to provide a qualitative assessment of each response, highlighting its strengths and weaknesses.

## Evaluation Set

The evaluation will be conducted on a small, curated set of 100-200 prompts. These prompts will be designed to cover a range of topics and question types, and will include:

*   **In-domain questions:** Questions that are specific to the domain of the DAPT corpus.
*   **Out-of-domain questions:** Questions that are not related to the domain of the DAPT corpus.
*   **Hallucination probes:** Questions with known ground truth, designed to test the model's factuality.
*   **Safety prompts:** "Red-team" prompts designed to test the model's safety and alignment.

## Inter-Rater Reliability

To ensure the consistency and reliability of the evaluation results, we will measure the inter-rater reliability using Krippendorff's alpha or Cohen's kappa. A high level of agreement between raters will give us confidence in the validity of our findings.
