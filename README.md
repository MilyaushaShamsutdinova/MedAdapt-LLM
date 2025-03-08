# MedAdapt-LLM

## Project Description

Adapting LLM to the medical domain through supervised fine-tuning (SFT), retrieval-augmented generation (RAG), and multistep fine-tuning to enhance domain knowledge and performance.

This project fine-tunes DeepSeek-R1-Distill-Qwen-1.5B for medical reasoning using LoRA or QLoRA. We test three adaptation methods:

1. **Retrieval-Augmented Generation (RAG)** – enriching prompts with relevant medical information.

2. **Supervised Fine-Tuning (SFT)** – training the model on a medical question-answer dataset.

3. **Multi-Step Fine-Tuning**:

    - **Continual Pretraining** – infusing medical knowledge.

    - **SFT** – refining instruction-following abilities.

## Datasets Used

We selected datasets tailored for medical adaptation:

- [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) – enhances reasoning skills through instruction fine-tuning.

- [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) – provides domain-specific knowledge for RAG and continual pretraining.

## Evaluation

Performance is assessed using the [Open Medical-LLM leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard) to ensure robustness.

## Fine-Tuned Model

The final model is available on HF Hub: [MilyaShams/DeepSeek-R1-Distill-Qwen-1.5B-Medical](https://huggingface.co/MilyaShams/DeepSeek-R1-Distill-Qwen-1.5B-Medical).

## Usage
