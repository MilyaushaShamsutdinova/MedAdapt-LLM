# MedAdapt-LLM

## Project Description

This project's primary objective is **to explore methods for adapting LLM to the medical domain and assess their effectiveness**. We decide to implement the following methods:

1. **Retrieval-Augmented Generation (RAG)** – enriching prompts with relevant medical information.

2. **Supervised Fine-Tuning (SFT)** – training the model on a medical question-answer dataset.

3. **Multi-Step Fine-Tuning**:

    - Step 1: **Continual Pretraining** – infusing medical knowledge.

    - Step 2: **SFT** – refining instruction-following abilities.

By evaluating these approaches, we aim to determine the most effective strategy for domain-specific LLM adaptation. The insights gained will contribute to understanding how different fine-tuning techniques impact model performance and whether hybrid approaches like RAG can enhance domain relevance without excessive computational overhead.

## Datasets & model

**Model**: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) \
We selected this model because it is a recently released, trending model that leverages modern NLP insights, demonstrates strong benchmark performance, and offers advanced reasoning capabilities. Additionally, its compact size is ideal for training it on the Kaggle platform, balancing performance and efficiency.

We selected datasets tailored for medical adaptation:

- [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) – open-sourced the medical reasoning dataset for SFT, built on medical verifiable problems and an LLM verifier. It contains of triples of question, reasoning, and response. We use this dataset for the SFT phase.

- [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) – dataset of the chunked snippets from the Textbooks corpus used in MedRAG. It's perfect for building our own medical RAG. Moreover, we will use it to serve as a corpus for continual pretraining.

## Implementation details

### RAG

RAG is implementation is in the [rag folder](https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/tree/main/src/rag).

- `db.py` - a class for loading the dataset into a vector store and querying the vector store. The dataset [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) was used as the source of relevant medical documents. The dataset of size 127,847 snippets was loaded into a vector database in about 8 minutes, requiring ~1 GB of memory.

- `llm.py` - a class for [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) inference.

- `rag.py` - a class that implements basic RAG logic.

### SFT

The SFT training notebook is available [here](https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/notebooks/sft_training.ipynb).

We fine-tuned the (4-bit) quantized [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) on a subset of the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset (first 20,000 samples) using QLoRA.

SFT training setting:

| Hyperparameter              | Value              |
|-----------------------------|--------------------|
| LoRA rank                   | 16                 |
| LoRA alpha                  | 32                 |
| LoRA dropout                | 0.05               |
| Number of epochs            | 1                  |
| Learning rate (lr)          | 1e-4               |
| Learning rate scheduler     | cosine             |
| Optimizer                   | adamw_torch_fused  |
| Weight decay                | 0.01               |
| Warmup steps                | 300                |
| Per device train batch size | 2                  |
| Gradient accumulation steps | 4                  |
| FP16                        | True               |
| BF16                        | False              |
| seed                        | 4242               |


Training loss graphs:


<p align="center">
  <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/sft_train_loss.png?raw=true" width="45%" />
  <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/sft_eval_loss.png?raw=true" width="45%" />
</p>

Training took around 11 hours on Kaggle with GPU P100. The fine-tuned model is available on HF Hub: [MilyaShams/DeepSeek-R1-Distill-Qwen-1.5B-Medical](https://huggingface.co/MilyaShams/DeepSeek-R1-Distill-Qwen-1.5B-Medical).

### Multi-Step Fine-Tuning

Work in progress...

## Evaluation

Performance will be assessed using benchmarks from the [Open Medical-LLM leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard) to fairly compare methods.

## How to run

0. Create a virtual environment.
1. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
2. Install the package in editable mode:  
   ```
   pip install -e .
   ```
3. Define `HF_TOKEN` in secrets.

