# MedAdapt-LLM

## Project Description

This project's primary objective is to **explore methods for adapting LLM to the medical domain and assess their effectiveness**. We decide to experiment with the following methods:

1. **Retrieval-Augmented Generation (RAG)** – enriching prompts with relevant medical information.

2. **Supervised Fine-Tuning (SFT)** – training the model on a medical question-answer dataset.

3. **Multi-Step Fine-Tuning**:

    - step 1: **Continual Pretraining** – infusing medical knowledge.

    - step 2: **SFT** – refining instruction-following abilities.

By evaluating these approaches, we aim to determine the most effective strategy for domain-specific LLM adaptation. The insights gained will contribute to understanding how different fine-tuning techniques impact model performance and whether hybrid approaches like RAG can enhance domain relevance without excessive computational overhead.

## Datasets & model

**Model**: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).  
We chose this model to work with because it provides a balance between performance and computational efficiency, making it a suitable choice for domain adaptation in the medical field.

We selected datasets tailored for medical adaptation:

- [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) – provides triples of question, reasoning, and response. We use this dataset for the SFT phase.

- [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) – provides chunked medical texts that are convenient for RAG. This dataset is used to create a knowledge database for RAG and also serves as a corpus for continual pretraining.

## Implementation details

### RAG

RAG is implemented in the [rag folder](https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/tree/main/src/rag).

- `db.py` - a class for loading the dataset into a vector store and querying the vector store. The dataset [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) was used as the source of relevant medical documents. The dataset of size 127,847 snippets was loaded into a vector database in about 8 minutes, requiring ~1 GB of memory.

- `llm.py` - a class for inference with [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).

- `rag.py` - a class that implements basic RAG logic.

### SFT

The SFT training notebook is available [here](https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/notebooks/sft_training.ipynb).

We fine-tuned the quantized (4-bit) [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) on a subset of the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset (first 20,000 samples) using QLoRA.

The training hyperparameters used:

| Parameter                   | Value               |
|-----------------------------|---------------------|
| Learning rate (lr)          | 1e-4               |
| Number of epochs            | 1                  |
| Optimizer                   | adamw_torch_fused  |
| Batch size                  | 2                  |
| Gradient accumulation steps | 4                  |
| Weight decay                | 0.01               |
| Warmup steps                | 300                |
| Logging steps               | 100                |
| Save steps                  | 200                |
| FP16                        | True               |
| BF16                        | False              |
| Per device train batch size | 2                  |
| Per device eval batch size  | 2                  |
| Learning rate scheduler     | Cosine             |

The fine-tuned model is available on HF Hub: [MilyaShams/DeepSeek-R1-Distill-Qwen-1.5B-Medical](https://huggingface.co/MilyaShams/DeepSeek-R1-Distill-Qwen-1.5B-Medical).

### Multi-Step Fine-Tuning

Work in progress...

## Evaluation

Performance will be assessed using datasets from the [Open Medical-LLM leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard) to enable a fair comparison of methods.

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
