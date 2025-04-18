# MedAdapt-LLM

## Introduction

This project's primary objective is **to explore methods for adapting LLM to the medical domain and assess their effectiveness**. We decide to implement the following methods:

1. **Retrieval-Augmented Generation (RAG)** – enriching prompts with relevant medical information.

2. **Supervised Fine-Tuning (SFT)** – training the model on a medical question-answer dataset.

3. **Multi-Step Fine-Tuning**:

    - Step 1: **Continual Pretraining** – infusing medical knowledge.

    - Step 2: **SFT** – refining instruction-following abilities.

By evaluating these approaches, we aim to determine the most effective strategy for domain-specific LLM adaptation. The insights gained will contribute to understanding how different fine-tuning techniques impact model performance and whether hybrid approaches like RAG can enhance domain relevance without excessive computational overhead.

## Related Work



## Methodology

### Datasets & model

**Model**: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) \
We selected this model because it is a recently released, trending model that leverages modern NLP insights, demonstrates strong benchmark performance, and offers advanced reasoning capabilities. Additionally, its compact size is ideal for training it on the Kaggle platform, balancing performance and efficiency.

We selected datasets tailored for medical adaptation:

- [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) – open-sourced the medical reasoning dataset for SFT, built on medical verifiable problems and an LLM verifier. It contains of triples of question, reasoning, and response. We use this dataset for the SFT phase.

- [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) – dataset of the chunked snippets from the Textbooks corpus used in MedRAG. It's perfect for building our own medical RAG. Moreover, we will use it to serve as a corpus for continual pretraining.

### Implementation details

#### RAG

RAG is implementation is in the [rag folder](https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/tree/main/src/rag).

- `db.py` - a class for loading the dataset into a vector store and querying the vector store. The dataset [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) was used as the source of relevant medical documents. The dataset of size 127,847 snippets was loaded into a vector database in about 8 minutes, requiring ~1 GB of memory.

- `llm.py` - a class for [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) inference.

- `rag.py` - a class that implements basic RAG logic.

#### SFT

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

#### Multi-Step Fine-Tuning

Multi-step fine-tuning consists of two phases:

1. Continual Pretraining on medical textbooks to inject domain knowledge. 
2. Supervised Fine-Tuning (SFT) for task alignment.

We first continued pretraining the base model on the entire MedRAG/textbooks corpus (~127k snippets), using Masked Language Modeling (MLM). This step helps the model better understand domain-specific vocabulary and context before instruction tuning.

After continual pretraining, we fine-tuned the model on the same 20,000 QA pairs from the medical-o1-reasoning-SFT dataset using the same QLoRA configuration as in the SFT-only approach.

This two-step pipeline helps improve both factual grounding and instruction-following performance in the medical domain.

Training and evaluation loss showed slightly improved convergence compared to direct SFT.

## GitHub link
[MedAdapt-LLM](https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM)

### How to run

0. Create a virtual environment.
1. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
2. Install the package in editable mode:  
   ```
   pip install -e .
   ```
3. Define `HF_TOKEN` in environment variables.


## Experiments and Evaluation

Performance will be assessed using benchmarks from the [Open Medical-LLM leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard) to fairly compare methods.

## Experiments and Evaluation

### 1. Continual Pretraining

We performed continual pretraining on the [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) model using medical documents from the [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) dataset to enhance the model's domain-specific knowledge prior to supervised fine-tuning.

#### Setup

- **Dataset**: `MedRAG/textbooks` (127,847 medical text snippets)
- **Model**: 4-bit quantized `DeepSeek-R1-Distill-Qwen-1.5B`
- **Pretraining technique**: Causal Language Modeling (CLM)
- **Training environment**: Kaggle GPU (P100)
- **Framework**: Hugging Face Transformers + PEFT + BitsAndBytes

#### Hyperparameters

| Parameter             | Value              |
|-----------------------|--------------------|
| Number of epochs      | 1                  |
| Learning rate         | 5e-5               |
| Batch size            | 1                  |
| Gradient accumulation | 4                  |
| Warmup steps          | 300                |
| Max steps             | 5000               |
| FP16                  | True               |
| LoRA config           | r=16, α=32, dropout=0.05 |

#### Results

- Training time: ~6 hours
- Final training loss: **1.36**
- Final evaluation loss: **1.41**

<p align="center">
  <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/pretrain_train_loss.png?raw=true" width="45%" />
  <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/pretrain_eval_loss.png?raw=true" width="45%" />
</p>

Continual pretraining helped the model better internalize domain-specific language patterns and terminology prior to instruction tuning, which we expect to yield stronger downstream performance in SFT and RAG scenarios.

### 2. SFT



## Analysis and Observations

## Conclusion

## References

[1] Guo, D., Yang, D., Zhang, H., *et al.* (2024). [**DeepSeek-R1-Distill: Democratizing Open-Source Models for Efficient Reasoning**](https://arxiv.org/pdf/2501.12948).
We fine-tuned this model for both supervised and continual medical domain adaptation.

[2] Xiong, G., Jin, Q., Lu, Z., Zhang, A., *et al.* (2024). [**MedRAG: A Benchmark for Evaluating Domain-Specific Retrieval-Augmented Generation in the Medical Domain**](https://arxiv.org/pdf/2402.13178).
We used datasets from this benchmark for both RAG document retrieval and continual pretraining.

[3] Chen, J., Cai, Z., Ji, K., Wang, X., *et al.* (2024). [**Medical Reasoning Instruction Tuning**](https://arxiv.org/pdf/2412.18925).
We used the FreedomIntelligence/medical-o1-reasoning-SFT dataset introduced in this paper for supervised fine-tuning (SFT).


