# MedAdapt-LLM: Adapting Large Language Models to the Medical Domain

**Authors:** \
Milyausha Shamsutdinova \
Milana Sirozhova \
Diana Vostrova


## Introduction

The rapid advancement of Large Language Models (LLMs) has revolutionized natural language processing. However, their application in specialized fields like medicine necessitates domain-specific adaptation to ensure accuracy and reliability. MedAdapt-LLM explores and compares various methodologies for effectively adapting LLMs to the medical domain, aiming to enhance their performance in medical tasks.

This project focuses on adapting the **DeepSeek-R1-Distill-Qwen-1.5B** model, a compact yet capable 1.5B parameter LLM, chosen for its strong benchmark results and suitability for training on limited GPU resources.


## Methodology

We implemented and compared three distinct approaches for medical domain adaptation:

1.  **Retrieval-Augmented Generation (RAG):** Enriching prompts with relevant medical information fetched dynamically from a medical corpus.
    *   **Knowledge Base:** `MedRAG/textbooks` and `MedRAG/statpearls` datasets (460K text snippets).
    *   **Vector Store:** ChromaDB.
2.  **Supervised Fine-Tuning (SFT):** Directly modifying the LLM's parameters by training on a medical question-answer dataset to align its behavior with specific tasks and response formats (Q-CoT-A: Question, Chain-of-Thought, Answer).
    *   **Dataset:** `FreedomIntelligence/medical-o1-reasoning-SFT` (20K samples).
3.  **Multi-Step Fine-Tuning (MSFT):** A two-stage approach:
    *   **Stage 1: Continual Pretraining (CP):** Infusing medical knowledge by further pretraining the base LLM on a large corpus of medical texts.
        *   **Dataset:** Combined `MedRAG/textbooks` and `MedRAG/statpearls` (460K text snippets).
    *   **Stage 2: Supervised Fine-Tuning (SFT):** Subsequent instruction tuning on the medical QA dataset to enhance instruction-following specific to medical queries.
        *   **Dataset:** `FreedomIntelligence/medical-o1-reasoning-SFT` (20K samples).

All fine-tuning (SFT and MSFT stages) utilized QLoRA with 4-bit quantization and Unsloth optimization for efficient training.

## Models and Datasets used

*   **Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
*   **For SFT (Standalone and MSFT Stage 2):**
    *   `FreedomIntelligence/medical-o1-reasoning-SFT`: Open-sourced medical reasoning dataset containing triples of (question, reasoning, response).
*   **For RAG Knowledge Base & Continual Pretraining (MSFT Stage 1):**
    *   `MedRAG/textbooks`: Chunked snippets from medical textbooks.
    *   `MedRAG/statpearls`: Chunked snippets from the StatPearls book corpus.

## Experimental Setup

*   **Hardware:** Single NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM).
*   **Key Libraries:** Hugging Face (transformers, datasets, trl), PyTorch, bitsandbytes, accelerate, wandb, unsloth.
*   **Evaluation Benchmarks:** Subsets from the Open Medical-LLM Leaderboard (200 samples each):
    *   MedMCQA (MCQ)
    *   MedQA (MCQ)
    *   MMLU (MCQ, 6 medical subsets)
    *   PubMedQA (QA, pqa_labeled subset)

## Results and Analysis

Performance was assessed on the benchmarks mentioned above.

### I. Quantitative Results

| Model         | MedMCQA | MedQA   | Med MMLU | PubMedQA | **Avg**   |
| :------------ | :------ | :------ | :------- | :------- | :-------- |
| Base Model    | 30.5%   | 22.5%   | 34%      | 38%      | 31.25%    |
| **RAG Model** | 29%     | 21.5%   | 30.5%    | **56.5%**| **34.375%**|
| SFT Model     | 30.5%   | 25.5%   | 28.5%    | 36%      | 30.125%   |
| MSFT Model    | 23%     | **28%** | 29%      | 41.5%    | 30.375%   |

### II. Key Observations

*   **RAG:** Demonstrated the highest average performance, excelling significantly on PubMedQA. This highlights the effectiveness of dynamic external knowledge retrieval for tasks amenable to it.
*   **SFT:** Showed slight improvement on MedQA but resulted in a small decrease in average performance compared to the base model. This suggests that while SFT aligned the model to the Q-CoT-A format, it might have led to some knowledge degradation or overfitting.
*   **MSFT:** Outperformed SFT on MedQA, Med MMLU, and PubMedQA, indicating some benefit from continual pretraining. However, it underperformed the base model on average and showed a notable drop on MedMCQA.
*   **Computational Cost:**
    *   RAG: Corpus indexing took ~40 minutes (offline). Inference has retrieval latency.
    *   SFT: Training took ~4.5 hours.
    *   Continual Pretraining (MSFT Stage 1): ~65 hours.
    *   MSFT (Total): ~70 hours.

### III. Training Dynamics (Loss Curves)

All training stages successfully converged within a single epoch.


|  Loss | SFT    | Continual Pretraining   | MSFT (SFT Stage) |
| :----:| :-------------: | :---------: | :------------: |
| Train | <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/sft_train_loss.png?raw=true" width="200"> | <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/cont_pretrain_train_loss.png?raw=true" width="200"> | <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/msft_train_loss.png?raw=true" width="200"> |
| Eval | <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/sft_eval_loss.png?raw=true" width="200"> | <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/cont_pretrain_eval_loss.png?raw=true" width="200"> | <img src="https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM/blob/main/assets/msft_eval_loss.png?raw=true" width="200"> |


The final evaluation loss for the SFT stage of MSFT (1.65) was slightly lower than standalone SFT (1.67), hinting at potentially better optimization after continual pretraining.

## Conclusion

The effectiveness of LLM adaptation strategies is highly task-dependent.

*   **RAG** emerged as the top performer on average, especially for retrieval-heavy tasks (PubMedQA), offering a computationally cheaper alternative to full fine-tuning.
*   **SFT and MSFT** demonstrated the feasibility of fine-tuning a 1.5B model on consumer hardware. However, they did not consistently outperform the base model across all benchmarks.
    *   SFT can align model behavior but may risk knowledge degradation.
    *   MSFT, despite the computationally intensive continual pretraining phase, yielded only marginal average improvements over SFT for this model size and dataset configuration.

This study underscores the trade-offs between computational cost, implementation complexity, and performance gains. For smaller models, the benefits of continual pretraining might require larger-scale data or more sophisticated integration.

## Limitations

*   Single, relatively small base model (1.5B parameters).
*   Training for only one epoch for each fine-tuning stage.
*   Specific choices of datasets and their sizes.
*   Evaluation on limited subsets of benchmarks.
*   Qualitative aspects like reasoning coherence and safety were not formally evaluated.

## Accessing Models and Code

*   **Source Code:** [https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM](https://github.com/MilyaushaShamsutdinova/MedAdapt-LLM)
    *   RAG implementation: `src/rag/`
    *   Training notebooks: `notebooks/` (e.g., `sft_training.ipynb`, `continual_pretrain.ipynb`, `msft_training.ipynb`)
*   **Fine-Tuned Models on Hugging Face Hub:**
    *   [DeepSeek-R1-Distill-Qwen-1.5B-medical-SFT-merged](https://huggingface.co/MilyaShams/DeepSeek-R1-Distill-Qwen-1.5B-medical-sft-merged)
    *   [DeepSeek-R1-Distill-Qwen-1.5B-medical-continual-pretrain-merged](https://huggingface.co/MilyaShams/DeepSeek-R1-Distill-Qwen-1.5B-medical-continual-pretrain-merged)
    *   [DeepSeek-R1-Distill-Qwen-1.5B-medical-MSFT-merged](https://huggingface.co/MilyaShams/DeepSeek-R1-Distill-Qwen-1.5B-medical-msft-merged)
    *   **Collection**: [DeepSeek-R1-Distill-Qwen-1.5B-medical models](https://huggingface.co/collections/MilyaShams/deepseek-r1-distill-qwen-15b-medical-models-67cc39de5fd9f4c5d8fd837b)

## References

[1] Guo, D., Yang, D., Zhang, H., *et al.* (2024). [**DeepSeek-R1-Distill: Democratizing Open-Source Models for Efficient Reasoning**](https://arxiv.org/pdf/2501.12948).
We fine-tuned this model for both supervised and continual medical domain adaptation.

[2] Xiong, G., Jin, Q., Lu, Z., Zhang, A., *et al.* (2024). [**MedRAG: A Benchmark for Evaluating Domain-Specific Retrieval-Augmented Generation in the Medical Domain**](https://arxiv.org/pdf/2402.13178).
We used datasets from this benchmark for both RAG document retrieval and continual pretraining.

[3] Chen, J., Cai, Z., Ji, K., Wang, X., *et al.* (2024). [**Medical Reasoning Instruction Tuning**](https://arxiv.org/pdf/2412.18925).
We used the FreedomIntelligence/medical-o1-reasoning-SFT dataset introduced in this paper for supervised fine-tuning (SFT).