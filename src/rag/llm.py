from vllm import LLM, SamplingParams
from typing import List, Optional
import torch


class LLMInference:
    """
    A class to handle text generation using the vLLM inference engine.
    """

    def __init__(self,
                 model_name: str,
                 dtype=torch.float16,
                 trust_remote_code: bool=True,
                 quantization: Optional[str]=None,
                 tensor_parallel_size=1,
        ):
        """
        Initializes the VLLMInference object with the specified parameters and loads the model.

        Args:
            model_name (str): The name or path of the model to be loaded.
            dtype (str, optional): The data type to use for the model. Defaults to 'auto'.
            trust_remote_code (bool, optional): Whether to trust remote code when loading the model. Defaults to True.
            quantization (Optional[str], optional): The quantization mode to use for the model. Defaults to None.
            tensor_parallel_size (int, optional): The number of tensor parallelism to use. Defaults to 1.
        """
        self.model_name = model_name
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.quantization = quantization
        self.seed = 4242

        self.llm = LLM(
            model=self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization=self.quantization,
            seed=self.seed,
            tensor_parallel_size=tensor_parallel_size,
        )

    def generate(self,
                 prompts: List[str],
                 max_tokens: int=4096,
                 temperature: float=0.01,
                 top_p: float=1.0,
                 top_k: int=-1,
                 **kwargs
        ) -> List[str]:
        """
        Generates text based on the provided prompts.

        Args:
            prompts (List[str]): A list of input prompts for text generation.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.
            temperature (float, optional): Sampling temperature to use. Defaults to 0.01.
            top_p (float, optional): Nucleus sampling probability. Defaults to 1.0.
            top_k (int, optional): Top-k sampling. Defaults to -1.

        Returns:
            List[str]: A list of generated texts corresponding to each prompt.
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=self.seed,
            **kwargs
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


# # usage
# if __name__ == "__main__":
#     model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#     prompts = [
#         """
#         You are an expert in solving multiple-choice questions accurately and explaining your reasoning clearly.
#         Given a question and a list of answer choices (A, B, C, D), your task is to:
#         1. Reason shortly about the question and answer choices to find evidances to support your answer.
#         2. Identify the correct answer.
#         3. Output the final answer in the format: Answer: [Option Letter]

#         Here is a question: Which vitamin is supplied from only animal source?
#         A. Vitamin C
#         B. Vitamin B7
#         C. Vitamin B12
#         D. Vitamin D

#         Reasoning:
#         """
#     ]
#     llm = LLMInference(model_name=model_name)
#     response = llm.generate(prompts)
#     print(response)
