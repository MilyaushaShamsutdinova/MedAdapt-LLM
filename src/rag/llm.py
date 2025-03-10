from typing import Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMInference:
    """
    A class for performing LLM inference using HuggingFace Transformers.
    
    This class loads a pre-trained language model and tokenizer, formats chat prompts,
    and generates responses based on the given prompts.
    """

    def __init__(self,
                 model_name: str="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                 device: Optional[str] = None) -> None:
        """
        Initialize the LLMInference instance with the specified model and tokenizer.

        :param model_name: The HuggingFace model name or path.
        :param device: The device to run the model on (e.g., "cuda" or "cpu").
                       If None, uses 'cuda' if available, otherwise 'cpu'.
        """
        self.model_name = model_name
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate_response(self,
                          prompt: str,
                          system_prompt: str = "",
                          max_new_tokens: int = 1024,
                          **generate_kwargs: Any) -> str:
        """
        Generate a text response based on a user prompt and an optional system prompt.

        :param prompt: The main user prompt.
        :param system_prompt: A guiding system prompt for the model (optional).
        :param max_new_tokens: Maximum number of tokens to generate.
        :param generate_kwargs: Additional keyword arguments for the model.generate() method.
        :return: The generated text response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


# # usage
# if __name__ == "__main__":
#     system_prompt = """You are a medical expert with advanced knowledge 
#     in clinical reasoning, diagnostics, and treatment planning.
#     Please answer the following medical question.\n"""

#     user_query = """What type of cement bonds to tooth structure, provides an anticariogenic effect, 
#                 has a degree of translucency, and is non-irritating to the pulp?"""
    
#     llm_inference = LLMInference()
#     response = llm_inference.generate_response(user_query, system_prompt=system_prompt)
#     print("------------------")
#     print(response)

