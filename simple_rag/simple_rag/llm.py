import torch

from openai import AzureOpenAI, OpenAI
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        pass

class LocalModelProvider(BaseLLMProvider):
    """
    Loads and runs a HuggingFace model locally for response generation.
    """

    def __init__(self, model_name: str, hf_token:str= None, dtype=torch.float16, device_map="auto"):
        self.model_name = model_name
        
        if hf_token:
            try:
                login(hf_token)
            except Exception as e:
                raise RuntimeError(f"Error defining HuggingFace token: {e}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map)
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}: {e}")



    def test(self):
        """
        Quick test to check tokenizer special tokens and device.
        """
        print(f"EOS token: {self.tokenizer.eos_token}")
        print(f"BOS token: {self.tokenizer.bos_token}")
        print(f"UNK token: {self.tokenizer.unk_token}")
        print(f"Model device: {self.model.device}")

    def build_prompt(self, query: str, context: str):
        """
        Builds the prompt, using chat template if available.
        """
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."
                    "Provide one Answer ONLY the following query based on the context provided below. "
                    "Do not generate or answer any other questions. "
                    "Do not make up or infer any information that is not directly stated in the context. "
                    "Provide a concise answer."
                    f"{context}"},
            {"role": "user", "content": query}
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        else:
            prompt = f"Context:\n{context}\n\nUser: {query}\nAssistant:"
        
        return prompt


    def __call__(self, query: str, context: str = "", max_new_tokens: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        prompt = self.build_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id

        )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


class OpenAIProvider(BaseLLMProvider):
    """
    Loads and runs a OpenAI model locally for response generation.
    """
    def __init__(self, api_key: str, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def __call__(self, query: str, context: str = "") -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Provide one Answer ONLY the following query based on the context provided below. "
                    "Do not generate or answer any other questions. "
                    "Do not make up or infer any information that is not directly stated in the context. "
                    "Provide a concise answer."
                    f"{context}"
                )
            },
            {"role": "user", "content": query}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content



class AzureOpenAIProvider(BaseLLMProvider):
    """
    Loads and runs a Azure OpenAI model locally for response generation.
    """
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, deployment_name: str):
        self.deployment_name = deployment_name
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )

    def __call__(self, query: str, context: str = "") -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Provide one Answer ONLY the following query based on the context provided below. "
                    "Do not generate or answer any other questions. "
                    "Do not make up or infer any information that is not directly stated in the context. "
                    "Provide a concise answer."
                    f"{context}"
                )
            },
            {"role": "user", "content": query}
        ]
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content