from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class LocalModelProvider(BaseLLMProvider):
    """
    Loads and runs a HuggingFace model locally for response generation.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def test(self):
        print(self.tokenizer.eos_token)
        print(self.tokenizer.bos_token)
        print(self.tokenizer.unk_token)

    def generate(self, prompt: str) -> str:
        return prompt


class OpenAIProvider(BaseLLMProvider):
    """
    Loads and runs a OpenAI model locally for response generation.
    """


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Loads and runs a Azure OpenAI model locally for response generation.
    """





