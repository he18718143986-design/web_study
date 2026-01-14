from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Return model output for the given prompt."""
        raise NotImplementedError
