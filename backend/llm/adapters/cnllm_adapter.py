import os
import requests

class CNLLMAdapter:
    """简单封装国内免费 LLM API"""
    def __init__(self, model: str = "ziya-13b", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("CNLLM_API_KEY")
        if not self.api_key:
            raise ValueError("请在环境变量 CNLLM_API_KEY 中设置你的 API Key")
        self.endpoint = "https://api.ziyangpt.com/v1/completions"  # 示例地址

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        r = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()
