# test_hf_model.py
from transformers import pipeline
import os

# -----------------------------
# 1️⃣ 确认 Hugging Face API Token
# -----------------------------
api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("请先设置环境变量 HUGGINGFACEHUB_API_TOKEN")

# -----------------------------
# 2️⃣ 选择模型
# -----------------------------
model_name = "distilgpt2"  # 小型文本生成模型

# -----------------------------
# 3️⃣ 创建 pipeline
# 不需要传 use_auth_token，环境变量已经够
# -----------------------------
try:
    generator = pipeline(
        "text-generation",
        model=model_name,
    )
except Exception as e:
    print("创建 pipeline 失败:", e)
    exit(1)

# -----------------------------
# 4️⃣ 测试生成文本
# 注意这里加 truncation=True 避免警告
# -----------------------------
prompt = "Hello, can you explain what Hugging Face is?"
try:
    output = generator(prompt, max_length=50, truncation=True)
    print("模型输出:", output)
except Exception as e:
    print("模型调用失败:", e)
