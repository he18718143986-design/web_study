"""
test_hf_model.py

用途：
- 独立测试 Hugging Face Inference API 是否真实可用
- 明确区分：环境变量问题 / 网络问题 / 模型问题
- 不依赖 FastAPI / orchestrator / session
"""

import asyncio
import time
import os
import traceback

from dotenv import load_dotenv

# ✅ 显式加载 .env（非常关键）
load_dotenv()

from backend.llm.adapters.hf_adapter import HuggingFaceAdapter

# 固定测试问题（不要改，保证可重复）
TEST_QUESTION = "请用 3 点简要说明什么是在线教学平台。"

# ✅ 明确指定一个「确定可用」的 Hugging Face 模型
HF_MODEL_ID = "google/flan-t5-small"


async def test_hf():
    print("=" * 70)
    print("🚀 开始测试 Hugging Face Inference API")
    print("=" * 70)

    # 1️⃣ 明确检查环境变量
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_key:
        print("❌ 未检测到环境变量 HUGGINGFACE_API_KEY")
        print("👉 请先 export 或写入 .env 文件")
        return
    else:
        print("✅ 已检测到 HUGGINGFACE_API_KEY（已隐藏）")

    # 2️⃣ 初始化 HF Adapter
    print(f"\n📦 使用模型: {HF_MODEL_ID}")
    adapter = HuggingFaceAdapter(model_id=HF_MODEL_ID)

    # 3️⃣ 构造最小 prompt
    prompt = f"""
你是一个助手，请用中文回答。

问题：
{TEST_QUESTION}
""".strip()

    print("\n📨 Prompt:")
    print(prompt)
    print("-" * 70)

    # 4️⃣ 发起真实调用并计时
    start_time = time.time()

    try:
        response = await adapter.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=256,
        )

        elapsed = time.time() - start_time

        print("✅ Hugging Face 调用成功")
        print(f"⏱️ 用时: {elapsed:.2f} 秒")

        # 5️⃣ 判断是否“真的调用了远程模型”
        if elapsed < 1.0:
            print("⚠️ 警告：返回过快，可能未走真实 HF 请求")
        else:
            print("🎯 判断：这是一次真实的 HF Inference 请求")

        print("\n📤 模型原始输出:")
        print(response)

    except Exception as e:
        elapsed = time.time() - start_time

        print("❌ Hugging Face 调用失败")
        print(f"⏱️ 用时: {elapsed:.2f} 秒")
        print("\n错误信息:")
        print(str(e))

        print("\n完整异常栈（用于调试）:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_hf())
