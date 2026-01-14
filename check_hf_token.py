import os
import requests

token = os.environ.get("HUGGINGFACE_API_KEY")
model = os.environ.get("HF_MODEL_ID", "bigscience/bloom-560m")

if not token:
    print("❌ HUGGINGFACE_API_KEY 未设置！")
    exit(1)

url = f"https://api-inference.huggingface.co/models/{model}"
headers = {"Authorization": f"Bearer {token}"}

resp = requests.get(url, headers=headers)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:300]}")
if resp.status_code == 401:
    print("❌ Token 无效或无权限！请检查 HUGGINGFACE_API_KEY")
elif resp.status_code == 200:
    print("✅ Token 有效，模型可访问")
else:
    print("⚠️ 其他响应，请检查模型名和 token 权限")
