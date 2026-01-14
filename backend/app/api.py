# backend/app/api.py
import sys
from pathlib import Path

# Ensure project root is importable when running as a module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import APIRouter, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import asyncio
import logging

# 导入项目内部模块（路径基于你 repo 的结构）
from backend.services.orchestrator import multi_model_query
from backend.services.iteration_controller import run_iterations
from backend.storage.simple_store import save_structured_session, load_structured_session, append_iteration_round
from backend.prompt.registry import get_prompt

logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-LLM Arbiter API")

# --- CORS: 允许前端开发服务器访问（Vite 默认 http://localhost:5173） ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 简单健康检查根路由（方便确认服务在线）
@app.get("/")
async def root():
    return {"status": "ok", "service": "Multi-LLM Arbiter API"}

router = APIRouter(prefix="/v1")

class QueryRequest(BaseModel):
    question: str
    models: Optional[List[str]] = ["mock", "mock"]
    prompt_id: Optional[str] = "answerer_v1"
    prompt_version: Optional[str] = "v1"
    max_rounds: Optional[int] = 3

@router.post("/query")
async def post_query(req: QueryRequest):
    """
    接收前端问题，运行初轮多模型并发（structured=True），
    保存会话初始信息到存储，并异步触发迭代控制器（run_iterations）。
    """
    session_id = str(uuid4())

    # 尝试读取 prompt（如果找不到，get_prompt 可抛异常或返回默认）
    try:
        _ = get_prompt(req.prompt_id, req.prompt_version)
    except Exception as e:
        logger.warning("Prompt not found or failed to load: %s", e)

    # 调用 orchestrator 获取初轮结果（structured=True）
    # 添加超时保护，避免单个慢速模型阻塞整个请求
    try:
        result = await asyncio.wait_for(
            multi_model_query(
                req.question,
                req.models,
                structured=True,
                prompt_id=req.prompt_id,
                prompt_version=req.prompt_version
            ),
            timeout=100.0  # 100 秒超时
        )
    except asyncio.TimeoutError:
        logger.error("multi_model_query timed out after 100 seconds")
        raise HTTPException(
            status_code=504,
            detail="Request timeout: LLM API calls took too long. Please try again or use faster models."
        )
    except Exception as e:
        logger.exception("multi_model_query failed")
        raise HTTPException(status_code=500, detail=f"multi_model_query failed: {e}")

    # 最小化持久化会话结构（后续轮次会追加）
    session_obj = {
        "session_id": session_id,
        "state": "running",
        "rounds": [{
            "round": 1,
            "multi": result,
            "report": None,
            "contradictions": 0,
            "agreement_score": None
        }],
        "final_report": None
    }
    try:
        save_structured_session(session_id, session_obj)
    except Exception:
        logger.exception("Failed to save structured session")

    # 异步触发迭代（不阻塞请求返回）
    try:
        asyncio.create_task(
            run_iterations(req.question, req.models, req.max_rounds, req.prompt_id, req.prompt_version, session_id=session_id)
        )
    except Exception:
        logger.exception("Failed to schedule run_iterations task")

    # 返回初步响应（包含 session_id 与初轮 multi 结果）
    return {
        "session_id": session_id,
        "state": "running",
        "rounds": [
            {
                "round": 1,
                "multi": result
            }
        ]
    }

@router.get("/session/{session_id}")
def get_session(session_id: str):
    sess = load_structured_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    return sess

class FollowupRequest(BaseModel):
    session_id: str
    followup_question: str
    models: Optional[List[str]] = None
    prompt_id: Optional[str] = "answerer_v1"
    prompt_version: Optional[str] = "v1"

@router.post("/followup")
async def post_followup(req: FollowupRequest):
    """
    对既有 session 发起单轮 followup（同步调用 orchestrator 并把结果追加到会话）
    """
    sess = load_structured_session(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")

    # 如果 caller 没指定 models，则使用上轮的模型列表（从 saved session 中读取）
    try:
        prev_responses = sess["rounds"][-1]["multi"]["responses"]
        prev_model_ids = [r.get("model_id") for r in prev_responses if r.get("model_id")]
    except Exception:
        prev_model_ids = req.models or ["mock"]

    models = req.models or prev_model_ids

    try:
        result = await multi_model_query(
            req.followup_question,
            models,
            structured=True,
            prompt_id=req.prompt_id,
            prompt_version=req.prompt_version
        )
    except Exception as e:
        logger.exception("multi_model_query in followup failed")
        raise HTTPException(status_code=500, detail=f"multi_model_query failed: {e}")

    # 将新一轮追加到存储
    try:
        append_iteration_round(req.session_id, {
            "round": len(sess["rounds"]) + 1,
            "multi": result,
            "report": None,
            "contradictions": 0,
            "agreement_score": None
        })
    except Exception:
        logger.exception("Failed to append round to session")

    return {"session_id": req.session_id, "round": len(sess["rounds"]), "multi": result}

@router.get("/models")
def get_models():
    # 从配置或文件返回当前可选模型（前端可调用列出）
    models = [
        {"id":"mock","name":"Mock Adapter","version":"v1","status":"available"},
        {"id":"hf","name":"Hugging Face","version":"default","status":"available"},
        {"id":"ollama","name":"Ollama Local","version":"local","status":"available"}
    ]
    return {"models": models}

# 把 router 注册到 app（必须）
app.include_router(router)
