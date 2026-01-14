import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.llm.adapters.hf_adapter import HuggingFaceAdapter
from backend.llm.adapters.mock_adapter import MockAdapter
from backend.llm.adapters.ollama_adapter import OllamaAdapter
from backend.prompt.registry import get_prompt
from backend.services.orchestrator import multi_model_query
from backend.services.semantic import cluster_points, embed_points, extract_points
from backend.services.iteration_controller import run_iterations
from backend.services.aggregator import aggregate_structured_responses
from backend.storage.simple_store import load_structured_session, save_structured_session
from backend.storage.vector_store import search_similar
from backend.services.orchestrator import multi_model_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM client CLI")
    parser.add_argument("--question", "-q", help="Prompt to send", default="Hello from Phase 1")
    parser.add_argument(
        "--backend",
        "-b",
        choices=["mock", "hf", "ollama"],
        default=os.getenv("LLM_BACKEND", "mock"),
        help="Which backend adapter to use",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("HF_MODEL_ID", "gpt2"),
        help="Hugging Face model id (used when backend=hf)",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Enable multi-model querying",
    )
    parser.add_argument(
        "--models",
        default=os.getenv("LLM_MODELS", "mock"),
        help="Comma-separated model ids when --multi is set",
    )
    parser.add_argument(
        "--prompt-id",
        default=os.getenv("PROMPT_ID", "answerer_v1"),
        help="Prompt id to use from registry",
    )
    parser.add_argument(
        "--prompt-version",
        default=os.getenv("PROMPT_VERSION", "v1"),
        help="Prompt version to use from registry",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Parse responses as structured JSON when supported",
    )
    parser.add_argument(
        "--cluster-session",
        help="Session id to load structured responses and perform clustering",
    )
    parser.add_argument(
        "--aggregate-session",
        help="Session id to aggregate (nli + cross-eval + summary)",
    )
    parser.add_argument(
        "--run-query",
        action="store_true",
        help="Run iterative controller end-to-end",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum rounds for iterative controller",
    )
    parser.add_argument(
        "--search-history",
        help="Search historical Q&A in vector store",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K results to return when searching history",
    )
    return parser.parse_args()


def build_client(backend_name: str, model_id: Optional[str]) -> MockAdapter | HuggingFaceAdapter:
    if backend_name == "mock":
        return MockAdapter()
    if backend_name == "hf":
        return HuggingFaceAdapter(model_id=model_id or "gpt2")
    if backend_name == "ollama":
        return OllamaAdapter(model_id=model_id or "llama3.2")
    raise ValueError(f"Unsupported backend: {backend_name}")


async def run() -> None:
    args = parse_args()
    if args.run_query:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        result = await run_iterations(
            args.question,
            models,
            max_rounds=args.max_rounds,
            prompt_id=args.prompt_id,
            prompt_version=args.prompt_version,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.search_history:
        hits = search_similar(args.search_history, top_k=args.top_k)
        printable = [
            {
                "score": round(score, 4),
                "text": item.get("text"),
                "meta": item.get("meta", {}),
                "session_id": item.get("session_id"),
            }
            for score, item in hits
        ]
        print(json.dumps({"query": args.search_history, "results": printable}, ensure_ascii=False, indent=2))
        return

    if args.aggregate_session:
        session = load_structured_session(args.aggregate_session)
        if not session:
            print(json.dumps({"error": f"session {args.aggregate_session} not found"}, ensure_ascii=False))
            return
        structured_items = [
            {"model_id": r.get("model_id"), "parsed": r.get("parsed")}
            for r in session.get("responses", [])
            if r.get("parsed")
        ]
        report = aggregate_structured_responses(structured_items)
        print(json.dumps({"session_id": args.aggregate_session, **report}, ensure_ascii=False, indent=2))
        return

    if args.cluster_session:
        session = load_structured_session(args.cluster_session)
        if not session:
            print(json.dumps({"error": f"session {args.cluster_session} not found"}, ensure_ascii=False))
            return
        responses = session.get("responses", [])
        structured_items = [
            {"model_id": r.get("model_id"), "parsed": r.get("parsed")}
            for r in responses
            if r.get("parsed")
        ]
        points = extract_points(structured_items)
        embeddings = embed_points(points)
        clusters = cluster_points(points, embeddings)
        print(json.dumps({"session_id": args.cluster_session, "clusters": clusters}, ensure_ascii=False, indent=2))
        return

    if args.multi:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        result = await multi_model_query(
            args.question,
            models,
            structured=args.structured,
            prompt_id=args.prompt_id,
            prompt_version=args.prompt_version,
        )
        if args.structured:
            session_id = str(uuid.uuid4())
            result["session_id"] = session_id
            save_structured_session(session_id, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    client = build_client(args.backend, args.model_id)
    try:
        output = await client.generate(args.question)
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"[error] {exc}")
        return

    print(output)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
