import json
import uuid
from typing import Any, Dict, List, Optional

from backend.services.aggregator import aggregate_structured_responses
from backend.services.orchestrator import multi_model_query
from backend.storage.vector_store import add_documents
from backend.storage.simple_store import (
    append_iteration_round,
    finalize_iteration_session,
    save_iteration_session,
)


async def run_iterations(
    question: str,
    models: List[str],
    max_rounds: int = 3,
    prompt_id: str = "answerer_v1",
    prompt_version: str = "v1",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    from backend.storage.simple_store import load_iteration_session
    
    if session_id is None:
        session_id = str(uuid.uuid4())
        session_meta = {"session_id": session_id, "question": question, "models": models, "rounds": []}
        save_iteration_session(session_id, session_meta)
    else:
        # 如果 session 已存在，读取它（可能已有第一轮数据）
        existing = load_iteration_session(session_id)
        if existing:
            # 使用已存在的 session，但确保 question 和 models 一致
            pass
        else:
            # Session 不存在，创建新的
            session_meta = {"session_id": session_id, "question": question, "models": models, "rounds": []}
            save_iteration_session(session_id, session_meta)

    state = "running"
    prev_contradictions = None
    final_report: Dict[str, Any] = {}

    for round_idx in range(1, max_rounds + 1):
        multi = await multi_model_query(
            question,
            models,
            structured=True,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
        )
        structured_items = [
            {"model_id": r.get("model_id"), "parsed": r.get("parsed")}
            for r in multi.get("responses", [])
            if r.get("parsed")
        ]
        report = aggregate_structured_responses(structured_items)

        contradictions = len(report.get("contradictions", []))
        cluster_total = len(report.get("contradictions", [])) + len(report.get("confirmed", []))
        agreement_score = 1.0 if cluster_total == 0 else len(report.get("confirmed", [])) / cluster_total

        round_entry = {
            "round": round_idx,
            "multi": multi,
            "report": report,
            "contradictions": contradictions,
            "agreement_score": agreement_score,
        }
        append_iteration_round(session_id, round_entry)

        # Persist to vector store for later semantic retrieval.
        docs_to_add: List[Dict[str, Any]] = [
            {"text": question, "meta": {"role": "question", "round": round_idx}},
        ]
        for r in structured_items:
            model_id = r.get("model_id")
            for sp in r.get("parsed", {}).get("summary_points", []) or []:
                docs_to_add.append(
                    {
                        "text": sp.get("text", ""),
                        "meta": {"model_id": model_id, "round": round_idx, "point_id": sp.get("id", "")},
                    }
                )
        add_documents(session_id, docs_to_add)

        converged = False
        if prev_contradictions is not None and prev_contradictions > 0 and contradictions <= prev_contradictions * 0.5:
            converged = True
        if agreement_score >= 0.8:
            converged = True
        if round_idx == max_rounds:
            state = "max_rounds_reached"
            final_report = report
            break

        if converged:
            state = "converged"
            final_report = report
            break

        prev_contradictions = contradictions

    if not final_report:
        final_report = report  # last report

    finalize_iteration_session(session_id, state, final_report)
    return {
        "session_id": session_id,
        "state": state,
        "rounds": load_rounds(session_id),
        "final_report": final_report,
    }


def load_rounds(session_id: str) -> List[Dict[str, Any]]:
    from backend.storage.simple_store import load_iteration_session

    data = load_iteration_session(session_id) or {}
    return data.get("rounds", [])
