import json
from pathlib import Path
from typing import Any, Dict, List, Optional

STORE_DIR = Path(__file__).resolve().parent / "data"
STORE_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return STORE_DIR / f"{session_id}.json"


def save_structured_session(session_id: str, payload: Dict[str, Any]) -> None:
    _session_path(session_id).write_text(json.dumps(payload), encoding="utf-8")


def load_structured_session(session_id: str) -> Optional[Dict[str, Any]]:
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# Iteration sessions

def save_iteration_session(session_id: str, payload: Dict[str, Any]) -> None:
    _session_path(session_id).write_text(json.dumps(payload), encoding="utf-8")


def load_iteration_session(session_id: str) -> Optional[Dict[str, Any]]:
    return load_structured_session(session_id)


def append_iteration_round(session_id: str, round_entry: Dict[str, Any]) -> None:
    data = load_iteration_session(session_id) or {}
    rounds: List[Dict[str, Any]] = data.get("rounds", [])
    rounds.append(round_entry)
    data["rounds"] = rounds
    save_iteration_session(session_id, data)


def finalize_iteration_session(session_id: str, state: str, final_report: Dict[str, Any]) -> None:
    data = load_iteration_session(session_id) or {}
    data["state"] = state
    data["final_report"] = final_report
    save_iteration_session(session_id, data)
