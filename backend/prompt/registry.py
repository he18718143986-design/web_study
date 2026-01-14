import yaml
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "prompts" / "prompt_registry.yaml"


def get_prompt(prompt_id: str, version: str = "v1") -> Optional[str]:
    if not REGISTRY_PATH.exists():
        return None
    entries = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or []
    for item in entries:
        if item.get("id") == prompt_id and item.get("version") == version:
            return item.get("template")
    return None
