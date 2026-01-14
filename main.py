import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure repo root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.services.iteration_controller import run_iterations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-LLM Arbiter CLI (root entrypoint)")
    parser.add_argument("question", nargs="?", default=None, help="Question text")
    parser.add_argument(
        "--models",
        default=os.getenv("LLM_MODELS", "mock,mock"),
        help="Comma-separated model ids (default: mock,mock)",
    )
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum rounds (default: 3)")
    return parser.parse_args()


async def _main_async() -> None:
    args = parse_args()
    if not args.question:
        print("placeholder")
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    result = await run_iterations(args.question, models, max_rounds=args.max_rounds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
