import asyncio
from typing import Dict, Optional

import httpx

# Simple async POST with retry/backoff for transient errors.
async def post_json(
    url: str,
    payload: Dict,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,  # 增加到 120 秒，因为 LLM API 调用可能需要较长时间
    retries: int = 2,
    backoff: float = 0.5,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in status_forcelist:
                    raise httpx.HTTPStatusError("retryable status", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            await asyncio.sleep(backoff * (2**attempt))
    assert last_exc is not None
    raise last_exc
