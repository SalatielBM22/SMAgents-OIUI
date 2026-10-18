"""The combos the UI may switch between: OmniRoute's sm-* models that can call tools.

Only meaningful behind a proxy (OPENAI_BASE_URL); without one the list is empty and the picker hides itself.  The answer is cached
briefly so opening the page does not hit the proxy on every render.
"""

import os
import time

import httpx

PREFIX = "sm-"
TTL_SECONDS = 60.0
_cache: dict = {"at": 0.0, "value": None}


def _flag(entry: dict, *names: str) -> bool | None:
    caps = entry.get("capabilities") if isinstance(entry.get("capabilities"), dict) else {}
    for name in names:
        for source in (caps, entry):
            if name in source and isinstance(source[name], bool):
                return source[name]
    return None


def _modalities(entry: dict) -> list[str]:
    for source in (entry, entry.get("capabilities") or {}, entry.get("architecture") or {}):
        value = source.get("input_modalities") if isinstance(source, dict) else None
        if isinstance(value, list):
            return [str(v).lower() for v in value]
    return []


def summarize(entries: list) -> list[dict]:
    """sm-* entries that do not say they lack tool calling, with a vision flag; sorted by id."""
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", ""))
        if not model_id.startswith(PREFIX):
            continue
        if _flag(entry, "tool_calling", "tools", "function_calling") is False:
            continue
        vision = _flag(entry, "vision") or "image" in _modalities(entry)
        out.append({"id": model_id, "vision": bool(vision)})
    return sorted(out, key=lambda m: m["id"])


def default_model() -> str:
    return os.environ.get("LLM_MODEL", "").strip() or "sm-frontier"


async def list_models(fetch=None) -> dict:
    base = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    if not base:
        return {"default": default_model(), "models": []}
    now = time.monotonic()
    if _cache["value"] is not None and now - _cache["at"] < TTL_SECONDS:
        return _cache["value"]
    if fetch is None:
        fetch = _fetch
    entries = await fetch(base, os.environ.get("OPENAI_API_KEY", ""))
    value = {"default": default_model(), "models": summarize(entries)}
    _cache.update(at=now, value=value)
    return value


async def _fetch(base: str, key: str) -> list:
    # The first /models after OmniRoute boots can take several seconds.
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        return response.json().get("data") or []
