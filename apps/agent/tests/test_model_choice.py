import asyncio

import pytest
from langchain_openai import ChatOpenAI

from src import model_catalog
from src.credentials import current_credentials, current_model, parse_model
from src.model import with_requested_model


def test_parse_model_accepts_combo_ids_only():
    assert parse_model([(b"x-oiu-model", b"sm-coding")]) == "sm-coding"
    assert parse_model([(b"x-oiu-model", b"bad id;rm")]) is None
    assert parse_model([]) is None


def _model():
    return ChatOpenAI(model="sm-frontier", base_url="http://127.0.0.1:1/v1", api_key="x")


def test_requested_combo_overrides_server_model(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    token = current_model.set("sm-coding")
    try:
        assert with_requested_model(_model()).model_name == "sm-coding"
    finally:
        current_model.reset(token)


def test_no_override_without_proxy_or_with_visitor_keys(monkeypatch):
    token = current_model.set("sm-coding")
    try:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert with_requested_model(_model()).model_name == "sm-frontier"
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
        cred = current_credentials.set(object())
        try:
            assert with_requested_model(_model()).model_name == "sm-frontier"
        finally:
            current_credentials.reset(cred)
    finally:
        current_model.reset(token)


def test_summarize_keeps_tool_capable_sm_combos():
    entries = [
        {"id": "sm-frontier", "capabilities": {"tool_calling": True}, "input_modalities": ["text", "image"]},
        {"id": "sm-cheap", "capabilities": {"tool_calling": False}},
        {"id": "sm-coding"},
        {"id": "gpt-4o"},
    ]
    assert model_catalog.summarize(entries) == [
        {"id": "sm-coding", "vision": False},
        {"id": "sm-frontier", "vision": True},
    ]


def test_list_models_without_proxy_is_empty(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "sm-frontier")
    assert asyncio.run(model_catalog.list_models()) == {"default": "sm-frontier", "models": []}


def test_list_models_caches(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy/v1")
    model_catalog._cache.update(at=0.0, value=None)
    calls = []

    async def fetch(base, key):
        calls.append(base)
        return [{"id": "sm-frontier"}]

    first = asyncio.run(model_catalog.list_models(fetch))
    second = asyncio.run(model_catalog.list_models(fetch))
    assert first == second and len(calls) == 1
    model_catalog._cache.update(at=0.0, value=None)
