"""ChatOpenAI for OpenAI-compatible proxies (OmniRoute) whose tool calls arrive whole in one stream delta.

OpenAI streams a tool call as a first delta with id and name (empty arguments) and then argument deltas.  Some proxies send id, name
and the full arguments in a single delta.  ag_ui_langgraph emits TOOL_CALL_START for a delta that carries a name and returns without
emitting its arguments, and the finish chunk never closes the call, so the next run fails with "Cannot send 'TOOL_CALL_START' event:
A tool call with ID ... is already in progress" and the widget gets empty arguments.  This class re-emits such deltas the way OpenAI
does.  Merging the split chunks gives back exactly the original message (string fields concatenate, usage is kept once).
"""

from typing import Any, AsyncIterator, Iterator, List

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI


def _raw_tool_calls(message: AIMessageChunk) -> List[dict]:
    return list((message.additional_kwargs or {}).get("tool_calls") or [])


def split_tool_call_chunk(chunk: ChatGenerationChunk) -> List[ChatGenerationChunk]:
    message = chunk.message
    if not isinstance(message, AIMessageChunk):
        return [chunk]
    tcs = list(message.tool_call_chunks or [])
    whole = [tc for tc in tcs if tc.get("name") and tc.get("args")]
    if not whole and len(tcs) <= 1:
        return [chunk]

    raw_by_index = {}
    for raw in _raw_tool_calls(message):
        raw_by_index[raw.get("index")] = raw
    other_kwargs = {k: v for k, v in (message.additional_kwargs or {}).items() if k != "tool_calls"}

    out: List[ChatGenerationChunk] = []
    pieces = []
    for tc in tcs:
        idx = tc.get("index")
        raw = raw_by_index.get(idx) or {}
        args = tc.get("args") or ""
        if tc.get("name") and args:
            head_raw = {**raw, "function": {**(raw.get("function") or {}), "arguments": ""}}
            tail_raw = {"index": idx, "function": {"arguments": args}}
            pieces.append(({**tc, "args": ""}, head_raw))
            pieces.append(({"name": None, "args": args, "id": None, "index": idx, "type": "tool_call_chunk"}, tail_raw))
        else:
            pieces.append((tc, raw))

    last = len(pieces) - 1
    for i, (tc, raw) in enumerate(pieces):
        kwargs = dict(other_kwargs) if i == 0 else {}
        if raw:
            kwargs["tool_calls"] = [raw]
        msg = AIMessageChunk(
            content=message.content if i == 0 else "",
            additional_kwargs=kwargs,
            tool_call_chunks=[tc],
            id=message.id,
            # usage and finish metadata once, on the last piece, so merging does not double count
            response_metadata=message.response_metadata if i == last else {},
            usage_metadata=message.usage_metadata if i == last else None,
        )
        out.append(ChatGenerationChunk(message=msg, generation_info=chunk.generation_info if i == last else None))
    return out


def closing_chunk() -> ChatGenerationChunk:
    """An empty delta after the last tool-call delta.  OpenAI ends a stream with a usage-only chunk (no tool call, no finish_reason),
    which is what makes ag_ui_langgraph emit TOOL_CALL_END; OmniRoute ends with the finish chunk, which ag_ui_langgraph skips, so
    the call stays open and CopilotKit's ManuallyEmitToolCall for the same id fails with "already in progress".  Merging it is a no-op."""
    return ChatGenerationChunk(message=AIMessageChunk(content=""))


class SplitToolCallChatOpenAI(ChatOpenAI):
    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        saw_tool_call = False
        for chunk in super()._stream(*args, **kwargs):
            for piece in split_tool_call_chunk(chunk):
                saw_tool_call = saw_tool_call or bool(getattr(piece.message, "tool_call_chunks", None))
                yield piece
        if saw_tool_call:
            yield closing_chunk()

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        saw_tool_call = False
        async for chunk in super()._astream(*args, **kwargs):
            for piece in split_tool_call_chunk(chunk):
                saw_tool_call = saw_tool_call or bool(getattr(piece.message, "tool_call_chunks", None))
                yield piece
        if saw_tool_call:
            yield closing_chunk()
