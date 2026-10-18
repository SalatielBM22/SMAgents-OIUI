from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from src.proxy_openai import split_tool_call_chunk


def _whole_chunk():
    raw = [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "plan_visualization", "arguments": '{"a":1}'}}]
    msg = AIMessageChunk(
        content="",
        additional_kwargs={"tool_calls": raw},
        tool_call_chunks=[{"name": "plan_visualization", "args": '{"a":1}', "id": "call_1", "index": 0, "type": "tool_call_chunk"}],
        id="run-1",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    return ChatGenerationChunk(message=msg)


def test_whole_tool_call_is_split_like_openai():
    pieces = split_tool_call_chunk(_whole_chunk())
    assert len(pieces) == 2
    head, tail = (p.message for p in pieces)
    # ag_ui_langgraph: START from the head (name, no args), ARGS from the tail (no name)
    assert head.tool_call_chunks[0]["name"] == "plan_visualization" and head.tool_call_chunks[0]["args"] == ""
    assert tail.tool_call_chunks[0]["name"] is None and tail.tool_call_chunks[0]["args"] == '{"a":1}'
    assert head.usage_metadata is None and tail.usage_metadata["total_tokens"] == 15


def test_merging_the_pieces_gives_the_original_message():
    original = _whole_chunk().message
    pieces = [p.message for p in split_tool_call_chunk(_whole_chunk())]
    merged = pieces[0]
    for p in pieces[1:]:
        merged = merged + p
    assert merged.tool_calls == original.tool_calls
    assert merged.additional_kwargs["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
    assert merged.usage_metadata["total_tokens"] == 15


def test_openai_style_chunks_pass_through_unchanged():
    msg = AIMessageChunk(content="", tool_call_chunks=[{"name": "x", "args": "", "id": "c", "index": 0, "type": "tool_call_chunk"}])
    chunk = ChatGenerationChunk(message=msg)
    assert split_tool_call_chunk(chunk) == [chunk]
    text = ChatGenerationChunk(message=AIMessageChunk(content="oi"))
    assert split_tool_call_chunk(text) == [text]


def test_stream_ends_with_a_closing_chunk_after_tool_calls():
    import asyncio

    from src.proxy_openai import SplitToolCallChatOpenAI

    finish = ChatGenerationChunk(message=AIMessageChunk(content="", response_metadata={"finish_reason": "tool_calls"}))

    class Fake(SplitToolCallChatOpenAI):
        pass

    async def fake_parent(self, *a, **k):
        yield _whole_chunk()
        yield finish

    async def collect():
        orig = SplitToolCallChatOpenAI.__mro__[1]._astream
        SplitToolCallChatOpenAI.__mro__[1]._astream = fake_parent
        try:
            model = Fake(model="x", api_key="test", base_url="http://127.0.0.1:9/v1")
            return [c async for c in model._astream([])]
        finally:
            SplitToolCallChatOpenAI.__mro__[1]._astream = orig

    out = asyncio.run(collect())
    # head, tail, finish, closing: the closing chunk has no tool call and no finish_reason, so ag_ui_langgraph emits TOOL_CALL_END
    assert len(out) == 4
    last = out[-1].message
    assert not last.tool_call_chunks and not last.response_metadata.get("finish_reason") and last.content == ""
