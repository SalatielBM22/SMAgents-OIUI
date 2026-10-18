"""Chat model factory for the agent."""

import os
from contextlib import contextmanager

from langchain.agents.middleware import AgentMiddleware
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages import AIMessageChunk
from pydantic import Field
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_MODEL = "chat-latest"

# langchain-anthropic defaults max_tokens to 4096, which truncates
# generateSandboxedUi tool args mid-stream (jsFunctions/jsExpressions never
# arrive and the widget is stuck in its preview sandbox). Widget generation
# routinely needs tens of thousands of output tokens.
MAX_TOKENS = 64000


def build_model(*, allow_unconfigured: bool = False) -> BaseChatModel:
    model_name = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    is_anthropic = model_name.startswith("claude-") and not base_url
    is_openai = not is_anthropic
    if not is_anthropic and not (model_name == "chat-latest" or model_name.startswith("gpt-") or base_url):
        raise ValueError(
            "LLM_MODEL must name a supported provider: claude-* (Anthropic) "
            "or gpt-*/chat-latest (OpenAI). Unset LLM_MODEL to use the default."
        )
    key_name = "ANTHROPIC_API_KEY" if is_anthropic else "OPENAI_API_KEY"
    if not os.environ.get(key_name, "").strip():
        if allow_unconfigured:
            return UnconfiguredModel()
        raise ValueError(
            f"{key_name} is required for the selected LLM_MODEL provider. "
            "Set it in the agent environment and restart the agent."
        )
    if is_anthropic:
        return ChatAnthropic(
            model=model_name,
            max_tokens=MAX_TOKENS,
        )
    from langchain_openai import ChatOpenAI

    if base_url:
        # OpenAI-compatible proxy (OmniRoute): split whole tool-call deltas for ag_ui_langgraph, and ask for room for widget code.
        from src.proxy_openai import SplitToolCallChatOpenAI

        return SplitToolCallChatOpenAI(
            model=model_name,
            base_url=base_url,
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "32000")),
        )
    return ChatOpenAI(model=model_name, base_url=base_url)


class UnconfiguredModel(BaseChatModel):
    """Compile a BYOK-only graph without constructing a provider with fake credentials."""

    @property
    def _llm_type(self):
        return "unconfigured"

    def _generate(self, *args, **kwargs):
        raise ValueError("Provider credentials are required. Add OpenAI and Jev keys.")

    def bind_tools(self, tools, **kwargs):
        return self


def request_model(fallback):
    from src.credentials import current_credentials
    credentials = current_credentials.get()
    if credentials is None:
        return fallback
    from langchain_openai import ChatOpenAI
    from openai import AsyncOpenAI, OpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient

    # SDK None defaults inherit server organization/project settings. Explicit
    # clients with empty scopes and isolated transports prevent that inheritance,
    # including server HTTP(S)_PROXY and OPENAI_PROXY settings.
    options = {
        "api_key": credentials.openai,
        "base_url": "https://api.openai.com/v1",
        "organization": "",
        "project": "",
    }
    sync_client = OpenAI(**options, http_client=DefaultHttpxClient(trust_env=False))
    async_client = AsyncOpenAI(
        **options, http_client=DefaultAsyncHttpxClient(trust_env=False)
    )
    # After construction, None omits these headers without rereading env vars.
    for client in (sync_client, async_client):
        client.organization = None
        client.project = None
        client.admin_api_key = None
        client.webhook_secret = None
        # The SDK merges OPENAI_CUSTOM_HEADERS even with default_headers={}.
        # Clear that merge before any request so host Authorization cannot win.
        client._custom_headers = {}
    model = ChatOpenAI(
        model=DEFAULT_MODEL,
        api_key=credentials.openai,
        base_url=options["base_url"],
        openai_proxy="",
        root_client=sync_client,
        client=sync_client.chat.completions,
        root_async_client=async_client,
        async_client=async_client.chat.completions,
    )
    # LangChain's metadata field separately inherits organization environment
    # values even when prebuilt SDK clients are supplied. Clear the inert field.
    model.openai_organization = None
    return model


@contextmanager
def provider_error_boundary():
    from src.credentials import current_credentials
    try:
        yield
    except Exception:
        if current_credentials.get() is None:
            raise
        raise ValueError("OpenAI request failed. Check your key and model access, or retry.") from None


class RequestModelMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        with provider_error_boundary():
            return handler(request.override(model=request_model(request.model)))

    async def awrap_model_call(self, request, handler):
        with provider_error_boundary():
            return await handler(request.override(model=request_model(request.model)))


class CredentialScopedModel(BaseChatModel):
    """Route DeepAgents' internal summary/subagent model calls through BYOK too."""

    fallback: BaseChatModel = Field(exclude=True)

    @property
    def _llm_type(self):
        return "credential-scoped"

    def bind_tools(self, tools, **kwargs):
        return self.bind(_credential_tools=tools, _credential_tool_options=kwargs)

    def _target(self, kwargs):
        model = with_requested_model(request_model(self.fallback))
        options = dict(kwargs)
        tools = options.pop("_credential_tools", None)
        tool_options = options.pop("_credential_tool_options", {})
        if tools is not None:
            model = model.bind_tools(tools, **tool_options)
        return model, options

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with provider_error_boundary():
            model, options = self._target(kwargs)
            message = model.invoke(messages, stop=stop, **options)
            return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        with provider_error_boundary():
            model, options = self._target(kwargs)
            message = await model.ainvoke(messages, stop=stop, **options)
            return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        with provider_error_boundary():
            model, options = self._target(kwargs)
            inner, bound = _unbind(model)
            for chunk in inner._stream(messages, stop=stop, **{**bound, **options}):
                yield ChatGenerationChunk(message=as_chunk(chunk.message))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        with provider_error_boundary():
            model, options = self._target(kwargs)
            inner, bound = _unbind(model)
            async for chunk in inner._astream(messages, stop=stop, **{**bound, **options}):
                yield ChatGenerationChunk(message=as_chunk(chunk.message))


def with_requested_model(model):
    """The server model with the combo the UI picked for this request (header x-oiu-model), if any.

    Only for server credentials behind a proxy (OmniRoute): a visitor's own keys keep their fixed model.  The copy shares the
    HTTP clients; only the model name changes, so switching combos mid-conversation needs no restart.
    """
    from src.credentials import current_credentials, current_model

    name = current_model.get()
    if not name or current_credentials.get() is not None or not os.environ.get("OPENAI_BASE_URL", "").strip():
        return model
    if getattr(model, "model_name", None) in (None, name):
        return model
    return model.model_copy(update={"model_name": name})


def _unbind(model):
    """The provider model and its bound kwargs (tools, tool_choice) when it was wrapped by bind_tools.

    Streaming goes through the provider's own _stream/_astream instead of its public stream(): the public method opens a second
    callback run under this wrapper's, LangGraph streams both, and ag_ui_langgraph saw every tool-call delta twice -- two
    TOOL_CALL_START for one id ("A tool call with ID ... is already in progress"), or an END consumed by the hidden copy ("Cannot send
    'RUN_FINISHED' while tool calls are still active").  This wrapper's own run already emits every chunk once.
    """
    from langchain_core.runnables import RunnableBinding

    if isinstance(model, RunnableBinding):
        return model.bound, dict(model.kwargs)
    return model, {}


def as_chunk(message):
    if isinstance(message, AIMessageChunk):
        return message
    return AIMessageChunk(**message.model_dump(exclude={"type"}))
