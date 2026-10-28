"""OpenAI-compatible models with explicit, accounted retries and ordered fallbacks."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_openai import ChatOpenAI

from nasdaq_agent.config import Settings
from nasdaq_agent.infra.usage import BudgetExceeded, estimate_tokens
from nasdaq_agent.llm import tracked


def _model(settings: Settings, name: str, primary: bool = False) -> BaseChatModel:
    """SDK retries are disabled so every actual attempt reserves budget first."""
    creds = settings.providers[name]
    token_parameter = creds.output_token_parameter or ("max_tokens" if name == "deepseek" else "max_completion_tokens")
    return ChatOpenAI(
        model=creds.model,
        api_key=creds.api_key,
        base_url=creds.base_url,
        temperature=settings.llm.temperature,
        timeout=settings.llm.primary_timeout_s if primary else settings.llm.timeout_s,
        max_retries=0,
        # DeepSeek documents max_tokens; MiMo/OpenAI use max_completion_tokens.
        # extra_body avoids ChatOpenAI rewriting the former to the latter.
        max_tokens=settings.llm.max_output_tokens if token_parameter == "max_completion_tokens" else None,
        extra_body={"max_tokens": settings.llm.max_output_tokens} if token_parameter == "max_tokens" else None,
        tags=[f"provider:{name}"],
        metadata={"provider": name, "model": creds.model},
    )


def provider_order(settings: Settings) -> list[str]:
    order = [settings.llm.provider, *settings.llm.fallback]
    seen: list[str] = []
    for name in order:
        if name in settings.providers and name not in seen:
            seen.append(name)
    if not seen:
        raise RuntimeError(
            "No LLM provider configured. Set e.g. MIMO_API_KEY/MIMO_BASE_URL/MIMO_MODEL in .env."
        )
    return seen


def _build(settings, *, tools=None, schema=None, stage):
    order = provider_order(settings)
    models = []
    tool_specs = [convert_to_openai_tool(t) for t in tools] if tools else []
    if schema:
        tool_specs = [convert_to_openai_tool(schema)]
    for index, name in enumerate(order):
        fast_primary = index == 0 and len(order) > 1
        model = _model(settings, name, primary=fast_primary)
        if schema:
            # Invoke the raw model directly: RunnableParallel in include_raw=True
            # drops invoke kwargs, including the remaining-deadline timeout.
            model = model.bind_tools([schema], tool_choice=tool_specs[0]["function"]["name"], parallel_tool_calls=False)
        elif tools:
            model = model.bind_tools(tools, parallel_tool_calls=True)
        models.append((name, model, fast_primary))

    def invoke(messages, config: RunnableConfig):
        input_estimate = estimate_tokens({"messages": messages, "tools": tool_specs})
        for index, (name, model, fast_primary) in enumerate(models):
            retries = 0 if fast_primary else settings.llm.max_retries
            for attempt in range(1, retries + 2):
                try:
                    response = tracked.request(
                        settings, provider=name, model=settings.providers[name].model, stage=stage,
                        attempt=attempt, fallback=index > 0, input_estimate=input_estimate,
                        output_estimate=settings.llm.max_output_tokens,
                        timeout_s=settings.llm.primary_timeout_s if fast_primary else settings.llm.timeout_s,
                        invoke=lambda timeout: model.invoke(messages, config=config, timeout=timeout))
                    if schema:
                        parsed = PydanticToolsParser(tools=[schema], first_tool_only=True).invoke(response)
                        if parsed is None:
                            raise ValueError("structured model returned no schema tool call")
                        return parsed
                    return response
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    if attempt <= retries and tracked.retryable(exc):
                        tracked.wait_before_retry(exc, attempt)
                        continue
                    if index == len(models) - 1:
                        raise
                    break
        raise RuntimeError("all model providers failed")

    return RunnableLambda(invoke)


def build_chat(settings: Settings, tools: Sequence[Any] | None = None) -> Runnable:
    return _build(settings, tools=tools, stage="agent")


def build_structured(settings: Settings, schema: type) -> Runnable:
    return _build(settings, schema=schema, stage="sentiment")


def describe(settings: Settings) -> str:
    return " -> ".join(f"{n}:{settings.providers[n].model}" for n in provider_order(settings))
