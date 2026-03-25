"""OpenAI provider implementation."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from llm.provider import LLMProvider, LLMResponse, ToolCall


class OpenAIProvider(LLMProvider):
    """OpenAI SDK implementation for the common provider interface."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or llm.openai.api_key in config.")

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        input_messages: list[dict[str, Any]] = []

        if system_prompt:
            input_messages.append({
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            })

        input_messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        })

        request: dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "temperature": self.temperature,
        }
        if tools:
            request["tools"] = self._build_tools(tools)

        response = self._client.responses.create(**request)
        response_text = getattr(response, "output_text", "") or ""
        tool_calls = self._extract_tool_calls(response)
        response_raw = response.model_dump() if hasattr(response, "model_dump") else {}

        return LLMResponse(
            text=response_text.strip(),
            tool_calls=tool_calls,
            raw=response_raw,
        )

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            spec: dict[str, Any] = {
                "type": "function",
                "name": name,
                "description": str(tool.get("description", "")),
            }
            parameters = tool.get("parameters")
            if isinstance(parameters, dict):
                spec["parameters"] = parameters
            prepared.append(spec)
        return prepared

    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
        calls: list[ToolCall] = []
        output_items = getattr(response, "output", None)
        if not isinstance(output_items, list):
            return calls

        for item in output_items:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type != "function_call":
                continue

            name = item.get("name") if isinstance(item, dict) else getattr(item, "name", "")
            raw_arguments = (
                item.get("arguments") if isinstance(item, dict) else getattr(item, "arguments", {})
            )
            parsed_arguments: dict[str, Any] = {}
            if isinstance(raw_arguments, str):
                try:
                    parsed = json.loads(raw_arguments)
                    if isinstance(parsed, dict):
                        parsed_arguments = parsed
                except json.JSONDecodeError:
                    parsed_arguments = {}
            elif isinstance(raw_arguments, dict):
                parsed_arguments = raw_arguments

            if isinstance(name, str) and name:
                raw_item = item
                if not isinstance(item, dict):
                    raw_item = item.model_dump() if hasattr(item, "model_dump") else {}
                calls.append(ToolCall(
                    name=name,
                    arguments=parsed_arguments,
                    raw=raw_item,
                ))

        return calls
