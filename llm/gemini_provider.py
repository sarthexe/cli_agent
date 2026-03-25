"""Gemini provider implementation."""

from __future__ import annotations

from typing import Any

import httpx

from llm.provider import LLMProvider, LLMResponse, ToolCall


class GeminiProvider(LLMProvider):
    """Google Generative AI implementation for the common provider interface."""

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
            raise ValueError("Missing API key. Set GEMINI_API_KEY or llm.gemini.api_key in config.")

        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout)

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": self._build_function_declarations(tools)}]

        response = self._client.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_response(data)

    def _build_function_declarations(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            declaration: dict[str, Any] = {
                "name": str(tool.get("name", "")),
                "description": str(tool.get("description", "")),
            }
            parameters = tool.get("parameters")
            if isinstance(parameters, dict):
                declaration["parameters"] = parameters
            declarations.append(declaration)
        return declarations

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return LLMResponse(text="", tool_calls=[], raw=data)

        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content", {}) if isinstance(first, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []

        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    text_chunks.append(text)

                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    name = function_call.get("name")
                    args = function_call.get("args", {})
                    if not isinstance(args, dict):
                        args = {}
                    if isinstance(name, str) and name:
                        tool_calls.append(ToolCall(name=name, arguments=args, raw=function_call))

        return LLMResponse(
            text="\n".join(text_chunks).strip(),
            tool_calls=tool_calls,
            raw=data,
        )
