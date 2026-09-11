"""Thin, provider-agnostic wrapper around Anthropic / OpenAI / Groq.

Groq exposes an OpenAI-compatible chat completions API, so it reuses the
OpenAI SDK with a different base_url. `instructor` patches whichever
underlying client we build so we can request a Pydantic model back from
any of the three providers with the same call shape.
"""

from __future__ import annotations

import base64
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import settings

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or settings.llm_provider
        self._instructor_client = None
        self._raw_client = None

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------
    def _build_raw_client(self) -> Any:
        if self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        if self.provider == "openai":
            import openai

            return openai.OpenAI(api_key=settings.openai_api_key)
        if self.provider == "groq":
            import openai

            return openai.OpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    @property
    def raw_client(self) -> Any:
        if self._raw_client is None:
            self._raw_client = self._build_raw_client()
        return self._raw_client

    @property
    def instructor_client(self) -> Any:
        if self._instructor_client is None:
            import instructor

            if self.provider == "anthropic":
                self._instructor_client = instructor.from_anthropic(self.raw_client)
            else:  # openai + groq both speak the OpenAI protocol
                self._instructor_client = instructor.from_openai(self.raw_client)
        return self._instructor_client

    @property
    def model(self) -> str:
        return {
            "anthropic": settings.anthropic_model,
            "openai": settings.openai_model,
            "groq": settings.groq_model,
        }[self.provider]

    @property
    def vision_model(self) -> str:
        if self.provider == "groq":
            return settings.groq_vision_model
        return self.model

    # ------------------------------------------------------------------
    # Structured extraction (instructor-enforced Pydantic schema)
    # ------------------------------------------------------------------
    def structured_extract(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> T:
        if self.provider == "anthropic":
            system = "\n".join(m["content"] for m in messages if m["role"] == "system")
            user_messages = [m for m in messages if m["role"] != "system"]
            return self.instructor_client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or None,
                messages=user_messages,
                response_model=response_model,
            )
        return self.instructor_client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            response_model=response_model,
        )

    # ------------------------------------------------------------------
    # Vision: extract structured text from a page image
    # ------------------------------------------------------------------
    def vision_extract_text(self, image_bytes: bytes, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        if self.provider == "anthropic":
            response = self.raw_client.messages.create(
                model=self.vision_model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return response.content[0].text

        # OpenAI + Groq (OpenAI-compatible vision message format)
        response = self.raw_client.chat.completions.create(
            model=self.vision_model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""


_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
