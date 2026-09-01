from __future__ import annotations

import os
from typing import Any, Optional

from groq import Groq

from pandasai.agent.state import AgentState
from pandasai.core.prompts.base import BasePrompt
from pandasai.exceptions import APIKeyNotFoundError
from pandasai.llm.base import LLM


class GroqLLM(LLM):
    """Groq LLM integration for PandasAI.

    This implementation communicates directly with the
    Groq API through the official Groq Python SDK.

    LiteLLM is intentionally not used.
    """

    model: str = "openai/gpt-oss-20b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the Groq LLM.

        Args:
            api_key: Groq API key. If omitted, GROQ_API_KEY
                is read from the environment.
            model: Groq model identifier.
            **kwargs: Additional parameters passed to Groq.
        """

        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or None
        )

        if not self.api_key:
            raise APIKeyNotFoundError(
                "Groq API key is required"
            )

        if model:
            self.model = model

        self.params = kwargs

        self.client = Groq(
            api_key=self.api_key
        )

    @property
    def type(self) -> str:
        """Return the provider type."""

        return "groq"

    def call(
        self,
        instruction: BasePrompt,
        context: AgentState = None,
    ) -> str:
        """Send a prompt to Groq and return its response."""

        memory = context.memory if context else None

        self.last_prompt = self.prepend_system_prompt(
            instruction.to_string(),
            memory,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.last_prompt,
                }
            ],
            **self.params,
        )

        return response.choices[0].message.content