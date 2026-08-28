"""
Legacy LLM client module.

The actual LLM is now managed by the modified PandasAI
GroqLLM implementation in:

    G:/pandas-ai/pandasai/llm/groq.py

This module is intentionally kept temporarily so that
other parts of Kavion_small do not break if they still
import core.llm.client.
"""

import os

from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )