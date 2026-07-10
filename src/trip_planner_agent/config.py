# Copyright (c) 2026 Fabian Paul. All rights reserved.

"""Environment configuration and OpenAI client construction.

Nothing here reaches out to the network at
import time; the client is built lazily via :func:`get_client`.
"""

from __future__ import annotations

import os
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI

# Load variables from a local .env file (if present) into the environment.
load_dotenv()


class OpenAIModel(str, Enum):
    """Models available for this project.

    Values are the exact model ids passed to the OpenAI API.
    """

    GPT_41 = "gpt-4.1"  # Strong general-purpose reasoning; good default for development.
    GPT_41_MINI = "gpt-4.1-mini"  # Fast and affordable; good for drafting/brainstorming.
    GPT_41_NANO = "gpt-4.1-nano"  # Fastest/cheapest; good for high-frequency calls.


# Default model used across the project (balance of speed and cost).
DEFAULT_MODEL = OpenAIModel.GPT_41_MINI


def get_client() -> OpenAI:
    """Build an OpenAI client from environment variables.

    Environment variables:
        OPENAI_API_KEY:  Required. Your API (or Vocareum) key.
        OPENAI_BASE_URL: Optional. Set to ``https://openai.vocareum.com/v1``
                         when using the Vocareum endpoint; omit for standard OpenAI.

    Returns:
        A configured :class:`openai.OpenAI` client.

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    kwargs: dict[str, str] = {"api_key": api_key}

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)
