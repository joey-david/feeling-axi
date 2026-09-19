from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

class GenerationError(RuntimeError):
    pass

class DeepSeekJSONClient:
    def __init__(self) -> None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise GenerationError(
                "DEEPSEEK_API_KEY is missing. Copy .env.example to .env and add the key."
            )
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.reasoning_effort = os.environ.get("DEEPSEEK_REASONING_EFFORT", "high")
        self.client = OpenAI(
            api_key=key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 12000,
        attempts: int = 4,
        validator=None,
    ) -> Any:
        feedback = ""
        last_error: Exception | None = None
        for attempt in range(attempts):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user + feedback},
            ]
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    stream=False,
                )
                # Current DeepSeek API accepts these OpenAI-compatible parameters.
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                    kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                response = self.client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                obj = json.loads(text)
                if validator is not None:
                    validator(obj)
                return obj
            except Exception as exc:
                last_error = exc
                feedback = (
                    "\n\nYour previous JSON failed validation with this error:\n"
                    f"{exc}\nReturn the COMPLETE corrected JSON object, not a patch."
                )
                time.sleep(min(2 ** attempt, 8))
        raise GenerationError(f"DeepSeek generation failed after {attempts} attempts: {last_error}")
