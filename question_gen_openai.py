"""
Shared OpenAI helpers for document-grounded question generation (JSON output).
Used by create_test_cases.py and run_markdown_pipeline.py.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import aiohttp

log = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.environ.get("OPENAI_COMPLETENESS_MODEL", "gpt-4o")

QUESTION_GEN_PROMPT = """\
You are given a document in Markdown (it may contain Markdown tables — treat table cells as first-class facts).

**Document:**
{document}

Generate exactly {n} distinct questions about this document.
Requirements:
- Each question must be answerable only from the document.
- At least half of the questions should explicitly require using information from tables or numeric/tabular detail when tables exist.
- Questions should be concrete (not "summarize the document").

Return a JSON object with exactly this shape and no other keys:
{{"questions": ["...", "..."]}}
"""


def truncate_document(text: str, max_chars: int = 120_000) -> str:
    if len(text) <= max_chars:
        return text
    log.warning(
        "Document truncated from %d to %d chars for API prompt limits.",
        len(text),
        max_chars,
    )
    return text[:max_chars] + "\n\n[... truncated ...]"


async def chat_completion_text(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    *,
    max_tokens: int = 2000,
    response_format: Optional[dict[str, str]] = None,
) -> str:
    payload: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if response_format is not None:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with session.post(
        OPENAI_URL,
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        body_text = await resp.text()
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError:
            data = {}

    if resp.status != 200:
        raise RuntimeError(f"OpenAI HTTP {resp.status}: {body_text[:2000]}")

    err = data.get("error")
    if err:
        raise RuntimeError(f"OpenAI error: {err}")

    choices = data.get("choices")
    if not choices:
        raise RuntimeError(f"No choices in response: {str(data)[:2000]}")

    content = choices[0].get("message", {}).get("content")
    if content is None:
        raise RuntimeError(f"Missing message content: {choices[0]}")

    return content.strip()


async def generate_questions_for_document(
    session: aiohttp.ClientSession,
    api_key: str,
    document_md: str,
    n: int,
) -> list[str]:
    doc = truncate_document(document_md)
    prompt = QUESTION_GEN_PROMPT.format(document=doc, n=n)
    raw = await chat_completion_text(
        session,
        api_key,
        prompt,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    obj = json.loads(raw)
    qs = obj.get("questions") or []
    cleaned = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
    if len(cleaned) < n:
        log.warning("Expected %d questions, got %d.", n, len(cleaned))
    return cleaned[:n]
