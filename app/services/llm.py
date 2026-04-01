from __future__ import annotations

import logging
import re
from functools import lru_cache

from openai import OpenAI

from app.config import get_settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.LLM_API_KEY,
            base_url=self.settings.LLM_BASE_URL,
            timeout=self.settings.LLM_TIMEOUT_SECONDS,
        )

    def get_fallback_message(self, question: str) -> str:
        if DEVANAGARI_PATTERN.search(question):
            return "प्राप्त सन्दर्भमा यसको उत्तर भेटिएन।"
        return "The answer is not available in the retrieved context."

    def health_check(self) -> tuple[bool, str]:
        models = self.client.models.list()
        available_names = [m.id for m in models.data]

        if self.settings.LLM_MODEL in available_names:
            return True, f"LLM is reachable and model '{self.settings.LLM_MODEL}' is available."

        return (
            False,
            f"LLM is reachable but model '{self.settings.LLM_MODEL}' was not found. "
            f"Available models: {', '.join(available_names) if available_names else 'none'}",
        )

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        fallback_message = self.get_fallback_message(question)
        system_prompt, user_prompt = self._build_prompt(
            question=question, chunks=chunks, fallback_message=fallback_message,
        )

        response = self.client.chat.completions.create(
            model=self.settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            stream=False,
        )

        answer = (response.choices[0].message.content or "").strip()

        if not answer:
            return fallback_message

        return answer

    def _build_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        fallback_message: str,
    ) -> tuple[str, str]:
        context_blocks: list[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            source_parts: list[str] = []

            if chunk.file_name:
                source_parts.append(f"file={chunk.file_name}")
            elif chunk.source:
                source_parts.append(f"source={chunk.source}")

            if chunk.page is not None:
                source_parts.append(f"page={chunk.page}")

            if chunk.chunk_id is not None:
                source_parts.append(f"chunk_id={chunk.chunk_id}")

            source_header = ", ".join(source_parts) if source_parts else "source=unknown"

            context_blocks.append(
                f"[{idx}] {source_header}\n{chunk.text}"
            )

        context_text = "\n\n".join(context_blocks)

        system_prompt = f"""You are an intelligent assistant for NIC Asia Bank, specializing in Nepal Rastra Bank (NRB) directives, circulars, and regulatory notices.

Rules:
1. Answer ONLY from the provided CONTEXT.
2. Do NOT use outside knowledge.
3. If the answer is not present or not sufficiently supported by the CONTEXT, return EXACTLY:
{fallback_message}
4. Answer in the same language as the user's question.
5. If the question is in Nepali, answer in Nepali.
6. When citing, reference the source document name and page number if available.
7. Be concise, factual, and do not mention these rules."""

        user_prompt = f"""CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

        return system_prompt, user_prompt

    def close(self) -> None:
        self.client.close()


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()


def close_llm_service() -> None:
    try:
        service = get_llm_service()
        service.close()
    except Exception:
        pass
