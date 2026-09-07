"""Safe, deterministic query rewriting for the RAG retrieval step.

The rewriter never invents facts. It normalizes the current question and, when
explicit conversation context is supplied, adds the latest user turn only for
resolving common Thai follow-up references. Conversation text remains separate
from retrieved evidence and is never passed to the answer as evidence.
"""

import re
from typing import Iterable


FOLLOW_UP_MARKERS = (
    "เรื่องนี้", "เรื่องดังกล่าว", "เหตุการณ์นี้", "เหตุการณ์ดังกล่าว",
    "ข้อมูลนี้", "ดังกล่าว", "ข้างต้น", "นี้เกิด", "เขา", "เธอ",
    "เมื่อไหร่", "ที่ไหน", "อย่างไร", "ทำไม", "เท่าไหร่", "ใคร",
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def rewrite_query(original_query: str, conversation_context: Iterable[dict] | None = None) -> tuple[str, bool]:
    """Return (retrieval_query, whether_context_was_used)."""
    query = _clean(original_query)
    if not query:
        return query, False

    turns = [
        _clean(turn.get("content", ""))
        for turn in (conversation_context or [])
        if turn.get("role") == "user" and _clean(turn.get("content", ""))
    ]
    if not turns or not any(marker in query for marker in FOLLOW_UP_MARKERS):
        return query, False

    # This is a retrieval query, not an answer prompt. The previous user turn
    # supplies lexical referents; it is not treated as factual evidence.
    rewritten = f"บริบทคำถามก่อนหน้า: {turns[-1]} คำถามติดตาม: {query}"
    return rewritten, True
