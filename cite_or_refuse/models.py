"""Core types — the structure that makes grounding checkable.

Forcing answers into (kind, claims-with-citations) is the whole trick: it turns
"is this answer faithful?" from a vibe into something a machine can check.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum


class ResponseKind(str, Enum):
    ANSWER = "answer"                  # grounded answer; every claim cites a source
    NOT_IN_SOURCES = "not_in_sources"  # honest refusal: the docs don't support an answer
    OUT_OF_SCOPE = "out_of_scope"      # outside what this assistant answers


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Claim:
    """One assertion in an answer. chunk_ids = supporting sources (empty => ungrounded)."""

    text: str
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Answer:
    kind: ResponseKind
    claims: tuple[Claim, ...] = ()
    message: str = ""  # human-facing text (e.g. the refusal reason)


# The assistant contract shared by the answerer and the eval runner:
# a question maps to an answer plus the chunks retrieval actually provided.
AssistantFn = Callable[[str], "tuple[Answer, Sequence[Chunk]]"]
