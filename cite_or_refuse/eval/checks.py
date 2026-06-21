"""Mechanical (deterministic) checks — the plumbing of grounding.

These are pure Python: same input, same result, no model, no network. They verify the
*form* of grounding. The semantic question ("does the chunk actually support the claim?")
is left to the optional LLM-as-Judge in judge.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..models import Answer, Chunk, ResponseKind


@dataclass(frozen=True)
class CheckResult:
    criterion: str
    passed: bool
    detail: str = ""


def check_citation_coverage(answer: Answer) -> CheckResult:
    """Every claim must carry at least one citation."""
    uncited = [c.text for c in answer.claims if not c.chunk_ids]
    return CheckResult(
        "citation_coverage",
        not uncited,
        ("uncited: " + " / ".join(uncited)) if uncited else "",
    )


def check_grounding(answer: Answer, retrieved: Sequence[Chunk]) -> CheckResult:
    """Cited chunk ids must exist in what retrieval actually provided (no phantom sources)."""
    known = {c.chunk_id for c in retrieved}
    unknown = [cid for cl in answer.claims for cid in cl.chunk_ids if cid not in known]
    return CheckResult(
        "grounding",
        not unknown,
        ("phantom citations: " + ", ".join(unknown)) if unknown else "",
    )


def check_expected_kind(expected: ResponseKind, answer: Answer) -> CheckResult:
    """Right response kind — AND a refusal must not smuggle in fabricated claims.

    This is where "refusal as a first-class eval" lives: NOT_IN_SOURCES / OUT_OF_SCOPE
    only pass if the system actually refused *and* made no claims.
    """
    if answer.kind is not expected:
        return CheckResult("expected_kind", False, f"expected {expected.value}, got {answer.kind.value}")
    if expected is not ResponseKind.ANSWER and answer.claims:
        return CheckResult("expected_kind", False, f"{expected.value} carried {len(answer.claims)} claim(s)")
    return CheckResult("expected_kind", True)
