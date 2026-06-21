"""Faithfulness LLM-as-Judge — does the cited chunk *actually support* the claim?

The mechanical checks verify the plumbing: a citation exists, and the cited chunk was
retrieved. They cannot catch a *valid-citation hallucination* — a claim that cites a
real, retrieved chunk but misrepresents what it says. Catching that needs a model.

Discipline (kept in this docstring on purpose): an LLM judge is itself an unvalidated
model. Never trust it as a measurement until you have *calibrated* it against human
labels. A judge you haven't checked is just another hallucination you've promoted to a
gate. So this judge fails *closed*: any output it can't parse is treated as unsupported,
never as a pass. It is pluggable, so it runs with any provider — and the tests exercise
it deterministically with a fake, fully offline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .eval.checks import CheckResult
from .models import Answer, Chunk, ResponseKind


@dataclass(frozen=True)
class Verdict:
    supported: bool
    reason: str = ""


class FaithfulnessJudge(Protocol):
    def assess(self, claim: str, evidence: Sequence[str]) -> Verdict:
        """Does `claim` follow from `evidence` (the cited chunk texts)?"""
        ...


_PROMPT = """You grade the faithfulness of a RAG answer.
Decide whether the CLAIM follows using ONLY the explicit content of the EVIDENCE.
No outside knowledge, no guessing. If the evidence does not support the claim, supported=false.

EVIDENCE:
{evidence}

CLAIM:
{claim}

Reply with one JSON line: {{"supported": true or false, "reason": "<one sentence>"}}
"""


def parse_verdict(raw: str) -> Verdict:
    """Parse a model's reply into a Verdict, tolerant of real-world LLM output.

    Handles markdown code fences, leading/trailing prose, and a `supported` value that
    comes back as a JSON string instead of a bool. Anything it cannot parse — or a missing
    `supported` key — fails closed to `supported=False` (see the module docstring).
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)  # first {...}, ignoring fences/prose
    if not match:
        return Verdict(False, "judge output contained no JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(False, "judge output was not valid JSON")

    value = data.get("supported")
    if isinstance(value, bool):
        supported = value
    elif isinstance(value, str):
        supported = value.strip().lower() == "true"
    else:
        supported = False  # missing/unknown -> conservative: treat as unsupported
    return Verdict(supported, str(data.get("reason", "")))


@dataclass(frozen=True)
class LLMFaithfulnessJudge:
    """Vendor-neutral judge: inject any `prompt -> text` callable (Anthropic, etc.)."""

    complete: Callable[[str], str]

    def assess(self, claim: str, evidence: Sequence[str]) -> Verdict:
        body = "\n".join(f"- {e}" for e in evidence) or "(no evidence)"
        return parse_verdict(self.complete(_PROMPT.format(evidence=body, claim=claim)))


def check_faithfulness(
    answer: Answer, retrieved: Sequence[Chunk], judge: FaithfulnessJudge
) -> CheckResult:
    """Semantic grounding — each claim's cited chunk must actually support it."""
    if answer.kind is not ResponseKind.ANSWER:
        return CheckResult("faithfulness", True)

    text_by_id = {c.chunk_id: c.text for c in retrieved}
    failures = []
    for claim in answer.claims:
        evidence = [text_by_id[cid] for cid in claim.chunk_ids if cid in text_by_id]
        verdict = judge.assess(claim.text, evidence)
        if not verdict.supported:
            failures.append(f"[{', '.join(claim.chunk_ids)}] {claim.text} — {verdict.reason}")

    return CheckResult(
        "faithfulness",
        not failures,
        ("unsupported: " + " / ".join(failures)) if failures else "",
    )
