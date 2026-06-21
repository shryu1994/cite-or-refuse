"""Run the golden set through the assistant and report.

The golden set mixes three expected kinds — answer / not_in_sources / out_of_scope —
so "honestly refusing when the docs don't support an answer" is scored as a PASS, not
ignored. A RAG system that only measures answer accuracy is blind to its own hallucinations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import Answer, AssistantFn, ResponseKind

if TYPE_CHECKING:
    from ..judge import FaithfulnessJudge
from .checks import (
    CheckResult,
    check_citation_coverage,
    check_expected_kind,
    check_grounding,
)


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    question: str
    expected_kind: ResponseKind
    note: str = ""


@dataclass(frozen=True)
class CaseResult:
    case: GoldenCase
    answer: Answer
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass(frozen=True)
class Report:
    results: tuple[CaseResult, ...]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def to_text(self) -> str:
        lines = [f"Eval: {self.passed}/{self.total} passed", ""]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"[{status}] {r.case.case_id} ({r.case.expected_kind.value}) — {r.case.question}"
            )
            for c in r.answer.claims:
                lines.append(f"         · {c.text} [{', '.join(c.chunk_ids)}]")
            for ch in r.checks:
                if not ch.passed:
                    lines.append(f"         x {ch.criterion}: {ch.detail}")
        return "\n".join(lines)


def run_eval(
    cases: Sequence[GoldenCase],
    assistant: AssistantFn,
    judge: "FaithfulnessJudge | None" = None,
) -> Report:
    """Run every case through the mechanical checks, plus the faithfulness judge if given.

    The judge is optional because it needs a model: `make eval` runs the deterministic
    checks offline, and you pass a judge to add the semantic faithfulness gate.
    """
    results = []
    for case in cases:
        answer, retrieved = assistant(case.question)
        checks = [
            check_expected_kind(case.expected_kind, answer),
            check_citation_coverage(answer),
            check_grounding(answer, retrieved),
        ]
        if judge is not None:
            from ..judge import check_faithfulness

            checks.append(check_faithfulness(answer, retrieved, judge))
        results.append(CaseResult(case, answer, tuple(checks)))
    return Report(tuple(results))


def load_golden(path: Path) -> tuple[GoldenCase, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        GoldenCase(
            case_id=i["case_id"],
            question=i["question"],
            expected_kind=ResponseKind(i["expected_kind"]),
            note=i.get("note", ""),
        )
        for i in raw
    )
