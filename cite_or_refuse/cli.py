"""Command line: `ask` a question, or run the `eval` over the golden set.

    python -m cite_or_refuse.cli ask "How much storage does the Free plan include?"
    python -m cite_or_refuse.cli eval
"""

from __future__ import annotations

import sys
from pathlib import Path

from .answerer import build_assistant
from .data import HARBOR_CHUNKS
from .eval.runner import load_golden, run_eval
from .models import ResponseKind

_GOLDEN = Path(__file__).resolve().parents[1] / "evalset" / "golden.json"


def _ask(question: str) -> int:
    assistant = build_assistant(HARBOR_CHUNKS)
    answer, _ = assistant(question)
    print(f"Q: {question}")
    print(f"kind: {answer.kind.value}")
    if answer.kind is ResponseKind.ANSWER:
        for c in answer.claims:
            print(f"  - {c.text}  [{', '.join(c.chunk_ids)}]")
    else:
        print(f"  {answer.message}")
    return 0


def _eval() -> int:
    assistant = build_assistant(HARBOR_CHUNKS)
    report = run_eval(load_golden(_GOLDEN), assistant)
    print(report.to_text())
    return 0 if report.all_passed else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "ask" and rest:
        return _ask(" ".join(rest))
    if cmd == "eval":
        return _eval()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
