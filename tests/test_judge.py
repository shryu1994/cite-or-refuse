"""The LLM-as-Judge catches what mechanical checks can't: valid-citation hallucination.

Run fully offline with a deterministic fake judge — no API key, no network.
"""

from collections.abc import Sequence

from cite_or_refuse.judge import (
    LLMFaithfulnessJudge,
    Verdict,
    check_faithfulness,
    parse_verdict,
)
from cite_or_refuse.models import Answer, Chunk, Claim, ResponseKind

CHUNK = Chunk("H1", "Harbor Docs — Storage", "The Free plan includes 5 GB of storage per user.")


class FakeJudge:
    """Scripted claim -> Verdict, so we test the check logic, not a model."""

    def __init__(self, verdicts: dict[str, Verdict]):
        self._v = verdicts

    def assess(self, claim: str, evidence: Sequence[str]) -> Verdict:
        return self._v[claim]


def test_valid_citation_but_unsupported_claim_fails():
    # cites a real, retrieved chunk (passes mechanical checks) but the chunk says
    # nothing about encryption — only a semantic judge can catch this.
    answer = Answer(
        kind=ResponseKind.ANSWER,
        claims=(Claim("The Free plan includes end-to-end encryption.", ("H1",)),),
    )
    judge = FakeJudge(
        {"The Free plan includes end-to-end encryption.": Verdict(False, "chunk mentions storage, not encryption")}
    )
    result = check_faithfulness(answer, [CHUNK], judge)
    assert result.passed is False
    assert "H1" in result.detail


def test_supported_claim_passes():
    answer = Answer(
        kind=ResponseKind.ANSWER,
        claims=(Claim("The Free plan includes 5 GB of storage per user.", ("H1",)),),
    )
    judge = FakeJudge(
        {"The Free plan includes 5 GB of storage per user.": Verdict(True, "stated verbatim")}
    )
    assert check_faithfulness(answer, [CHUNK], judge).passed is True


def test_refusals_skip_the_judge():
    answer = Answer(kind=ResponseKind.NOT_IN_SOURCES)
    assert check_faithfulness(answer, [CHUNK], FakeJudge({})).passed is True


# --- the real LLM path must survive messy, real-world model output ---

def test_judge_parses_markdown_fenced_output():
    judge = LLMFaithfulnessJudge(lambda _p: '```json\n{"supported": false, "reason": "no"}\n```')
    assert judge.assess("claim", ["evidence"]).supported is False


def test_judge_handles_stringified_booleans():
    assert parse_verdict('{"supported": "false", "reason": "x"}').supported is False
    assert parse_verdict('{"supported": "true", "reason": "x"}').supported is True


def test_judge_fails_closed_on_unparseable_or_missing_field():
    assert parse_verdict("the model rambled with no JSON at all").supported is False
    assert parse_verdict('{"reason": "no supported key"}').supported is False

