"""The eval harness must pass on the golden set — including the refusal cases."""

from pathlib import Path

from cite_or_refuse.answerer import build_assistant
from cite_or_refuse.data import HARBOR_CHUNKS
from cite_or_refuse.eval.checks import check_expected_kind, check_grounding
from cite_or_refuse.eval.runner import GoldenCase, load_golden, run_eval
from cite_or_refuse.judge import Verdict
from cite_or_refuse.models import Answer, Chunk, Claim, ResponseKind

_GOLDEN = Path(__file__).resolve().parents[1] / "evalset" / "golden.json"


def test_golden_set_all_pass():
    assistant = build_assistant(HARBOR_CHUNKS)
    report = run_eval(load_golden(_GOLDEN), assistant)
    assert report.all_passed, "\n" + report.to_text()


def test_refusal_cases_are_actually_refusals():
    # refusal-as-eval: the golden set really does exercise non-answer kinds
    cases = load_golden(_GOLDEN)
    kinds = {c.expected_kind for c in cases}
    assert ResponseKind.NOT_IN_SOURCES in kinds
    assert ResponseKind.OUT_OF_SCOPE in kinds


def test_grounding_check_catches_phantom_citation():
    # a claim citing a chunk that was NOT retrieved must fail grounding
    bad = Answer(kind=ResponseKind.ANSWER, claims=(Claim("anything", ("GHOST",)),))
    retrieved = [Chunk("H1", "s", "t")]
    assert check_grounding(bad, retrieved).passed is False


def test_refusal_carrying_a_claim_fails_expected_kind():
    # the project's core invariant: a refusal must not smuggle in a fabricated claim
    sneaky = Answer(kind=ResponseKind.NOT_IN_SOURCES, claims=(Claim("smuggled", ("H1",)),))
    assert check_expected_kind(ResponseKind.NOT_IN_SOURCES, sneaky).passed is False


def test_eval_can_actually_go_red():
    # an assistant that always answers must FAIL the refusal cases -> report not all_passed
    def yes_man(_q):
        return (
            Answer(kind=ResponseKind.ANSWER, claims=(Claim("anything", ("H1",)),)),
            [Chunk("H1", "s", "t")],
        )

    cases = (GoldenCase("X", "q", ResponseKind.NOT_IN_SOURCES),)
    report = run_eval(cases, yes_man)
    assert report.all_passed is False
    assert report.passed < report.total


def test_run_eval_with_judge_catches_unfaithful_answer():
    # wire the judge into the runner and prove a valid-citation hallucination fails the case
    def bad_assistant(_q):
        return (
            Answer(kind=ResponseKind.ANSWER, claims=(Claim("Harbor offers encryption.", ("H1",)),)),
            [Chunk("H1", "Storage", "The Free plan includes 5 GB of storage per user.")],
        )

    class NeverSupports:
        def assess(self, claim, evidence):
            return Verdict(False, "not stated in the chunk")

    report = run_eval((GoldenCase("X", "q", ResponseKind.ANSWER),), bad_assistant, judge=NeverSupports())
    assert report.all_passed is False
    failed = {c.criterion for r in report.results for c in r.checks if not c.passed}
    assert "faithfulness" in failed
