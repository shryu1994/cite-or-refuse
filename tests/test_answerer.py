"""Behavior tests — the three outcomes, on the synthetic Harbor corpus."""

from cite_or_refuse.answerer import build_assistant
from cite_or_refuse.data import HARBOR_CHUNKS
from cite_or_refuse.models import ResponseKind

assistant = build_assistant(HARBOR_CHUNKS)


def test_in_source_question_is_answered_with_citation():
    answer, retrieved = assistant("How much storage does the Free plan include?")
    assert answer.kind is ResponseKind.ANSWER
    assert answer.claims, "expected at least one grounded claim"
    # every claim cites a chunk that was actually retrieved
    known = {c.chunk_id for c in retrieved}
    for claim in answer.claims:
        assert claim.chunk_ids and set(claim.chunk_ids) <= known
    assert "H1" in answer.claims[0].chunk_ids


def test_missing_topic_is_refused_not_hallucinated():
    answer, _ = assistant("Does Harbor support end-to-end encryption?")
    assert answer.kind is ResponseKind.NOT_IN_SOURCES
    assert answer.claims == ()  # a refusal must make no claims


def test_account_action_is_out_of_scope():
    answer, _ = assistant("Can you reset my password?")
    assert answer.kind is ResponseKind.OUT_OF_SCOPE
    assert answer.claims == ()


def test_paraphrase_with_no_real_answer_is_refused():
    # regression: this once matched the unrelated Sync chunk on "files"/"Harbor"
    answer, _ = assistant("Does Harbor encrypt my files?")
    assert answer.kind is ResponseKind.NOT_IN_SOURCES
    assert answer.claims == ()


def test_nonexistent_entity_is_refused_not_answered_with_neighbouring_facts():
    answer, _ = assistant("How much storage does the enterprise plan include?")
    assert answer.kind is ResponseKind.NOT_IN_SOURCES


def test_in_scope_question_with_the_word_compare_is_not_blackholed():
    # the out-of-scope rules must not refuse answerable docs questions
    answer, _ = assistant("How do I compare file versions?")
    assert answer.kind is not ResponseKind.OUT_OF_SCOPE


def test_undocumented_subfeature_of_a_documented_topic_is_refused():
    # "share link" / "version history" ARE documented; "password-protect" / "tagging"
    # are not — the guard must refuse rather than cite the generic sentence.
    for q in ("Can I password-protect a share link?", "Does version history support tagging?"):
        answer, _ = assistant(q)
        assert answer.kind is ResponseKind.NOT_IN_SOURCES, q


def test_morphological_variant_question_still_answers():
    # "syncs" in the docs vs "sync" in the question must match after stemming
    answer, _ = assistant("How quickly does Harbor sync files?")
    assert answer.kind is ResponseKind.ANSWER
