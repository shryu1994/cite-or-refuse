"""The grounding contract: cite a *relevant* source, or refuse. Never fabricate.

Three outcomes:
  ANSWER         — a retrieved sentence actually addresses the question; it is cited.
  NOT_IN_SOURCES — the docs don't support an answer; honestly say so.
  OUT_OF_SCOPE   — the question is outside what this assistant answers.

Two cheap, transparent gates keep confident hallucination out — *and neither is perfect*,
which is exactly why the faithfulness judge (judge.py) is the real semantic backstop:

  1. Undocumented-term guard. If a question contains a content word that appears NOWHERE
     in the corpus (e.g. "password", "enterprise", "encryption"), it is asking about
     something the docs don't cover, so we refuse instead of matching on the generic words
     it happens to share ("share", "link", "plan"). This is what stops "Can I
     password-protect a share link?" from being answered with the plain share-link
     sentence. It is deliberately biased toward refusing: a question phrased with synonyms
     the docs don't use ("operating systems" vs "Windows/macOS") will also be refused.

  2. IDF-weighted relevance floor. Among in-vocabulary questions, the best sentence must
     clear a relevance bar so loosely-related matches don't become answers.

Answers are *extractive* (verbatim sentences), so provenance grounding holds by
construction. Lexical relevance cannot judge meaning — that is the judge's job.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import Answer, AssistantFn, Chunk, Claim, ResponseKind
from .retriever import BM25Retriever, Retrieved, _stem, tokenize

# Function words / degree / manner / indefinite words that carry no topical meaning, so
# their absence from the docs is not a signal that the question is undocumented.
_STOPWORDS_RAW = """
    a an the this that these those
    i you he she it we they me him her us them my your his its our their
    is am are was were be been being do does did doing done have has had having
    can could will would shall should may might must
    of to in on at for with from by about as into than then over under up down out off
    and or but if so because while
    how what which who whom whose when where why
    much many more most less least some any all no none both each every
    not yes get got here there
    please just also only really very quite too able
    long short quickly fast slow slowly soon often always never sometimes far near
    big small good bad better worse best worst new old high low
    someone something anyone anything everyone everything somebody anybody nobody nothing
""".split()
# Match the stemmed token stream (tokenize stems, so "does" -> "doe", etc.).
_STOP = frozenset(_stem(w) for w in _STOPWORDS_RAW)

# Out-of-scope rules. Kept narrow on purpose: only patterns almost never part of a genuine
# docs question, so answerable questions are not blackholed.
_OUT_OF_SCOPE_PATTERNS = [
    r"\breset my password\b",
    r"\bmy account\b",
    r"\bcancel my\b",
    r"\brefund\b",
    r"\bspeak to (a )?human\b",
    r"\b(better|worse) than\b",
]

_SENT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


class Answerer:
    def __init__(
        self,
        retriever: BM25Retriever,
        min_score: float = 2.0,
        top_k: int = 3,
        cite_margin: float = 0.65,
        answer_floor: float = 2.5,
    ):
        self.retriever = retriever
        self.min_score = min_score
        self.top_k = top_k
        self.cite_margin = cite_margin
        self.answer_floor = answer_floor

    def _content_terms(self, question: str) -> list[str]:
        return [t for t in tokenize(question) if t not in _STOP]

    def answer(self, question: str) -> tuple[Answer, Sequence[Retrieved]]:
        if self._is_out_of_scope(question):
            return (
                Answer(
                    kind=ResponseKind.OUT_OF_SCOPE,
                    message="That's outside what the docs assistant answers — please contact support.",
                ),
                [],
            )

        # Gate 1 — undocumented-term guard (see module docstring).
        if any(self.retriever.idf(t) == 0.0 for t in self._content_terms(question)):
            return (
                Answer(
                    kind=ResponseKind.NOT_IN_SOURCES,
                    message="The question mentions something the documents don't cover.",
                ),
                [],
            )

        hits = self.retriever.search(question, top_k=self.top_k)
        if not hits or hits[0].score < self.min_score:
            return (
                Answer(kind=ResponseKind.NOT_IN_SOURCES, message="The answer isn't in the provided documents."),
                hits,
            )

        claims = self._extract_claims(question, hits)
        if not claims:
            return (
                Answer(kind=ResponseKind.NOT_IN_SOURCES, message="The documents are related but don't directly answer this."),
                hits,
            )
        return (Answer(kind=ResponseKind.ANSWER, claims=tuple(claims)), hits)

    def _is_out_of_scope(self, question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q) for p in _OUT_OF_SCOPE_PATTERNS)

    def _relevance(self, q_terms: set[str], sentence: str) -> float:
        """IDF-weighted overlap: how much *distinctive* question vocabulary the sentence shares."""
        shared = q_terms & set(tokenize(sentence))
        return sum(self.retriever.idf(t) for t in shared)

    def _extract_claims(self, question: str, hits: Sequence[Retrieved]) -> list[Claim]:
        q_terms = set(self._content_terms(question))
        cite_floor = max(self.min_score, hits[0].score * self.cite_margin)
        claims: list[Claim] = []
        for hit in hits:
            if hit.score < cite_floor:
                continue
            best_sent, best_rel = None, 0.0
            for sent in _split_sentences(hit.chunk.text):
                rel = self._relevance(q_terms, sent)
                if rel > best_rel:
                    best_rel, best_sent = rel, sent
            if best_sent is not None and best_rel >= self.answer_floor:
                claims.append(Claim(text=best_sent, chunk_ids=(hit.chunk.chunk_id,)))

        seen: set[str] = set()
        unique: list[Claim] = []
        for c in claims:
            if c.text not in seen:
                seen.add(c.text)
                unique.append(c)
        return unique[:2]


def build_assistant(chunks: Sequence[Chunk], **kwargs) -> AssistantFn:
    retriever = BM25Retriever(chunks)
    answerer = Answerer(retriever, **kwargs)

    def assistant(question: str) -> tuple[Answer, Sequence[Chunk]]:
        answer, hits = answerer.answer(question)
        return answer, [h.chunk for h in hits]

    return assistant
