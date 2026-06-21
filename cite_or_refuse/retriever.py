"""A tiny BM25 retriever — standard library only, no embeddings, fully offline.

Kept intentionally simple: the point of this repo is the grounding + refusal + eval
discipline, not the retrieval model. Replace BM25 with a vector store and nothing
else in the pipeline has to change.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .models import Chunk

_WORD = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Crude, dependency-free normalizer: collapse simple plural / 3rd-person 's'
    (files->file, syncs->sync, includes->include) so morphological variants match.

    Not a real stemmer — just enough that "include" and "includes" are the same token,
    which matters for the undocumented-term guard in the answerer.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower())]


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._docs = [tokenize(c.text) for c in self.chunks]
        self._len = [len(d) for d in self._docs]
        self._avglen = (sum(self._len) / len(self._docs)) if self._docs else 0.0
        self._tf = [Counter(d) for d in self._docs]
        df: Counter[str] = Counter()
        for d in self._docs:
            for term in set(d):
                df[term] += 1
        n = len(self._docs)
        self._idf = {
            term: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5)) for term, dfi in df.items()
        }

    def idf(self, term: str) -> float:
        """Inverse document frequency of a term (0.0 if it never appears in the corpus).

        Exposed so the answerer can weight relevance by how *distinctive* a word is —
        a shared "the" or "files" should count for far less than a shared "encryption".
        """
        return self._idf.get(term, 0.0)

    def _score(self, query_terms: Sequence[str], i: int) -> float:
        tf = self._tf[i]
        dl = self._len[i]
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = self._idf.get(term, 0.0)
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avglen or 1))
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 3) -> list[Retrieved]:
        q = tokenize(query)
        scored = [Retrieved(self.chunks[i], self._score(q, i)) for i in range(len(self.chunks))]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
