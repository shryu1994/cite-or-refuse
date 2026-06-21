# cite-or-refuse

> A documentation Q&A assistant that **cites its sources or honestly refuses** — and scores the refusing as a first-class eval.

The most common way a RAG system fails in production isn't a wrong answer — it's a
**confident hallucination**: a fluent reply to a question the documents don't actually
answer. `cite-or-refuse` is a small, readable reference implementation that treats *"I
don't know"* as a measurable product requirement, not an afterthought.

Everything here runs on your machine with **no API key and no network** — clone it and
`make eval` in seconds. (Synthetic data only; the corpus is a fictional product.)

## The idea in one screen

Every reply is forced into one of three shapes, so faithfulness becomes *checkable*:

| Kind | When | Example |
|---|---|---|
| `answer` | a retrieved sentence actually addresses it | *"The Free plan includes 5 GB of storage per user."* `[H1]` |
| `not_in_sources` | the docs don't support an answer | *"The answer isn't in the provided documents."* |
| `out_of_scope` | the question isn't this assistant's job | *"That's outside what the docs assistant answers — contact support."* |

```console
$ python -m cite_or_refuse.cli ask "How much storage does the Free plan include?"
Q: How much storage does the Free plan include?
kind: answer
  - The Free plan includes 5 GB of storage per user.  [H1]

$ python -m cite_or_refuse.cli ask "Can I password-protect a share link?"
Q: Can I password-protect a share link?
kind: not_in_sources
  The question mentions something the documents don't cover.
```

That second question is the hard case: *share links* are documented, but
*password-protecting* them is not — so a naive system happily cites the generic
share-link sentence (it shares the words "share" and "link"). This one refuses. See
[Two cheap gates](#two-cheap-gates-and-why-neither-is-enough).

## Refusal as a first-class eval

Most RAG evals only measure *answer accuracy* — which is blind to the failure that
matters most. Here the golden set mixes all three kinds, so **honestly refusing scores
as a PASS**:

```console
$ make eval
Eval: 11/11 passed
```

The 11 cases are 3 `answer`, 6 `not_in_sources`, 2 `out_of_scope` — including the exact
paraphrases that an adversarial review caught this system getting wrong (`N3`–`N6`),
pinned so they can't regress. Three deterministic checks run on every case
(`cite_or_refuse/eval/checks.py`):

- **citation_coverage** — every claim carries at least one source.
- **grounding** — cited chunk ids exist in what retrieval returned (no phantom citations).
- **expected_kind** — the right kind *and* a refusal smuggles in no fabricated claims.

These are pure Python — same input, same result, no model, no network.

## Two cheap gates (and why neither is enough)

A claim is only made if the question passes two transparent, lexical gates
(`answerer.py`):

1. **Undocumented-term guard.** If a question contains a content word that appears
   *nowhere* in the corpus (e.g. `password`, `enterprise`, `encryption`), it's asking
   about something the docs don't cover — so it refuses instead of matching on the
   generic words it happens to share. This is what stops the `password-protect` example.
2. **IDF-weighted relevance floor.** Among in-vocabulary questions, the best sentence
   must clear a relevance bar (shared words weighted by how *distinctive* they are) so
   loosely-related matches don't become answers.

**Neither gate understands meaning, and that's the point.** They are biased toward
*refusing*: a question phrased with synonyms the docs don't use ("operating systems" vs.
the documented "Windows / macOS") will also be refused. Over-refusing is the safe
direction for a tool called *cite-or-refuse* — but the real semantic check is the judge.

## The semantic backstop: an optional faithfulness LLM-as-Judge

The lexical gates and mechanical checks verify *form*. They can't catch a
**valid-citation hallucination** — a claim that cites a real, retrieved chunk but
misrepresents what it says — and they can't reason about meaning. That needs a model
(`cite_or_refuse/judge.py`), so it's **optional**: `make eval` runs the deterministic
checks offline; you pass a judge to add the semantic gate:

```python
report = run_eval(cases, assistant, judge=my_judge)   # adds a faithfulness check per case
```

The judge is **vendor-neutral** (inject any `prompt -> text` callable) and **fails
closed**: any output it can't parse — markdown fences, stray prose, a stringified
`"false"`, a missing field — is treated as *unsupported*, never as a pass. The tests
exercise it deterministically with a fake, fully offline. The discipline is in the
docstring on purpose:

> An LLM judge is itself an unvalidated model. Never trust it as a measurement until you
> have **calibrated** it against human labels. A judge you haven't checked is just
> another hallucination you've promoted to a gate.

## Run it

```bash
make eval                       # run the golden set (exits non-zero if any case fails)
make test                       # pytest — 20 tests, fully offline
make demo                       # ask a sample question
python -m cite_or_refuse.cli ask "your question here"

pip install pytest              # the only dev dependency, if you don't have it
```

No runtime dependencies beyond the standard library. Runs on Python 3.9+. The CLI is run
from the cloned source tree (it reads `evalset/golden.json` relative to the repo).

## Architecture

```
question
   │
   ▼
out-of-scope rules ─────────────► OUT_OF_SCOPE
   │
   ▼
undocumented-term guard ────────► NOT_IN_SOURCES   (a content word is absent from the docs)
   │
   ▼
BM25 retriever  ── top score < threshold ──► NOT_IN_SOURCES
   │
   ▼
extractive answerer
   │   best sentence's IDF-weighted relevance < floor ──► NOT_IN_SOURCES
   ▼
ANSWER  (claims, each citing a source chunk)
   │
   ▼
eval:  mechanical checks  ( + optional faithfulness LLM-as-Judge )
```

Each layer is swappable without touching the rest: replace BM25 with a vector store, the
extractive answerer with an LLM synthesizer, the fake judge with a real provider.

## Honest limitations

A deliberately small reference, not a product. Stated plainly because pretending
otherwise is the exact failure mode the project is about:

- **The gates are lexical, not semantic.** They reliably refuse undocumented *topics*,
  but they cannot recognize synonyms (a question about "operating systems" is refused
  even though the docs list "Windows / macOS"). They err toward refusal; the faithfulness
  judge is the intended semantic gate.
- **Thresholds are corpus-tuned.** `answer_floor` and the stemming are sized to the
  bundled corpus; real use wants semantic retrieval.
- **Answers are extractive** (verbatim sentences): grounding by construction, not fluent
  synthesis. The synthesis seam is pluggable for that reason.
- **The judge is uncalibrated here.** A calibration harness (judge vs. human labels) is
  the intended next step before trusting any judged score.
- **Synthetic, single-corpus data.** The fictional "Harbor" docs exist to make the
  behavior legible and the repo safe to publish — not to benchmark anything.

## A note on how this was built

Before publishing, this repo went through **two rounds of adversarial review** against a
skeptical hiring bar. The first caught the system committing the **exact
confident-hallucination the project exists to prevent**. The second caught the first fix
for what it was — a *patch*: it silenced two specific phrasings but left the whole class
open (`"Can I password-protect a share link?"` was still answered with the generic
share-link sentence, because the distinctive word `password` carried zero lexical
weight). The root cause — lexical relevance giving no weight to exactly the terms that
prove a question is unanswerable — is now addressed by the undocumented-term guard, and
verified across 14 adversarial probes with **zero** confident hallucinations. The residual
limitation (synonym over-refusal) is documented above and is the safe direction. Finding
and closing your own system's failure mode — and being honest about what's *still*
imperfect — is the point.

## Layout

```
cite_or_refuse/
  models.py      response types (kind + claims-with-citations)
  retriever.py   BM25 + light stemming, stdlib-only
  answerer.py    the cite-or-refuse contract + two lexical gates
  judge.py       faithfulness LLM-as-Judge (pluggable, fails closed)
  eval/          mechanical checks + golden-set runner (optional judge)
  data.py        synthetic "Harbor" corpus
evalset/golden.json   11 cases: answer / not_in_sources / out_of_scope
tests/                20 offline tests
```

## License

MIT — see [LICENSE](LICENSE).
