# P14 — Evidence atlas / research notebook

Status: local cited-answer workflow implemented in
[`applications/evidence`](../../applications/evidence/index.ts) and
[`applications/evidence/index.ts`](../../applications/evidence/index.ts).
Trace-corpus ingestion and live-model grounding review remain open.
[Shared capabilities](README.md).

## Natlang prerequisites

C0 expresses question decomposition and evidence assessment. C1 exposes search/read as crisp environment operations. C7 supplies ordinary source/span/claim records. C4 is relevant when the corpus includes natlang execution traces, but is not required to search plain documents. No new global search action or vector-store core facility.

## Programme and typed boundary

`answer.nl(question, collection, policy) -> EvidenceAnswer` calls `plan_search.nl`, `assess_passage.nl`, `compare_claims.nl`, `identify_gaps.nl` and `compose_answer.nl`. Exact helpers retrieve and verify source spans.

Return supported claims, citations, conflicts and unknowns as records. Each citation identifies a source revision and span. Model-written summaries stay distinct from source observations. Collection IDs refer to an explicitly bound corpus, not arbitrary private caller state.

## Crisp environment

Start with `corpus.search(query, filters)` and `corpus.read(id, revision, range)` over supplied documents. Native document objects and optional indexes live in that environment. Add full-text or vector implementations after measuring retrieval failures; the natlang workflow need not change its tool inventory.

Meta-search of execution history uses the trace data APIs through eval. A trace descriptor that says native data was not captured cannot answer a question about that data. Source deletions and permission changes must be reflected in reads/index access rather than hidden by old cached snippets.

## Reduction and stream shape

Finite search/assess/revise uses bounded Iterate. A search result is evidence with a version, not an instruction. The programme may stop with an unresolved answer when further retrieval cannot establish a claim.

Live collection updates later arrive as stream events that invalidate affected answer revisions. Running answers pin a snapshot; they do not receive invisible corpus mutation. A retained native collection must provide a versioned view or label results as observations from a changing source.

## Delivery and checks

The first implementation uses exact token search over versioned paragraph
spans, explicit retrieval truncation, and a natlang claim composition step.
The host checks citation identity, revision, passage authenticity and quote
presence. Tests cover an answer, fabricated quotes, changed passages and stale
collection revisions. `citation-checked` is intentionally narrower than
semantic support; independent review and retrieval-recall measurement are
still needed before admitting teacher answers as training gold.

1. Index selected natlang docs and answer a fixed question set with exact citations.
2. Add conflicting revisions, absent answers and misleading near-matches. Gate: cited spans exist and substantiate the stated claims.
3. Ingest C4 run evidence and answer why an execution failed. Gate: distinguish actual action/validation facts from hypotheses about the model.
4. Evaluate an optional retrieval index on held-out questions before adding it to the default host.

Measure retrieval recall separately from grounded answer quality. Test inaccessible/missing spans, duplicate documents, stale summaries and fabricated citations.

## Trace and teacher

Capture queries, result identities, actual read passages and source revision. Frozen corpus fixtures support replay; arbitrary live search does not automatically do so. Teacher references include justified uncertainty and preserved contradictions. The first implementation is intentionally code-accessible search over ordinary host data, with no universal host object protocol and no expansion of the natlang type algebra.
