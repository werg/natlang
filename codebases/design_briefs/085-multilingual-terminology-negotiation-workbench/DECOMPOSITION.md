# 085 — Multilingual Terminology Negotiation Workbench

## Source-derived procedure

Five domain stages: extractSenses, alignConcepts, constructTermOptions, negotiateDecision, propagateTerminology. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **extractSenses** — Collect terminology in actual use across expert documents and translations, preserving grammatical and domain context. Separate different senses of the same word and identify cases where different words denote one concept rather than assuming one-to-one lexical equivalence. Gate: Every sense has contextual evidence.
- **alignConcepts** — Compare conceptual boundaries across languages and teams, identifying exact, broader, narrower, and partial equivalence. Preserve concepts that do not map cleanly and avoid forcing one language’s categorization onto all others. Gate: Concept alignment is supported by definitions and counterexamples, not dictionary matching alone.
- **constructTermOptions** — Propose terms and scoped usage rules considering approved domain conventions, grammatical forms, and audience needs. Explain tradeoffs among clarity, continuity, and precision, and retain regional or contextual variants where one universal term would change meaning. Gate: Each option preserves the intended concept in its stated context.
- **negotiateDecision** — Present alternatives and concrete usage examples to the relevant experts, reconcile their comments, and distinguish stylistic preference from conceptual disagreement. Record scoped acceptance and unresolved dissent rather than treating a majority vote as proof of semantic equivalence. Gate: Decision authority and applicability are explicit; unresolved conceptual conflict remains visible.
- **propagateTerminology** — Identify content using the affected concept and sense, propose context-aware revisions, and preserve unrelated homographs. Revalidate grammar and meaning after replacement and version the glossary and dependent release artifacts together. Gate: No global search-and-replace across different senses; affected content passes fidelity checks.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
