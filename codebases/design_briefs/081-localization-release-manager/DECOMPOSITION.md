# 081 — Localization Release Manager

## Source-derived procedure

Five domain stages: segmentWithContext, resolveTerminology, transformContent, validateLocale, assembleLocaleRelease. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **segmentWithContext** — Parse source content into translatable units while preserving structured markup, placeholders, plurals, and surrounding context. Distinguish identical strings used for different purposes and attach the product state or audience needed to choose the correct meaning. Gate: Exact token and markup structure preserved; context is not discarded during deduplication.
- **resolveTerminology** — Match terms to concepts and approved locale guidance, considering polysemy, grammatical role, and product meaning. Preserve scoped exceptions and route genuine terminology conflicts to the owner rather than choosing whichever translation appears most often. Gate: Term choice is concept- and context-specific, with versioned authority.
- **transformContent** — Translate or adapt according to communicative purpose while preserving product facts, obligations, and constraints. Generate alternatives for culturally or linguistically sensitive phrasing using supplied guidance, and keep unresolved meaning questions visible rather than hiding them in fluent text. Gate: Meaning, protected tokens, and required product terminology survive transformation.
- **validateLocale** — Run exact placeholder, markup, length, plural-form, and link checks, then review semantic fidelity and naturalness independently. Inspect cross-segment consistency and context screenshots or metadata, distinguishing harmless variation from contradictory terminology or behavior. Gate: Exact structural gates pass and no unresolved material semantic defect remains.
- **assembleLocaleRelease** — Build a locale bundle bound to source, glossary, context, and review versions. Identify stale translations after source changes, apply explicit fallback rules, and release only approved segments without mixing incompatible editions in one user journey. Gate: Bundle is internally version-consistent and every fallback is declared and safe.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
