# 088 — Historical Record Linkage Atelier

## Source-derived procedure

Five domain stages: normalizeRecords, generateLinks, judgeIdentity, reconcileNetworks, prepareResearchView. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **normalizeRecords** — Preserve original spellings, date expressions, and geographic descriptions while creating normalized search forms. Distinguish transcription uncertainty from historical variation, and avoid converting approximate dates or changing place names into falsely precise modern facts. Gate: Original representation and uncertainty remain available.
- **generateLinks** — Generate candidate links using names, chronology, places, relationships, and contextual descriptions. Include plausible aliases while preserving hard incompatibilities, and audit records excluded by blocking so unusual spellings do not disappear from consideration. Gate: Candidate generation is bounded and recall is independently assessed.
- **judgeIdentity** — Compare evidence for identity versus kinship, institutional succession, or coincidence. Weigh source reliability only within its actual scope, preserve competing candidates, and avoid treating a plausible story as proof of identity. Gate: Established links require discriminating evidence and compatible chronology.
- **reconcileNetworks** — Check proposed identity and relationship links globally for impossible dates, incompatible simultaneous identities, and circular assumptions. Prevent one uncertain link from becoming evidence that confirms itself through a derived family or institutional network. Gate: No self-supporting evidence cycle or established chronological contradiction.
- **prepareResearchView** — Publish a reviewable linkage view with source-level explanations and explicit alternatives. Support corrections and reversible splits, preserve stable source identifiers, and expose uncertainty to downstream researchers instead of exporting only a deceptively clean graph. Gate: Accepted and provisional relationships remain distinguishable in every export.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
