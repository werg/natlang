# 079 — Health Record Request Fulfillment Workbench

## Source-derived procedure

Five domain stages: interpretAuthorization, locateRecords, reviewContent, assembleFulfillment, verifyDelivery. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **interpretAuthorization** — Read the request and authorization together to determine recipient, purpose, categories, dates, and any restrictions. Distinguish a requested scope from an authorized scope and route ambiguous authority to the records officer before retrieval. Gate: Current authorization covers the exact recipient and proposed disclosure purpose.
- **locateRecords** — Search authorized inventories using verified patient and episode identifiers, preserving date and category boundaries. Distinguish unavailable records from records outside scope and prevent similar identities or merged historical accounts from contaminating the packet. Gate: Correct patient identity and explicit coverage manifest.
- **reviewContent** — Assess responsiveness and identify third-party or otherwise restricted material under supplied disclosure rules. Preserve clinically relevant context where authorized, and route nuanced disclosure judgments to qualified records reviewers rather than making autonomous legal decisions. Gate: Every exclusion or redaction has a supplied policy basis and review status.
- **assembleFulfillment** — Build a complete or explicitly partial packet with a manifest and explanation of outstanding items. Apply approved redactions through trusted rendering services, verify extracted content and metadata, and bind approval to exact artifact hashes. Gate: No recoverable restricted content, missing unexplained item, or mismatched patient document.
- **verifyDelivery** — Recheck authorization immediately before recipient-bound delivery, record actual receipt where available, and track partial fulfillment or corrections. Preserve an audit trail without exposing sensitive record bodies in operational logs or notifications. Gate: Correct recipient, current authority, and exact released artifact identity are verified.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
