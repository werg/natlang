# 095 — Culinary Concept Development Kitchen

## Source-derived procedure

Five domain stages: interpretBrief, generateDishes, constructMenu, planAndCost, reviewTastings. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **interpretBrief** — Translate the culinary brief into desired experience, service style, supplied dietary requirements, equipment, and budget constraints. Distinguish preferences from mandatory restrictions and use only verified ingredient and allergen information rather than inferring safety from a dish name. Gate: Mandatory dietary and operational constraints are explicit and source-backed.
- **generateDishes** — Invent distinct dishes using available ingredients and approved culinary procedures, explaining flavor, texture, and service intent. Propose substitutions only when their properties and dietary implications are verified, and preserve specialist food-safety review where needed. Gate: Concepts respect verified restrictions and documented kitchen capability.
- **constructMenu** — Assemble a balanced menu whose dishes work together across preparation effort, equipment, pacing, and customer choice. Identify shared components and bottlenecks, preserve meaningful alternatives for required diets, and avoid creating variety only through superficial garnish changes. Gate: Menu-level feasibility and required dietary coverage pass.
- **planAndCost** — Use exact recipe scaling, inventory, costing, and scheduling services to build preparation and service plans. Account for yield, waste, shared equipment, and approved handling requirements; preserve traceability from purchased ingredients to served dishes. Gate: Exact quantities, costs, equipment capacity, and approved handling constraints reconcile.
- **reviewTastings** — Interpret structured tasting and service rehearsal notes to identify creative and practical improvements. Distinguish subjective preference from repeatable defects, revise the smallest relevant components, and revalidate cost, dietary information, and service dependencies after every change. Gate: Revised menu passes hard checks; culinary quality remains chef-reviewed rather than self-certified.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
