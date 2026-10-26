# 077 — Care Transition Logistics Coordinator

## Source-derived procedure

Five domain stages: extractLogistics, mapParticipants, constructSequence, resolveConflicts, verifyReadiness. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **extractLogistics** — Read the approved plan to identify logistical tasks and dependencies, preserving its wording for clinical requirements. Separate administrative coordination from treatment decisions and route any ambiguous clinical instruction back to the authorized team. Gate: No generated change to clinical content or priority.
- **mapParticipants** — Identify transport providers, equipment suppliers, caregivers, and receiving services with their actual responsibilities. Distinguish availability from acceptance, verify representation and information-sharing authority, and avoid assuming family relationships imply consent or capacity. Gate: Every responsibility has explicit acceptance and permitted information scope.
- **constructSequence** — Build an exact dependency plan for equipment, access, transport, appointments, and acknowledgments. Identify conditions that must be ready before transfer and expose impossible timing or missing ownership rather than quietly relaxing the approved plan. Gate: All logistical prerequisites are feasible and traceable to the approved plan.
- **resolveConflicts** — Propose permitted logistical alternatives when providers or times change, explaining consequences for the approved plan. Escalate any alternative that could alter clinical requirements and keep the transition blocked until the appropriate team resolves the conflict. Gate: Administrative replanning never substitutes for clinical approval.
- **verifyReadiness** — Recheck actual delivery, provider acceptance, access, and participant acknowledgments before the transition. Distinguish scheduled from completed prerequisites and record the authorized release decision separately from the program’s readiness summary. Gate: No transition-ready claim while a required logistical prerequisite is unverified.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
