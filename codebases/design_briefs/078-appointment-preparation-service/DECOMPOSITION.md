# 078 — Appointment Preparation Service

## Source-derived procedure

Five domain stages: recoverInstructions, interpretCircumstances, composeChecklist, routeQuestions, trackAcknowledgment. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **recoverInstructions** — Parse the approved instruction set into actions, timing, exceptions, and contact routes. Preserve clinical wording where transformation could change meaning, and identify ambiguity for staff resolution rather than interpreting medical instructions independently. Gate: Every consequential instruction and exception remains traceable to the approved source.
- **interpretCircumstances** — Read the patient’s stated practical circumstances and communication preferences without inferring diagnoses or capability. Identify logistical barriers and questions requiring staff attention, and verify authority before involving a representative. Gate: Context uses volunteered facts and does not change the clinical plan.
- **composeChecklist** — Produce an accessible checklist using approved terminology and exact timing services. Adapt layout, language level, or format while preserving all clinical conditions; separate logistical suggestions from clinician instructions and flag unresolved questions visibly. Gate: Semantic fidelity to approved instructions passes independent review.
- **routeQuestions** — Form focused questions for the appropriate staff when instructions conflict with practical circumstances or remain unclear. Track authoritative responses, avoid answering medical questions from general model knowledge, and update the checklist only from approved clarification. Gate: Clinical clarification comes from an authorized source.
- **trackAcknowledgment** — Record receipt, acknowledgment, and reported completion distinctly, respecting contact preferences and quiet periods. Escalate unresolved administrative barriers under supplied procedures without assuming acknowledgment proves comprehension or medical readiness. Gate: No false readiness claim and no unapproved clinical recommendation.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
