# 080 — Patient Experience Improvement Laboratory

## Source-derived procedure

Five domain stages: extractExperiences, linkJourneyEvents, formMechanisms, designImprovements, reviewResults. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **extractExperiences** — Process surveys, complaints, and calls to identify administrative experiences and their context. Separate access or communication issues from clinical conclusions, preserve mixed experiences, and avoid inferring diagnoses or treatment quality from administrative feedback. Gate: Findings stay within observed administrative experience.
- **linkJourneyEvents** — Relate feedback to appointment, contact, and referral events using authorized identifiers and time windows. Distinguish repeated reports of one episode from independent experiences, and preserve uncertain linkage rather than exposing unrelated patient records. Gate: Patient isolation and evidence-backed episode linkage.
- **formMechanisms** — Identify plausible administrative mechanisms such as unclear instructions or failed callbacks. Seek contradictory cases and alternative explanations, distinguish correlation from causality, and define evidence needed to test the mechanism without making clinical claims. Gate: Mechanisms remain appropriately qualified and source-supported.
- **designImprovements** — Propose bounded communication or workflow improvements within approved service authority. Define measurable administrative outcomes, guard against shifting burden to another team, and specify how results would support adoption, revision, or abandonment. Gate: Experiment does not alter clinical care and has an independent outcome measure.
- **reviewResults** — Reconcile computed administrative metrics with follow-up feedback and implementation evidence. Check changed denominators, observation channels, and minority experiences, then recommend next actions without claiming clinical benefit beyond the study’s scope. Gate: Reported improvements match exact measures and preserve limits and counterevidence.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
