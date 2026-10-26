# 078 — Appointment Preparation Service

## Distinct algorithm: compile an appointment instruction into versioned relative-time reminders with fidelity checks

The service starts from the latest authorized clinical instruction and the exact appointment time. It computes relative actions only from explicit intervals and records the source instruction version on every checklist item. An appointment change invalidates dependent relative times and reissues only those items. Conflicting patient-reported instructions are not resolved by choosing one: the service routes the discrepancy to clinic staff.

Translation is a fidelity-sensitive projection. Any wording change to an exception, negation, or timing condition blocks release until reviewed by a qualified language reviewer. A representative receives the checklist only under current, appointment-specific consent and communication scope. Delivery has explicit states; an undelivered message triggers an approved fallback contact route, while a missing acknowledgment is never treated as receipt.

The resumable checkpoint stores instruction version, appointment version, original time fields, exact computed reminder offsets, translation review, consent scope, channel attempts and acknowledgment receipts. The tool prepares patient-facing material; it does not add or alter clinical instructions.
