# 076 — Referral Packet Completeness Manager

## Source-derived procedure

Five domain stages: interpretRequirements, matchPatientRecords, assessCompleteness, requestMaterial, releaseAndTrack. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **interpretRequirements** — Read the receiving service’s supplied administrative requirements and the clinician’s referral instructions. Identify required records, forms, dates, and signatures while preserving clinician-provided priority and avoiding any independent judgment about clinical necessity. Gate: Requirements are current, service-specific, and administrative in scope.
- **matchPatientRecords** — Verify patient and episode identity using approved exact matching and reviewed evidence. Distinguish similar names, historical identifiers, and unrelated episodes; quarantine mismatches before content is included in a packet. Gate: No cross-patient attachment or unsupported identity merge.
- **assessCompleteness** — Compare each requirement with submitted documents, checking scope, dates, signatures, and readability. Distinguish document presence from actual required content and route clinical-content questions to authorized staff rather than interpreting them autonomously. Gate: Every administrative requirement is met or explicitly unresolved.
- **requestMaterial** — Group missing-material requests by authorized source, ask only for necessary records, and preserve patient consent and permitted disclosure scope. Avoid repeated requests for existing evidence and track who is responsible for each unresolved gap. Gate: Requests are minimal, authorized, and tied to a specific requirement.
- **releaseAndTrack** — Assemble a recipient-bound packet, verify exact patient and service identifiers, and obtain required review before transmission. Track delivery, receipt, and administrative acceptance separately, preserving rejection reasons and follow-up obligations. Gate: No claim of appointment or clinical acceptance from a transport receipt alone.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
