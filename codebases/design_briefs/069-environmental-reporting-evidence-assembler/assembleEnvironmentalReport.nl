---
description: Assemble one immutable environmental report edition from a frozen boundary and reconciled physical activity ledger.
args: { dossier: ReportingDossier; change: ReportChange; officer: ReportingOfficer }
returns: ReportingProposal
---
Freeze the current approved reporting boundary and effective period. If a proposed boundary differs, preserve both source versions and block totals until the effective date and reviewer authority are established.

Bind boundary = `interpretReportingBoundary(dossier, change)`. Bind inclusions, exclusions, measurement rules and scope to the approved source. Do not aggregate while boundary or period precedence remains unresolved.

Bind measures = `mapEnvironmentalMeasurements(dossier, change, boundary)`. Retain source method, period, unit, direct measurement versus estimate, calibration and uncertainty. Keep replacement meter readings separate until the continuity method is supported.

Bind reconciliation = `reconcilePhysicalAttribution(dossier, boundary, measures)`. First determine which reports describe the same physical event. Distinguish transferred responsibility from duplicate measurement. Allocate a shared activity only under its stated rule, and verify exact unit conversion and conservation. Leave unresolved events outside totals and visible as gaps.

Bind claimAssessments = `testEnvironmentalClaims(dossier, measures, reconciliation, change)`. Compare each proposed statement with the actual measured outcome and its scope. Initiative completion alone does not substantiate benefit. Qualify estimates and block claims that exceed their evidence or boundary.

Bind edition = `freezeReportEdition(dossier, boundary, reconciliation, claimAssessments, change)`. Preserve schedules, methods, boundary identity, claim links, limitations, restatement bridge and reviewer binding as a new edition. Return `ReportingProposal` for `officer.id` with exact totals and gaps in a resumable checkpoint. A boundary change is not reported as operational improvement, and this proposal does not claim a public submission occurred.
