---
description: Audit complete argument coverage, reconcile participation counts, anonymize examples and freeze a versioned report.
args: { docket: ConsultationDocket; delta: SubmissionDelta; update: ConsultationUpdate; arguments: ScopedArgument[]; campaigns: CampaignRelationGraph; themes: ThemeCoverage[]; responses: ResponseDraft }
returns: ConsultationReport
---
Call `traceArgumentCoverage(arguments, themes, responses, docket)`. Then call `reconcileParticipationDenominators(docket, delta, campaigns)`; do not substitute the number of template clusters for participants or submissions. Review every illustrative quote for identifying personal circumstances. Redact details unless a supplied consent covers disclosure. When a report is challenged, preserve its prior edition, show the exact argument/theme links reopened, and record any revised disposition. Freeze only after omissions, denominator and privacy findings are explicit.
