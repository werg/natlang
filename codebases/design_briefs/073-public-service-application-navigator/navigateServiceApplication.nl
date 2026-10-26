---
description: Navigate one public-service application through rule-versioned evidence routes, truthful forms, delegated consent and receipt reconciliation.
args: { dossier: NavigationDossier; change: ServiceChange; navigator: NavigatorAgent }
returns: NavigatorProposal
---
Use the applicant's stated situation and current service-rule version. Preserve unknown facts; do not turn a missing document into an eligibility judgment.

Bind situation = `interpretApplicantSituation(dossier, change)` and bind facts, uncertainties and any urgent-routing flag. When an urgent flag requires staff support, keep that route visible and do not delay it behind routine form work.

Bind pathways = `buildEvidencePathways(dossier, change, situation)`. A standard document may be replaced only through an alternative explicitly accepted by the current rule. Keep verification pending until an authorized intake source confirms it. If a rule changed, revalidate only pathways whose condition or rule version changed and explain the delta.

Bind questions = `askDecisionChangingQuestions(dossier, change, pathways)` for only the missing facts that could change a supported route. If a form cannot represent the applicant's truth, create a clarification task instead of forcing a false answer.

Bind tasks = `prepareTruthfulApplicationTasks(dossier, change, pathways, questions)`. Bind each task to required evidence, owner and completion condition. A helper may act only within the applicant's current documented consent scope and channel.

Bind receipts = `reconcileAgencyReceipts(dossier, change, tasks)`. If a receipt is delayed or absent, check supplied agency-status evidence first. Keep submission status unknown until an authoritative receipt or status response exists; do not resubmit a possibly received application.

Return `NavigatorProposal` for `navigator.id` with applicable rule versions, allowed evidence alternatives, truthful clarification tasks, delegated scope, unresolved outcome and type-compatible resume state. A prepared task is not a submitted application or eligibility decision.
