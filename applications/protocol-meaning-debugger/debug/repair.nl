---
description: Propose the smallest definition repair that restores the intended proposition.
args:
  context: InvestigationContext
  state: InvestigationState
  diagnosis: Diagnosis
returns: RepairReplay | null
---
1. If diagnosis has an unresolved necessary cause or no source-backed edit scope, return null and
   preserve that uncertainty for report. Do not edit an arbitrary module to restore one headline number.
2. For each root cause locate the selected right-implementation definition and its exact original span.
   Form a bounded RepairEdit replacing that role's meaning with the controlling target requirement.
3. Keep unchanged roles, execution bindings, raw trial data and archive definitions intact. A repair
   changes the used predicate/anchor/numerator, not the evidence that made the baseline misleading.
4. Count edited definition sites under request.objective only after semantic coverage. A one-line
   metric inversion cannot repair a population predicate or event anchor that independently differs.
5. Create one RepairProposal with exact source identity, allowed path, oldQuote, replacement and
   claimed repaired contrast IDs. Include source witnesses for each changed role and its authority.
6. Call auditRepair(context, state, diagnosis, proposal). Consume its reinterpreted baseline, probe
   replay and findings; return that RepairReplay rather than declaring success from the proposal text.
7. A repair that restores baseline50percent but still answers the wrong probe question is violated.
   The intended proposition must agree on population, anchors and numerator across authorized tests.
8. Effects are proposed definition edits. This function does not modify the original files or deploy
   a research pipeline; exact application is a private counterfactual used by the audit child.
