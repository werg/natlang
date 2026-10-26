---
description: reconcileNetworks domain stage for Historical Record Linkage Atelier.
args:
  goal: string
  records: HistoricalIdentityFinding[]
  evidence: EvidenceItem[]
returns: StageResultHistoricalNetwork
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftHistoricalNetwork values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Check proposed identity and relationship links globally for impossible dates, incompatible simultaneous identities, and circular assumptions. Prevent one uncertain link from becoming evidence that confirms itself through a derived family or institutional network. The stage gate is: No self-supporting evidence cycle or established chronological contradiction.
