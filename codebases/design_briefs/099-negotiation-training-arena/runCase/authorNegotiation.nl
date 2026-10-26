---
description: authorNegotiation domain stage for Negotiation Training Arena.
args:
  goal: string
  records: NegotiationExercise[]
  evidence: EvidenceItem[]
returns: StageResultNegotiationScenario
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftNegotiationScenario values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Construct a fictional negotiation with multiple issues, private priorities, constraints, and opportunities for mutually beneficial trades. Ensure the scenario has meaningful information gathering and feasible outcomes without requiring prohibited tactics or real-world manipulation. The stage gate is: Exact feasibility and learning-objective coverage verified before use.
