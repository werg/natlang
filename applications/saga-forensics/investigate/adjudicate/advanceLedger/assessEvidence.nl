---
description: Classify observations under the source evidence contract and preserve genuine outcome uncertainty.
args:
  candidate: EffectCandidate
  index: EvidenceIndex
  correlation: Correlation
returns: EvidenceAssessment
---
1. Select the candidate's relevant atoms plus the evidence-strength contract and any explicit snapshot boundary.
   Keep each atom separate so repeated observations cannot masquerade as independent effects.
2. Define a local ObservationJudgment type with atomId, strength, and witnesses using the EvidenceAssessment
   observation element shape. Create a scoped NL literal capturing only the applicable evidence clauses and
   candidate identity. Directly await this literal on each relevant atom and collect its returned judgments.
3. The literal judges what this assertion establishes for this candidate: commit, replay, intent, acceptance,
   rejection or bounded-unknown. It must retain negation and processing coverage in the selected witness.
4. Consume the classifications: a valid commit or replay of a committed effect supports committed; an explicit
   authoritative no-effect rejection supports absent; mere intention, missing reply or acceptance does not.
5. When the source explicitly bounds an uncertain outcome to committed or absent, return both statuses once.
   If no finite outcome scope is established, preserve that insufficiency in conflicts rather than inventing one.
6. A repeated effect ID with a compatible amount reports the same effect. Different compatible source reports
   strengthen evidence; they never add a second monetary or stock quantity to this candidate.
7. If assertions appear contradictory, compare their scope and time coverage under the contract. An earlier
   empty history can coexist with a later commit. Keep unresolved same-scope contradictions in conflicts.
8. Return the candidate, all relevant observation judgments, the supported status set and conflicts. A source
   atom that only supplies business fields need not be misclassified as a commit; use its intent-level role.
