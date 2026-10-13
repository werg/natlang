---
description: Choose source-authorized queries whose answer partitions are sufficient for the repair.
args:
  index: EvidenceIndex
  correlation: Correlation
  ledger: EffectLedger
  obligations: ObligationGraph
returns: ObservationPlan
---
1. For each uncertainRefundLedgerId, retrieve that effect, its processing boundary and the source-listed query
   alternatives. Candidate queries must concern the same tenant, key and outcome proposition.
2. Call partitionAnswers(queryAtoms, effect, correlation, index.atoms) once per candidate query. This child
   returns the source answer partition, temporal coverage and whether each answer justifies a repair decision.
3. A query is sufficient only if each possible answer either confirms the refund or establishes final absence
   after its processing boundary. An early empty history leaves both worlds together and cannot authorize retry.
4. Among sufficient choices for this uncertainty, choose minimum source observation cost. Break an exact tie
   with queryId lexical order only to select a canonical witness; equivalent tied choices remain valid policies.
5. Retain all rejected alternatives with their reasons and witnesses. If none is sufficient, record unresolved
   rather than assuming that the existing refund did not happen or inventing an unlisted query service.
6. Join selections across independent uncertainties. If one query explicitly covers several, avoid charging or
   invoking it twice while preserving its answer conditions for every affected ledger entry.
7. Return selected, rejected and unresolved. Empty selected is correct only when no repair-relevant unknown
   exists; an unresolved ledger with no candidate query must have an explicit unresolved explanation.
