---
description: Own one consequence/defeat delta in the finite stratified justification frontier.
args:
  state: ClosureState
returns: ClosureState
---
1. If pendingApplicationIds is empty return state. Select its first ready application in the caller's
   dependency order. Remove the ID from a copied pending list and append it to processed exactly once.
2. Call deriveConsequences(application, state). Consume selectedTerms, proofs, defeatCandidates
   and unresolved. Append settled terms before later duty applications read them. Deduplicate only identical norm/key/premise combinations, preserving independent source support.
3. Add the new proofs before processing that application's defeat candidates. For each candidate call
   resolveDefeat(candidate, stateWithNewProofs) and append its actual DefeatDecision.
4. A true defeat marks only its target proof under the stated scope. An unknown defeat leaves a
   possible surviving obligation and must remain explicit in classification. False defeats change no proof.
5. Keep the structural premise DAG: a proof may reference only settled prerequisites or source facts.
   A negative cycle already classified unresolved cannot become positive support through this step.
6. Return the copied state with one fewer pending application and the actual accumulated deltas.
   A conclusion reused from a previous snapshot must still have every dependency satisfied here.
