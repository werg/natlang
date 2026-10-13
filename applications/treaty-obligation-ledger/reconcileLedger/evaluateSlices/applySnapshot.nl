---
description: Seed current applicability, retract invalid premises and settle the affected obligation closure.
args:
  previous: SliceLedger | null
  snapshot: Snapshot
  archive: Archive
  compiled: CompiledNorms
  authority: AuthoritySet
  tripIds: string[]
returns: SliceLedger
---
1. For each requested trip call seedApplicability(snapshot, trip, compiled, archive, authority).
   Consume every returned norm/event judgment, including false and unknown conditions for audit provenance.
2. Compare these actual applicability values and visible evidence premises with previous when present.
   Identify old proofs whose factual condition, selected amount or authorized defeater status changed.
3. Initialize ClosureState with the new snapshot, applications and empty selectedTerms. Retain only old proofs whose
   premises remain valid; call propagateRetraction(state, invalidatedProofIds) to remove their dependent cone.
4. Use the returned survivingProofs. Preserve independent proofs for the same business identity;
   invalidating one supporting clause does not retract a separately grounded notice duty.
5. Build pendingApplicationIds in source-justified dependency order: settle applicable rate terms
   first, then materialize positive duties using those returned terms, then scoped exceptions/reservations
   and closure decisions. A rate term is not a second competing obligation proof.
6. Detect cyclic negative-dependence components before settlement. If source priority does not resolve
   such a component, record it unresolved instead of repeatedly toggling its truth status.
7. Invoke settleFrontier.iterateOn(state) with measure pendingApplicationIds.length until empty.
   Each application is consumed once in its settled dependency stratum; its actual proof delta is retained.
8. Enumerate the stable business keys across requested trips and source-applicable actions. For each
   call classifyObligation(key, proofs, defeats, snapshot) using the returned complete closure.
9. Return snapshot, entries, proofs, defeats and unresolved. Do not fill an unknown exception from
   a later snapshot or delete rejected normative evidence merely because the current duty is clear.
