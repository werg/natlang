---
description: Replay one obligation's premise DAG and challenge each authority dependence.
args:
  entry: LedgerEntry
  ledger: SliceLedger
  archive: Archive
  authority: AuthoritySet
returns: ProofAudit
---
1. Resolve the entry's supportingProofIds and defeatingDecisionIds. Traverse typed proof references
   with a visited set and active recursion path; an active-path repeat is circular support, not a root premise.
2. For each source fact check identity, truth and availability in ledger.snapshot. Reconstruct the
   governing event-time interval separately; a source fact can be newly known about an old trip.
3. For each normative proof call challengeAuthority(proof, authority, archive). Consume its actual
   permitted result and source witnesses. A null result keeps this audit unresolved.
4. Replay conjunction and scoped defeat relationships using the same proposition meanings carried
   by the proof graph. Require an actual source-backed premise for every quantity and permission alternative.
5. Recompute whether at least one independent definite proof survives. Compare that classification
   with entry.status and business fields, preserving source-equivalent proof ordering or generated IDs.
6. Return ProofAudit with snapshotId=ledger.snapshot.id, visitedProofIds, authorityFindings, issues and witnesses. valid=true means
   all inspected dependencies support this certificate; false or null exposes the precise first failed premise.
