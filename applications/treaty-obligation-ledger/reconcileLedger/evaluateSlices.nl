---
description: Evaluate ordered knowledge snapshots while preserving historical transit and norm times.
args:
  request: LedgerRequest
  archive: Archive
  compiled: CompiledNorms
  authority: AuthoritySet
returns: SliceLedger[]
---
1. Order requested snapshots by knowledgeTime, retaining their source IDs. Confirm every requested
   trip exists; a missing trip produces unresolved evidence, not an empty successful ledger.
2. Begin with previous=null. For each snapshot call applySnapshot(previous, snapshot, archive,
   compiled, authority, request.tripIds), and append the actual returned SliceLedger before advancing previous.
3. A new snapshot changes evidence availability only. A trip's operator, destination and event time
   remain fixed unless newly available source evidence explicitly contradicts those facts.
4. Keep each snapshot's proof and defeat objects distinct so later evidence cannot retroactively
   rewrite the earlier certificate. Immutable historical uncertainty remains visible in the returned list.
5. Return ledgers in requested time order. Norm effectiveness is tested against each trip's event time;
   evidence availability is tested against snapshot.knowledgeTime. Neither comparison substitutes for the other.
