---
args: { adaptation: InstructionAdaptationCase, release: InstructionReleasePacket, ledger: FloorCopyLedger, receipt: FloorCopyReceipt }
returns: CopyDispositionAssessment
---
1. Find the floor copy by exact copyId and require its current version to equal receipt.sourceVersion and adaptation.sourceVersion. A newer instruction release does not itself withdraw an older posted copy.
2. A "withdrawn" outcome requires evidence that the physical copy was collected, disabled, or visibly marked obsolete, plus a witnessRef. Without that evidence the copy remains active.
3. An "unavailable" result records a failed location search; leave the old copy active and return a named owner/search task. Do not represent loss as withdrawal.
4. A "replaced" outcome requires physical withdrawal evidence for the old copy and proof that a new copy with a distinct copyId and the approved released version was posted. A draft or unreleasable variant cannot replace an active instruction.
5. An unknown outcome stays held. Return whether the old copy is actually withdrawn, whether the replacement can enter the inventory, and each missing evidence item.
