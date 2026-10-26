---
args: { cells: TenderCoverageCell[], clarifications: TenderClarification[], revisions: TenderRevision[] }
returns: ClarificationInvalidation
---
Find each clarification whose target cell's revision was superseded before the clarification's effective cutoff or by a later removal of its inclusion. Return exact invalidated cell, clarification, and revision IDs; do not carry forward the former coverage verdict.
