---
args: { prior: DisputeCaseCheckpoint|null, request: DisputeBuildRequest, records: DisputeRecordSet, review: DisputeProgramResult }
returns: DisputeCaseCheckpoint
---
Create a case checkpoint bound to one transaction and party pair. Preserve each party’s submissions and response windows by original source version. On a changed appeal evidence set, reopen only questions whose decisive premise or governing transaction-time rule is affected; do not erase prior balanced dossiers or imply an adjudicator accepted the recommendation. Record payment reversals only from exact ledger receipts. Duplicate source and appeal evidence replay is a no-op; advance revision for a new source edition, changed finding or actual receipt.
