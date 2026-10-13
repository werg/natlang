---
description: Identify actual obligation and justification changes without conflating new IDs with new meaning.
args:
  before: SliceLedger
  after: SliceLedger
returns: CertificateDelta[]
---
1. Form the union of business keys in both ledgers. Compare status, amount alternatives and permission
   values for each key, treating amount sets as unordered and debtor/beneficiary direction as exact.
2. Match proof meaning by norm, key and typed premise identities. Newly generated proof ID spelling
   alone is not a changed obligation or an invalidated source justification.
3. For each key collect genuinely removed/invalidated and new proof IDs, preserving surviving
   alternative support. Mark changed only when business consequence or justified status changes.
4. Return CertificateDelta with both snapshots, source-backed reason and support=null for now.
   The parent supplies minimal changed-premise support only after its separate replay analysis.
