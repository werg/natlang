---
description: Compare adjacent evidence snapshots by stable business identity and proof provenance.
args:
  ledgers: SliceLedger[]
  archive: Archive
  compiled: CompiledNorms
returns: CertificateDelta[]
---
1. For each adjacent ledger pair call compareCertificates(before, after). Match entries by the full
   business key, not array position, proof spelling or current amount.
2. For every changed result call minimizeSupport(delta, before, after, archive). Consume its actual
   newPremiseIds, backgroundPremises and minimality qualification as the delta's support.
3. Keep unchanged entries in the comparison. Different source evidence can leave a duty unchanged;
   surviving independent notice proofs explain why a rescue exception does not erase that obligation.
4. A change in knowledge about a past event is not a change in that event's time. Preserve the same
   business key and show which newly available certification or scope premise changed its status.
5. Return all deltas with exact before/after snapshot IDs. If no sufficient changed-premise support
   can be established, retain minimality unresolved rather than inventing an explanatory source fact.
