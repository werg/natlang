---
description: Independently audit authority paths, proof closure and snapshot-preserving ledger changes.
args:
  request: LedgerRequest
  archive: Archive
  authority: AuthoritySet
  compiled: CompiledNorms
  ledgers: SliceLedger[]
  deltas: CertificateDelta[]
returns: LedgerReport
---
1. Check every requested snapshot/trip business identity is accounted for under the source action
   contract. Preserve unsupported and unresolved entries instead of hiding them from the audit inventory.
2. For each entry call replayJustification(entry, ledger, archive, authority). Consume its actual
   proof audit, including authority challenges and unresolved premises.
3. Compare each delta with the two immutable ledgers. Verify historical event time and party roles
   are unchanged; a late receipt cannot retroactively move the trip across an amendment boundary.
4. Check every independent notice proof remains represented. A reservation must follow the debtor,
   and an order's scope may not exceed the narrower charter competence even if its text claims more.
5. Verify source witnesses by exact id/path/quotation membership. This exact check supplements the
   children's semantic support judgment; quote presence alone cannot establish a duty.
6. Return LedgerReport containing actual authority, compiled norms, ledgers, deltas, audits and source
   coverage. Aggregate unresolved source/audit questions honestly; no effects are applied to real institutions.
