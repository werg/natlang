---
description: Read nested instrument and event collections without collapsing their distinct date roles.
kind: directory-reducer
args:
  request: LedgerRequest
returns: Archive
---
1. Read manifest.json as SourceManifest and require caseId=request.caseId. Visit each listed path
   once, grouped by its actual parent directory, preserving source id, path, roleHint and exact text.
2. Extract visibly labeled clauses with exact Witness quotations. A label names a source assertion;
   the folder name only guides reading and cannot decide authority or whether an instrument is adopted.
3. For each document call interpretInstrument(document, clauses) and interpretEvent(document, clauses).
   A mixed document may contribute both instrument metadata and evidence facts; consume both deltas.
4. Join clauses and instruments by explicit source identity. Keep adoptedAt, publishedAt, effective
   interval, eventTime and availableFrom as separate fields. An earlier signature need not mean earlier knowledge.
5. Reconcile the control document's snapshot visibility with each document's receipt assertions.
   Annotate delayed clauses with availableFrom and availabilityPremiseIds naming the exact receipt assertions.
   Preserve unknown receipt; a derived date without its source premise cannot justify knowledge.
6. Join trips only by explicit trip ID and consistent operator/destination/time. If two assertions
   conflict, retain the exact propositions in unresolved instead of overwriting whichever was read first.
7. Record sourceCoverage for every document and clause. Return complete Archive with documents,
   interpreted clauses, instruments, trips, facts and unresolved questions. Unknown certification is not false.
