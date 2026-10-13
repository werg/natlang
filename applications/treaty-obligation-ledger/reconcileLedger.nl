---
description: Reconcile authority, event-time duties and knowledge-snapshot changes in a fictional federation.
kind: directory-reducer
args:
  request: LedgerRequest
returns: LedgerReport
---
1. Call indexArchive(folder, request). Consume its clauses, instruments, trips, evidence facts and
   knowledge-availability annotations. Keep publication, effectiveness and archive receipt separate.
2. Call buildAuthority(archive). A claimed grant is usable only with an actual supported charter path;
   a pair of mutually citing offices does not become a root source of competence.
3. Call compileNorms(archive, authority). Consume typed norms and change relations, including rejected
   clauses. An out-of-scope circular remains evidence of a failed claim rather than a controlling rule.
4. Call evaluateSlices(request, archive, compiled, authority). Each returned ledger refers to one
   knowledge snapshot while preserving every trip's historical event time and the full justification graph.
5. Call explainDelta(ledgers, archive, compiled). Consume its stable-identity comparisons and minimal
   changed-premise supports, including unchanged obligations with different or retained proofs.
6. Call auditLedger(request, archive, authority, compiled, ledgers, deltas). Return its actual report.
   Preserve unresolved source scope; the root cannot replace a failed child with its own unaudited answer.
7. All consequences are proposed data over original fictional documents. Do not contact a port,
   apply real law, collect payment or mutate the evidence snapshot.
8. Generated norm, application and proof IDs are arbitrary unique references. Preserve source clause,
   trip, party and action identities exactly; different proof spellings do not imply different obligations.
