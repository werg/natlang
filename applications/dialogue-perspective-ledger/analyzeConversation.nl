---
description: Derive time-scoped beliefs and commitments without leaking global truth into private views.
kind: directory-reducer
args:
  request: DialogueRequest
  snapshot: DialogueSnapshot
returns: PerspectiveReport
---
1. Read the visible Folder inventory and compare each supplied snapshot document with its original
   relative path and exact text. Keep all nested source paths; the root is a pure directory reducer and
   commits no file changes. A mismatch identifies that original path for repair. Check request.caseId and snapshot.caseId identify the same closed source world. Retain the immutable
   snapshot document objects. Source identities and text are provenance, not conclusions to re-create.
2. Call intake(snapshot, request). Consume the returned source roles, original turns, participant scopes,
   temporal registry, receipts, physical facts and visible proposition vocabulary as separate contributions.
3. Call scopeTurns(source). Keep its actual quotation trees, reference bindings and act readings. A quoted
   original and a reporter's own speech occupy different frames even when their strings are adjacent.
4. Call assembleEpisodes(source, scoped). Its original episode order and availability effects determine
   when participant knowledge can change. Do not substitute document traversal order for the clock.
5. Call evolveViews(source, scoped, episodePlan). Retain actual per-episode deltas and view history; the
   root does not replace a false endorsed belief with the application's independently verified truth.
6. Call answerQuestions(source, scoped, episodePlan, evolution, request). The question's dimension and
   participant/time scope determine which ledger branch is admissible, not which convenient fact is true.
7. Call auditReport(source, scoped, episodePlan, evolution, answers, request). Return that actual audited
   report, including unresolved meanings and failed audits. An audit failure is not a license to fabricate
   a repaired trace. No message, key release or other external action is performed by this application.
