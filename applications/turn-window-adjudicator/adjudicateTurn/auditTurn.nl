---
description: Audit returned timing, identity and effect histories without replacing them with a reconstructed answer.
args:
  request: TurnRequest
  source: TurnSource
  captured: CapturedTurn
  settled: WindowState
returns: TurnCertificate
---
1. Read the actual settled windows in committed order. Check exact snapshot continuity, slot uniqueness
   and source-inventoried capacities. Preserve any unfinished or unresolved state instead of claiming a
   completed turn because the current board happens to resemble one possible expected result.
2. Call replayIdentity(captured, settled, source) and consume all returned findings. This independent
   join tests instance continuity, controller basis and old/new garden stays against actual changes.
3. For each committed window call challengeTiming(window, settled.windows, captured, source). Consume
   the scoped judgment's defended, defect and unresolved findings into this certificate's audit.
4. Inspect that each final resolving part has a source-linked replacement history, each departure has
   contributing resolved damage, and each queued consequence has an actual eligible TriggerDecision.
   A valid negative trigger judgment is evidence, not a missing event to fabricate for completeness.
5. Verify source coverage against the original manifest and exact clause IDs. These mechanical domain
   checks do not establish entailment; preserve semantic objections returned by either child review.
6. Return caseId, actual captured announcements, settled finalBoard/windows, all actual trigger decisions,
   findings, unresolved and sourceCoverage. Keep original IDs exact and generated trace references
   internally consistent. The certificate reports this execution proposal, including its defects.
