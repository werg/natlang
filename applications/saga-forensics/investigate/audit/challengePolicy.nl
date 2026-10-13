---
description: Seek source-backed counterexamples to identity, repair safety and minimality.
args:
  request: InvestigationRequest
  index: EvidenceIndex
  correlation: Correlation
  ledger: EffectLedger
  repairResult: RepairPolicy
  accounting: BranchAccounting[]
returns: AuditFinding[]
---
1. Challenge key identity: inspect same raw keys across tenants and epochs, semantic retries after restarts and
   incompatible reuse. Check that one source clause, rather than string equality, justifies each merge/separation.
2. Challenge evidence strength: compare accepted requests, repeated receipts, printed manifests and signed
   handovers. Identify any ledger quantity that relies on a weaker observation than its source contract requires.
3. Challenge time-sensitive entitlement: compare cancellation and handover in the same normalized clock.
   A local timestamp that looks earlier must not overturn a later global cancellation without a mapping witness.
4. Challenge compensation scope: try the tempting full refund, deposit refund, whole-reservation release and
   maintenance release. For each relevant temptation, identify a concrete rule or counterexample forbidding it.
5. Challenge uncertainty: try skipping observation, using an early empty query or blindly retrying. If a supported
   world would then double-refund or fail an entitlement, return that world and its source-backed violation.
6. Challenge optimality: derive a lower bound from independent required release/refund writes in each outcome
   world. Compare the maximum lower bound to candidate.worstCaseWrites. A lower bound counts distinct business
   effects, not an arbitrary number of generated action nodes or a preference for immediate execution.
7. Check each accounting result satisfies the postconditions and preserves protected ownership. Equivalent
   independent action timing is allowed when dependencies and outcomes are unchanged.
8. Return defect for a concrete contradiction, unresolved for a missing deciding premise, and defended for a
   source-backed challenge that the report survives. Keep exact witnesses; do not declare success from style,
   the number of functions called, or agreement with a hidden expected-answer file.
