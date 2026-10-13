---
description: Derive rollback obligations, select discriminating observations and search legal repair policies.
args:
  request: InvestigationRequest
  index: EvidenceIndex
  correlation: Correlation
  ledger: EffectLedger
returns: RepairPolicy
---
1. Call deriveObligations(request, index, correlation, ledger). Its result owns cancellation quantities, protected
   effects, current stock and confirmed cash. Do not derive a new obligation inside policy search.
2. Call chooseObservation(index, correlation, ledger, obligationResult). This selects only queries whose possible
   answers distinguish the repairs needed by the unresolved outcome worlds.
3. Assemble SearchContext from these results and call searchPolicy(context). Keep the returned candidate and
   all blocked reasons. A missing observation or unresolved entitlement can make the repair partially blocked.
4. Return the selected observation plan, candidate policy, obligation graph and blocked reasons. The final audit
   will independently evaluate each branch; this stage does not mark its own policy verified.
5. Source obligations are binding even when a different policy uses fewer writes. Optimize only among policies
   that satisfy them and preserve the protected inventory, custody and deposits in every supported world.
6. No policy action is executed here. Effects remain proposal data with source-based quantities and references.
