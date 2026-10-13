---
description: Discharge one obligation in every supported outcome branch using assessed action extensions.
args:
  state: RepairSearchState
returns: RepairSearchState
---
1. If no obligation remains, return state. Otherwise remove its first ID and retrieve that Obligation. Keep all
   already-covered obligation actions and every observation branch intact while expanding this one.
2. For each surviving candidate, propose a release of exactly releaseQuantity when positive. Bind it to the
   reservation effect and cancellation cause; an unused reservation is the subject, not the whole warehouse.
3. For each candidate branch, compute the outstanding refund: total entitlement minus confirmed same-cause
   refund minus any uncertain refund whose branch condition says committed. Treat absent as zero only here,
   after its observation establishes final absence. A negative outstanding amount is a conflict, not a new charge.
4. For a positive outstanding amount, derive the refund key and tenant from the source-authorized new action or
   original uncertain attempt. A retry after final absence preserves its original key and referenced payment.
5. Construct proposed RepairAction records with unique actionIds and exact cause/effect references. A common
   release or refund appears in commonActions only if it is valid in every branch with identical business fields.
6. For each proposed action, call assessAction(action, obligation, branchConditions, state.context). Consume
   permitted and necessaryQuantity: retain only authorized quantities, otherwise reject this candidate extension
   and append the source-backed reason. Do not silently clamp a disallowed quantity to make it pass.
7. When there are multiple source-authorized extensions, retain each separately for later objective comparison.
   Independent action order need not be permuted; it represents the same candidate policy.
8. Set dependency IDs only for actual source prerequisites. A refund contingent on an observation remains in
   that answer branch; actions with no order dependency have empty dependsOn even if listed after another.
9. Mark the obligation covered only after every supported branch meets its required release and refund outcome.
   Recompute worstCaseWrites from the constructed actions. Keep source witnesses on each action.
10. Return state with the shorter remaining list, surviving candidates and accumulated rejected extensions.
    If all candidates are rejected, later steps may shrink the remaining list while preserving those reasons.
