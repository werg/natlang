---
description: Classify a business obligation from all surviving proofs and scoped defeat alternatives.
args:
  key: ObligationKey
  proofs: Justification[]
  defeats: DefeatDecision[]
  snapshot: Snapshot
returns: LedgerEntry
---
1. Select proofs for this exact trip/debtor/beneficiary/action key and their targeted defeat decisions.
   Keep independent proofs separate until classifying the shared required performance.
2. If a definitely supported proof survives every relevant definite and possible defeater, the duty
   is active. Defeating another alternative proof does not remove this surviving justification.
3. If all source-supported duty proofs are definitely defeated, classify defeated. If decisive support
   or defeat is null and no definite survivor settles the result, classify unresolved.
4. If no supported or possible source proof exists, classify unsupported and preserve the failed
   authority/scope explanation. Do not confuse unsupported with a valid duty that was later exempted.
5. For tolls collect distinct source-supported amount alternatives, including0 for a supported
   exemption possibility. A defeated toll has[0]. Never average uncertain amounts into an invented debt.
6. For passage return permissionRequired=true for an active permission duty,false for a definitely
   defeated one and null when unresolved. Notice/translation use status; their amount fields are null.
7. Return supportingProofIds, defeatingDecisionIds, reason and witnesses. Include every independent
   notice justification even when one performance satisfies them all.
