---
description: Apply namespace guarantees without conflating restart, tenant and epoch.
args:
  left: Attempt
  right: Attempt
  payload: PayloadRelation
  clauses: EvidenceAtom[]
returns: IdentityDecision
---
1. Select the identity guarantee for the requests' system. Interpret its named scope fields exactly, including
   whether a process restart preserves or changes the namespace. Keep the supporting clause witnesses.
2. Compare the required fields on left and right. A known different field makes the requests independent even
   when raw key bytes and business payload happen to match. State the distinguishing field in reason.
3. If a required scope field is unknown, return unresolved unless an explicit source statement settles identity.
4. When all required scope fields and key match, use payload.sameMeaning: true gives duplicate; false gives
   incompatible; null gives unresolved. Carry the payload and namespace witnesses together.
5. Incompatible reuse does not replace a prior successful request. Whether the incompatible attempt was
   rejected is an observation checked by the parent and ledger, not an effect invented by this function.
6. Return leftAttemptId, rightAttemptId, relation, scopeFields, reason and witnesses. Do not combine unrelated
   bank and warehouse namespaces merely because their source records use similar key strings.
