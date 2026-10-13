---
description: Judge business payload equivalence, then apply the separately interpreted key scope.
args:
  left: Attempt
  right: Attempt
  clauses: EvidenceAtom[]
returns: IdentityDecision
---
1. Select only the contract atoms defining business payload meaning and the two attempts' witnesses.
   Form a compact local context with the applicable clauses and the operation fields that can affect meaning.
2. In eval, declare a local PayloadJudgment type with sameMeaning: boolean | null, differingFields: string[],
   explanation: string and witnesses: Witness[]. Create a dynamic NL literal with nl.with<PayloadJudgment>
   capturing that local context. Give it the changing input {left, right} and directly await its returned object.
3. Its instructions are: compare these two business requests under the captured payload rules; distinguish
   cosmetic wording from changed order, operation kind, amount or quantity; retain uncertainty when a required
   field is unstated; return the fields that make the requests meaningfully different and exact witnesses.
4. Consume the literal's actual result as PayloadRelation. A same-meaning result is necessary for a duplicate,
   but it is not sufficient: equal payloads can occur in independent tenants or namespace epochs.
5. Call resolveKeyScope(left, right, payloadResult, clauses). Return that child's IdentityDecision unchanged.
6. Do not replace the literal with a string-equality helper or invoke a finite-choice operation on its structured
   object. The direct awaited judgment supplies the payload premise used by the scope resolver.
7. When no payload rule applies, the literal still returns an explicit unknown or justified ordinary request
   interpretation; the caller may not turn an empty decision list into permission to merge.
