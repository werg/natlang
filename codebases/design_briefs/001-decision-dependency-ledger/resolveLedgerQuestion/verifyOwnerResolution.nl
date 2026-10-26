---
args: {"ledger": "DecisionLedger", "response": "LedgerOwnerResponse"}
returns: "ResolutionBinding"
---
Compare response.questionKey and every expected decision revision with the current open question and current ledger. A concurrent newer revision makes the response stale; return the actual intervening changes without overwriting. Verify signed owner or successor authority for the scope and effective date using the supplied evidence. Distinguish approval of wording, acknowledgment of receipt, and adoption of the actual alternative. Missing signature verification or ambiguous meaning remains unresolved. Return adoptedMeaning only when all bindings and the substantive response agree.
